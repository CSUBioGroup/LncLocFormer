from torch import nn as nn
from torch.nn import functional as F
import torch,time,os,random
import numpy as np
from collections import OrderedDict
from math import floor

class TextEmbedding(nn.Module):
    def __init__(self, embedding, dropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding,dtype=torch.float32), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=dropout/2)
        self.dropout2 = nn.Dropout(p=dropout/2)
        self.p = dropout
    def forward(self, x):
        # x: batchSize × seqLen
        if self.p>0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x

class BatchNorm1d(nn.Module):
    def __init__(self, inSize, name='batchNorm1d'):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(inSize)
        self.name = name
    def forward(self, x):
        return self.bn(x)

class TextCNN(nn.Module):
    def __init__(self, featureSize, filterSize, contextSizeList, reduction='pool', actFunc=nn.ReLU, bn=False, ln=False, name='textCNN'):
        super(TextCNN, self).__init__()
        moduleList = []
        bns,lns = [],[]
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Conv1d(in_channels=featureSize, out_channels=filterSize, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2),
            )
            bns.append(nn.BatchNorm1d(filterSize))
            lns.append(nn.LayerNorm(filterSize))
        if bn:
            self.bns = nn.ModuleList(bns)
        if ln:
            self.lns = nn.ModuleList(lns)
        self.actFunc = actFunc()
        self.conv1dList = nn.ModuleList(moduleList)
        self.reduction = reduction
        self.batcnNorm = nn.BatchNorm1d(filterSize)
        self.bn = bn
        self.ln = ln
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [conv(x).transpose(1,2) for conv in self.conv1dList] # => scaleNum * (batchSize × seqLen × filterSize)

        if self.bn:
            x = [b(i.transpose(1,2)).transpose(1,2) for b,i in zip(self.bns,x)]
        elif self.ln:
            x = [l(i) for l,i in zip(self.lns,x)]
        x = [self.actFunc(i) for i in x]

        if self.reduction=='pool':
            x = [F.adaptive_max_pool1d(i.transpose(1,2), 1).squeeze(dim=2) for i in x]
            return torch.cat(x, dim=1) # => batchSize × scaleNum*filterSize
        elif self.reduction=='None':
            return x # => scaleNum * (batchSize × seqLen × filterSize)

class TextLSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, ln=False, reduction='none', name='textBiLSTM'):
        super(TextLSTM, self).__init__()
        self.name = name
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=True, batch_first=True, num_layers=num_layers, dropout=dropout)
        if ln:
            self.layerNorm =nn.LayerNorm(hiddenSize*2)
        self.ln = ln
        self.reduction = reduction

    def forward(self, x):
        # x: batchSizeh × seqLen × feaSize
        output, hn = self.biLSTM(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if self.ln:
            output = self.layerNorm(output)
        if self.reduction=='pool':
            return torch.max(output, dim=1)[0]
        elif self.reduction=='none':
            return output # output: batchSize × seqLen × hiddenSize*2
    def orthogonalize_gate(self):
        nn.init.orthogonal_(self.biLSTM.weight_ih_l0)
        nn.init.orthogonal_(self.biLSTM.weight_hh_l0)
        nn.init.ones_(self.biLSTM.bias_ih_l0)
        nn.init.ones_(self.biLSTM.bias_hh_l0)

class FastText(nn.Module):
    def __init__(self, feaSize, name='fastText'):
        super(FastText, self).__init__()
        self.name = name
    def forward(self, x, xLen):
        # x: batchSize × seqLen × feaSize; xLen: batchSize
        x = torch.sum(x, dim=1) / xLen.float().view(-1,1)
        return x

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.0, bnEveryLayer=False, dpEveryLayer=False, inBn=False, outBn=False, outAct=False, outDp=False, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        hiddens,bns = [],[]
        if inBn:
            self.startBN = nn.BatchNorm1d(inSize)
        for i,os in enumerate(hiddenList):
            hiddens.append( nn.Sequential(
                nn.Linear(inSize, os),
            ) )
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.inBn = inBn
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
    def forward(self, x):
        if self.inBn: x = self.startBN(x)
        for h,bn in zip(self.hiddens,self.bns):
            x = h(x)
            if self.bnEveryLayer:
                if len(x.shape)==3:
                    x = bn(x.transpose(1,2)).transpose(1,2)
                else:
                    x = bn(x)
            x = self.actFunc(x)
            if self.dpEveryLayer:
                x = self.dropout(x)
        x = self.out(x)
        if self.outBn: x = self.bns[-1](x)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x

class DeepPseudoLabelwiseAttention(nn.Module):
    def __init__(self, inSize, classNum, L=1, M=64, hdnDropout=0.1, actFunc=nn.ReLU, dkEnhance=4, recordAttn=False, name='DPLA'):
        super(DeepPseudoLabelwiseAttention, self).__init__()
        if L>-1:
            self.inLWA = nn.Linear(inSize, M)

            hdnLWAs,hdnFCs,hdnBNs,hdnActFuncs = [],[],[],[]
            for i in range(L):
                hdnFCs.append(nn.Linear(inSize,inSize))
                hdnBNs.append(nn.BatchNorm1d(inSize))
                hdnActFuncs.append(actFunc())
                hdnLWAs.append(nn.Linear(inSize, M))
            self.hdnLWAs = nn.ModuleList(hdnLWAs)
            self.hdnFCs = nn.ModuleList(hdnFCs)
            self.hdnBNs = nn.ModuleList(hdnBNs)
            self.hdnActFuncs = nn.ModuleList(hdnActFuncs)

        self.outFC = nn.Linear(inSize, inSize*dkEnhance)
        self.outBN = nn.BatchNorm1d(inSize*dkEnhance)
        self.outActFunc = actFunc()
        self.outLWA = nn.Linear(inSize*dkEnhance, classNum)

        self.dropout = nn.Dropout(p=hdnDropout)
        self.name = name
        self.L = L

        self.recordAttn = recordAttn
    def forward(self, x):
        # x: batchSize × seqLen × inSize
        if self.recordAttn:
            attn = None
        if self.L>-1:
            # input layer
            score = self.inLWA(x) # => batchSize × seqLen × M
            alpha = self.dropout(F.softmax(score,dim=1)) # => batchSize × seqLen × M
            if self.recordAttn:
                attn = alpha.detach().cpu().data.numpy()
            a_nofc = alpha.transpose(1,2) @ x # => batchSize × M × inSize

            # hidden layers
            score = 0
            for i,(lwa,fc,bn,act) in enumerate(zip(self.hdnLWAs,self.hdnFCs,self.hdnBNs,self.hdnActFuncs)):
                a = fc(a_nofc) # => batchSize × M × inSize
                a = bn(a.transpose(1,2)).transpose(1,2) # => batchSize × M × inSize
                a_pre = self.dropout(act(a)) #  + a_nofc # => batchSize × M × inSize

                score = lwa(a_pre)# + score
                alpha = self.dropout(F.softmax(score,dim=1))
                if self.recordAttn:
                    attn @= alpha.detach().cpu().data.numpy()
                a_nofc = alpha.transpose(1,2) @ a_pre + a_nofc # => batchSize × M × inSize

            a_nofc = self.dropout(a_nofc)
        else:
            a_nofc = x 

        # output layers
        if self.L>-1:
            a = self.outFC(a_nofc) # => batchSize × M × inSize
            a = self.outBN(a.transpose(1,2)).transpose(1,2) # => batchSize × M × inSize
            a = self.dropout(self.outActFunc(a)) # => batchSize × M × inSize
        else:
            a = a_nofc

        score = self.outLWA(a) # => batchSize × M × classNum
        alpha = self.dropout(F.softmax(score,dim=1)) # => batchSize × M × classNum
        if self.recordAttn:
            if attn is None:
                attn = alpha.detach().cpu().data.numpy()
            else:
                attn @= alpha.detach().cpu().data.numpy()
        x = alpha.transpose(1,2) @ a # => batchSize × classNum × inSize
        
        return x,attn if self.recordAttn else None

class SelfAttention_Realformer(nn.Module):
    def __init__(self, feaSize, dk, multiNum, maxRelativeDist=7, dkEnhance=1, dropout=0.1, name='selfAttn'):
        super(SelfAttention_Realformer, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, dkEnhance*self.dk*multiNum)
        self.WK = nn.Linear(feaSize, dkEnhance*self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.dropout = nn.Dropout(p=dropout)
        if maxRelativeDist>0:
            self.relativePosEmbK = nn.Embedding(2*maxRelativeDist+1, multiNum)
            self.relativePosEmbB = nn.Embedding(2*maxRelativeDist+1, multiNum)
        self.maxRelativeDist = maxRelativeDist
        self.dkEnhance = dkEnhance
        self.name = name
    def forward(self, qx, kx, vx, preScores=None, maskPAD=None, posIdx=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen; posIdx: batchSize × seqLen
        B,L,C = qx.shape

        queries = self.WQ(qx).reshape(B,L,self.multiNum,self.dk*self.dkEnhance).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        keys    = self.WK(kx).reshape(B,L,self.multiNum,self.dk*self.dkEnhance).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        values  = self.WV(vx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        
        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × seqLen × seqLen
        
        # relative position embedding
        if self.maxRelativeDist>0:
            if posIdx is None:
                relativePosTab = torch.abs(torch.arange(0,L).reshape(1,-1,1) - torch.arange(0,L).reshape(1,1,-1)).float() # 1 × L × L
            else:
                relativePosTab = torch.abs(posIdx.reshape(B,L,1) - posIdx.reshape(B,1,L)).float() # B × L × L
            relativePosTab[relativePosTab>self.maxRelativeDist] = self.maxRelativeDist+torch.log2(relativePosTab[relativePosTab>self.maxRelativeDist]-self.maxRelativeDist).float()
            relativePosTab = torch.clip(relativePosTab,min=0,max=self.maxRelativeDist*2).long().to(qx.device)
            scores = scores * self.relativePosEmbK(relativePosTab).transpose(1,-1).reshape(-1,self.multiNum,L,L) + self.relativePosEmbB(relativePosTab).transpose(1,-1).reshape(-1,self.multiNum,L,L)

        # residual attention
        if preScores is not None:
            scores = scores + preScores

        if maskPAD is not None:
            #scores = scores*maskPAD.unsqueeze(dim=1)
            scores = scores.masked_fill((maskPAD==0).unsqueeze(dim=1), -2**32+1) # -np.inf

        alpha = self.dropout(F.softmax(scores, dim=3))

        z = alpha @ values # => batchSize × multiNum × seqLen × dk
        z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × seqLen × multiNum*dk

        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z,scores

class FFN_Realformer(nn.Module):
    def __init__(self, feaSize, dropout=0.1, actFunc=nn.GELU, name='FFN'):
        super(FFN_Realformer, self).__init__()
        self.layerNorm1 = nn.LayerNorm([feaSize])
        self.layerNorm2 = nn.LayerNorm([feaSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(feaSize, feaSize*4), 
                        actFunc(),
                        nn.Linear(feaSize*4, feaSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = self.layerNorm1(x + self.dropout(z)) # => batchSize × seqLen × feaSize

        ffnx = self.Wffn(z) # => batchSize × seqLen × feaSize
        return self.layerNorm2(z+self.dropout(ffnx)) # => batchSize × seqLen × feaSize
    
class Transformer_Realformer(nn.Module):
    def __init__(self, feaSize, dk, multiNum, maxRelativeDist=7, dropout=0.1, dkEnhance=1, actFunc=nn.GELU):
        super(Transformer_Realformer, self).__init__()
        self.selfAttn = SelfAttention_Realformer(feaSize, dk, multiNum, maxRelativeDist, dkEnhance, dropout)
        self.ffn = FFN_Realformer(feaSize, dropout, actFunc)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx,preScores,maskPAD,posIdx = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z,preScores = self.selfAttn(qx,kx,vx,preScores,maskPAD,posIdx) # => batchSize × seqLen × feaSize
        x = self.ffn(vx, z)
        return (x, x, x, preScores,maskPAD,posIdx) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class TransformerLayers_Realformer(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, maxRelativeDist=7, hdnDropout=0.1, dkEnhance=1, 
                 actFunc=nn.GELU, name='textTransformer'):
        super(TransformerLayers_Realformer, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_Realformer(feaSize, dk, multiNum, maxRelativeDist, hdnDropout, dkEnhance, actFunc)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
    def forward(self, x, maskPAD, posIdx):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maxPAD,posIdx = self.transformerLayers((x, x, x, None, maskPAD, posIdx))
        return (qx,kx,vx,scores,maxPAD,posIdx)# => batchSize × seqLen × feaSize

def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class LayerNormAndDropout(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='layerNormAndDropout'):
        super(LayerNormAndDropout, self).__init__()
        self.layerNorm = nn.LayerNorm(feaSize)
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x):
        return self.dropout(self.layerNorm(x))

class FFN(nn.Module):
    def __init__(self, featureSize, dropout=0.1, name='FFN'):
        super(FFN, self).__init__()
        self.layerNorm1 = nn.LayerNorm(featureSize)
        self.layerNorm2 = nn.LayerNorm(featureSize)
        self.Wffn = nn.Sequential(
                        nn.Linear(featureSize, featureSize*4), 
                        nn.ReLU(),
                        nn.Linear(featureSize*4, featureSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = x + self.dropout(self.layerNorm1(z)) # => batchSize × seqLen × feaSize
        ffnx = self.Wffn(z) # => batchSize × seqLen × feaSize
        return z+self.dropout(self.layerNorm2(ffnx)) # => batchSize × seqLen × feaSize

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gama=2, weight=-1, logit=True):
        super(FocalCrossEntropyLoss, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32), requires_grad=False)
        self.gama = gama
        self.logit = logit
    def forward(self, Y_pre, Y):
        if self.logit:
            Y_pre = F.softmax(Y_pre, dim=1)
        P = Y_pre[list(range(len(Y))), Y]
        if self.weight.shape!=torch.Size([]):
            w = self.weight[Y]
        else:
            w = torch.tensor([1.0 for i in range(len(Y))], device=self.weight.device)
        w = (w/w.sum()).reshape(-1)
        return (-w*((1-P)**self.gama * torch.log(P))).sum()

class MultiLabelCircleLoss(nn.Module):
    def __init__(self, transpose=False):
        super(MultiLabelCircleLoss, self).__init__()
        self.transpose = transpose
    def forward(self, Y_logit, Y):
        if self.transpose:
            Y_logit = Y_logit.transpose(0,1)
            Y = Y.transpose(0,1)
        loss,cnt = 0,0
        for yp,yt in zip(Y_logit,Y):
            neg = yp[yt==0]
            pos = yp[yt==1]
            loss += torch.log(1+torch.exp(neg).sum()) + torch.log(1+torch.exp(-pos).sum())
            #loss += torch.log(1+(F.sigmoid(neg)**2*torch.exp(neg)).sum()) + torch.log(1+((1-F.sigmoid(pos))**2*torch.exp(-pos)).sum())
            #loss += len(yp) * (torch.log(1+torch.exp(neg).sum()/len(neg)) + torch.log(1+torch.exp(-pos).sum()/len(pos)))
            cnt += 1
        return loss/cnt