import numpy as np
import pandas as pd
import torch,time,os,pickle,random
from torch import nn as nn
from nnLayer import *
from metrics import *
from collections import Counter,Iterable
from sklearn.model_selection import StratifiedKFold,KFold
from torch.backends import cudnn
from tqdm import tqdm
from torchvision import models
from pytorch_lamb import lamb
from torch.utils.data import DataLoader,Dataset
import torch.distributed

# 对抗训练
class FGM():
    def __init__(self, model, emb_name='emb'):
        self.model = model
        self.emb_name = emb_name
        self.backup = {}
 
    def attack(self, epsilon=1.):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
 
    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def dict_to_device(data, device):
    for k in data:
        if data[k] is not None:
            data[k] = data[k].to(device)
    return data

class BaseClassifier:
    def __init__(self, model):
        pass
    def calculate_y_logit(self, X):
        pass
    def calculate_y_prob(self, X):
        pass
    def calculate_y(self, X):
        pass
    def calculate_y_prob_by_iterator(self, dataStream):
        pass
    def calculate_y_by_iterator(self, dataStream):
        pass
    def calculate_loss(self, X, Y):
        pass
    def train(self, optimizer, trainDataSet, validDataSet=None, otherDataSet=None,
              batchSize=256, epoch=100, earlyStop=10, saveRounds=1, 
              isHigherBetter=False, metrics="LOSS", report=["LOSS"], 
              attackTrain=False, attackLayerName='emb', useEMA=False, prefetchFactor=2,
              savePath='model', shuffle=True, dataLoadNumWorkers=0, pinMemory=False, 
              trainSampler=None, validSampler=None, warmupEpochs=0, doEvalTrain=True, doEvalValid=True, doEvalOther=False):
        if attackTrain:
            self.fgm = FGM(self.model, emb_name=attackLayerName)
        if useEMA:
            ema = EMA(self.model, 0.999)
            ema.register()

        metrictor = self.metrictor if hasattr(self, "metrictor") else Metrictor()
        device = next(self.model.parameters()).device
        worldSize = torch.distributed.get_world_size() if self.mode>0 else 1
        # schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if isHigherBetter else 'min', factor=0.5, patience=20, verbose=True)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        itersPerEpoch = (len(trainDataSet)+batchSize-1)//batchSize
        warmSteps = int(itersPerEpoch * warmupEpochs / worldSize)
        decaySteps = int(itersPerEpoch*epoch / worldSize) - warmSteps
        schedulerRLR = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i:i/warmSteps if i<warmSteps else (decaySteps-(i-warmSteps))/decaySteps)
        trainStream = DataLoader(trainDataSet, batch_size=batchSize, shuffle=shuffle, num_workers=dataLoadNumWorkers, pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=trainSampler, prefetch_factor=prefetchFactor)
        evalTrainStream = DataLoader(trainDataSet, batch_size=batchSize, shuffle=False, num_workers=dataLoadNumWorkers, pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=trainSampler, prefetch_factor=prefetchFactor)

        mtc,bestMtc,stopSteps = 0.0,0.0 if isHigherBetter else 9999999999,0
        if validDataSet is not None: validStream = DataLoader(validDataSet, batch_size=batchSize, shuffle=False, num_workers=dataLoadNumWorkers, pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=validSampler, prefetch_factor=prefetchFactor)
        if otherDataSet is not None: otherStream = DataLoader(otherDataSet, batch_size=batchSize, shuffle=False, num_workers=dataLoadNumWorkers, pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=validSampler, prefetch_factor=prefetchFactor)
        st = time.time()
        for e in range(epoch):
            pbar = tqdm(trainStream)
            self.to_train_mode()
            for data in pbar:
                data = dict_to_device(data, device=device)
                loss = self._train_step(data, optimizer, attackTrain)
                if useEMA:
                    ema.update()
                schedulerRLR.step()
                pbar.set_description(f"Training Loss: {loss}; Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
            if useEMA:
                ema.apply_shadow()
            if ((self.mode>0 and torch.distributed.get_rank() == 0) or self.mode==0):
                if (validDataSet is not None) and ((e+1)%saveRounds==0):
                    print(f'========== Epoch:{e+1:5d} ==========')
                    with torch.no_grad():
                        self.to_eval_mode()
                        if doEvalTrain:
                            print(f'[Total Train]',end='')
                            # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                            # metrictor.show_res(res)
                            data = self.calculate_y_prob_by_iterator(evalTrainStream)
                            metrictor.set_data(data)
                            # print()
                            metrictor(report)
                        if doEvalValid:
                            print(f'[Total Valid]',end='')
                            # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                            # metrictor.show_res(res)
                            data = self.calculate_y_prob_by_iterator(validStream)
                            metrictor.set_data(data)
                            res = metrictor(report)
                            mtc = res[metrics]
                            print('=================================')
                            if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                                if (self.mode>0 and torch.distributed.get_rank() == 0) or self.mode==0:                
                                   print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                                   bestMtc = mtc
                                   self.save("%s.pkl"%savePath, e+1, bestMtc)
                                stopSteps = 0
                            else:
                                stopSteps += 1
                                if stopSteps>=earlyStop:
                                    print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                                    break
                        if doEvalOther:
                            print(f'[Total Other]',end='')
                            # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                            # metrictor.show_res(res)
                            data = self.calculate_y_prob_by_iterator(otherStream)
                            metrictor.set_data(data)
                            # print()
                            metrictor(report)
            if useEMA:
                ema.restore()
        if (self.mode>0 and torch.distributed.get_rank() == 0) or self.mode==0:
            with torch.no_grad():
                self.load("%s.pkl"%savePath)
                self.to_eval_mode()
                os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
                print(f'============ Result ============')
                print(f'[Total Train]',end='')
                # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                # metrictor.show_res(res)
                data = self.calculate_y_prob_by_iterator(evalTrainStream)
                metrictor.set_data(data)
                metrictor(report)
                print(f'[Total Valid]',end='')
                # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                # metrictor.show_res(res)
                data = self.calculate_y_prob_by_iterator(validStream)
                metrictor.set_data(data)
                res = metrictor(report)
                if otherDataSet is not None:
                    print(f'[Total Other]',end='')
                    # res = self.calculate_metrics_by_iterator(ds, metrictor, ignoreIdx, report)
                    # metrictor.show_res(res)
                    data = self.calculate_y_prob_by_iterator(otherStream)
                    metrictor.set_data(data)
                    metrictor(report)
                #metrictor.each_class_indictor_show(dataClass.id2lab)
                print(f'================================')
                return res
    def to_train_mode(self):
        self.model.train()  #set the module in training mode
        if self.collateFunc is not None:
            self.collateFunc.train = True
    def to_eval_mode(self):
        self.model.eval()
        if self.collateFunc is not None:
            self.collateFunc.train = False
    def _train_step(self, data, optimizer, attackTrain):
        loss = self.calculate_loss(data)
        loss.backward()
        if attackTrain:
            self.fgm.attack()
            lossAdv = self.calculate_loss(data)
            lossAdv.backward()
            self.fgm.restore()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        optimizer.zero_grad()
        return loss
    def save(self, path, epochs, bestMtc=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc, 'model':self.model.state_dict()}
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None):
        parameters = torch.load(path, map_location=map_location)
        self.model.load_state_dict(parameters['model'])
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))

class PadAndTknizeCollateFunc:
    def __init__(self, tokenizer, maskProb=0.15, groups=-1, duplicate=False, randomSample=False, dataEnhance=False, dataEnhanceRatio=0.1):
        self.tokenizer = tokenizer
        self.maskProb = maskProb
        self.train = False
        self.duplicate = duplicate
        self.randomSample = randomSample
        self.dataEnhance = dataEnhance
        self.dataEnhanceRatio = dataEnhanceRatio
        self.groups = groups
    def __call__(self, data):
        tmp = [i['sequence'] for i in data]

        if data[0]['tokenizedKgpSeqArr'] is not None:
            tokenizedKgpSeqArr = torch.cat([i['tokenizedKgpSeqArr'].unsqueeze(0) for i in data], dim=0)
            tokenizedSeqArr,maskPAD,posIdxArr = None,None,None
        elif self.groups>-1:
            if self.randomSample and self.train and random.random()<0.2: # 随机丢失15的核苷酸
                tmp = [[j for j in i if random.random()>0.15] for i in tmp]
            tokenizedKgpSeqArr = torch.tensor(self.tokenizer.tokenize_sentences_to_k_group(tmp, self.groups), dtype=torch.float32)
            tokenizedSeqArr,maskPAD,posIdxArr = None,None,None
        else:
            tokenizedKgpSeqArr = None

            if self.randomSample and self.train:
    #             posIdxArr = [np.sort(np.random.permutation(len(i))[:self.tokenizer.seqMaxLen]) for i in tmp]
                posIdxArr = [[np.int(random.random()*(len(i)-self.tokenizer.seqMaxLen)), self.tokenizer.seqMaxLen] if len(i)>self.tokenizer.seqMaxLen else [0,len(i)] for i in tmp]
                posIdxArr = [np.arange(i,i+j) for i,j in posIdxArr]
            else:
                posIdxArr = None
            tokenizedSeqArr,maskPAD = self.tokenizer.tokenize_sentences([np.array(i)[posIdx] for i,posIdx in zip(tmp,posIdxArr)] if (self.randomSample and posIdxArr is not None) else tmp, train=self.train) # batchSize × seqLen
            if posIdxArr is not None:
                seqMaxLen = min(max([len(i) for i in tmp]), self.tokenizer.seqMaxLen) if self.train else self.tokenizer.seqMaxLen
                posIdxArr = [[0]+(i+1).tolist()+(list(range(j['sLen']+1,seqMaxLen+1)) if (len(tmp)>1 or self.train) else []) for i,j in zip(posIdxArr,data)]
                posIdxArr = torch.tensor(posIdxArr, dtype=torch.float32)

            tokenizedSeqArr,maskPAD = torch.tensor(tokenizedSeqArr, dtype=torch.long),torch.tensor(maskPAD, dtype=torch.bool)
            if self.duplicate:
                tokenizedSeqArr = torch.cat([tokenizedSeqArr,tokenizedSeqArr], dim=0)
                maskPAD = torch.cat([maskPAD,maskPAD], dim=0)
                posIdxArr = torch.cat([posIdxArr,posIdxArr], dim=0)
            maskPAD = maskPAD.reshape(len(tokenizedSeqArr), 1, -1) & maskPAD.reshape(len(tokenizedSeqArr), -1, 1)

            seqLens = torch.tensor([min(i['sLen'],self.tokenizer.seqMaxLen) for i in data], dtype=torch.int32)
            if self.dataEnhance:
                for i in range(len(tokenizedSeqArr)): # 数据增强
                    if random.random()<self.dataEnhanceRatio/2: # 随机排列
                        tokenizedSeqArr[i][:seqLens[i]] = tokenizedSeqArr[i][:seqLens[i]][np.random.permutation(int(seqLens[i]))]
                    if random.random()<self.dataEnhanceRatio:  # 逆置
                        tokenizedSeqArr[i][:seqLens[i]] = tokenizedSeqArr[i][:seqLens[i]][range(int(seqLens[i]))[::-1]]

        tmp = self.tokenizer.tokenize_labels([i['label'] for i in data])
        labArr = np.zeros((len(tmp), self.tokenizer.labNum))
        for i in range(len(tmp)):
            labArr[i,tmp[i]] = 1
        tokenizedLabArr = torch.tensor(labArr, dtype=torch.float32) # batchSize
        if self.duplicate:
            tokenizedLabArr = torch.cat([tokenizedLabArr,tokenizedLabArr], dim=0)

        if self.tokenizer.useAAC:
            aacFea = torch.tensor(self.tokenizer.transform_to_AAC([i['sequence_'] for i in data]), dtype=torch.float32)
        else:
            aacFea = None

        return {'tokenizedSeqArr':tokenizedSeqArr, 'tokenizedKgpSeqArr':tokenizedKgpSeqArr, 'maskPAD':maskPAD, 'posIdxArr':posIdxArr,
                'tokenizedLabArr':tokenizedLabArr, 'aacFea':aacFea} # 

class SequenceClassifier(BaseClassifier):
    def __init__(self, model, collateFunc=None, mode=0, criterion=None):
        self.model = model
        self.collateFunc = collateFunc
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.mode = mode
        if mode==2:
            self.scaler = torch.cuda.amp.GradScaler()
        elif mode==3:
            import apex
    def calculate_y_logit(self, data):
        return self.model(data)
    def calculate_y_prob(self, data):
        Y_pre = self.calculate_y_logit(data)['y_logit']
        return {'y_prob':F.softmax(Y_pre, dim=-1)}
    def calculate_y(self, data):
        Y_pre = self.calculate_y_prob(data)['y_prob']
        return {'y_pre':(Y_pre>0.5).astype('int32')}
    def calculate_loss_by_iterator(self, dataStream):
        loss,cnt = 0,0
        for data in dataStream:
            loss += self.calculate_loss(data) * len(data['tokenizedSeqArr'])
            cnt += len(data['tokenizedSeqArr'])
        return loss / cnt
    def calculate_y_prob_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr,Y_preArr = [],[]
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            Y_pre,Y = self.calculate_y_prob(data)['y_prob'].detach().cpu().data.numpy().astype('float32'),data['tokenizedLabArr'].detach().cpu().data.numpy().astype('int32')
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return {'y_prob':Y_preArr, 'y_true':YArr}
    def calculate_loss(self, data):
        out = self.calculate_y_logit(data)
        Y = data['tokenizedLabArr']
        Y_logit = out['y_logit'].reshape(len(Y),-1)
        return self.criterion(Y_logit, Y)
    def _train_step(self, data, optimizer, attackTrain):
        optimizer.zero_grad()
        loss = self.calculate_loss(data)
        loss.backward()
        if attackTrain:
            self.fgm.attack()
            lossAdv = self.calculate_loss(data)
            lossAdv.backward()
            self.fgm.restore()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        return loss
    def save(self, path, epochs, bestMtc=None):
        if self.mode==0:
            model = self.model.state_dict()
        else:
            model = self.model.module.state_dict()
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc, 'model':model}
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None):
        parameters = torch.load(path, map_location=map_location)
        if self.mode==0:
            self.model.load_state_dict(parameters['model'])
        else:
            self.model.module.load_state_dict(parameters['model'])
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))

class SequenceMultiLabelClassifier(SequenceClassifier):
    def __init__(self, model, collateFunc=None, mode=0, criterion=None):
        self.model = model
        self.collateFunc = collateFunc
        self.criterion = nn.MultiLabelSoftMarginLoss() if criterion is None else criterion
        self.mode = mode
        if mode==2:
            self.scaler = torch.cuda.amp.GradScaler()
        elif mode==3:
            import apex
    def calculate_y_prob(self, data):
        Y_pre = self.calculate_y_logit(data)['y_logit']
        return {'y_prob':F.sigmoid(Y_pre)}

class SequenceMultiLabelClassifierWithContrastLearning(SequenceMultiLabelClassifier):
    def __init__(self, model, gama=0.2, alpha=0.1, collateFunc=None, mode=0):
        self.model = model
        self.collateFunc = collateFunc
        self.criterion = MultiLabelSoftMarginLossWithContrastLearning(gama, alpha)
        self.mode = mode
        if mode==2:
            self.scaler = torch.cuda.amp.GradScaler()
        elif mode==3:
            import apex
    def calculate_y_prob_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr,Y_preArr = [],[]
        vecArr, famArr = [],[]
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            tmp = self.calculate_y_logit(data)
            Y_pre,Y = F.sigmoid(tmp['y_logit']).detach().cpu().data.numpy().astype('float32'),data['tokenizedLabArr'].detach().cpu().data.numpy().astype('int32')
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        
            vec = tmp['p_vector'].detach().cpu().data.numpy().astype('float32')
            vecArr.append(vec)

        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        vecArr = np.vstack(vecArr).astype('float32')
        return {'y_prob':Y_preArr, 'y_true':YArr, 'p_vector':vecArr}
    def calculate_loss(self, data):
        out = self.calculate_y_logit(data)
        pVec = out['p_vector']
        Y = data['tokenizedLabArr']
        Y_logit = out['y_logit'].reshape(len(Y),-1)
        return self.criterion(pVec, Y_logit, Y)

class KGPDPLAM_alpha(nn.Module):
    def __init__(self, classNum, tknEmbedding=None, tknNum=None, 
                 embSize=64, dkEnhance=1, freeze=False, 
                 L=4, H=256, A=4, maxRelativeDist=7,
                 embDropout=0.2, hdnDropout=0.15, paddingIdx=-100):
        super(KGPDPLAM_alpha, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(tknEmbedding, freeze=freeze) if tknEmbedding is not None else nn.Embedding(tknNum, embSize, padding_idx=paddingIdx)
        self.dropout = nn.Dropout(p=embDropout)
        
        self.backbone = TransformerLayers_Realformer(L, feaSize=embSize if tknEmbedding is None else tknEmbedding.shape[1], dk=H//A, multiNum=A, maxRelativeDist=maxRelativeDist, hdnDropout=0.1, dkEnhance=dkEnhance)
        self.deepPseudoLabelwiseAttn = DeepPseudoLabelwiseAttention(embSize, classNum, L=-1, hdnDropout=hdnDropout, dkEnhance=1)
        self.fcLinear = MLP(embSize, 1)

    def forward(self, data):
        x = data['tokenizedKgpSeqArr'] @ self.embedding.weight # => batchSize × seqLen × embSize
        x = self.dropout(x)
        x,_,_,_,_,_ = self.backbone(x, None, None) # => batchSize × seqLen × embSize
        pVec = torch.mean(x, dim=1) # => batchSize × embSize
        pVec = pVec / torch.sqrt(torch.sum(pVec**2, dim=1, keepdim=True))
        x,attn = self.deepPseudoLabelwiseAttn(x) # => batchSize × classNum × embSize
        x = self.fcLinear(x).squeeze(dim=2) # => batchSize × classNum

        return {'y_logit':x, 'p_vector':pVec, 'attn':attn}

class KGPM_alpha(nn.Module):
    def __init__(self, classNum, tknEmbedding=None, tknNum=None, 
                 embSize=64, dkEnhance=1, freeze=False, 
                 L=4, H=256, A=4, maxRelativeDist=7,
                 embDropout=0.2, hdnDropout=0.15, paddingIdx=-100):
        super(KGPM_alpha, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(tknEmbedding, freeze=freeze) if tknEmbedding is not None else nn.Embedding(tknNum, embSize, padding_idx=paddingIdx)
        self.dropout = nn.Dropout(p=embDropout)
        
        self.backbone = TransformerLayers_Realformer(L, feaSize=embSize if tknEmbedding is None else tknEmbedding.shape[1], dk=H//A, multiNum=A, maxRelativeDist=maxRelativeDist, hdnDropout=0.1, dkEnhance=dkEnhance)
        self.fcLinear = MLP(embSize, classNum)

    def forward(self, data):
        x = data['tokenizedKgpSeqArr'] @ self.embedding.weight # => batchSize × seqLen × embSize
        x = self.dropout(x)
        x,_,_,_,_,_ = self.backbone(x, None, None) # => batchSize × seqLen × embSize
        pVec = torch.mean(x, dim=1) # => batchSize × embSize
        pVec = pVec / torch.sqrt(torch.sum(pVec**2, dim=1, keepdim=True))
        x,_ = torch.max(x, dim=1) # => batchSize × embSize
        x = self.fcLinear(x) # => batchSize × classNum

        return {'y_logit':x, 'p_vector':pVec}

class CNN(nn.Module):
    def __init__(self, classNum, tknEmbedding=None, tknNum=None, 
                 embSize=64, hdnSize=64, contextSizeList=[1,3,5], freeze=False,
                 embDropout=0.2, hdnDropout=0.15, paddingIdx=-100):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(tknEmbedding, freeze=freeze) if tknEmbedding is not None else nn.Embedding(tknNum, embSize, padding_idx=paddingIdx)
        self.dropout = nn.Dropout(p=embDropout)
        self.cnn = TextCNN(embSize if tknEmbedding is None else tknEmbedding.shape[1], hdnSize, contextSizeList, reduction='pool', ln=True, actFunc=nn.ReLU, name='textCNN')
        self.fcLinear = MLP(hdnSize*len(contextSizeList), classNum)
    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = self.dropout(x)
        x = self.cnn(x)
        x = self.fcLinear(x)
        return {'y_logit':x}

class RNN(nn.Module):
    def __init__(self, classNum, tknEmbedding=None, tknNum=None, 
                 embSize=64, hdnSize=64, freeze=False,
                 embDropout=0.2, hdnDropout=0.15, paddingIdx=-100):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(tknEmbedding, freeze=freeze) if tknEmbedding is not None else nn.Embedding(tknNum, embSize, padding_idx=paddingIdx)
        self.dropout = nn.Dropout(p=embDropout)
        self.lstm = TextLSTM(embSize if tknEmbedding is None else tknEmbedding.shape[1], hdnSize, num_layers=1, ln=True)
        self.fcLinear = MLP(hdnSize*2, classNum)
    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = self.dropout(x)
        x = self.lstm(x)
        x,_ = torch.max(x, dim=1)
        x = self.fcLinear(x)
        return {'y_logit':x}

class FastText(nn.Module):
    def __init__(self, classNum, tknEmbedding=None, tknNum=None, 
                 embSize=64, freeze=False,
                 embDropout=0.2, paddingIdx=-100):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(tknEmbedding, freeze=freeze) if tknEmbedding is not None else nn.Embedding(tknNum, embSize, padding_idx=paddingIdx)
        self.dropout = nn.Dropout(p=embDropout)
        self.fcLinear = MLP(embSize, classNum)
    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr'])
        x = self.dropout(x)
        x,_ = torch.max(x, dim=1)
        x = self.fcLinear(x)
        return {'y_logit':x}

class BPNN(nn.Module):
    def __init__(self, classNum, inSize, hdnList=[128], dropout=0.2):
        super(BPNN, self).__init__()
        self.fcLinear = MLP(inSize, classNum, hiddenList=hdnList, dropout=dropout, inBn=True, dpEveryLayer=True)
    def forward(self, data):
        x = self.fcLinear(data['aacFea'])
        return {'y_logit':x}
