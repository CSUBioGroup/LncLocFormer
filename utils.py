import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import torch,random,os
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from itertools import permutations
from sklearn.feature_extraction.text import CountVectorizer
# from gensim.models import Word2Vec,FastText
# from glove import Glove, Corpus
import logging,pickle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Tokenizer:
    def __init__(self, sequences, labels, seqMaxLen=512, maxTknNum=-1, useAAC=False, sequences_=None):
        print('Tokenizing the data...')
        cnt = 3
        id2tkn,tkn2id = ["[MASK]","[PAD]","[CLS]"],{"[MASK]":0,"[PAD]":1,"[CLS]":2}
        for seq in tqdm(sequences):
            for tkn in seq:
                if tkn not in tkn2id:
                    tkn2id[tkn] = cnt
                    id2tkn.append(tkn)
                    cnt += 1
            if maxTknNum>0 and len(id2tkn)>=maxTknNum:
                break
        self.id2tkn,self.tkn2id = id2tkn,tkn2id
        self.tknNum = cnt
        self.seqMaxLen = min(max([len(s) for s in sequences]), seqMaxLen)
        cnt = 0
        id2lab,lab2id = [],{}
        for labs in labels:
            for lab in labs:
                if lab not in lab2id:
                    lab2id[lab] = cnt
                    id2lab.append(lab)
                    cnt += 1
        labNum = cnt
        self.id2lab,self.lab2id = id2lab,lab2id
        self.labNum = labNum

        if useAAC:
            ctr = CountVectorizer(ngram_range=(1, 3), analyzer='char')
            pContFeat = ctr.fit_transform(sequences_).toarray().astype('float32')
            k1,k2,k3 = [len(i)==1 for i in ctr.get_feature_names()],[len(i)==2 for i in ctr.get_feature_names()],[len(i)==3 for i in ctr.get_feature_names()]

            pContFeat[:,k1] = (pContFeat[:,k1] - pContFeat[:,k1].mean(axis=1).reshape(-1,1))/(pContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
            pContFeat[:,k2] = (pContFeat[:,k2] - pContFeat[:,k2].mean(axis=1).reshape(-1,1))/(pContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
            pContFeat[:,k3] = (pContFeat[:,k3] - pContFeat[:,k3].mean(axis=1).reshape(-1,1))/(pContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
            mean,std = pContFeat.mean(axis=0),pContFeat.std(axis=0)+1e-8
            pContFeat = (pContFeat-mean) / std
            self.pContFeatVectorizer = {'transformer':ctr, 
                                        'mean':mean, 'std':std}
        self.useAAC = useAAC

        # test new trick
        from sklearn.preprocessing import OneHotEncoder
        self.ohe = OneHotEncoder()
        self.ohe.fit([[i] for i in range(self.tknNum)])

    def tokenize_sentences(self, sequences, train=False):
        seqMaxLen = min(max([len(i) for i in sequences]), self.seqMaxLen) if (train or len(sequences)==1) else self.seqMaxLen
        return [[self.tkn2id['[CLS]']]+[self.tkn2id[tkn] if tkn in self.tkn2id else self.tkn2id["[MASK]"] for tkn in seq[:seqMaxLen]]+[self.tkn2id['[PAD]']]*(seqMaxLen-len(seq)) for seq in sequences],[[1]+[1]*len(seq[:seqMaxLen])+[0]*(seqMaxLen-len(seq)) for seq in sequences]
    def tokenize_sentences_to_k_group(self, sequences, k):
        tmp = [F.adaptive_avg_pool1d(torch.tensor(self.ohe.transform([[self.tkn2id[tkn] if tkn in self.tkn2id else self.tkn2id["[MASK]"]] for tkn in seq]).toarray(),dtype=torch.float32).transpose(-1,-2), output_size=k).transpose(-1,-2).unsqueeze(0) for seq in sequences]
        return torch.cat(tmp, dim=0)
    def tokenize_labels(self, labels):
        return [[self.lab2id[lab] for lab in labs] for labs in labels]
    def transform_to_AAC(self, sequences):
        ctr = self.pContFeatVectorizer['transformer']
        mean,std = self.pContFeatVectorizer['mean'],self.pContFeatVectorizer['std']

        pContFeat = ctr.transform(sequences).toarray().astype('float32')
        k1,k2,k3 = [len(i)==1 for i in ctr.get_feature_names()],[len(i)==2 for i in ctr.get_feature_names()],[len(i)==3 for i in ctr.get_feature_names()]

        pContFeat[:,k1] = (pContFeat[:,k1] - pContFeat[:,k1].mean(axis=1).reshape(-1,1))/(pContFeat[:,k1].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k2] = (pContFeat[:,k2] - pContFeat[:,k2].mean(axis=1).reshape(-1,1))/(pContFeat[:,k2].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat[:,k3] = (pContFeat[:,k3] - pContFeat[:,k3].mean(axis=1).reshape(-1,1))/(pContFeat[:,k3].std(axis=1).reshape(-1,1)+1e-8)
        pContFeat = (pContFeat-mean) / std

        return pContFeat

    # def vectorize(self, sequences, method=["skipgram"], embSize=64, window=7, iters=10, batchWords=10000,
    #               workers=8, loadCache=True, suf=""):
    #     path = f'cache/{"-".join(method)}_d{embSize*len(method)}_{suf}.pkl'
    #     if os.path.exists(path) and loadCache:
    #         with open(path, 'rb') as f:
    #             self.embedding = pickle.load(f)
    #         print('Loaded cache from cache/%s'%path)
    #     else:
    #         corpus = [i+['[PAD]'] for i in sequences]
    #         embeddings = []
    #         if 'skipgram' in method:
    #             model = Word2Vec(corpus, min_count=0, window=window, vector_size=embSize, workers=workers, sg=1, epochs=iters, batch_words=batchWords)
    #             word2vec = np.zeros((self.tknNum, embSize), dtype=np.float32)
    #             for i in range(self.tknNum):
    #                 if self.id2tkn[i] in model.wv:
    #                     word2vec[i] = model.wv[self.id2tkn[i]]
    #                 else:
    #                     print('word %s not in word2vec.'%self.id2tkn[i])
    #                     word2vec[i] =  np.random.random(embSize)
    #             embeddings.append(word2vec)
    #         if 'glove' in method:
    #             gCorpus = Corpus()
    #             gCorpus.fit(corpus, window=window)
    #             model = Glove(no_components=embSize)
    #             model.fit(gCorpus.matrix, epochs=iters, no_threads=workers, verbose=True)
    #             model.add_dictionary(gCorpus.dictionary)
    #             word2vec = np.zeros((self.tknNum, embSize), dtype=np.float32)
    #             for i in range(self.tknNum):
    #                 if self.id2tkn[i] in model.dictionary:
    #                     word2vec[i] = model.word_vectors[model.dictionary[self.id2tkn[i]]]
    #                 else:
    #                     print('word %s not in word2vec.'%self.id2tkn[i])
    #                     word2vec[i] =  np.random.random(embSize)
    #             embeddings.append(word2vec)
    #         if 'fasttext' in method:
    #             model = FastText(corpus, vector_size=embSize, window=window, min_count=0, epochs=iters, sg=1, workers=workers, batch_words=batchWords)
    #             word2vec = np.zeros((self.tknNum, embSize), dtype=np.float32)
    #             for i in range(self.tknNum):
    #                 if self.id2tkn[i] in model.wv:
    #                     word2vec[i] = model.wv[self.id2tkn[i]]
    #                 else:
    #                     print('word %s not in word2vec.'%self.id2tkn[i])
    #                     word2vec[i] =  np.random.random(embSize)
    #             embeddings.append(word2vec)
    #         self.embedding = np.hstack(embeddings)

    #         with open(path, 'wb') as f:
    #             pickle.dump(self.embedding, f, protocol=4)

class lncRNA_loc_dataset(Dataset):
    def __init__(self, dataPath, k=1, mode="csv"):
        self.dataPath = dataPath
        if mode=="csv":
            data = pd.read_csv(dataPath)
            self.ids = data['Gene_ID'].tolist()
            tmp = ['-'*(k//2)+i+'-'*(k//2) for i in data['Sequence']]
            self.sequences = [[i[j-k//2:j+k//2+1] for j in range(k//2,len(i)-k//2)] for i in tmp]
            self.sequences_ = [i for i in data['Sequence']]
            self.labels = [i.split(';') for i in data['SubCellular_Localization'].tolist()]
        elif mode=="fasta":
            with open(dataPath, 'r') as f:
                tmp = f.readlines()
            names = [i.strip() for i in tmp[::2]]
            self.ids = ["".join(i[1:].split('|')[:-1]) for i in names]
            self.labels = [i.split('|')[-1].split(';') for i in names]
            tmp = [i.strip() for i in tmp[1::2]]
            self.sequences_ = tmp
            tmp = ['-'*(k//2)+i+'-'*(k//2) for i in tmp]
            self.sequences = [[i[j-k//2:j+k//2+1] for j in range(k//2,len(i)-k//2)] for i in tmp]
        self.sLens = [len(i) for i in tqdm(self.sequences)]

    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return {'id':self.ids[index],
                'sequence':self.sequences[index], 
                'sequence_':self.sequences_[index],
                'sLen':self.sLens[index],
                'label':self.labels[index],
                'tokenizedKgpSeqArr':self.tokenizedKgpSeqArr[index] if hasattr(self, 'tokenizedKgpSeqArr') else None}
    def cache_tokenizedKgpSeqArr(self, tokenizer, groups):
        self.groups = groups
        self.tokenizedKgpSeqArr = tokenizer.tokenize_sentences_to_k_group(self.sequences, groups)

class lncRNA_loc_dataset_old(Dataset):
    def __init__(self, dataPath, k=1):
        self.dataPath = dataPath
        with open(dataPath, 'r') as f:
            data = [i.split() for i in f.readlines()]
        tmp = [i[1] for i in data]
        self.sequences = [[i[j-k//2:j+k//2+1] for j in range(k//2,len(i)-k//2)] for i in tmp]
        self.labels = [[i[2]] for i in data]
        self.sLens = [len(i) for i in tqdm(self.sequences)]
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return {'sequence':self.sequences[index], 
                'sLen':self.sLens[index],
                'label':self.labels[index],
                'tokenizedKgpSeqArr':self.tokenizedKgpSeqArr[index] if hasattr(self, 'tokenizedKgpSeqArr') else None}
    def cache_tokenizedKgpSeqArr(self, tokenizer, groups):
        self.groups = groups
        self.tokenizedKgpSeqArr = tokenizer.tokenize_sentences_to_k_group(self.sequences, groups)

class DM3Loc_lncRNA_dataset(Dataset):
    def __init__(self, dataPath, labIndex):
        self.dataPath = dataPath
        with open(dataPath, 'r') as f:
            lines = f.readlines()
        self.names = [i for i in lines[::2]]
        self.sequences = [i for i in lines[1::2]]
        self.labels = [[k for j,k in zip(list(i.split(',')[0][1:]),labIndex) if j=='1'] for i in self.names]
        self.sLens = [len(i) for i in tqdm(self.sequences)]
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, index):
        return {'name':self.names[index],
                'sequence':self.sequences[index],
                'sLen':self.sLens[index],
                'label':self.labels[index]}
