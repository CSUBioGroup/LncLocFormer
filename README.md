# LncLocFormer
a transformer-based deep learning model for multi-label lncRNA subcellular localization prediciton by using localization-specific mechanism.

## Requirements
numpy==1.22.3

scikit-learn==1.1.1

torch==1.13.1+cu117

## Usage

### 1. Import the package
```python
from utils import *
from DL_ClassifierModel import *
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold,KFold
from functools import reduce
SEED = 388014 # fix the random seed
```

### 2. Load the dataset
```python
totalDS = lncRNA_loc_dataset('../dataset/data.csv', k=3, mode='csv')
tokenizer = Tokenizer(totalDS.sequences, totalDS.labels, seqMaxLen=8196, sequences_=totalDS.sequences_)
# transform the binary label vector into decimal int and use it to do the stratified split
tknedLabs = []
for lab in totalDS.labels:
    tmp = np.zeros((tokenizer.labNum))
    tmp[[tokenizer.lab2id[i] for i in lab]] = 1
    tknedLabs.append( reduce(lambda x,y:2*x+y, tmp)//4 )
# get the test set
for i,j in StratifiedShuffleSplit(test_size=0.1, random_state=SEED).split(range(len(tknedLabs)), tknedLabs):
    testIdx = j
    break
testIdx_ = set(testIdx)
restIdx = np.array([i for i in range(len(totalDS)) if i not in testIdx_])

# cache the subsequence one-hot
totalDS.cache_tokenizedKgpSeqArr(tokenizer, groups=512)
```

### 3. Train the model by 5-fold cross-validation
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) # 5-fold cross validation
for i,(trainIdx,validIdx) in enumerate(skf.split(restIdx, np.array(tknedLabs)[restIdx])):
    # get the training set and validation set of fold i
    trainIdx,validIdx = restIdx[trainIdx],restIdx[validIdx]
    trainDS,validDS,testDS = torch.utils.data.Subset(totalDS,trainIdx),torch.utils.data.Subset(totalDS,validIdx),torch.utils.data.Subset(totalDS,testIdx)

    # initialize the LncLocFormer
    backbone = KGPDPLAM_alpha(tokenizer.labNum, tknEmbedding=None, tknNum=tokenizer.tknNum,  # torch.tensor(np.eye(tokenizer.tknNum), dtype=torch.float32)
                              embSize=128, dkEnhance=1, freeze=False,
                              L=8, H=256, A=8, maxRelativeDist=25,
                              embDropout=0.2, hdnDropout=0.1, paddingIdx=tokenizer.tkn2id['[PAD]']).cuda()
    model = SequenceMultiLabelClassifier(backbone, collateFunc=PadAndTknizeCollateFunc(tokenizer), mode=0)

    # set the learning rate and weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in backbone.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.001},
            {'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=3e-4, weight_decay=0.001)

    # train the model of fold i
    model.train(optimizer=optimizer, trainDataSet=trainDS, validDataSet=validDS, otherDataSet=testDS,
                batchSize=64, epoch=256, earlyStop=64, saveRounds=1, 
                isHigherBetter=True, metrics="MaAUC", report=["LOSS", "AvgF1", 'MiF', 'MaF', "LOSS", "MaAUC", 'MiAUC', 'MiP', 'MaP', 'MiR', 'MaR', "EachAUC", "EachAUPR"], 
                savePath=f'models/KGPDPLAM_alpha_cv{i}', shuffle=True, dataLoadNumWorkers=0, pinMemory=True, 
                warmupEpochs=4, doEvalTrain=False, prefetchFactor=2)
```

## Citation
Min Zeng, Yifan Wu, Yiming Li, Rui Yin, Chengqian Lu, Junwen Duan, Min Li*, “LncLocFormer: a Transformer-based deep learning model for multi-label lncRNA subcellular localization prediction by using localization-specific attention mechanism”.
