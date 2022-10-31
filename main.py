# from datasets import load_dataset
from nltk.tokenize import wordpunct_tokenize
from transformers import BertConfig
from model import Model
import torch
import torch.nn as nn
import time
from mind_data import load_mind


# dataset = load_dataset('imdb')
dataset = load_mind('mind')

text=[]
label=[]
for row in dataset['train']['text']+dataset['test']['text']:
    text.append(wordpunct_tokenize(row.lower()))
for row in dataset['train']['label']+dataset['test']['label']:
    label.append(row)

word_dict={'PADDING':0}
for sent in text:    
    for token in sent:        
        if token not in word_dict:
            word_dict[token]=len(word_dict)

news_words = []
for sent in text:       
    sample=[]
    for token in sent:     
        sample.append(word_dict[token])
    sample = sample[:256]
    news_words.append(sample+[0]*(256-len(sample)))

import numpy as np
news_words=np.array(news_words,dtype='int32') 
label=np.array(label,dtype='int32') 

index=np.arange(len(label))
print(index.shape)
train_index=index[:len(dataset['train']['label'])]
np.random.shuffle(train_index)
test_index=index[len(dataset['train']['label']):]

import os
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config=BertConfig.from_json_file('fastformer.json')
config.num_labels = dataset['class_num']

def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot

model = Model(config, word_dict, True)
print(model)
import torch.optim as optim
optimizer = optim.Adam([ {'params': model.parameters(), 'lr': 1e-3}])
model.cuda()

best_acc = 0
for epoch in range(3):
    loss = 0.0
    accuary = 0.0

    start = time.time()
    for cnt in range(len(train_index)//64):

        log_ids=news_words[train_index][cnt*64:cnt*64+64,:256]
        targets= label[train_index][cnt*64:cnt*64+64]

        log_ids = torch.LongTensor(log_ids).cuda(non_blocking=True)
        targets = torch.LongTensor(targets).cuda(non_blocking=True)
        bz_loss, y_hat = model(log_ids, targets)
        loss += bz_loss.data.float()
        accuary += acc(targets, y_hat)
        unified_loss=bz_loss
        optimizer.zero_grad()
        unified_loss.backward()
        optimizer.step()

        if cnt % 100== 0:
            print( ' Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(cnt * 64, loss.data / (cnt+1), accuary / (cnt+1)))
    end = time.time()
    print('time cost:', end-start)

    model.eval()
    allpred=[]
    for cnt in range(len(test_index)//64+1):
    
        log_ids=news_words[test_index][cnt*64:cnt*64+64,:256]
        targets= label[test_index][cnt*64:cnt*64+64]
        log_ids = torch.LongTensor(log_ids).cuda(non_blocking=True)
        targets = torch.LongTensor(targets).cuda(non_blocking=True)
        bz_loss2, y_hat2 = model(log_ids, targets)
        allpred+=y_hat2.to('cpu').detach().numpy().tolist()
        
    y_pred=np.argmax(allpred,axis=-1)
    y_true=label[test_index]
    from sklearn.metrics import *
    new_acc = accuracy_score(y_true, y_pred)
    print(new_acc)

    if new_acc > best_acc:
        best_acc = new_acc
        torch.save(model.state_dict(), 'ff.pt')
        print('save model ...')

    model.train()