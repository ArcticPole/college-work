import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from timeit import default_timer as timer

feature_col=['firstBlood','firstTower',
     'firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills',
     't1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills',
     't2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
label_col=['winner']
lol=pd.read_csv("new_data.csv",header=0)
lol_test=pd.read_csv("test_set.csv",header=0)

#DT
###########
tic = timer()

train_set=lol[feature_col]
label_set=lol[label_col]-1

train_set_test=lol_test[feature_col]
label_set_test=lol_test[label_col]-1

DT=DecisionTreeClassifier()
dt=DT.fit(train_set,label_set)
y_pred=dt.predict(train_set_test)
dt_proba=dt.predict_proba(train_set_test)
print("accuracy of DT is :",accuracy_score(label_set_test,y_pred))
toc = timer()
print("the time consuming for DT is ",toc - tic)
###########

#ANN
###########
tic = timer()

ann_train_set_old=lol.drop(['gameId','creationTime','seasonId','winner'], axis=1).values
ann_label_set=lol['winner'].values-1

ann_train_set_old_test=lol_test.drop(['gameId','creationTime','seasonId','winner'], axis=1).values
ann_label_set_test=lol_test['winner'].values-1


#归一化
new=[]
for i in range(17):
    new.append(ann_train_set_old[:,i]/np.max(ann_train_set_old[:,i]))
ann_train_set=np.transpose(new).copy()
new_test=[]
for i in range(17):
    new_test.append(ann_train_set_old_test[:,i]/np.max(ann_train_set_old_test[:,i]))
ann_train_set_test=np.transpose(new_test).copy()

ann_x_train=torch.FloatTensor(ann_train_set)
ann_x_test=torch.FloatTensor(ann_train_set_test)
ann_y_train=torch.LongTensor(ann_label_set)
ann_y_test=torch.LongTensor(ann_label_set_test)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(in_features=17,out_features=30)
        self.output=nn.Linear(in_features=30,out_features=2)
    def forward(self,x):
        x=torch.sigmoid(self.fc1(x))
        x=self.output(x)
        x=F.softmax(x)
        return x

model=ANN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


epochs = 200
#loss_arr = []
for i in range(epochs):
    y_hat = model.forward(ann_x_train)
    loss = criterion(y_hat, ann_y_train)
    #loss_arr.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

predict_out=model(ann_x_test)
_,predict_y=torch.max(predict_out,1)
print("the accuracy of ann is ",accuracy_score(ann_y_test,predict_y))
toc = timer()
print("the time consuming for ann is ",toc - tic)
###########

#ensemble
###########
pro_mean_old=dt_proba+predict_out.detach().numpy()

pro_mean=[]
for i in range(len(pro_mean_old)):
    if pro_mean_old[i][0]<=pro_mean_old[i][1]:
        pro_mean.append(1)
    else:
        pro_mean.append(0)
#print(pro_mean)
pro_max=[]
for i in range(len(pro_mean)):
    try1=[dt_proba[i][1],predict_out[i][1]]
    max=try1[np.argmax(try1)]
    if max>=0.5:
        pro_max.append(1)
    else:
        pro_max.append(0)

print("the accuracy of mean ensemble is ",accuracy_score(ann_y_test,pro_mean))
print("the accuracy of max ensemble is ",accuracy_score(ann_y_test,pro_max))
###########