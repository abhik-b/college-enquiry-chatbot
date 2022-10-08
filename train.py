import json
from nlp_utils import tokenize,stem,bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from neural_net import NeuralNet


with open('intents.json','r') as json_data:
    intents = json.load(json_data)


all_words=[]
tags=[intent['tag'] for intent in intents['intents']]
xy=[]
puncts=['?','!','.',',']


for intent in intents['intents']:
    for pattern in intent['patterns']: 
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,intent['tag']))

all_words=[stem(w) for w in all_words if w not in puncts]

all_words=sorted(set(all_words))
tags=sorted(set(tags))



X_train=[]
Y_train=[]


for (pattern_sentence,tag) in xy:
    bag=bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    label=tags.index(tag)
    Y_train.append(label)


X_train=np.array(X_train)
Y_train=np.array(Y_train)



class ChatDataSet(Dataset):

    def __init__(self, *args, **kwargs):
        self.n_samples=len(X_train)
        self.x_data=X_train
        self.y_data=Y_train

    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset=ChatDataSet()
train_loader=DataLoader(dataset=dataset,batch_size=8,shuffle=True,num_workers=0)



input_size=len(X_train[0])
hidden_size=8
output_size=len(tags)


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size,output_size)
model.to(device)


criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


for epoch in range(1000):
    for (words,labels) in train_loader:
        words=words.to(device)
        labels=labels.to(device)
        labels=labels.to(torch.long)
        outputs=model(words)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{8}, loss={loss.item():.4f}')


print(f'final loss,loss={loss.item():.4f}')



data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags
}


FILE='data.pth'
torch.save(data,FILE)
print(f'training complete. File saved to {FILE}')

