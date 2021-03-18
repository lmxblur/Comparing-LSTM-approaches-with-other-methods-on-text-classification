import torch
import torch.nn as nn

class RNN_classifier(nn.Module):
    def __init__(self,hidden,layers_size,binary = True):
        super(RNN_classifier,self).__init__()

        #self.batchsize = batchsize
        self.hidden = hidden
        self.layers_size = layers_size

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_model.vectors))
        self.hidden=300

        self.lstm = nn.LSTM(input_size = self.hidden, hidden_size = self.hidden,num_layers= self.layers_size,batch_first= True)

        self.linear1 = nn.Linear(self.hidden,128)
        self.linear2 = nn.Linear(128,1)
        self.linear3 = nn.Linear(128,3)


        self.binary = binary

    def forward(self,input):
        h = torch.zeros((self.layers_size,input.size(0),self.hidden))
        c = torch.zeros((self.layers_size,input.size(0),self.hidden))

        # init the h0 and c0 using xavier normal

        nn.init.xavier_normal_(h)
        nn.init.xavier_normal_(c)

        output = self.embedding(input)

        output,_ = self.lstm(output)

        output = nn.Dropout(0.5)(output)

        output = nn.ReLU()(self.linear1(output[:,-1,:]))

        output = nn.Dropout(0.5)(output)

        if self.binary:
            output = nn.Sigmoid()(self.linear2(output))
        else:
            output = self.linear3(output)

        return output 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

import gensim

data = pd.read_csv('winemag-data-130k-v2.csv').iloc[:,1:]
data2 = pd.read_csv('winemag-data_first150k.csv').iloc[:,1:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.model_selection import train_test_split
score_index = data2.points.notnull()
score = data2[score_index].iloc[:,[1,3]]
tag = [1 if single >=90 else 0 for single in score.points]
tag1 = [3 if single >=90  else 2 if single>=85 else 1 for single in score.points]
score['tag'] = tag
score['tag1'] = tag1
plt.figure()
sns.countplot(x= "tag",data=score)
plt.title('Tag counts')
plt.show()
plt.close()

Tr_vali, test_data = train_test_split(score, train_size = 0.7)

train_data, validation_data = train_test_split(Tr_vali, train_size = 0.8)

X_train = train_data.description
Y_train = train_data.tag

X_vali = validation_data.description
Y_vali = validation_data.tag

X_test = test_data.description
Y_test = test_data.tag


# switch to a different method of 

test_data.to_csv('/home/max/python/text_mining/project_2/data/test1.csv',index=False)
train_data.to_csv('/home/max/python/text_mining/project_2/data/train1.csv',index=False)
validation_data.to_csv('/home/max/python/text_mining/project_2/data/vali1.csv',index=False)

from torchtext.data import TabularDataset,Field,BucketIterator

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
tag1_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)

text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)

fields = [('description', text_field),(None,None) ,('tag', label_field),('tag1',tag1_field)]


folder = '/home/max/python/text_mining/project_2/data'


train,vali ,test = TabularDataset.splits(path=folder ,train='train1.csv',validation='vali1.csv' 
,test='test1.csv',format='CSV', fields=fields, skip_header=True)

# A larger batch_size means a faster training process 

# we need to load the pretrained word2vec wordembedding using gensim package, the bin is downloaded from google
embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
    '/home/max/Downloads/GoogleNews-vectors-negative300.bin',binary=True)
train_iterator = BucketIterator(train,batch_size=300,device=device,sort=False)
test_iterator = BucketIterator(test,batch_size=32,device=device,sort=False)
validation_iterator = BucketIterator(vali,batch_size=256,device=device,sort=False)

text_field.build_vocab(train)

def train(model,optimizer,loss_function=nn.BCELoss(),
          train_loader = train_iterator,epochs = 15,
          vali_loader  = validation_iterator,best_loss = 1e9):
    validation_cycle =300
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    model.train()
    for epoch in range(epochs):
        for ((description,_),tag,tag1),_ in train_loader:
            description = description.to(device)
            # reshape the label to two dimension using view(), original one dimension
            tag = tag.to(device).view(-1,1)
            tag1 = tag1.to(device).view(-1,1)

            output = model(description)
            if model.binary:
                loss = loss_function(output,tag)
            else:
                loss = loss_function(output,tag1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1
            # validation step:
            if global_step % validation_cycle==0:
                # eval mode
                model.eval()
                with torch.no_grad():
                    for ((description,_),tag,tag1),_ in vali_loader:
                        description = description.to(device)
                        tag = tag.to(device).view(-1,1)
                        tag1 = tag1.to(device).view(-1,1)
                        output = model(description)
                        loss = loss_function(output,tag)
                        valid_running_loss += loss
                
                training_loss = running_loss/validation_cycle

                valid_loss = valid_running_loss/len(vali_loader)

                train_loss_list.append(training_loss)
                valid_loss_list.append(valid_loss)
                global_steps_list.append(global_step)

                running_loss = 0
                valid_running_loss = 0
                model.train()
                
                print(
                    f'Epoch {epoch+1}, Global step {global_step}, Train loss:{training_loss}, Valid loss:{valid_loss}'
                    )
                
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    save_checkpoint('model.pt', model, optimizer, valid_loss)
        


        #print(running_loss)
    
def save_checkpoint(path, model, optimizer, valid_loss):    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, path)
    print('Model saved')


    
    
def load_checkpoint(path, model, optimizer):
    state_dict = torch.load(path, map_location=device)
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    print('loaded')
    return state_dict['valid_loss']    




model = RNN_classifier(200,1).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.003)


train(model,optimizer)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
def evaluate(model,test_loader):
  y_pred = []
  y_true = []

  model.eval()
  with torch.no_grad():
      for ((description,_),tag,tag1),_ in test_loader:
          description = description.to(device)
          tag = tag.to(device).view(-1,1)
          tag1 = tag1.to(device).view(-1,1)
          output = model(description)
          output = (output > 0.5).int()
          y_pred.extend(output.tolist())
          y_true.extend(tag.tolist())
      
      y_pred = ['Perfect' if float(single[0]) ==1 else 'With Drawbacks' for single in y_pred]
      y_true = ['High End' if float(single[0]) ==1 else 'Middle Range' for single in y_true]
      print(classification_report(y_true, y_pred, digits=4))
      
      print('#'*50)
      ef = confusion_matrix(y_true, y_pred)
      print(confusion_matrix(y_true, y_pred))
      ef = pd.DataFrame(ef,index=['Perfect','With Drawbacks'],columns=['Perfect','With Drawbacks'])
      plt.figure(figsize=(8,8))
      sns.heatmap(ef,annot=True,fmt='d')
      plt.xlabel('Predicted Label')
      plt.ylabel('True Label')
      plt.show()
      plt.close()
      
      print(pd.DataFrame(classification_report(y_true, y_pred,digits=2,output_dict=True)).T.to_latex())
      #return y_pred,y_true
      
device = torch.device('cpu')
model = RNN_classifier(200,1).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.003)

load_checkpoint('model.pt', model, optimizer)




evaluate(model,test_iterator)