import torch
from torch import nn, optim
import pandas as pd
import torch.nn.functional as F 

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        x = x.view(x.shape[0], -1)
        x = x.type(torch.FloatTensor)
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        return F.log_softmax(self.output(x) , dim=1)

def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()
    return test_loss, accuracy

def training(model, trainloader, testloader, criterion, optimizer, epochs = 10):
    valid_loss = []
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()
            out = model.forward(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            with torch.no_grad():
                test_loss, test_accuracy = validation(model, testloader, criterion)
                valid_loss.append(test_loss / len(trainloader))
            print("epoch {0}".format(e))
            print("Training loss = {0} ".format(running_loss / len(trainloader)))
            print("Test loss = {0} ".format(test_loss / len(trainloader)))
            print("Test accuracy = {0} % \n".format((test_accuracy / len(trainloader))*100))
    return valid_loss


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_set = torch.tensor(train.values)
test_set = torch.tensor(test.values)

#Splitting training set into batches of size 128
batch_size = 512
trainloader = []
n = len(train_set)//batch_size 
for i in range(n):
    trainloader.append( (train_set[i * batch_size : (i+1) * batch_size, 1::], train_set[i * batch_size : (i+1) * batch_size, 0]))
if(len(train_set)%batch_size != 0):
    trainloader.append((train_set[batch_size * n : len(train_set), 1::], train_set[batch_size * n : len(train_set), 0]))

#defining Network, loss function, optimizer
model = Classifier(784, 10, [512, 256, 128, 64], 0.2)
criterion = nn.NLLLoss()
#Adam optimization algorithm with learning rate = 0.001 and Beta1 = 0.9 and Beta2 = 0.999
optimizer = optim.Adam(model.parameters(), lr = 0.001) 

valid_loss = training(model, trainloader, trainloader, criterion, optimizer, 100 )
loss , acc = validation(model,trainloader, criterion)

import matplotlib.pyplot as plt
plt.plot(valid_loss, label='valid_loss')
plt.legend(frameon=False)       

print(acc / len(trainloader))

checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'state_dict': model.state_dict() }
torch.save(checkpoint, "checkpoint.pth")
output = model.forward(test_set)
ps = torch.exp(output)
pred = ps.max(1)[1]

f = open("submission.txt","w+")
f.write("ImageId,Label\n")
for i in range(len(test_set[:,0])):
    f.write("{0},{1}\n".format(i+1,pred[i]))









