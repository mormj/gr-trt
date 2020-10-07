#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import pickle
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from random import shuffle
import matplotlib.pyplot as plt


# In[2]:


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Xd = pickle.load(open("RML2016.10a_dict.pkl", 'rb'), encoding='latin1')
snrs, mods = map(lambda j: sorted(
    list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
snrs =  [10,12,14,16,18]
# test_snrs =  [18]

# In[3]:


for mod in mods:
    for snr in snrs:
        # interleave I and Q
        X.append(Xd[(mod, snr)].transpose((0,2,1)).reshape((-1,256))  )
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)
print(X.shape)


# In[4]:


np.random.seed(2019)
n_examples = X.shape[0]
n_train = int(round(n_examples * 0.5))


# train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
# test_idx = list(set(range(0, n_examples))-set(train_idx))
# X_train = X[train_idx]
# X_test = X[test_idx]


# In[5]:

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)), yy] = 1  # ?
    return yy1

modidx = np.array([mods.index(x[0]) for x in lbl])
snr = np.array([x[1] for x in lbl])

idx = list(range(len(modidx)))
shuffle(idx)

X_train = X[idx[:n_train]]
Y_train = modidx[idx[:n_train]]
snr_train = snr[idx[:n_train]]

X_test = X[idx[n_train:]]
Y_test = modidx[idx[n_train:]]
snr_test = snr[idx[n_train:]]

X_train = torch.Tensor(X_train).to(DEVICE)
Y_train = torch.LongTensor(Y_train).to(DEVICE)
X_test = torch.Tensor(X_test).to(DEVICE)
Y_test = torch.LongTensor(Y_test).to(DEVICE)


# In[7]:

dr = 0.5
class VTCNN2(nn.Module):
    def __init__(self):
        super(VTCNN2, self).__init__()
        # self.reshape = torch.reshape((nsamples, 1, X_train.shape[1], X_train.shape[2]))
        self.conv1 = nn.Conv2d(1,256,(1,3),padding=(0,2))
        # nn.init.xavier_uniform_(self.conv1.weight, gain=1.0)
        self.drop1 = nn.Dropout(dr)
        self.drop2 = nn.Dropout(dr)
        self.drop3 = nn.Dropout(dr)
        self.conv2 = nn.Conv2d(256,80,(2,3),padding=(0,2))
        # nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(10560, 256)
        # nn.init.kaiming_normal_(self.dense1.weight)
        self.dense2 = nn.Linear(256, 11)
        # nn.init.kaiming_normal_(self.dense2.weight)
        # self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = x/x.max()
        x = torch.reshape(x,(-1,1,128,2))
        x = torch.transpose(x,2,3)
        x = self.drop1(F.relu(self.conv1(x)))
        x = self.drop2(F.relu(self.conv2(x)))
        x = self.flat(x)
        x = self.drop3(F.relu(self.dense1(x)))
        x = self.dense2(x)
        # x = self.sm(x)
        return x

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.dense1 = nn.Linear(256, 4096)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(4096, 64)
        self.dense3 = nn.Linear(64, 256)
        self.dense4 = nn.Linear(256, 16)
        self.dense5 = nn.Linear(16, 11)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = x/x.max()
        x = self.relu1(self.dense1(x))
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        # x = self.sm(x)
        return x


class FCNSMALL(nn.Module):
    def __init__(self):
        super(FCNSMALL, self).__init__()
        self.dense1 = nn.Linear(256, 512)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(512, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, 16)
        self.dense5 = nn.Linear(16, 11)
        # self.sm = nn.Softmax()

    def forward(self, x):
        x = x/x.max()
        x = self.relu1(self.dense1(x))
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        # x = self.sm(x)
        return x

# In[13]:


def train():
    net = model.train()
    nsamples = X_train.shape[0]
    current_sample = 0
    while(current_sample < nsamples):
        batch = min(batch_size, nsamples-current_sample)
        xx = X_train[current_sample:(current_sample+batch)]
        yy = Y_train[current_sample:(current_sample+batch)]
        # yy = cls_train[current_sample:(current_sample+batch)]
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()
        
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = net(xx)

        # Compute and print loss.
        loss = loss_fn(y_pred, yy)
        # loss = nn.NLLLoss()(torch.log(y_pred), yy)
        
        if current_sample == 0:
            print(loss.item())

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        current_sample += batch


# In[14]:


def test(epoch):
    net = model.eval()
    test_loss = 0
    correct = 0
    nsamples = X_test.shape[0]
    with torch.no_grad():
        current_sample = 0
        while(current_sample < nsamples):
            batch = min(batch_size, nsamples-current_sample)
            xx = X_test[current_sample:(current_sample+batch)]
            yy = Y_test[current_sample:(current_sample+batch)]
            # zz = cls_test[current_sample:(current_sample+batch)]

            output = net(xx)
            
            test_loss += loss_fn(output, yy.long()).item()  # sum up batch loss
            pred = output.argmax(dim=1)  # get the index of the max log-probability
            # correct += pred.eq(yy).sum().item()
            correct += pred.eq(yy).sum().item()
            current_sample += batch


    test_loss /= nsamples

    print('\n({}) Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, nsamples,
        100. * correct / nsamples))


def predict(X_test):
    net = model.eval()
    correct = 0
    nsamples = X_test.shape[0]
    ret = []
    with torch.no_grad():
        current_sample = 0
        while(current_sample < nsamples):
            batch = min(batch_size, nsamples-current_sample)
            xx = X_test[current_sample:(current_sample+batch)]
            yy = Y_test[current_sample:(current_sample+batch)]

            output = net(xx)
            ret += output.argmax(dim=1)
            current_sample += batch
            
    return ret


# In[ ]:

loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.MSELoss(reduction='sum')

# model = VTCNN2().to(DEVICE)
# model = FCN().to(DEVICE)
model = FCNSMALL().to(DEVICE)
print(model)
batch_size = 1024
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
n_epochs = 100

for t in range(n_epochs):
    train()
    test(t)

# In[ ]:

dummy_input = torch.randn(1, 256, device='cuda')
ONNX_FILE_PATH = model._get_name() + '.onnx'
torch.onnx.export(model, dummy_input, ONNX_FILE_PATH, input_names=['input'],
                  output_names=['output'], export_params=True)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


classes = mods
# Plot confusion matrix
test_Y_hat = predict(X_test)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = int(Y_test[i])
    k = int(test_Y_hat[i])
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)

# Save the test sets
X_test.cpu().numpy().astype(np.float32).tofile('RML2016.10a.test.fc32')
Y_test.cpu().numpy().astype(np.int16).tofile('RML2016.10a.test.lbl')

# Plot confusion matrix
acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_X_i = X_test[np.where(snr_test==snr)]
    test_Y_i = Y_test[np.where(snr_test==snr)]    

    # estimate classes
    test_Y_i_hat = predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = int(test_Y_i[i])
        k = int(test_Y_i_hat[i])
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)

    # Save the test sets per SNR
    test_X_i.cpu().numpy().astype(np.float32).tofile('RML2016.10a.test_snr={}.fc32'.format(snr))
    test_Y_i.cpu().numpy().astype(np.int16).tofile('RML2016.10a.test_snr={}.lbl'.format(snr))


plt.show()
