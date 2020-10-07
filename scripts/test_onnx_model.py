import numpy as np
from collections import Counter
import onnx
from onnx import numpy_helper
import torch
from torchvision import models

from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_test = torch.Tensor(np.fromfile('RML2016.10a.test.fc32', dtype=np.float32).reshape((-1,256))).to(DEVICE)
Y_test = torch.LongTensor(np.fromfile('RML2016.10a.test.lbl', dtype=np.int16)).to(DEVICE)

classes = list(Counter(Y_test.cpu().numpy()).keys())
classes.sort()


def predict(X_test, model, batch_size):
    net = model.eval()
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

dr = 0.5
class VTCNN2(nn.Module):
    def __init__(self):
        super(VTCNN2, self).__init__()
        # self.reshape = torch.reshape((nsamples, 1, X_train.shape[1], X_train.shape[2]))
        self.conv1 = nn.Conv2d(1,256,(1,3),padding=(0,2))
        self.drop1 = nn.Dropout(dr)
        self.drop2 = nn.Dropout(dr)
        self.drop3 = nn.Dropout(dr)
        self.conv2 = nn.Conv2d(256,80,(2,3),padding=(0,2))
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(10560, 256)
        self.dense2 = nn.Linear(256, 11)
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

onnx_model = onnx.load('_VTCNN2.onnx')
graph = onnx_model.graph
initalizers = dict()
for init in graph.initializer:
    initalizers[init.name] = numpy_helper.to_array(init)

model = VTCNN2()
for name, p in model.named_parameters():
    p.data = (torch.from_numpy(initalizers[name])).data
model = model.to(DEVICE)
batch_size = 1024

# Plot confusion matrix
test_Y_hat = predict(X_test, model, batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = int(Y_test[i])
    k = int(test_Y_hat[i])
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])


plot_confusion_matrix(confnorm, labels=classes)

plt.show()
