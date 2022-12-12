from torch.nn.modules.activation import Softmax
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor() # 이미지, 배열을 텐서로 변경하고, 픽셀값을 정규하시켜줌
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor() 
)

train_dataloader = DataLoader(training_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)

class Lenet_5(nn.Module):
  def __init__(self):
    super(Lenet_5, self).__init__()
    self.flatten = nn.Flatten()
    self.conv_sigmoid_stack = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2), # input 1*32*32, output 6*28*28, same padding
        nn.Sigmoid(),
        nn.AvgPool2d(2),
        nn.Conv2d(6, 16, 5),
        nn.Sigmoid(),
        nn.AvgPool2d(2),
    )

    self.linear_sigmoid_stack = nn.Sequential(
      nn.Linear(16*5*5, 84),
      nn.Sigmoid(),
      nn.Linear(84, 10)
    )

  def forward(self, x):
    x1 = self.conv_sigmoid_stack(x)
    x2 = self.flatten(x1)
    logits = self.linear_sigmoid_stack(x2)
    return logits

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  for batch, (X, y) in enumerate(dataloader, 1):

    X = X.to(device)
    y = y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 1000 == 0:
      loss, current = loss.item(), batch * len(X)
      print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
    for X, y in dataloader:
      X = X.to(device)
      y = y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f'Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n')

model = Lenet_5().to(device)
print(model)

learning_rate = 1e-3
batch_size = 1
epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
  print(f'Epoch {t+1}\n-----------------------------------')
  train_loop(train_dataloader, model, loss_fn, optimizer)
  test_loop(test_dataloader, model, loss_fn)
print('Done!!!')