import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms

from colored import ColoredMNIST


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(3 * 28 * 28, 512)
    self.fc2 = nn.Linear(512, 512)
    self.fc3 = nn.Linear(512, 1)

  def forward(self, x):
    x = x.view(-1, 3 * 28 * 28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    logits = self.fc3(x).flatten()
    return logits


class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5, 1,2)
    self.conv2 = nn.Conv2d(6, 16, 5, 1)
    self.fc1 = nn.Linear(400, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84,1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2, 2)
    x = x.view(-1, 400)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    logits = self.fc3(x).flatten()
    return logits


def test_model(model, device, test_loader, set_name="test set"):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
      pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                         torch.Tensor([1.0]).to(device),
                         torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    set_name, test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

  return 100. * correct / len(test_loader.dataset)


def erm_train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device).float()
    optimizer.zero_grad()
    output = model(data)
    loss = F.binary_cross_entropy_with_logits(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 10 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def train_and_test_erm():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  all_train_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='all_train',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
    batch_size=64, shuffle=True, **kwargs)

  test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    ])),
    batch_size=1000, shuffle=True, **kwargs)

  model = ConvNet().to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.01)
  acc1 = []
  acc2 = []

  for epoch in range(1, 20):
    erm_train(model, device, all_train_loader, optimizer, epoch)
    acc_1 = test_model(model, device, all_train_loader, set_name='train set')
    acc_2 = test_model(model, device, test_loader)
    acc1.append(acc_1)
    acc2.append(acc_2)
  return acc1,acc2

def compute_irm_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]
  return (g1 * g2).sum()


def irm_train(model, device, train_loaders, optimizer, epoch):
  model.train()

  train_loaders = [iter(x) for x in train_loaders]

  dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)

  batch_idx = 0
  penalty_multiplier = epoch ** 1.6
  print(f'Using penalty multiplier {penalty_multiplier}')
  while True:
    optimizer.zero_grad()
    error = 0
    penalty = 0
    for loader in train_loaders:
      data, target = next(loader, (None, None))
      if data is None:
        return
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      loss_erm = F.binary_cross_entropy_with_logits(output * dummy_w, target, reduction='none')
      penalty += compute_irm_penalty(loss_erm, dummy_w)
      error += loss_erm.mean()
    (error + penalty_multiplier * penalty).backward()
    optimizer.step()
    if batch_idx % 2 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tERM loss: {:.6f}\tGrad penalty: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loaders[0]._dataset),
               100. * batch_idx / len(train_loaders[0]), error.item(), penalty.item()))
      print('First 20 logits', output.data.cpu().numpy()[:20])

    batch_idx += 1


def train_and_test_irm():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train1_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='train1',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
    batch_size=2000, shuffle=True, **kwargs)

  train2_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='train2',
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ])),
    batch_size=2000, shuffle=True, **kwargs)

  test_loader = torch.utils.data.DataLoader(
    ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    ])),
    batch_size=1000, shuffle=True, **kwargs)

  model = ConvNet().to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  acc1 = []
  acc2 = []
  acc3 = []

  for epoch in range(1, 200):
    irm_train(model, device, [train1_loader, train2_loader], optimizer, epoch)
    train1_acc = test_model(model, device, train1_loader, set_name='train1 set')
    train2_acc = test_model(model, device, train2_loader, set_name='train2 set')
    test_acc = test_model(model, device, test_loader)
    acc1.append(train1_acc)
    acc2.append(train2_acc)
    acc3.append(test_acc)
    if train1_acc > 69 and train2_acc > 69 and test_acc > 59:
      print('found acceptable values. stopping training.')
      return acc1,acc2,acc3
  return acc1,acc2,acc3


def plot_dataset_digits(dataset):
  fig = plt.figure(figsize=(13, 16))
  columns = 6
  rows = 6
  # ax enables access to manipulate each of subplots
  ax = []

  for i in range(columns * rows):
    img, label = dataset[i]
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("Label: " + str(label))  # set title
    plt.imshow(img)

  plt.show()  # finally, render the plot


def main():
  acc1,acc2,acc3 = train_and_test_irm()
  epoch = len(acc1)
  acc_1,acc_2 = train_and_test_erm()
  x = np.linspace(1,epoch,epoch)
  fig1 = plt.figure()
  plt.plot(x,acc1)
  plt.plot(x,acc2)
  plt.plot(x,acc3)
  plt.legend(['train1','train2','test'])
  plt.xlabel('Epoch')
  plt.ylabel('Acc(IRM)')
  plt.savefig('plots\\IRM')
  fig1.clf()
  epoch = len(acc_1)
  x = np.linspace(1,epoch,epoch)
  fig2 = plt.figure()
  plt.plot(x,acc_1)
  plt.plot(x,acc_2)
  plt.legend(['train','test'])
  plt.xlabel('Epoch')
  plt.ylabel('Acc(ERM)')
  plt.savefig('plots\\ERM')



if __name__ == '__main__':
  main()