---
title: Classificazione di immagini 
date: 2020-01-29
tags: ["markdown"]
image : "post/img/cnn.jpg"
Description  : "Classificazione di immagini di abiti e accessori in bassa risoluzione tramite reti convoluzionali..."
---
# Presentazione

Avvio presentazione fullscreen

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQAO3RBGqYbmRqohruiuVRg3hR5IFxAqxV2PFOHD29-iPJLiZS8tuWXGFJPPrRMa7mNcHRT5FwqTmYQ/embed?start=false&loop=false&delayms=3000" frameborder="0" width="820" height="498" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

# Notebook
```
import torchvision.transforms as transforms
import torchvision
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='data/', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='data/', train=False, download=True, transform=transform)
```

```
val_ratio = 0.2
val_size = int(val_ratio * len(train_dataset))
train_size = len(train_dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
```

```
import torch.nn as nn

class GrayScaleCNN(nn.Module):
    def __init__(self):
        super(GrayScaleCNN, self).__init__()
        
        # Strato di convolution con kernel 3x3, output di 64 filtri
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        # Strato di pooling con kernel 2x2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Strato di convolution con kernel 3x3, output di 128 filtri
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Strato di pooling con kernel 2x2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Strato fully connected con output di 128 unità
        self.fc1 = nn.Linear(in_features=128 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()
        
        # Strato di output con 10 unità, una per ogni classe
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(-1, 128 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        return x
```

```
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output

device = 'cuda'
print(torch.cuda.is_available())
model = GrayScaleCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

epochs = 30
train_losses = []
val_losses = []
lr = 0.02
weight_decay = 0.001
optimizer = optim.ASGD(model.parameters(), lr = lr , weight_decay = weight_decay)
for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0
    
    # Addestramento sui dati di train
    model.train()
    for data, target in train_loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_losses.append(train_loss / len(train_loader))
    
    # Valutazione sui dati di validation
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            
    val_losses.append(val_loss / len(val_loader))
    clear_output(wait=True)
    print("Epoch {}/{} - Train Loss: {:.4f} - Val Loss: {:.4f}".format(epoch+1, epochs, train_loss, val_loss), "  lr = ",lr,"  weight_decay = ",weight_decay)
    
    # Plot della curva di addestramento
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()

print("Addestramento completato!")
```

```
Epoch 30/30 - Train Loss: 143.4445 - Val Loss: 43.1652   lr =  0.02   weight_decay =  0.001
```

[![png](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.2-classificatore-cnn/notebook-di-lavoro/output_3_1.png)](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.2-classificatore-cnn/notebook-di-lavoro/output_3_1.png)

```
Addestramento completato!
```

```
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader)
accuracy = 100. * correct / len(test_dataset)
print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_dataset), accuracy))
#Test set: Average loss: 0.2864, Accuracy: 8953/10000 (90%)
```

```
Test set: Average loss: 0.2481, Accuracy: 9095/10000 (91%)
```

```

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchviz import make_dot
from IPython.display import SVG

model = GrayScaleCNN()
vis_graph = make_dot(model(torch.randn(1,1,28,28)), params=dict(model.named_parameters()))
#vis_graph.view()
SVG(vis_graph.render(format='svg'))
```

```
model = GrayScaleCNN()
param_count = sum(p.numel() for p in model.parameters())
print("Il modello ha un totale di {} parametri.".format(param_count))
```

```
model = GrayScaleCNN()
for name, param in model.named_parameters():
    print("Strato: {}\tNumero di parametri: {}".format(name, param.numel()))
```

```
import torch

device = torch.device("cuda:0")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'myimage.svg'

plt.savefig(image_name, format=image_format, dpi=1200)
```

```
True
Tesla T4
```

[![png](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.2-classificatore-cnn/notebook-di-lavoro/output_8_1.png)](https://automataia.github.io/AutoGnosis/4-progetti/4.1-supervised-learning/4.1.2-classificatore-cnn/notebook-di-lavoro/output_8_1.png)