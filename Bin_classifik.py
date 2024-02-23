from data_apple import train_loader, test_loader, X_train, y_train, X_test, y_test

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
f = next(iter(train_loader))

input_size = 7

class Bin_classifik(nn.Module):
    def __init__(self):
        super(Bin_classifik, self).__init__()
        self.fc1 = nn.Linear(input_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) 
        return x.squeeze()

model = Bin_classifik()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())


total_step = len(train_loader)
loss_list = []
acc_list = []

num_epochs = 150

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        total = labels.size(0)
        predicted = torch.round(outputs.data)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)
    


correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
      .format(epoch + 1, num_epochs, loss.item(), (correct / total) * 100))

         
            

            
            
            
            
            
            
            
            
            
