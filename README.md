# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: M THEJESWARAN
### Register Number: 212223240168
```python

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_excel('dldata.xlsx')
X = dataset1[['Input']].values
y = dataset1[['Output']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

ai_teja= NeuralNet ()
criterion = nn. MSELoss ()
optimizer = optim.RMSprop (ai_teja. parameters(), lr=0.001)

def train_model(ai_teja, X_train, y_train, criterion, optimizer, epochs=4000) :
  for epoch in range (epochs) :
    optimizer. zero_grad()
    loss = criterion(ai_teja(X_train), y_train)
    loss. backward()
    optimizer.step()
    ai_teja. history['loss'] .append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_teja, X_train_tensor, y_train_tensor, criterion, optimizer)



with torch.no_grad():
    test_loss = criterion(ai_teja(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')



loss_df = pd.DataFrame(ai_teja.history)


import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_teja(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```
## Dataset Information

<img width="182" height="504" alt="Screenshot (53)" src="https://github.com/user-attachments/assets/d3f39b04-83ff-4021-8d97-085f7a69529d" />

## OUTPUT

<img width="362" height="463" alt="Screenshot (51)" src="https://github.com/user-attachments/assets/cf3a9d19-22a0-4668-a8f4-132eff7591b1" />

### Training Loss Vs Iteration Plot

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/a706ece3-fbe6-4324-8651-3497ac933ebe" />

### New Sample Data Prediction

<img width="297" height="25" alt="PREDICTION" src="https://github.com/user-attachments/assets/859a997f-034f-4e70-9170-eb3cd3950fc7" />

## RESULT
Thus the neural network regression model is devoloped , trained and tested sucessfully
