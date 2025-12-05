from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

########### TORCH BASELINE MODEL ##############
class BaselineModel(nn.Module):
  def __init__(self):
    super(BaselineModel, self).__init__()
    self.fc1 = nn.Linear(64, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, 1)

  def forward(self, x):
    return self.fc3(self.fc2(self.fc1(x)))



################### DATASET HELPERS ##############

def load_data():
  # Load the dataset
  digits = load_digits()
  X = digits.data    # Images, flattened into vectors
  y = digits.target  # Target digits
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
  return X_train, X_test, y_train, y_test



################### MODEL HELPERS #################################

def init_model():
  mat_1 = np.random.random((128, 64)) - 0.5
  mat_2 = np.random.random((128, 128)) - 0.5
  mat_3 = np.random.random((1, 128)) - 0.5
  return np.concatenate((mat_1.flatten(), mat_2.flatten(), mat_3.flatten()), axis=0)


def model_from_weights(weights):
  mat_1 = np.array(weights[:64*128].reshape((128, 64)))
  mat_2 = np.array(weights[64*128:64*128+128*128].reshape((128, 128)))
  mat_3 = np.array(weights[64*128+128*128:].reshape((1, 128)))
  return (mat_1, mat_2, mat_3)




############### LOSS HELPERS #################

def get_loss(x, label, model):
  mat_1, mat_2, mat_3 = model
  prediction = mat_3 @ (mat_2 @ (mat_1 @ x))
  return ((prediction - label)**2)[0]


def loss_fn(batch_x, batch_label, weights):
  total_loss = 0
  for x, label in zip(batch_x, batch_label):
    model = model_from_weights(weights)
    total_loss += get_loss(x, label, model)
  return total_loss / len(batch_x)


def predict(x, weights):
  model = model_from_weights(weights)
  mat_1, mat_2, mat_3 = model
  prediction = mat_3 @ (mat_2 @ (mat_1 @ x))
  return prediction



def zero_gradient(x, label, weights, avgs, loss_fn, prev_loss=None, prev_weights=None, step_scaler=1, iters=50):
  # avg = 1 --> 0.01
  # 0.5 --> avg and then divide by 100

  best_loss = prev_loss or float("inf")
  best_weights = prev_weights
  for _ in range(iters):
    weights_attempt = np.zeros((weights.shape))
    for i, _ in enumerate(zip(weights, avgs)):
      weights_attempt[i] = weights[i] + (((random.random() - 0.5) * 2) * avgs[i]) * step_scaler

    loss = loss_fn(x, label, weights_attempt)
    if loss < best_loss:
      best_loss = loss
      best_weights = weights_attempt
  
  return best_weights, best_loss



def zero_grad_optimize():
  xTr, xTe, yTr, yTe = load_data()
  xTr = xTr[:100]
  yTr = yTr[:100]
  xTe = xTe[:10]
  yTe = yTe[:10]

  scalers = [0.1]*10 + [0.05]*10 + [0.01]*20 + [0.005]*50 + [0.0001]*50
  w = init_model()
  best_w = None
  best_loss = float("inf")
  for i, scaler in enumerate(scalers):
    avgs = np.ones((w.shape))
    label = int(random.random() * 10)
    best_w, best_loss = zero_gradient(xTr, yTr, w, avgs, loss_fn, step_scaler=scaler, prev_loss=best_loss, prev_weights=best_w)
    w = best_w
    print(f"FINISHED STEP {i} with scaler value {scaler}, BEST LOSS {best_loss}")


  for x, label in zip(xTr, yTr):
    pred = predict(x, best_w)
    print(pred, label)

  print()
  for x, label in zip(xTe, yTe):
    pred = predict(x, best_w)
    print(pred, label)



def torch_loss(output, target):
  loss = (output.flatten()-target)**2
  return sum(loss) / len(target)
  

def torch_baseline():
  digits = load_digits()
  X = digits.data.astype("float32")   # Images, flattened into vectors
  y = digits.target.astype("int")  # Target digits
  xTr, xTe, yTr, yTe = train_test_split(X, y, test_size = 0.2, random_state = 1)
  xTr = xTr[:100]
  yTr = yTr[:100]
  xTe = xTe[:10]
  yTe = yTe[:10]
  epochs = 20
  batch_size=10
  learning_rate = 0.001

  model = BaselineModel()

  # loss_function = nn.CrossEntropyLoss()  # For classification tasks
  loss_function = torch_loss
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Learning rate

  xTr = torch.asarray(xTr)
  xTe = torch.asarray(xTe)
  yTr = torch.asarray(yTr)
  yTe = torch.asarray(yTe)

  training_data = TensorDataset(xTr, yTr)
  test_data = TensorDataset(xTe, yTe)

  train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


  train_losses = []
  test_losses = []

  # Example training loop (you will need to add data loading and batching)
  for epoch in range(epochs):  # Number of epochs
      model.train()
      total_train_loss = 0

      for batch_idx, (data, target) in enumerate(train_loader):  # Define your train_loader
          optimizer.zero_grad()   # Clear gradients
          output = model(data)    # Compute the model output
          loss = loss_function(output, target)  # Calculate loss
          loss.backward()  # Compute gradients
          optimizer.step()  # Update weights
          total_train_loss += loss.item() * data.size(0)


      average_train_loss = total_train_loss / len(train_loader.dataset)
      train_losses.append(average_train_loss)

      # Testing loop
      model.eval()  # Set model to evaluation mode
      total_test_loss = 0

      with torch.no_grad():
          for data, target in test_loader:
              output = model(data)
              loss = loss_function(output, target)
              total_test_loss += loss.item() * data.size(0)

      average_test_loss = total_test_loss / len(test_loader.dataset)
      test_losses.append(average_test_loss)

      print(f'Epoch {epoch+1}, Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f}')


  model.eval()
  for x, label in zip(xTr, yTr):
    pred = model(x).detach()[0]
    print(pred, label, pred-label)

  print()
  for x, label in zip(xTe, yTe):
    pred = model(x).detach()[0]
    print(pred, label, pred-label)


torch_baseline()


# for _ in range(10):
#   x = np.random.random((64,)) - 0.5
#   label = int(random.random() * 10)


# x = [1,2,3,4]
# avgs = [0.5, 4, 6, 2]
# zero_gradient(x, avgs,)



