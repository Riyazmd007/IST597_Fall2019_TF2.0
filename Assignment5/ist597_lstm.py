#Lab 4

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image


class NotMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root="/content/notMNIST_small"):
        self.samples = []
        folders = sorted(os.listdir(root))
        self.label_map = {folder: i for i, folder in enumerate(folders)}

        for folder in folders:
            folder_path = os.path.join(root, folder)
            for fname in os.listdir(folder_path):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(folder_path, fname), self.label_map[folder]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("L")
        except:
            return self.__getitem__((idx + 1) % len(self.samples))

        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)
        return img, label

#GRUcell
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCellBasic(nn.Module):

    def __init__(self, hidden_size, input_size):
        super(GRUCellBasic, self).__init__()
        self.hidden_dim = hidden_size
        self.input_dim = input_size
        added_dim = hidden_size + input_size

        self.weight_update = nn.Parameter(torch.randn(added_dim, hidden_size) * 0.1)
        self.bias_update = nn.Parameter(torch.zeros(hidden_size))


        self.weight_reset = nn.Parameter(torch.randn(added_dim, hidden_size) * 0.1)
        self.bias_reset = nn.Parameter(torch.zeros(hidden_size))


        self.weight_candidate = nn.Parameter(torch.randn(added_dim, hidden_size) * 0.1)
        self.bias_candidate = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, s_prev):


        concat_hx = torch.cat([s_prev, x_t], dim=-1)


        z_t = torch.sigmoid(concat_hx @ self.weight_update + self.bias_update)


        r_t = torch.sigmoid(concat_hx @ self.weight_reset + self.bias_reset)


        gated_s = r_t * s_prev


        concat_candidate = torch.cat([gated_s, x_t], dim=-1)


        h_dash = torch.tanh(concat_candidate @ self.weight_candidate + self.bias_candidate)


        new_s = (1 - z_t) * s_prev + z_t * h_dash

        return new_s, new_s
class MGUCellBasic(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(MGUCellBasic, self).__init__()

        added_dim = hidden_size + input_size

        self.weight_update = nn.Parameter(torch.randn(added_dim, hidden_size) * 0.1)
        self.bias_update   = nn.Parameter(torch.zeros(hidden_size))

        self.weight_reset  = nn.Parameter(torch.randn(added_dim, hidden_size) * 0.1)
        self.bias_reset    = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, s_prev):


        concat = torch.cat([s_prev, x_t], dim=-1)


        f_t = torch.sigmoid(concat @ self.weight_update + self.bias_update)


        gated_s = f_t * s_prev

        concat_cand = torch.cat([gated_s, x_t], dim=-1)
        s_dash = torch.tanh(concat_cand @ self.weight_reset + self.bias_reset)

        new_s = (1 - f_t) * s_prev + f_t * s_dash

        return new_s


#RNN model
class CustomRNN(nn.Module):
    def __init__(self, cell_class, input_dim, hidden_dim, num_layers, output=10):
        super(CustomRNN, self).__init__()

        self.hidden = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            inp_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(cell_class(hidden_dim, inp_dim))

        self.fc = nn.Linear(hidden_dim, output)

    def forward(self, x):
        x = x.squeeze(1)  # (batch, 28, 28)
        batch = x.size(0)

        states = [torch.zeros(batch, self.hidden_dim, device=x.device)
                  for _ in range(self.num_layers)]

        for t in range(28):
            x_t = x[:, t, :]
            inp = x_t

            for layer_idx, cell in enumerate(self.layers):
                new_state = cell(inp, states[layer_idx])
                states[layer_idx] = new_state
                inp = new_state

        return self.fc(states[-1])


#Dataloading
dataset = NotMNISTDataset("/content/notMNIST_small")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


#choice between GRU or MGU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cell_class = MGUCellBasic

model = CustomRNN(cell_class, input_dim=28, hidden_dim=64, num_layers=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#accuracy
def accuracy(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


epochs = 10
train_curve, test_curve = [], []

for epoch in range(epochs):
    model.train()
    correct, total = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    test_acc = accuracy(test_loader)

    train_curve.append(train_acc)
    test_curve.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.4f} | Test: {test_acc:.4f}")


#plots
plt.plot(train_curve, label="Train")
plt.plot(test_curve, label="Test")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.legend()
plt.show()
