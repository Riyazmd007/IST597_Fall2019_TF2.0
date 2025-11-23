#Normalization Analysis
#Following the pyTorch template

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time  

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1234)

#Loading datasets

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=True,
                                           transform=transform,
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root="./data",
                                          train=False,
                                          transform=transform,
                                          download=True)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=64,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=64,
                                          shuffle=False)


#Given CNN class implementation

class CNN(nn.Module):
    def __init__(self, hidden_size=100, output_size=10):
        super().__init__()

        filter_h, filter_w, filter_c, filter_n = 5, 5, 1, 30

       
        self.W1 = nn.Parameter(torch.randn(filter_h, filter_w, filter_c, filter_n) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(filter_n))

     
        self.W2 = nn.Parameter(torch.randn(14 * 14 * filter_n, hidden_size) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(hidden_size))

        self.W3 = nn.Parameter(torch.randn(hidden_size, output_size) * 0.1)
        self.b3 = nn.Parameter(torch.zeros(output_size))

        #gamma to 1 and beta initialized to 0's for both BN and LN
        self.gamma_bn1 = nn.Parameter(torch.ones(1, hidden_size))
        self.beta_bn1  = nn.Parameter(torch.zeros(1, hidden_size))

        self.gamma_bn2 = nn.Parameter(torch.ones(1, output_size))
        self.beta_bn2  = nn.Parameter(torch.zeros(1, output_size))

       
        self.gamma_ln1 = nn.Parameter(torch.ones(1, hidden_size))
        self.beta_ln1  = nn.Parameter(torch.zeros(1, hidden_size))

        self.gamma_ln2 = nn.Parameter(torch.ones(1, output_size))
        self.beta_ln2  = nn.Parameter(torch.zeros(1, output_size))

        #WN parameters
        self.v1 = nn.Parameter(torch.randn(14 * 14 * filter_n, hidden_size) * 0.1)
        self.g1 = nn.Parameter(torch.ones(hidden_size))

        self.v2 = nn.Parameter(torch.randn(hidden_size, output_size) * 0.1)
        self.g2 = nn.Parameter(torch.ones(output_size))


    

    def batch_normalization(self, X, gamma, beta, epsilon=1e-5):
        result = torch.zeros_like(X)
        features = X.shape[1]  

        for j in range(features):             
          col = X[:, j]             
          mean = col.mean()
          var = col.var(unbiased=False)
          result[:, j] = (col - mean) / torch.sqrt(var + epsilon)

        return gamma * result + beta

    def layer_normalization(self, X, gamma, beta, epsilon=1e-5):
        samples, features = X.shape
        result = torch.zeros_like(X)

        for i in range(samples):     
          row = X[i]        
          mean = row.mean()
          var  = row.var(unbiased=False)
          result[i] = (row - mean) / torch.sqrt(var + epsilon)

        return gamma * result + beta

    def weight_normalization(X, v, g, b, epsilon=1e-6):
    
      input, output = v.shape
      weight = torch.zeros_like(v)

    # normalize each column manually
      for j in range(output):
        column = v[:, j]                 
        norm = torch.sqrt((column**2).sum() + epsilon)
        weight[:, j] = g[j] * (column / norm)   # normalized column

      return X @ weight + b



  

    def pad(self, X, padding):
        return torch.nn.functional.pad(X, (padding, padding, padding, padding))

    def extract_windows(self, X, window_h, window_w, stride, out_h, out_w):
        N, C, H, W = X.shape
        windows = []
        for y in range(out_h):
            for x in range(out_w):
                window = X[:, :, y*stride:y*stride+window_h, x*stride:x*stride+window_w]
                windows.append(window)
        return torch.stack(windows).reshape(-1, window_h * window_w * C)

    def convolution(self, X, W, b, padding=2, stride=1):
        N, C, H, W_in = X.shape
        f_h, f_w, f_c, f_n = W.shape

        out_h = (H + 2*padding - f_h)//stride + 1
        out_w = (W_in + 2*padding - f_w)//stride + 1

        X_padded = self.pad(X, padding)
        X_flat   = self.extract_windows(X_padded, f_h, f_w, stride, out_h, out_w)
        W_flat   = W.reshape(-1, f_n)

        z = X_flat @ W_flat + b
        return z.view(out_h, out_w, N, f_n).permute(2,0,1,3)

    def relu(self, X):
        return torch.clamp(X, min=0)

    def max_pool(self, X, pool_h=2, pool_w=2, stride=2):
        N, C, H, W = X.shape
        out_h = (H - pool_h)//stride + 1
        out_w = (W - pool_w)//stride + 1

        windows = []
        for y in range(out_h):
            for x in range(out_w):
                region = X[:, :, y*stride:y*stride+pool_h, x*stride:x*stride+pool_w]
                windows.append(region)

        stacked = torch.stack(windows)
        pooled = stacked.max(dim=-1)[0].max(dim=-1)[0]
        return pooled.permute(1,2,0).reshape(N, C, out_h, out_w)

    def affine(self, X, W, b):
        return X.reshape(X.shape[0], -1) @ W + b

   
    def forward(self, X):
        out = self.convolution(X, self.W1, self.b1)
        out = self.relu(out)
        out = self.max_pool(out)

        out = self.affine(out, self.W2, self.b2)
        out = self.relu(out)

        out = self.affine(out, self.W3, self.b3)
        return out

#training for 10 epochs
total_epochs=10
model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

start_time = time.time()   

for epoch in range(total_epochs):
    total_loss = 0
    total_correct = 0
    count = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass (manual)
        out = model.convolution(images, model.W1, model.b1)
        out = model.relu(out)
        out = model.max_pool(out)

        out = model.affine(out, model.W2, model.b2)

        #FC1 normalization
        out = model.manual_batch_norm_fc(out, model.gamma_bn1, model.beta_bn1)
        # out = model.manual_layer_norm_fc(out, model.gamma_ln1, model.beta_ln1)
        # out = model.manual_weight_norm_fc(out, model.v1, model.g1, model.b2)
      

        out = model.relu(out)
        out = model.affine(out, model.W3, model.b3)

        # FC2 normalization
        out = model.manual_batch_norm_fc(out, model.gamma_bn2, model.beta_bn2)
        # out = model.manual_layer_norm_fc(out, model.gamma_ln2, model.beta_ln2)
        # out = model.manual_weight_norm_fc(out, model.v2, model.g2, model.b3)

        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (out.argmax(1) == labels).sum().item()
        count += images.size(0)

    print(f"Epoch {epoch+1} | Loss: {total_loss/count:.4f} | Acc: {100*total_correct/count:.2f}%")

end_time = time.time()     
print(f"\nTotal Training Time: {end_time - start_time:.2f} seconds")


#test

correct, total = 0, 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100*correct/total:.2f}%")
