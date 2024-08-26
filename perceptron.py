import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
torch.cuda.set_device(0)

class model(nn.Module):
    def __init__(self, input_size, h_neurons, output_size):
        super(model, self).__init__()
        self.hidden_layer = nn.Linear(input_size, h_neurons).to(device)
        self.output_layer = nn.Linear(h_neurons, output_size).to(device)

    def forward(self, X):
        hx1 = F.sigmoid(self.hidden_layer(X)).to(device)
        hx2 = F.sigmoid(self.output_layer(hx1)).to(device)
        return hx2
    

