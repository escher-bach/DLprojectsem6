import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        # Rich multi-layer random CNN
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # output: out_channels * 4 * 4 = 256 * 16 = 4096
        
        # Freeze all
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # x: (B, C, H, W)
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1) # Flatten
        return x

class ReservoirEncoder(nn.Module):
    def __init__(self, reservoir_size=4096, spectral_radius=0.9, sparsity=0.9, embed_dim=192):
        super().__init__()
        
        # CNN to reduce pixels
        self.cnn = RandomCNN(in_channels=3, out_channels=256) # output dim 4096
        cnn_out_dim = 256 * 4 * 4
        
        # ESN parameters
        self.reservoir_size = reservoir_size
        
        # W_in: CNN out -> Reservoir
        W_in = torch.randn(reservoir_size, cnn_out_dim) * 0.1
        self.register_buffer("W_in", W_in)
        
        # W: Reservoir -> Reservoir
        W = torch.rand(reservoir_size, reservoir_size) - 0.5
        # Apply sparsity
        mask = torch.rand(reservoir_size, reservoir_size) > sparsity
        W[mask] = 0.0
        
        # Spectral radius scaling
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(W)
            max_eigenvalue = torch.max(torch.abs(eigenvalues))
            if max_eigenvalue > 0:
                W = W * (spectral_radius / max_eigenvalue)
                
        self.register_buffer("W", W.real) # taking real part
        
        # Projection to embed_dim (fixed orthogonal)
        W_out = torch.empty(embed_dim, reservoir_size)
        nn.init.orthogonal_(W_out)
        self.register_buffer("W_out", W_out)
        
        self.embed_dim = embed_dim
        
    def forward(self, pixels, interpolate_pos_encoding=False):
        with torch.no_grad():
            x = self.cnn(pixels)
            # ESN update
            h = torch.tanh(F.linear(x, self.W_in))
            h = torch.tanh(F.linear(x, self.W_in) + F.linear(h, self.W))
            
            # Project down to embed_dim
            emb = F.linear(h, self.W_out) # (B, embed_dim)
            
        return emb
