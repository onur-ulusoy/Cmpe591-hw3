import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class CNMP(nn.Module):
    def __init__(self, context_dim=6, query_dim=2, target_dim=4, hidden_size=128, num_layers=4):
        super(CNMP, self).__init__()
        
        # Encoder: (t, ey, ez, oy, oz, h) -> Representation
        layers = []
        layers.append(nn.Linear(context_dim, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        
        # Decoder: (Agg_R, t, h) -> (mean, std) for (ey, ez, oy, oz)
        layers = []
        layers.append(nn.Linear(hidden_size + query_dim, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 2 * target_dim))
        self.decoder = nn.Sequential(*layers)
        
        self.target_dim = target_dim
        self.min_std = 0.01

    def forward(self, context_x, context_y, query_x):
        # context_x: (batch, n_context, context_dim)
        # context_y: not needed if context_x already includes everything? 
        # Actually, let's keep it standard:
        # context_pts: (batch, n_context, 6)
        # query_x: (batch, n_query, 2) i.e. (t, h)
        
        # Encode context points
        r_i = self.encoder(context_x) # (batch, n_context, hidden_size)
        
        # Aggregate
        R = torch.mean(r_i, dim=1, keepdim=True) # (batch, 1, hidden_size)
        
        # Tile R for query points
        n_query = query_x.shape[1]
        R_tiled = R.repeat(1, n_query, 1) # (batch, n_query, hidden_size)
        
        # Decode
        decoder_input = torch.cat([R_tiled, query_x], dim=-1) # (batch, n_query, hidden_size + 2)
        output = self.decoder(decoder_input)
        
        mean = output[..., :self.target_dim]
        std = F.softplus(output[..., self.target_dim:]) + self.min_std
        
        return mean, std

    def nll_loss(self, mean, std, target):
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target).sum(dim=-1).mean()
        return nll

import os

def train():
    while not os.path.exists("trajectories.npy"): # Wait if file not yet created
        import time as ttime
        ttime.sleep(5)
    
    data = np.load("trajectories.npy") # (N, 100, 6) -> [t, ey, ez, oy, oz, h]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNMP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Split data
    num_train = int(len(data) * 0.8)
    train_data = data[:num_train]
    test_data = data[num_train:]
    
    losses = []
    num_iterations = 20000
    batch_size = 16
    
    train_data = torch.from_numpy(train_data).float().to(device)
    
    for iter_step in range(num_iterations):
        optimizer.zero_grad()
        
        # Sample batch
        batch_indices = np.random.choice(len(train_data), batch_size, replace=True)
        batch_traj = train_data[batch_indices] # (batch, 100, 6)
        
        # Sample number of context and target points for this batch
        n_context = np.random.randint(1, 40)
        n_target = np.random.randint(1, 40)
        
        # Sample indices for each traj in batch
        # For simplicity, we use the same number of points, but different indices for each?
        # Actually, let's use the same indices for the whole batch for efficiency
        context_ix = np.random.choice(100, n_context, replace=False)
        target_ix = np.random.choice(100, n_target, replace=False)
        
        context_pts = batch_traj[:, context_ix, :] # (batch, n_context, 6)
        query_x = batch_traj[:, target_ix, :][:, :, [0, 5]] # (batch, n_target, 2)
        target_y = batch_traj[:, target_ix, :][:, :, 1:5] # (batch, n_target, 4)
        
        mean, std = model(context_pts, None, query_x)
        loss = model.nll_loss(mean, std, target_y)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if iter_step % 1000 == 0:
            print(f"Iteration {iter_step}, Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), "cnmp_model.pth")
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("NLL Loss")
    plt.title("Training Loss Curve")
    plt.savefig("loss_curve.png")
    plt.close()
    print("Training complete. Model saved to cnmp_model.pth")

if __name__ == "__main__":
    train()
