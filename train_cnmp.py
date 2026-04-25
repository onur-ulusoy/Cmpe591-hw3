import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import hyperparams as hp

class CNMP(nn.Module):
    def __init__(self, context_dim=hp.CONTEXT_DIM, query_dim=hp.QUERY_DIM, 
                 target_dim=hp.TARGET_DIM, hidden_size=hp.HIDDEN_SIZE, 
                 num_layers=hp.NUM_LAYERS):
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
        self.min_std = hp.MIN_STD

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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(hp.LOG_PATH),
                        logging.StreamHandler()
                    ])

def train():
    while not os.path.exists(hp.DATA_PATH): # Wait if file not yet created
        import time as ttime
        ttime.sleep(5)
    
    data = np.load(hp.DATA_PATH) # (N, 100, 6) -> [t, ey, ez, oy, oz, h]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNMP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
    
    # Split data
    num_train = int(len(data) * hp.TRAIN_SPLIT)
    train_data = data[:num_train]
    test_data = data[num_train:]
    
    losses = []
    num_iterations = hp.NUM_ITERATIONS
    batch_size = hp.BATCH_SIZE
    
    train_data = torch.from_numpy(train_data).float().to(device)
    
    for iter_step in range(num_iterations):
        optimizer.zero_grad()
        
        # Sample batch
        batch_indices = np.random.choice(len(train_data), batch_size, replace=True)
        batch_traj = train_data[batch_indices] # (batch, 100, 6)
        
        # Sample number of context and target points for this batch
        n_context = np.random.randint(1, hp.MAX_STEPS // 2)
        n_target = np.random.randint(1, hp.MAX_STEPS // 2)
        
        # Sample indices for each traj in batch
        context_ix = np.random.choice(hp.MAX_STEPS, n_context, replace=False)
        target_ix = np.random.choice(hp.MAX_STEPS, n_target, replace=False)
        
        context_pts = batch_traj[:, context_ix, :] # (batch, n_context, 6)
        query_x = batch_traj[:, target_ix, :][:, :, [0, 5]] # (batch, n_target, 2)
        target_y = batch_traj[:, target_ix, :][:, :, 1:5] # (batch, n_target, 4)
        
        mean, std = model(context_pts, None, query_x)
        loss = model.nll_loss(mean, std, target_y)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if iter_step % 1000 == 0:
            logging.info(f"Iteration {iter_step}, Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), hp.MODEL_PATH)
    
    # Generate Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Linear Plot (New version)
    ax1.plot(losses)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Training Loss Curve (Linear Scale)")
    ax1.grid(True)
    
    # Log Plot (Old version - note: negative values will be hidden)
    ax2.plot(losses)
    ax2.set_yscale('log')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("NLL Loss")
    ax2.set_title("Training Loss Curve (Log Scale - Negative values hidden)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("loss_curves_comparison.png")
    plt.close()
    
    # Also save the linear one as the main loss_curve.png for consistency
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("NLL Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()

    logging.info("Training complete. Model saved to cnmp_model.pth, plots saved to loss_curves_comparison.png and loss_curve.png")

if __name__ == "__main__":
    train()
