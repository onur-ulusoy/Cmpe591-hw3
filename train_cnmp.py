import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import hyperparams as hp
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(hp.LOG_PATH),
                        logging.StreamHandler()
                    ])

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
        r_i = self.encoder(context_x) # (batch, n_context, hidden_size)
        R = torch.mean(r_i, dim=1, keepdim=True) # (batch, 1, hidden_size)
        n_query = query_x.shape[1]
        R_tiled = R.repeat(1, n_query, 1) # (batch, n_query, hidden_size)
        decoder_input = torch.cat([R_tiled, query_x], dim=-1) # (batch, n_query, hidden_size + 2)
        output = self.decoder(decoder_input)
        mean = output[..., :self.target_dim]
        std = F.softplus(output[..., self.target_dim:]) + self.min_std
        return mean, std

    def nll_loss(self, mean, std, target):
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target).sum(dim=-1).mean()
        return nll

def train():
    while not os.path.exists(hp.DATA_PATH):
        import time as ttime
        ttime.sleep(5)
    
    data = np.load(hp.DATA_PATH) # (N, 100, 6) -> [t, ey, ez, oy, oz, h]
    
    # Dynamic Normalization
    if hp.NORMALIZE:
        # Exclude time (index 0) from mean/std? Usually better to normalize all.
        mean = np.mean(data, axis=(0, 1))
        std = np.std(data, axis=(0, 1))
        std[std < 1e-6] = 1.0 # Avoid division by zero
        np.save(hp.NORM_PATH, {"mean": mean, "std": std})
        data = (data - mean) / std
        logging.info(f"Data normalized with mean: {mean}, std: {std}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNMP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.NUM_ITERATIONS//2, gamma=0.5)
    
    num_train = int(len(data) * hp.TRAIN_SPLIT)
    train_data = data[:num_train]
    
    losses = []
    num_iterations = hp.NUM_ITERATIONS
    batch_size = hp.BATCH_SIZE
    
    train_data = torch.from_numpy(train_data).float().to(device)
    
    for iter_step in range(num_iterations):
        optimizer.zero_grad()
        
        batch_indices = np.random.choice(len(train_data), batch_size, replace=True)
        batch_traj = train_data[batch_indices]
        
        n_context = np.random.randint(1, hp.MAX_STEPS // 2)
        n_target = np.random.randint(1, hp.MAX_STEPS // 2)
        
        context_ix = np.random.choice(hp.MAX_STEPS, n_context, replace=False)
        target_ix = np.random.choice(hp.MAX_STEPS, n_target, replace=False)
        
        context_pts = batch_traj[:, context_ix, :] # (batch, n_context, 6)
        query_x = batch_traj[:, target_ix, :][:, :, [0, 5]] # (batch, n_target, 2)
        target_y = batch_traj[:, target_ix, :][:, :, 1:5] # (batch, n_target, 4)
        
        mean, std = model(context_pts, None, query_x)
        loss = model.nll_loss(mean, std, target_y)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if iter_step % 1000 == 0:
            logging.info(f"Iteration {iter_step}, Loss: {loss.item():.4f}")
            
    torch.save(model.state_dict(), hp.MODEL_PATH)
    
    # Generate Plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(losses)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("NLL Loss")
    ax1.set_title("Training Loss Curve (Linear Scale)")
    ax1.grid(True)
    
    ax2.plot(losses)
    ax2.set_yscale('symlog') # Use symlog for better visibility of negative values
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("NLL Loss")
    ax2.set_title("Training Loss Curve (Symmetric Log Scale)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("loss_curves_comparison.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("NLL Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()

    logging.info(f"Training complete. Model saved to {hp.MODEL_PATH}")

if __name__ == "__main__":
    train()
