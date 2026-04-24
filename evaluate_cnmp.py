import torch
import numpy as np
import matplotlib.pyplot as plt
from train_cnmp import CNMP

def evaluate():
    data = np.load("trajectories.npy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNMP().to(device)
    model.load_state_dict(torch.load("cnmp_model.pth", map_location=device))
    model.eval()
    
    ee_mses = []
    obj_mses = []
    
    num_tests = 100
    data_torch = torch.from_numpy(data).float().to(device)
    
    for i in range(num_tests):
        # Pick a random trajectory
        idx = np.random.randint(0, data.shape[0])
        traj = data_torch[idx]
        
        # Random number of context points
        n_context = np.random.randint(1, 50)
        context_ix = np.random.choice(100, n_context, replace=False)
        
        context_pts = traj[context_ix].unsqueeze(0) # (1, n_context, 6)
        
        # Target points (evaluate on the whole trajectory)
        query_x = traj[:, [0, 5]].unsqueeze(0) # (1, 100, 2)
        target_truth = traj[:, 1:5].unsqueeze(0) # (1, 100, 4)
        
        with torch.no_grad():
            mean, std = model(context_pts, None, query_x)
            
        mse = torch.mean((mean - target_truth)**2, dim=1).cpu().numpy()[0] # (4,)
        
        ee_mse = np.mean(mse[0:2]) # ey, ez
        obj_mse = np.mean(mse[2:4]) # oy, oz
        
        ee_mses.append(ee_mse)
        obj_mses.append(obj_mse)
        
    ee_mses = np.array(ee_mses)
    obj_mses = np.array(obj_mses)
    
    mean_ee = np.mean(ee_mses)
    std_ee = np.std(ee_mses)
    mean_obj = np.mean(obj_mses)
    std_obj = np.std(obj_mses)
    
    print(f"EE MSE: {mean_ee:.6f} +/- {std_ee:.6f}")
    print(f"Obj MSE: {mean_obj:.6f} +/- {std_obj:.6f}")
    
    # Plotting
    labels = ["End-Effector", "Object"]
    means = [mean_ee, mean_obj]
    stds = [std_ee, std_obj]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, means, yerr=stds, capsize=10, color=['blue', 'red'], alpha=0.7)
    plt.ylabel("Mean Squared Error")
    plt.title("CNMP Prediction Error (100 Tests)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("mse_bar_plot.png")
    plt.close()
    
    print("Evaluation complete. Plot saved to mse_bar_plot.png")

if __name__ == "__main__":
    evaluate()
