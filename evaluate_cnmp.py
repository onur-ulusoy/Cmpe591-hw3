import torch
import numpy as np
import matplotlib.pyplot as plt
from train_cnmp import CNMP
import hyperparams as hp
import os

def evaluate():
    data_raw = np.load(hp.DATA_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Normalization Stats
    if hp.NORMALIZE:
        stats = np.load(hp.NORM_PATH, allow_pickle=True).item()
        mean = stats["mean"]
        std = stats["std"]
        data = (data_raw - mean) / std
    else:
        data = data_raw
    
    model = CNMP().to(device)
    model.load_state_dict(torch.load(hp.MODEL_PATH, map_location=device))
    model.eval()
    
    ee_mses = []
    obj_mses = []
    
    num_tests = 100
    data_torch = torch.from_numpy(data).float().to(device)
    
    # 1. Quantitative Evaluation on 100 tests
    for i in range(num_tests):
        idx = np.random.randint(0, data.shape[0])
        traj = data_torch[idx]
        
        n_context = np.random.randint(1, hp.MAX_STEPS // 2)
        context_ix = np.random.choice(hp.MAX_STEPS, n_context, replace=False)
        context_pts = traj[context_ix].unsqueeze(0)
        
        query_x = traj[:, [0, 5]].unsqueeze(0)
        target_truth = traj[:, 1:5].unsqueeze(0)
        
        with torch.no_grad():
            pred_mean, _ = model(context_pts, None, query_x)
            
        if hp.NORMALIZE:
            target_mean = mean[1:5]
            target_std = std[1:5]
            pred_rescaled = pred_mean.cpu().numpy()[0] * target_std + target_mean
            truth_rescaled = target_truth.cpu().numpy()[0] * target_std + target_mean
        else:
            pred_rescaled = pred_mean.cpu().numpy()[0]
            truth_rescaled = target_truth.cpu().numpy()[0]
            
        mse = np.mean((pred_rescaled - truth_rescaled)**2, axis=0)
        ee_mses.append(np.mean(mse[0:2]))
        obj_mses.append(np.mean(mse[2:4]))

    # 2. Qualitative Visualization (Top 3 cases with most movement)
    movements = []
    for i in range(len(data)):
        traj_raw = data_raw[i]
        mv = np.max(traj_raw[:, 3:5]) - np.min(traj_raw[:, 3:5])
        movements.append((mv, i))
    
    movements.sort(key=lambda x: x[0], reverse=True)
    
    for visual_id in range(3):
        idx = movements[visual_id][1]
        traj = data_torch[idx]
        
        n_context = 5 
        context_ix = np.random.choice(hp.MAX_STEPS, n_context, replace=False)
        context_pts = traj[context_ix].unsqueeze(0)
        
        query_x = traj[:, [0, 5]].unsqueeze(0)
        target_truth = traj[:, 1:5].unsqueeze(0)
        
        with torch.no_grad():
            pred_mean, _ = model(context_pts, None, query_x)
            
        if hp.NORMALIZE:
            pred_rescaled = pred_mean.cpu().numpy()[0] * target_std + target_mean
            truth_rescaled = target_truth.cpu().numpy()[0] * target_std + target_mean
            context_rescaled = traj[context_ix, 1:5].cpu().numpy() * target_std + target_mean
        else:
            pred_rescaled = pred_mean.cpu().numpy()[0]
            truth_rescaled = target_truth.cpu().numpy()[0]
            context_rescaled = traj[context_ix, 1:5].cpu().numpy()
            
        plot_trajectory(truth_rescaled, pred_rescaled, context_rescaled, visual_id)
        
    ee_mses = np.array(ee_mses)
    obj_mses = np.array(obj_mses)
    
    m_ee = np.mean(ee_mses)
    s_ee = np.std(ee_mses)
    m_obj = np.mean(obj_mses)
    s_obj = np.std(obj_mses)
    
    print(f"Final EE MSE: {m_ee:.8f} +/- {s_ee:.8f}")
    print(f"Final Obj MSE: {m_obj:.8f} +/- {s_obj:.8f}")
    
    # Premium Bar Plot
    labels = ["End-Effector", "Object"]
    means = [m_ee, m_obj]
    stds = [s_ee, s_obj]
    
    plt.figure(figsize=(10, 7))
    colors = ['#4A90E2', '#E35B5B'] # More premium colors
    bars = plt.bar(labels, means, yerr=stds, capsize=12, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=12, fontweight='bold')
    plt.title("CNMP Prediction Error Evaluation (100 Tests)", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{height:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                
    plt.savefig("mse_bar_plot.png", dpi=300)
    plt.close()
    print("Bar plot saved as mse_bar_plot.png")

def plot_trajectory(truth, pred, context, test_id):
    # truth, pred: (100, 4) -> [ey, ez, oy, oz]
    # context: (n_context, 4)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # End-Effector
    ax[0].plot(truth[:, 0], truth[:, 1], 'k-', label="Ground Truth", linewidth=1.5, alpha=0.9)
    ax[0].plot(pred[:, 0], pred[:, 1], 'b-', label="CNMP Prediction", linewidth=2.5, alpha=0.8)
    ax[0].scatter(context[:, 0], context[:, 1], color='cyan', edgecolors='black', s=100, label="Context Points", zorder=5)
    ax[0].set_title(f"End-Effector Trajectory (Test Case {test_id})", fontsize=13, fontweight='bold')
    ax[0].set_xlabel("Y Position", fontsize=11)
    ax[0].set_ylabel("Z Position", fontsize=11)
    ax[0].legend()
    ax[0].grid(True, linestyle=':', alpha=0.7)
    
    # Object
    ax[1].plot(truth[:, 2], truth[:, 3], 'k-', label="Ground Truth", linewidth=1.5, alpha=0.9)
    ax[1].plot(pred[:, 2], pred[:, 3], 'r-', label="CNMP Prediction", linewidth=2.5, alpha=0.8)
    ax[1].scatter(context[:, 2], context[:, 3], color='gold', edgecolors='black', s=100, label="Context Points", zorder=5)
    ax[1].set_title(f"Object Trajectory (Test Case {test_id})", fontsize=13, fontweight='bold')
    ax[1].set_xlabel("Y Position", fontsize=11)
    ax[1].set_ylabel("Z Position", fontsize=11)
    ax[1].legend()
    ax[1].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"trajectory_test_{test_id}.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    evaluate()
