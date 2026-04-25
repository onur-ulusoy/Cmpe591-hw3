import numpy as np
import torch
import environment
from homework4 import Hw5Env, bezier
import hyperparams as hp

def collect_data(num_trajectories=hp.NUM_TRAJECTORIES):
    env = Hw5Env(render_mode="offscreen")
    trajectories = []
    
    for i in range(num_trajectories):
        env.reset()
        p_1 = np.array([0.5, 0.3, 1.04])
        p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p_4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p_1, p_2, p_3, p_4], axis=0)
        curve = bezier(points)

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
        
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            states.append(env.high_level_state())
        
        states = np.stack(states) # (100, 5) -> [ey, ez, oy, oz, h]
        # Add time dimension
        time = np.linspace(0, 1, len(states)).reshape(-1, 1)
        traj = np.concatenate([time, states], axis=1) # (100, 6) -> [t, ey, ez, oy, oz, h]
        trajectories.append(traj)
        
        if (i + 1) % 1 == 0:
            print(f"Collected {i+1}/{num_trajectories} trajectories.")
            np.save(hp.DATA_PATH, np.array(trajectories))

    trajectories = np.array(trajectories)
    np.save(hp.DATA_PATH, trajectories)
    print(f"Data collection complete. Saved to {hp.DATA_PATH}")

if __name__ == "__main__":
    collect_data(hp.NUM_TRAJECTORIES)
