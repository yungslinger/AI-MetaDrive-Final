#file to visualize my trajectories; used for debugginfg

import numpy as np
from stable_baselines3 import SAC
from metadrive import MetaDriveEnv
import time

def visualize_trajectory(trajectory_file="trajectories/trajectories_10.pkl", trajectory_index=0): #change num at end of file to view specific
    env_config = dict(
        use_render=True, #stays on to see trajectories
        manual_control=False,
        traffic_density=0.1,
        num_scenarios=100,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        map="SSS",
        start_seed=10,
        horizon=1000,
        vehicle_config=dict(
            show_lidar=True,
            lidar=dict(num_lasers=180, distance=50, num_others=2, gaussian_noise=0.0, dropout_prob=0.0),
            lane_line_detector=dict(num_lasers=72, distance=50),
            show_side_detector=True,
            side_detector=dict(num_lasers=160, distance=50),
        ),
    )
    
    env = MetaDriveEnv(env_config)
    model = SAC.load("sac_metadrive_agent")
    
    import pickle
    with open(trajectory_file, 'rb') as f:
        trajectories = pickle.load(f)
    
    #get specific trajectory
    trajectory = trajectories[trajectory_index]
    obs, _ = env.reset()
    
    #viualization phase
    for step in range(len(trajectory['states'])):
        action, _ = model.predict(trajectory['states'][step], deterministic=True) #get action from model
        obs, reward, terminated, truncated, info = env.step(action) #step the environment

        env.render()
        if terminated or truncated:
            break
            
    env.close()

if __name__ == "__main__":
    visualize_trajectory()