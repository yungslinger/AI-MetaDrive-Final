# James Gunnlaugsson
# CPSC 4420 - AI Final Term Project
# File for collecting trajectories across all unique scenarios

import numpy as np
from stable_baselines3 import SAC
from metadrive import MetaDriveEnv
import pickle
import os
from tqdm import tqdm

class TrajectoryCollector:
    def __init__(self, env_config=None):

        #env copy and paste
        if env_config is None:
            self.env_config = dict(
                use_render=False,
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
        
        self.env = MetaDriveEnv(self.env_config)
        self.model = SAC.load("sac_metadrive_agent")

    #function for collecting the trajectories to be predicted
    def collect_trajectories(self, num_episodes=100, save_path="collected trajectories"): #save to trajectories folder, run 100 episodes for each scenario
        os.makedirs(save_path, exist_ok=True) #making trajectory path
        all_trajectories = []

        #state info we need to save
        for episode in tqdm(range(num_episodes)): 
            trajectory = {
                'states': [],
                'actions': [],
                'positions': [],
                'velocities': [],
                'headings': [],
                'timestamps': []
            }
            
            obs, _ = self.env.reset()
            done = False
            t = 0
            
            while not done:
                #getting the vehicles current state info
                vehicle = self.env.agent
                trajectory['states'].append(obs) #appending state
                trajectory['positions'].append(vehicle.position)
                trajectory['velocities'].append(vehicle.velocity)
                trajectory['headings'].append(vehicle.heading_theta)
                trajectory['timestamps'].append(t)
                
                #getting tyhe action from trained model
                action, _ = self.model.predict(obs, deterministic=True)
                trajectory['actions'].append(action)
                
                #same step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                t += 1
            
            #convert the list to a numpy array for manipulation
            for key in trajectory:
                trajectory[key] = np.array(trajectory[key])
            all_trajectories.append(trajectory)
            
            #saving trajectories to a file after every 10 episodes (pillow format)
            if (episode + 1) % 10 == 0:
                with open(os.path.join(save_path, f'trajectories_{episode+1}.pkl'), 'wb') as f:
                    pickle.dump(all_trajectories, f)

        return all_trajectories


if __name__ == "__main__":
    collector = TrajectoryCollector()
    trajectories = collector.collect_trajectories(num_episodes=100) #collect trajectories over 100eps
    print(f"Collected {len(trajectories)} trajectories")
    print(f"Average trajectory length: {np.mean([len(t['states']) for t in trajectories]):.2f} steps")