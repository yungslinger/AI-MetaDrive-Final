# James Gunnlaugsson
# CPSC 4420 - AI Final Term Project
# Custom SAC environment for RL training in MetaDrive

import argparse
import logging
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from metadrive import MetaDriveEnv
from metadrive.constants import HELP_MESSAGE
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

#tqdm to get a progress bar during training (thank god for documentation!!)
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Total Steps") #progress basr on start
        
    def _on_step(self):
        self.pbar.n = self.num_timesteps
        self.pbar.refresh() #refresh progress
        return True
        
    def _on_training_end(self): 
        self.pbar.close() #bye bye

class GymCompatibleMetaDriveEnv(MetaDriveEnv):
    def reset(self, seed=None, options=None):
        #override reseting to make compatible with gym API.
        if seed is not None:
            self.seed(seed)
        obs, info = super().reset()
        return obs, info

#custom environment setup for my agent, extending from MetaDriveEnv
def create_env():
    config = dict(
        use_render=True, #toggle on/off for visualization in metadrive
        manual_control=False,
        traffic_density=0.1, #smaller traffic size on bigger road for better training
        num_scenarios=100, #100 unique scenarios that display different traffic layouts
        random_agent_model=False,
        random_lane_width=False, 
        random_lane_num=False,  
        map="SSS",  #long straight road
        start_seed=10,
        horizon=1000, #episode limit to prune repetitive actions
        vehicle_config=dict(
            show_lidar=True,
            lidar=dict(num_lasers=180, distance=50, num_others=2, gaussian_noise=0.0, dropout_prob=0.0), #lidar for vehicle detection in front
            lane_line_detector=dict(num_lasers=72, distance=50), #lane line detector (duh)
            show_side_detector=True,
            side_detector=dict(num_lasers=160, distance=50), #side lidar
        ),
    )
    env = GymCompatibleMetaDriveEnv(config) #wrapped in gym to make stable-baselines SAC implementation possible
    env = TimeLimit(env, max_episode_steps=1000)
    return Monitor(env)

#function to train my SAC agent
def train_sac(env, timesteps=250000):
    check_env(env.envs[0])
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_metadrive/") #using MlpPolicy
    callback = ProgressCallback(timesteps)
    model.learn(total_timesteps=timesteps, callback=callback) #train model and callback for progress
    model.save("sac_metadrive_agent") #save agent after training
    print("Model training complete; saved as sac_metadrive_agent.")
    return model


#evaluation function to see how agent performed after training
def evaluate_agent(env, model, episodes=5): #can change based on how you wanna evaluate
    """
    Evaluate the trained agent in the environment.
    """
    for ep in range(episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = np.array(action).reshape(1, -1)
            try:
                #is it this gym API?
                obs, reward, done, truncated, info = env.step(action)
                done = done[0] or truncated[0]
            except ValueError:
                #or this gym API??
                obs, reward, done, info = env.step(action)
                done = done[0]
            total_reward += reward[0]
            env.render()
            
        print(f"Episode {ep + 1}: Total Reward = {total_reward}") #print results on console


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the SAC agent")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the SAC agent")
    parser.add_argument("--timesteps", type=int, default=250000, help="Training timesteps for SAC") #easy changing of timesteps
    args = parser.parse_args()

    env = DummyVecEnv([create_env]) #creating dummy vec environment for baselines3 training

    if args.train:
        print("Starting training...")
        sac_model = train_sac(env, timesteps=args.timesteps)
    elif args.evaluate:
        print("Evaluating trained model...")
        try:
            #load trained model and eval
            sac_model = SAC.load("sac_metadrive_agent")
            evaluate_agent(env, sac_model) 
        except FileNotFoundError:
            print("No trained model found.") #error checking

    else: #manual mode --taken from drive_in_single_agent_env
        print("Running environment in manual mode.")
        #same env for manual mode (early testing on system) 
        config = dict(
            use_render=True,
            manual_control=True,
            traffic_density=0.1,
            num_scenarios=1000,
            random_agent_model=False,
            random_lane_width=True,
            random_lane_num=False,
            map="SSS",
            start_seed=10,
            vehicle_config=dict(show_lidar=True),
        )
        env = MetaDriveEnv(config)
        try:
            obs, _ = env.reset(seed=21)
            print(HELP_MESSAGE)
            for _ in range(1000000000):
                action = [0, 0] 
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                if terminated or truncated:
                    env.reset()
        finally:
            env.close()