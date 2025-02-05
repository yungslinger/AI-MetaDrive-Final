#file to retreive results of SAC agent over unqiue scenarios

from stable_baselines3 import SAC
from metadrive import MetaDriveEnv
import numpy as np

model = SAC.load("sac_metadrive_agent") #loading model

#copy and paste environment extending MetadriveEnv
config = dict(
        use_render=False,
        manual_control=False,
        traffic_density=0.1, 
        num_scenarios=100, #spawns random scenarios of traffic and which lane the agent is spawned
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        map="SSS",  #longer straight
        start_seed=10,
        horizon=1000, #eipisode limit
        vehicle_config=dict(
            show_lidar=True,
            lidar=dict(num_lasers=180, distance=50, num_others=2, gaussian_noise=0.0, dropout_prob=0.0), #vehicle detection
            lane_line_detector=dict(num_lasers=72, distance=50), #lane line detection
            show_side_detector=True,
            side_detector=dict(num_lasers=160, distance=50), #side detection
        ),
)
env = MetaDriveEnv(config)

#vars
episodes = 100
success_count = 0
violations = []
rewards = []

for ep in range(episodes): #loop to store successes, rewards, and violations over all scenarios
    obs = env.reset()[0]
    done = False
    total_reward = 0
    ep_violations = 0
    
    while not done:
        action = model.predict(obs)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        
        if 'cost' in info:
            ep_violations += info['cost']
    
    success_count += info.get('arrive_dest', False) #did my agent arrive??
    violations.append(ep_violations)
    rewards.append(total_reward)

#print it out to my console
print(f"Success Rate: {success_count/episodes}")
print(f"Avg Safety Violations: {np.mean(violations):.2f} ±{np.std(violations):.2f}")
print(f"Avg Return: {np.mean(rewards):.2f} ±{np.std(rewards):.2f}")