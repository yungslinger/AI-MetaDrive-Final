# James Gunnlaugsson
# CPSC 4420 - AI Final Term Project
# File for LSTM construction for predicting trajectories and visualizing gifs

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

#custom PyTorch dataset for traj data
#converts traj files into sequence of inputs and targets for training and eval
class TrajectoryDataset(Dataset):
    def __init__(self, data_path, seq_length=10, pred_length=30): #taking 10 input from our agent and predicting 30 output
        self.trajectories = []

        #DONT FORGET THIS
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        #load trajectory files
        for file in glob.glob(f"{data_path}/trajectories_*.pkl"):
            with open(file, 'rb') as f:
                self.trajectories.extend(pickle.load(f))
        
        #taking trajectories and putting them into their sequences
        self.sequences = []
        for traj in self.trajectories:
            positions = traj['positions'] #pos for each timestep
            for i in range(len(positions) - (seq_length + pred_length)):
                self.sequences.append({
                    'input': positions[i:i+seq_length], #input (10 timesteps)
                    'target': positions[i+seq_length:i+seq_length+pred_length]
                })

    def __len__(self): #get sequences in dataset
        return len(self.sequences)

    def __getitem__(self, idx): #gets input-target pair at specific index
        seq = self.sequences[idx]

        #returns historacal posseq and future pos seq through pytorch
        return torch.FloatTensor(seq['input']), torch.FloatTensor(seq['target'])

#pytorch LSTM model for traj predicting; take past data and predict future
class TrajectoryPredictor(nn.Module): 
    def __init__(self, input_dim=2, hidden_dim=64, pred_length=30): #hidden dimension of 64 (capacity for LSTM to learn); may increase
        super().__init__()
        self.pred_length = pred_length
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_length * input_dim)
        )

    #forward pass of model (passed implciitly)
    #inputting tensor w/ shape (batch_size, seq_length, and input_dim)
    #outputs predicted positions of shape
    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x) #LTSM output for all timesteps
        lstm_last = lstm_out[:, -1, :] 
        output = self.fc(lstm_last)
        return output.view(batch_size, self.pred_length, -1) #reshaping for outputting

#training for the LTSM using collected traj data
def train_predictor(data_path="trajectories", epochs=50): #50 epochs for the LSTM
    
    #setup for prediction training
    device = torch.device("cpu") #god i wish i had cuda for this

    #create dataset and loader
    dataset = TrajectoryDataset(data_path) #trajectories
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #initialize model, adam optimizer, and loss func
    model = TrajectoryPredictor().to(device)
    optimizer = optim.Adam(model.parameters()) #adam optimizer
    criterion = nn.MSELoss() #mse loss func

    #train this darn thing
    for epoch in range(epochs):
        model.train()
        total_loss = 0 #tracking loss
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"): #visual check in console
            inputs, targets = inputs.to(device), targets.to(device) #move data to device
            
            #reset grads, forward pass, compute loss, and update weights
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() #tracking loss per epoch
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.6f}")
    
    torch.save(model.state_dict(), "trajectory_predictor.pth") #save the model as pth fiel (model-state dict)
    return model

def create_trajectory_gif(model, test_trajectory, output_path="trajectory.gif"):
    #set model to eval mode
    model.eval()
    
    #take first 10 positions as an input sequence
    input_seq = torch.FloatTensor(test_trajectory['positions'][:10]).unsqueeze(0)
    
    #generating prediction -- added no_grad for disabling gradient calculation for interference
    with torch.no_grad():
        predicted_trajectory = model(input_seq).squeeze(0).numpy()
    
    #create matplotlib figure and axes for animation
    fig, ax = plt.subplots(figsize=(10, 10))

    #get actual positions from 10-40 after input sequence (ground truth)
    actual = test_trajectory['positions'][10:40]
    
    def animate(i):
        ax.clear() #clear previous frame
        
        ax.plot(actual[:i+1, 0], actual[:i+1, 1], 'b-', label='Actual') #plot trajectory up to current frame

        ax.plot(predicted_trajectory[:i+1, 0], predicted_trajectory[:i+1, 1], 'r--', label='Predicted') #plot predicted trajectory up to current frame

        ax.legend() #legend to identify lines

        #axis limit dynamically based on data (trial-and-error!!!); padding of 5 on each sie
        ax.set_xlim([min(actual[:, 0].min(), predicted_trajectory[:, 0].min()) - 5,
                     max(actual[:, 0].max(), predicted_trajectory[:, 0].max()) + 5])
        ax.set_ylim([min(actual[:, 1].min(), predicted_trajectory[:, 1].min()) - 5,
                     max(actual[:, 1].max(), predicted_trajectory[:, 1].max()) + 5])
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        
        ax.set_title(f'Vehicle Trajectory Prediction - Frame {i}') #add frame counter
        
    #create 30fps animation saved using PIL
    anim = animation.FuncAnimation(fig, animate, frames=30, interval=100)
    anim.save(output_path, writer='pillow')
    plt.close()

if __name__ == "__main__":
    #trainin
    model = train_predictor()
    
    #loop to generate 1 visualization per traj file
    for i in range(10, 110, 10):  #named trajectories_10.pkl, trajectories_20.pkl, yadda yada ya...
        try:
            with open(f"trajectories/trajectories_{i}.pkl", 'rb') as f:
                trajectories = pickle.load(f)
                create_trajectory_gif(model, trajectories[0], f"trajectory_file_{i}.gif") #generate the gif
        except FileNotFoundError:
            print(f"File trajectories_{i}.pkl not found, skipping.")