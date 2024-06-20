from .agent import Agent 
from .snake_game import SnakeGame
import numpy as np
import matplotlib.pyplot as plt 
import torch
from torch import nn
from tqdm import tqdm
from pathlib import Path

class ReinforceAgent(Agent):
    def __init__(self, create_visual=False, model_path="/models/model.pth", perm="read"):
        super().__init__(create_visual)
        self.state = np.zeros(12)
        self.policy_net = PolicyNetwork(12, 64, 4)  # Policy network that maps states to action probabilities
        self.model_path = model_path
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)  # Optimizer for policy network
        self.num_episodes = 100  # Number of episodes to collect before updating policy
        self.gamma = 0.99  # Discount factor
        self.create_visual = create_visual # Whether to create visualizations
        if perm == "read":
            print("Loading model from file.")
            self.policy_net.load_state_dict(torch.load(str(Path(__file__).parent.absolute()) + model_path))

    def _learn(self):
        reward_list = []
        num_epochs = 150
        loss_list = []
        for epoch in tqdm(range(num_epochs)): # Don't allow the snake to kill itself if possible
            trajectories, total_rewards = self._collect_trajectories()
            reward_list.append((epoch + 1, total_rewards))
            
            discounted_rewards = self._compute_discounted_rewards(trajectories)
            
            policy_loss = self._compute_policy_gradient(trajectories, discounted_rewards)
            loss_list.append(policy_loss.item())
            
            # Update policy parameters
            self.policy_optimizer.zero_grad()
            policy_loss.backward()

            # Gradient Clipping
            max_grad_norm = 0.5
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)

            # Ensure minimum gradient values
            min_grad_value = 0.001
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data = torch.clamp(param.grad.data, min=min_grad_value)


            # L2 Regularization
            # l2_lambda = 0.1
            # for param in self.policy_net.parameters():
            #     policy_loss += l2_lambda * torch.norm(param)

            # Monitor gradients
            # for name, param in self.policy_net.named_parameters():
            #     print(f'Gradient - {name}: {param.grad}')

            # Monitor loss values
            # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {policy_loss.item()}')

            # Monitor parameter updates
            # for name, param in self.policy_net.named_parameters():
            #     print(f'Updated Parameter - {name}: {param.data}')

            self.policy_optimizer.step()

        self.plot(range(num_epochs), loss_list, "Epochs", "Loss", "Policy Loss")
        self.plot([x[0] for x in reward_list], [np.mean(x[1]) for x in reward_list], "Epochs", "Average Reward", "Average Reward vs Epochs")
        return


    def _collect_trajectories(self):
        # Collect trajectories by interacting with the environment
        trajectories = []
        total_rewards = []
        for episode in range(self.num_episodes):
            self.simulator = SnakeGame(self.create_visual)
            episode_trajectory = []  # List to store (state, action, reward) tuples for the current episode
            done = False
            total_reward = 0
            actions = []
            while done == False:
                # Select action using the policy network
                state_tensor = torch.FloatTensor(self.state).unsqueeze(0)  # Convert state to tensor
                if (torch.isnan(state_tensor).any()):
                    print("NaN detected in state tensor.")
                    exit()
                action_probs = self.policy_net(state_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()  # Sample action from action distribution

                # Take action in the environment
                actions.append(action)
                next_state, reward, done = self.simulator.step(action + 1, total_reward)

                # Calculate and store trajectory and rewards
                total_reward += reward
                episode_trajectory.append((self.state, action, reward))

                self._change_the_state_representation(next_state)

            total_rewards.append(total_reward)
            trajectories.append(episode_trajectory)

        return trajectories, total_rewards
    
    def _compute_discounted_rewards(self, trajectories):
        # Compute returns for each time step in the trajectories
        discounted_rewards_list = []
        for episode_trajectory in trajectories:
            episode_returns = []
            cumulative_return = 0
            for _, _, reward in reversed(episode_trajectory):
                cumulative_return = reward + self.gamma * cumulative_return # Discounted cumulative reward
                episode_returns.insert(0, cumulative_return) # Insert at the beginning of the list
            # all_returns.extend(episode_returns)
            discounted_rewards_list.append(episode_returns)
        return discounted_rewards_list
    
    def _compute_policy_gradient(self, trajectories, discounted_rewards_list):
        policy_losses = []
        for episode_trajectory, discounted_rewards in zip(trajectories, discounted_rewards_list):
            for episode_snapshot, discounted_reward in zip(episode_trajectory, discounted_rewards):
                state, action, _ = episode_snapshot
                state_tensor = torch.Tensor(state).unsqueeze(0)
                action_probs = self.policy_net(state_tensor)
                if torch.isnan(action_probs).any():
                    print("NaN detected in action probabilities.")
                    exit()
                # Log probability of the selected action
                log_prob = torch.log(action_probs.squeeze(0)[action])
                if torch.isnan(log_prob):
                    print("NaN detected in log probability.")
                    exit()
                policy_losses.append(-abs(log_prob) * discounted_reward)  # Use the episode return for all time steps in the episode
        loss = torch.stack(policy_losses).sum()

        # Entropy Regularization
        entropy = torch.distributions.Categorical(action_probs).entropy()
        entropy_loss = -0.01 * entropy.mean()
        loss += entropy_loss

        if torch.isnan(loss):
            print("NaN detected in loss.")

        return loss

    def _evaluate( self , trial_count , create_visual = False ) : 
        """ 
        1. Test the current model up to trial_count time. 
        2. Return the average result. 
        """ 
        pass 

    def _change_the_state_representation( self , state )  : 
        """ 
        Turn the raw state representation, to a useful state representation. 
        """ 
        if state[1] == "down":
            self.state[0] = 1
        elif state[1] == "up":
            self.state[1] = 1
        elif state[1] == "left":
            self.state[2] = 1
        elif state[1] == "right":
            self.state[3] = 1
        self.state[4] = state[9][0] # apple is to the right
        self.state[5] = state[9][1] # apple is to the left
        self.state[6] = state[9][2] # apple is down
        self.state[7] = state[9][3] # apple is up
        self.state[8] = state[8][0] # blocked right
        self.state[9] = state[8][1] # blocked left
        self.state[10] = state[8][2] # blocked down
        self.state[11] = state[8][3] # blocked up
        return
    
    def save(self):
        torch.save(self.policy_net.state_dict(), str(Path(__file__).parent.absolute()) + self.model_path)

    def plot(self, x, y, x_label, y_label, title):
        plt.plot(x, y, 'o', label='Original data')
        
        # Calculate the best-fit line
        coefficients = np.polyfit(x, y, 1)
        polynomial = np.poly1d(coefficients)
        regression_line = polynomial(x)
        plt.plot(x, regression_line, 'r--', label='Regression line')
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.show()

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)

        if torch.isnan(logits).any():
            print("NaN detected in logits.")
            exit()
        
        # Apply log-softmax for numerical stability
        log_probs = torch.log_softmax(logits, dim=-1)

        if torch.isnan(log_probs).any():
            print("NaN detected in log probabilities.")
            exit()
        
        # Compute softmax probabilities using log-softmax and exponentiation
        probs = torch.exp(log_probs)
        
        return probs