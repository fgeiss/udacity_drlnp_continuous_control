import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from configparser import ConfigParser


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(object):
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, hyperparameters):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            hyperparameters (dict): dictionary with h
        """
        
        # initialize the random generator to ensure reproducibility
        random.seed(random_seed)

        
        # Read hyperparameters from Config dict
        self.hyperparamaters = hyperparameters
        
        self.state_size = state_size
        self.action_size = action_size
        self.step_counter = 0
        self.epsilon = float(self.hyperparamaters['EPSILON_START'])

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), 
                                          lr=float(self.hyperparamaters['LR_ACTOR']))

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                           lr=float(self.hyperparamaters['LR_CRITIC']),
                                           weight_decay=float(self.hyperparamaters['WEIGHT_DECAY']))
        
        # Noise process
        self.noise = OUNoise(action_size)

        # Replay memory
        self.memory = ReplayBuffer(action_size, 
                                   int(self.hyperparamaters['BUFFER_SIZE']), 
                                   int(self.hyperparamaters['BATCH_SIZE']))
        
        # Hard update so that weights of local and target are identical 
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
    
    def step_add_to_memory(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        self.step_counter += 1
        # Learn, if enough samples are available in memory
        self.step_learn()

    def step_learn(self):
        if self.step_counter % int(self.hyperparamaters['LEARN_EVERY']) == 0:
            if len(self.memory) > int(self.hyperparamaters['BATCH_SIZE']):
                for _ in range(int(self.hyperparamaters['LEARN_TIMES'])):
                    experiences = self.memory.sample()
                    self.learn(experiences, float(self.hyperparamaters['GAMMA']))
        
            
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        
        scalar = False
        with torch.no_grad():
            if state.dim() == 1:
                state.unsqueeze_(0)
                scalar = True
            action = self.actor_local(state).cpu().data.numpy()
            if scalar:
                action =np.squeeze(action)
                
        self.actor_local.train()
        
        if add_noise:
            action += self.epsilon*self.noise.sample()
        
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()
        
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon*float(self.hyperparamaters['EPSILON_DECAY']),
                           float(self.hyperparamaters['EPSILON_END']))
        
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        indexes, states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # -----------------------------update td_error for ranking ------------- #
        deltas = list(torch.abs(Q_expected-Q_targets).cpu().detach().numpy().flatten())
        for index, delta in zip(indexes, deltas):
            self.memory.td_error_update(index,delta)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, float(self.hyperparamaters['TAU']))
        self.soft_update(self.actor_local, self.actor_target, float(self.hyperparamaters['TAU']))
        
        # ------------------------ update epsilon and noise -------------------- #
        self.update_epsilon()
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)            

            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["index", "state", "action", "reward", "next_state", "done"])
        self.td_errors = deque(maxlen=buffer_size)
        self.index = 0 #Global counter to help storing and updating td_error values
        self.prioritized_replay = False
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(self.index, state, action, reward, next_state, done)
        self.td_errors.append(0.0)
        self.memory.append(e)
        self.index+=1
        
        
    def td_error_update(self,index,td_error):
        start_index = self.memory[0].index
        self.td_errors[index-start_index] = td_error
    
    def sample(self):
        
        if self.prioritized_replay:
            start = time.time()
            """Compute probabilities according to prioritization """
            rank = [i+1.0 for i in range(len(self.memory))]
            D = np.reciprocal(rank)
            D_alpha = np.power(D,PRIO_ALPHA)
            D_alpha_sum = sum(D_alpha)
            ProbsRaw = D_alpha/D_alpha_sum

            L=list(zip(self.td_errors,range(len(self.td_errors))))
            Ls = sorted(L,reverse = True)
            Probs = np.zeros(len(self.td_errors))
            for i,ind in enumerate(Ls):
                Probs[ind[1]] = ProbsRaw[i]

            sample=np.random.choice(range(len(self.memory)),self.batch_size, p=Probs)

            """Randomly sample a batch of experiences from memory."""
            experiences = [self.memory[i] for i in sample]
            end = time.time()
        else:
            experiences = random.sample(self.memory, k=self.batch_size)
        
        indexes = [e.index for e in experiences if e is not None]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (indexes, states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)