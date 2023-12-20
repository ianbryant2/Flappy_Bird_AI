from collections import deque
import random
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
from helper import plot
from FlappyBird import flappy

#done is game over state
ALPHA = .001
GAMMA = .8
BATCH_SIZE = 32
BUFFER_SIZE = 700000
EPSILON_START = 1.0
EPSILON_END = .02
EPSILON_DECAY=1000

class Model(nn.Module):
    '''Represents the Deep Q Learning model'''
    def __init__(self, gm : flappy.GameManager):
        super().__init__()
        self.linear1 = nn.Linear(len(gm.get_state()),24)
        self.linear2 = nn.Linear(24,12)
        self.linear3 = nn.Linear(12,2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
        
    def save(self, run_num, type, file_name='best_weights.pt'):
        '''Will save the model in the folder model in the dir that the script was run in'''
        if run_num == None:
            model_folder_path = './model/' + str(type) 
        else: 
            model_folder_path = './model/' + str(type) + str(run_num)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)

        torch.save(self.state_dict(), file_name)

class ModelTrainer:
    '''Represents the trainer that will use the Q learning algorthim to train the model'''
    def __init__(self, model : Model, alpha=ALPHA, gamma=GAMMA):
        self.gamma = gamma
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr = alpha)
        self.criterion = nn.HuberLoss()

    def train_step(self, state, action, reward, new_state, done):
        '''Will train a single step'''
        state = torch.tensor(np.array(state), dtype=torch.float)
        new_state = torch.tensor(np.array(new_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state) # Returns a torch.Tensor
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(new_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target) #Returns a torch.Tensor
        loss.backward()
        self.optimizer.step()


class Agent:
    '''Represents an deep q agent that will interact in the game environment'''
    def __init__(self, gm : flappy.GameManager, model=Model, model_trainer=ModelTrainer, epsilon_start = EPSILON_START, epsilon_end = EPSILON_END, epsilon_decay= EPSILON_DECAY, buffer_size = BUFFER_SIZE):
        self.num_games = 0
        self.gamma = 0
        self.memory = deque(maxlen=buffer_size)
        self.gm = gm
        self.model = model(self.gm)
        self.trainer = model_trainer(self.model)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay


    def get_state(self) -> npt.NDArray:
        '''Will return an array that represents the current state'''
        return np.array(self.gm.get_state(), dtype=int)

    def remember(self, state : npt.NDArray, action : list[int], reward : float, new_state : npt.NDArray, done : bool) -> None:
        '''Will remeber information about current state and results from action'''
        self.memory.append((state, action, reward, new_state, done))

    def train_long_memory(self) -> None:
        '''Will train everything that is in the memory
        If there are more than BATCH_SIZE elements in the array then random elements will be picked from memory'''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        #TODO look into types of these variables
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def get_action(self, state, epsilon = True) -> list[int]:
        '''Will use the Epsilon-Greedy policy in order to determine what action to take
        An agent using this policy will gradually use less and less random inputs throughout training and 
        will more and more get inputs from the model'''
        self.epsilon = np.interp(self.num_games, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
        action = [0,0]
        if epsilon:
            if self.epsilon >= random.uniform(0.0,1.0):
                move = random.randint(0,1)
                action[move]=1
            else:
                state_t = torch.tensor(state, dtype=torch.float) #turn the state array into a tensor
                predict = self.model(state_t)
                move = torch.argmax(predict).item()
                action[move]=1
        else:
            state_t = torch.tensor(state, dtype=torch.float) #turn the state array into a tensor
            predict = self.model(state_t)
            move = torch.argmax(predict).item()
            action[move]=1
            self.gm.set_outputs(predict)
           
        return action
     
    def save_scores(self, record, total_score, run_num, type, file_name='scores.txt') -> None:
        '''Will save the information about scores in a directory called model in the running directory'''
        if run_num == None:
            model_folder_path = './model/' + str(type)
        else: 
            model_folder_path = './model/' + str(type) + str(run_num)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        with open(file_name, 'w') as r:
            r.write(str(record) + '\n' + str(total_score/self.num_games))

#TODO need to look into how pygame can stop this from happening
def train(agent : Agent, run_num : int=None, epochs:int=None, plotting_scores = False) -> None:
    '''Will train an agent for the given amount of epochs. The run num is how many training cycles'''
    total_score = 0
    record = 0
    agent = agent
    plot_scores = []
    plot_mean_scores = []
   
   #resets the enviroment
    agent.gm.reset()
    agent.num_games = 0

    while True:
        # get old state
        state = agent.get_state()
        #get move
        action = agent.get_action(state)
        # action then new state
        reward, done, score = agent.gm.action_sequence(action)
        new_state = agent.get_state() 
        agent.remember(state, action, reward, new_state, done)

        if done:
            total_score += score
            agent.gm.reset()
            agent.num_games+=1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(run_num, 'train')
                agent.save_scores(record, total_score, run_num, 'train')

            if plotting_scores: 
                plot_scores.append(score)
                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores, "Training...")
      
        if epochs == agent.num_games:
            agent.model.save(run_num, 'train')
            agent.save_scores(record, total_score, run_num, 'train')
            break

def evaluate(agent : Agent, run_num:int=None, epochs:int=None, plotting_scores=False) -> None:
    '''Will evaluate an agent given amount of epochs. The run num is how many evaluation cycles'''
    total_score = 0
    record = 0
    agent = agent
    plot_scores = []
    plot_mean_scores = []

    #resets the enviroment
    agent.gm.reset()
    agent.num_games = 0

    while True:
        # get old state
        state = agent.get_state()
        #get move
        action = agent.get_action(state, epsilon = False)
        # action then new state
        _, done, score = agent.gm.action_sequence(action)

        if done:
            total_score += score
            agent.gm.reset()
            agent.num_games+=1

            if score > record:
                record = score
                agent.save_scores(record, total_score, run_num, 'evaluate')
                
            if plotting_scores: 
                plot_scores.append(score)
                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores, "Evaluating...")

        if epochs == agent.num_games:
            agent.save_scores(record, total_score, run_num, 'evaluate')
            break

