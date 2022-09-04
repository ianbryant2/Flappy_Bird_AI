from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
from FlappyBird import flappy
#done is game over state
ALPHA = .001
GAMMA = .8
BATCH_SIZE = 32
BUFFER_SIZE = 700000
EPSILON_START = 1.0
EPSILON_END = .02
EPSILON_DECAY=1000
NUM_OF_FRAMES_TIL_CHECK = 2
STEPS_TIL_UPDATE = 10000

class model(nn.Module):
    def __init__(self, gm):
        super().__init__()
        self.linear1 = nn.Linear(len(gm.getState()),24)
        self.linear2 = nn.Linear(24,12)
        self.linear3=nn.Linear(12,2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
        
    def save(self, run_num, type, file_name='best_weights.pt'):
        if run_num == None:
            model_folder_path = './model/' + str(type) 
        else: 
            model_folder_path = './model/' + str(type) + str(run_num)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)

        torch.save(self.state_dict(), file_name)

class model_trainer:
    def __init__(self, model):
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.optimizer = torch.optim.Adam(model.parameters(), lr = self.alpha)
        self.criterion = nn.HuberLoss()

    def train_step(self, state, action, reward, new_state, done, target, model):
        state = torch.tensor(state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = model(state)
        # array holding format needed
        # array will be updated in for loop undeneath to get target network Q values
        tar = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(target(new_state[idx]))

            tar[idx][torch.argmax(action[idx]).item()] = Q_new
    
        
        self.optimizer.zero_grad()
        loss = self.criterion(tar, pred)
        loss.backward()
        self.optimizer.step()

#TODO make model trainers that are subclasses of model trainer that have target and non target networks
#TODO see if you can make AI learn based off of other experiences. Can put experience in replay buffer and combine seen and personal experience

class agent:
    def __init__(self, type_game, gm=flappy.gameManager, model=model, model_trainer=model_trainer):
        self.num_games = 0
        self.num_steps = 0
        self.gamma = 0
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.type_game = type_game
        self.gm = gm(type_game, NUM_OF_FRAMES_TIL_CHECK)
        self.model = model(self.gm)
        self.target = model(self.gm)
        self.trainer = model_trainer(self.model)
        self.epsilon_start = EPSILON_START
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY

    def get_state(self):
        return np.array(self.gm.getState(), dtype=int)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones, self.target, self.model)

    def get_action(self, state, epsilon = True):
        self.epsilon = np.interp(self.num_games, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
        action = [0,0]
        if epsilon:
            if self.epsilon >= random.uniform(0.0,1.0):
                move = random.randint(0,1)
                action[move]=1
            else:
                state_tensor = torch.tensor(state, dtype=torch.float) #turn the state array into a tensor
                predict = self.model(state_tensor)
                move = torch.argmax(predict).item()
                action[move]=1
        else:
            state_t = torch.tensor(state, dtype=torch.float) #turn the state array into a tensor
            predict = self.model(state_t)
            move = torch.argmax(predict).item()
            action[move]=1
        return action
    
    def save_scores(self, record, total_score, run_num, type, file_name = 'scores.txt'):
        if run_num == None:
            model_folder_path = './model/' + str(type)
        else: 
            model_folder_path = './model/' + str(type) + str(run_num)
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        with open(file_name, 'w') as r:
            r.write(str(record) + '\n' + str(total_score/self.num_games))

def train(agent, run_num = None, epochs = None):  #run num is the number of differnt training cycles. used to save correctly
    total_score = 0
    record = 0
    agent = agent
   
   #resets the enviroment
    agent.gm.reset()
    agent.num_games = 0
    print(agent.model.linear1.weight)

    while True:
        # get old state
        state = agent.get_state()
        #get move
        action = agent.get_action(state)
        # action then new state
        reward, done, score = agent.gm.actionSequence(action)
        #updates the counter of the number of steps taken
        agent.num_steps += 1
        #after certain number of steps update target network
        if agent.num_steps == STEPS_TIL_UPDATE:
            agent.target = agent.model

        new_state = agent.get_state() 

        agent.remember(state, action, reward, new_state, done)
        if done:
            total_score += score
            agent.gm.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(run_num, 'train')
                agent.save_scores(record, total_score, run_num, 'train')
                
        if epochs == agent.num_games:
            agent.model.save(run_num, 'train')
            agent.save_scores(record, total_score, run_num, 'train')
            break

def evaluate(agent, run_num = None, epochs = None):
    total_score = 0
    record = 0
    agent = agent

    #resets the enviroment
    agent.gm.reset()
    agent.num_games = 0

    while True:
        # get old state
        state = agent.get_state()
        #get move
        action = agent.get_action(state, epsilon = False)
        # action then new state
        _, done, score = agent.gm.actionSequence(action)

        if done:
            total_score += score
            agent.gm.reset()
            agent.num_games+=1

            if score > record:
                record = score
                agent.save_scores(record, total_score, run_num, 'evaluate')

        if epochs == agent.num_games:
            agent.save_scores(record, total_score, run_num, 'evaluate')
            break