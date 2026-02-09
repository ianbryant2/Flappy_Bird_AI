import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from copy import deepcopy
import os
from helper import plot

from FlappyBird.flappy import GameManger

# done is game over state
ALPHA = 0.001
GAMMA = 0.8
TAU = 0.001
MEM_ALPHA = .6
BETA = .6
BATCH_SIZE = 32
BUFFER_SIZE = 700000
EPSILON_START = .5
EPSILON_END = 0.02
EPSILON_DECAY = 1000

class QModel(nn.Module):
    def __init__(self, game_manager):
        super().__init__()
        self.linear1 = nn.Linear(len(game_manager.get_state()), 24)
        self.linear2 = nn.Linear(24, 12)
        self.linear3 = nn.Linear(12, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

    def save(self, run_num, mode, file_name="best_weights.pt"):
        if run_num is None:
            model_folder_path = "./model/" + str(mode)
        else:
            model_folder_path = "./model/" + str(mode) + str(run_num)

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)


class ModelLearner:
    def __init__(self, model, alpha=ALPHA, gamma=GAMMA, tau=TAU):
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.model = model
        self.target_model = deepcopy(model)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

    def train_step(self, batch : TensorDict):

        cur_q_value = torch.gather(self.model(batch['state']), 1, torch.argmax(batch['action'], dim =1).view(-1, 1))

        with torch.no_grad():
            next_q_value = self.target_model(batch['new_state'])
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        target_q_value = batch['reward'] + (1 - batch['done']) * self.gamma * next_q_value.max(dim=1).values.view(-1, 1)
        
        error = (target_q_value - cur_q_value) ** 2
        loss = error.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return error


class Agent:
    def __init__(
        self,
        game_manager : GameManger,
        model=QModel,
        learner=ModelLearner,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_size=BUFFER_SIZE,
        alpha=MEM_ALPHA,
        beta=BETA
    ):
        self.num_games = 0
        self.gamma = 0
        self.memory = TensorDictPrioritizedReplayBuffer(
            alpha=alpha,
            beta=beta,
            eps=1e-6,
            priority_key='td_error',
            storage=LazyTensorStorage(max_size=buffer_size),
            batch_size=BATCH_SIZE
        )
        self.max_priority = torch.tensor(1.0)

        self.game_manager = game_manager

        self.model = model(self.game_manager)
        self.learner = learner(self.model)

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start

    
    def new_game_manager(self, game_manager):
        self.game_manager = game_manager

    def get_state(self):
        return np.array(self.game_manager.get_state(), dtype=int)

    def remember(self, state, action, reward, new_state, done):
        td = TensorDict(
            {'state' : torch.tensor(state, dtype=torch.float),
             'action' : torch.tensor(action, dtype=torch.int),
             'reward' : torch.tensor(reward, dtype=torch.float),
             'new_state' : torch.tensor(new_state, dtype=torch.float),
             'done' : torch.tensor(done, dtype=torch.int),
             'td_error' : self.max_priority
             }
        )
        self.memory.add(td)

    def train(self):
        batch = self.memory.sample()
        error = self.learner.train_step(batch)
        self.max_priority = torch.max(self.max_priority, error.max())
        batch['td_error'] = error
        self.memory.update_tensordict_priority(batch)

    def get_action(self, state, use_epsilon=True):
        self.epsilon = np.interp(
            self.num_games,
            [0, self.epsilon_decay],
            [self.epsilon_start, self.epsilon_end],
        )

        action = [0, 0]

        if use_epsilon:
            if self.epsilon >= random.uniform(0.0, 1.0):
                move = random.randint(0, 1)
                action[move] = 1
            else:
                state_t = torch.tensor(state, dtype=torch.float)
                predict = self.model(state_t)
                move = torch.argmax(predict).item()
                action[move] = 1
        else:
            state_t = torch.tensor(state, dtype=torch.float)
            predict = self.model(state_t)
            move = torch.argmax(predict).item()
            action[move] = 1
            self.game_manager.set_outputs(predict)

        return action

    def save_scores(self, record, total_score, run_num, mode, file_name="scores.txt"):
        if run_num is None:
            model_folder_path = "./model/" + str(mode)
        else:
            model_folder_path = "./model/" + str(mode) + str(run_num)

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        with open(file_path, "w") as f:
            f.write(str(record) + "\n" + str(total_score / self.num_games))


def train(agent : Agent, run_num=None, epochs=None, plotting_scores=False):
    # run_num is the number of different training cycles, used to save correctly
    total_score = 0
    record = 0
    plot_scores = []
    plot_mean_scores = []

    # reset the environment
    agent.game_manager.reset()
    agent.num_games = 0
    num_steps = 0
    while True:
        # get old state
        state = agent.get_state()
        # get move
        action = agent.get_action(state)
        # action then new state
        reward, done, score = agent.game_manager.action_sequence(action)
        new_state = agent.get_state()

        agent.remember(state, action, reward, new_state, done)
        num_steps += 1 

        if num_steps > BATCH_SIZE:
            agent.train()

        if done:
            total_score += score
            agent.game_manager.reset()
            agent.num_games += 1

            if score > record:
                record = score
                agent.model.save(run_num, "train")
                agent.save_scores(record, total_score, run_num, "train")

            if plotting_scores:
                plot_scores.append(score)
                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores, "Training...")

        if epochs == agent.num_games:
            agent.model.save(run_num, "train")
            agent.save_scores(record, total_score, run_num, "train")
            break


def evaluate(agent, run_num=None, epochs=None, plotting_scores=False):
    total_score = 0
    record = 0
    plot_scores = []
    plot_mean_scores = []

    # reset the environment
    agent.game_manager.reset()
    agent.num_games = 0

    while True:
        # get old state
        state = agent.get_state()
        # get move
        action = agent.get_action(state, use_epsilon=False)
        # action then new state
        _, done, score = agent.game_manager.action_sequence(action)

        if done:
            total_score += score
            agent.game_manager.reset()
            agent.num_games += 1

            if score > record:
                record = score
                agent.save_scores(record, total_score, run_num, "evaluate")

            if plotting_scores:
                plot_scores.append(score)
                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores, "Evaluating...")

        if epochs == agent.num_games:
            agent.save_scores(record, total_score, run_num, "evaluate")
            break
