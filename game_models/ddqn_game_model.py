import numpy as np
import os
import random
import shutil
from statistics import mean

from tensorflow.python.keras.backend import dtype
from game_models.base_game_model import BaseGameModel
from convolutional_neural_network import ConvolutionalNeuralNetwork

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple, deque



GAMMA = 0.99
MEMORY_SIZE = 900000
# BATCH_SIZE = 128
BATCH_SIZE = 128
TRAINING_FREQUENCY = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 40000
# MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 5000
# REPLAY_START_SIZE = 50000
REPLAY_START_SIZE = 5000

# EXPLORATION_MAX = 1.0
EXPLORATION_MAX = 0.95
EXPLORATION_MIN = 0.1
EXPLORATION_TEST = 0.02
# EXPLORATION_STEPS = 850000
EXPLORATION_STEPS = 850000
EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class DDQNGameModel(BaseGameModel):

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path):
        BaseGameModel.__init__(self, game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        # self.ddqn = ConvolutionalNeuralNetwork(self.input_shape, action_space).model
        # if os.path.isfile(self.model_path):
        #     self.ddqn.load_weights(self.model_path)

    def _save_model(self):
        self.ddqn.save_weights(self.model_path)


class DDQNSolver(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space):
        testing_model_path = "./output/neural_nets/" + game_name + "/ddqn/testing/model.h5"
        assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN testing",
                               input_shape,
                               action_space,
                               "./output/logs/" + game_name + "/ddqn/testing/" + self._get_date() + "/",
                               testing_model_path)

    def move(self, state):
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)
        q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])


# class DDQNTrainer(DDQNGameModel):

#     def __init__(self, game_name, input_shape, action_space):
#         DDQNGameModel.__init__(self,
#                                game_name,
#                                "DDQN training",
#                                input_shape,
#                                action_space,
#                                "./output/logs/" + game_name + "/ddqn/training/" + self._get_date() + "/",
#                                "./output/neural_nets/" + game_name + "/ddqn/" + self._get_date() + "/model.h5")

#         if os.path.exists(os.path.dirname(self.model_path)):
#             shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
#         os.makedirs(os.path.dirname(self.model_path))

#         self.ddqn_target = ConvolutionalNeuralNetwork(self.input_shape, action_space).model
#         self._reset_target_network()
#         self.epsilon = EXPLORATION_MAX
#         self.memory = []

#     def move(self, state):
#         if np.random.rand() < self.epsilon or len(self.memory) < REPLAY_START_SIZE:
#             return random.randrange(self.action_space)
#         q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
#         return np.argmax(q_values[0])

#     def remember(self, current_state, action, reward, next_state, terminal):
#         self.memory.append({"current_state": current_state,
#                             "action": action,
#                             "reward": reward,
#                             "next_state": next_state,
#                             "terminal": terminal})
#         if len(self.memory) > MEMORY_SIZE:
#             self.memory.pop(0)

#     def step_update(self, total_step):
#         if len(self.memory) < REPLAY_START_SIZE:
#             return

#         if total_step % TRAINING_FREQUENCY == 0:
#             loss, accuracy, average_max_q = self._train()
#             self.logger.add_loss(loss)
#             self.logger.add_accuracy(accuracy)
#             self.logger.add_q(average_max_q)

#         self._update_epsilon()

#         if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
#             self._save_model()

#         if total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
#             self._reset_target_network()
#             print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
#             print('{{"metric": "total_step", "value": {}}}'.format(total_step))

#     def _train(self):
#         batch = np.asarray(random.sample(self.memory, BATCH_SIZE))
#         if len(batch) < BATCH_SIZE:
#             return

#         current_states = []
#         q_values = []
#         max_q_values = []


#         current_states = np.asarray(tuple(map(lambda s: np.asarray(s['current_state']),
#                                           batch))).astype(np.float64)
#         next_states = np.asarray(tuple(map(lambda s: np.asarray(s['next_state']),
#                                           batch))).astype(np.float64)
#         next_states_prediction = self.ddqn_target.predict(next_states)
#         next_q_value = np.max(next_states_prediction, axis=1)


#         for entry in batch:
#             current_state = np.expand_dims(np.asarray(entry["current_state"]).astype(np.float64), axis=0)
#             current_states.append(current_state)
#             next_state = np.expand_dims(np.asarray(entry["next_state"]).astype(np.float64), axis=0)
#             next_state_prediction = self.ddqn_target.predict(next_state).ravel()
#             next_q_value = np.max(next_state_prediction)
#             q = list(self.ddqn.predict(current_state)[0])
#             if entry["terminal"]:
#                 q[entry["action"]] = entry["reward"]
#             else:
#                 q[entry["action"]] = entry["reward"] + GAMMA * next_q_value
#             q_values.append(q)
#             max_q_values.append(np.max(q))

#         fit = self.ddqn.fit(np.asarray(current_states).squeeze(),
#                             np.asarray(q_values).squeeze(),
#                             batch_size=BATCH_SIZE,
#                             verbose=0)
#         loss = fit.history["loss"][0]
#         accuracy = fit.history["accuracy"][0]
#         return loss, accuracy, mean(max_q_values)



#     def _train(self):
#         # torch version of training
#         batch = np.asarray(random.sample(self.memory, BATCH_SIZE))
#         if len(batch) < BATCH_SIZE:
#             return

#         current_states = []
#         q_values = []
#         max_q_values = []

#         non_final_mask = torch.tensor(tuple(map(lambda s: s["terminal"] != True,
#                                             batch)), device=device, dtype=torch.bool)

#         non_final_next_states = torch.cat([s["next_state"] for s in batch
#                                                     if s["terminal"] != True])

#         state_batch = torch.tensor(list(map(lambda s: np.asarray(s['current_state']),
#                                           batch)), dtype=torch.float64)
#         action_batch = torch.tensor(list(map(lambda s: np.asarray(s['action']),
#                                           batch)), dtype=torch.Long)
#         reward_batch = torch.tensor(list(map(lambda s: np.asarray(s['reward']),
#                                           batch)), dtype=torch.float64)


#         state_action_values = self.ddqn(state_batch).gather(1, action_batch)

#         next_state_values = torch.zeros(BATCH_SIZE, device=device)
#         next_state_values[non_final_mask] = self.ddqn_target(non_final_next_states).max(1)[0].detach()

#         expected_state_action_values = (next_state_values * GAMMA) + reward_batch

#         # Compute Huber loss
#         criterion = nn.SmoothL1Loss()
#         loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

#         # Optimize the model
#         optimizer = optim.RMSprop(self.ddqn.parameters())

#         optimizer.zero_grad()
#         loss.backward()
#         for param in self.ddqn.parameters():
#             param.grad.data.clamp_(-1, 1)
#         optimizer.step()

#         return 0, 0, 0

#     def _update_epsilon(self):
#         self.epsilon -= EXPLORATION_DECAY
#         self.epsilon = max(EXPLORATION_MIN, self.epsilon)

#     def _reset_target_network(self):
#         self.ddqn_target.set_weights(self.ddqn.get_weights())


class DQNbn(nn.Module):

    def __init__(self, c, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class DQN(nn.Module):

    def __init__(self, c, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(c, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device) / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))

Transition = namedtuple('Transition',
                        ('current_state', 'action' , 'reward', 'next_state', 'terminal'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DDQNTrainer_torch(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space):
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN training",
                               input_shape,
                               action_space,
                               "./output/logs/" + game_name + "/ddqn/training/" + self._get_date() + "/",
                               "./output/neural_nets/" + game_name + "/ddqn/" + self._get_date() + "/model.pt")

        if os.path.exists(os.path.dirname(self.model_path)):
            shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
        os.makedirs(os.path.dirname(self.model_path))

        self.ddqn = DQN(*input_shape, action_space).to(device)
        self.optimizer = optim.RMSprop(self.ddqn.parameters(),lr=0.00025, alpha=0.95, eps=0.01)
        # self.optimizer = optim.Adam(self.ddqn.parameters(), lr=0.00025)
        self.ddqn_target = DQN(*input_shape, action_space).to(device)
        self._reset_target_network()
        self.epsilon = EXPLORATION_MAX
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.pre_run = 0

    def move(self, state):
        if np.random.rand() < self.epsilon or len(self.memory) < REPLAY_START_SIZE:
            return random.randrange(self.action_space)
        q_values = self.ddqn(torch.tensor(np.expand_dims(np.asarray(state).astype(np.float64)
                                , axis=0),device=device, dtype=torch.float32))

        return np.argmax(q_values[0].detach().cpu().numpy())

    def remember(self, current_state, action, reward, next_state, terminal):
        self.memory.push(torch.unsqueeze(torch.tensor(np.asarray(current_state), dtype=torch.float32),dim=0),
                            torch.unsqueeze(torch.tensor(np.asarray(action), dtype=torch.int64),dim=0),
                            torch.tensor(np.asarray(reward), dtype=torch.float32),
                            torch.unsqueeze(torch.tensor(np.asarray(next_state), dtype=torch.float32),dim=0),
                            torch.tensor(terminal != True, dtype=torch.bool))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def step_update(self, total_step, run):
        if len(self.memory) < REPLAY_START_SIZE:
            return

        if total_step % TRAINING_FREQUENCY == 0:
            loss, accuracy, average_max_q = self._train()
            self.logger.add_loss(loss)
            self.logger.add_accuracy(accuracy)
            self.logger.add_q(average_max_q)

        self._update_epsilon()

        if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
            self._save_model()

        if total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            self._reset_target_network()
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
            print('{{"metric": "total_step", "value": {}}}'.format(total_step))

    def _train(self):
        # torch version of training
        batch = self.memory.sample(BATCH_SIZE)
        if len(batch) < BATCH_SIZE:
            return
        batch = Transition(*zip(*batch))
        # non_final_mask = torch.tensor(tuple(map(lambda s: s["terminal"] != True,
        #                                     batch)), device=device, dtype=torch.bool)

        # non_final_next_states = torch.tensor([np.asarray(s["next_state"]) for s in batch
        #                                             if s["terminal"] != True], dtype=torch.float32, device=device)

        # state_batch = torch.tensor(list(map(lambda s: np.asarray(s['current_state']),
        #                                   batch)), dtype=torch.float32, device=device)
        # action_batch = torch.unsqueeze(torch.tensor(list(map(lambda s: s['action'],
        #                                   batch)), dtype=torch.int64, device=device),dim=1)
        # reward_batch = torch.tensor(list(map(lambda s: s['reward'],
        #                                   batch)), dtype=torch.float32, device=device)

        non_final_mask = torch.tensor(batch.terminal, device=device)

        non_final_next_states = torch.cat([s for s,t in zip(batch.next_state,batch.terminal)
                                                if t]).to(device)

        state_batch = torch.cat(batch.current_state).to(device)
        action_batch = torch.unsqueeze(torch.cat(batch.action),dim=1).to(device)
        reward_batch = torch.tensor(batch.reward, device=device)
        state_action_values = self.ddqn(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.ddqn_target(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.ddqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return float(loss.detach().cpu().numpy()), 0, float(torch.mean(next_state_values).detach().cpu().numpy())

    def _update_epsilon(self):
        self.epsilon -= EXPLORATION_DECAY
        self.epsilon = max(EXPLORATION_MIN, self.epsilon)

    def _reset_target_network(self):
        self.ddqn_target.load_state_dict(self.ddqn.state_dict())

    def _save_model(self):
        torch.save(self.ddqn.state_dict(), self.model_path)

    def _load_model(self):
        self.ddqn.load_state_dict(torch.load(self.model_path))
        self.ddqn.eval()


class DDQNSolver_torch(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space):
        testing_model_path = "./output/neural_nets/" + game_name + "/ddqn/testing/model.pt"
        assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN testing",
                               input_shape,
                               action_space,
                               "./output/logs/" + game_name + "/ddqn/testing/" + self._get_date() + "/",
                               testing_model_path)
        self.ddqn = DQN(*input_shape, action_space).to(device)
        self._load_model()

    def _load_model(self):
        self.ddqn.load_state_dict(torch.load(self.model_path))
        self.ddqn.eval()

    def move(self, state):
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)
        q_values = self.ddqn(torch.tensor(np.expand_dims(np.asarray(state).astype(np.float64)
                                , axis=0),device=device, dtype=torch.float32))

        return np.argmax(q_values[0].detach().cpu().numpy())

    def step_update(self, total_step, run):
        pass