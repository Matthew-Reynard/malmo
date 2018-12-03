import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class DeepQNetwork(nn.Module):
	def __init__(self, ALPHA):
		super(DeepQNetwork, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=0)
		# self.maxp1 = nn.MaxPool2d(3, stride=2)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=0)
		self.maxp2 = nn.MaxPool2d(2, stride=1)
		# self.conv3 = nn.Conv2d(64, 128, 3)
		self.fc1 = nn.Linear(2*2*32, 256)
		self.fc2 = nn.Linear(256, 5)

		self.optimiser = optim.SGD(self.parameters(), lr=ALPHA)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, observation):
		observation = T.Tensor(observation).to(self.device)
		observation = observation.view(-1, 3, 7, 7) # size of image 185 x 95
		observation = F.leaky_relu(self.conv1(observation))
		# observation = F.relu(self.maxp1(observation))
		observation = F.leaky_relu(self.conv2(observation))
		observation = F.leaky_relu(self.maxp2(observation))
		# observation = F.relu(self.conv3(observation))
		# print(observation)
		observation = observation.view(-1, 2*2*32) # flatten
		observation = F.leaky_relu(self.fc1(observation))
		
		actions = self.fc2(observation)

		return actions

class Agent(object):
	def __init__(self, gamma, epsilon, alpha, maxMemorySize, epsEnd=0.05, replace=10000, actionSpace=[0,1,2,3,4]):
		self.ALPHA = alpha
		self.GAMMA = gamma
		self.EPSILON = epsilon
		self.EPS_END = epsEnd
		self.memSize = maxMemorySize
		self.steps = 0
		self.learn_step_counter = 0
		self.memory = []
		self.memCntr = 0
		self.replace_target_cnt = replace
		self.Q_eval = DeepQNetwork(alpha)
		self.Q_next = DeepQNetwork(alpha)
		self.actionSpace = actionSpace


	def storeTransition(self, state, action, reward, terminal, state_):
		if self.memCntr < self.memSize:
			self.memory.append([state, action, reward, terminal, state_])
		else:
			self.memory[self.memCntr%self.memSize] = [state, action, reward, terminal, state_]
		self.memCntr += 1


	def chooseAction(self, observation):
		rand = np.random.random()
		actions = self.Q_eval.forward(observation)
		# print(actions)
		if rand < 1 - self.EPSILON:
			action = T.argmax(actions[0]).item()
		else:
			action = np.random.choice(self.actionSpace)
		self.steps += 1

		return action

	# Batch optimisation
	# Mitigate being trapped in a local minimum - break correlations between state transitions
	def learn(self, batch_size):
		self.Q_eval.optimiser.zero_grad() # batch optimisation instead of full optimisation
		
		if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
			self.Q_next.load_state_dict(self.Q_eval.state_dict()) # not used

		if self.memCntr+batch_size < self.memSize:
			memStart = int(np.random.choice(range(self.memCntr)))
		else:
			memStart = int(np.random.choice(range(self.memSize-batch_size-1)))

		miniBatch = self.memory[memStart:memStart+batch_size]
		# print(miniBatch)
		memory = np.array(miniBatch)
		# memory = miniBatch

		Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
		Qnext = self.Q_next.forward(list(memory[:, 4][:])).to(self.Q_eval.device)

		maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
		reward = T.Tensor(list(memory[:,2])).to(self.Q_eval.device)
		# Qtarget = Qpred
		Qtarget = Qpred.clone()
		# Qtarget[:, maxA] = reward + self.GAMMA*T.max(Qnext[1])
		for i in range(batch_size):
			if memory[:, 3][i]:
				Qtarget[i][maxA[i]] = reward[i]
			else:
				Qtarget[i][maxA[i]] = reward[i] + self.GAMMA*T.max(Qnext[i])

		# linear decrease of epsilon
		if self.steps > 500:
			if self.EPSILON - 2e-5 > self.EPS_END:
				self.EPSILON -= 2e-5
			else:
				self.EPSILON = self.EPS_END

		loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device) # calculate loss function
		loss.backward() # back propagate
		self.Q_eval.optimiser.step()
		self.learn_step_counter += 1

		return loss

	def save_model(self, name):
		try:
			os.makedirs('./Models/Torch', exist_ok=True)
		except Exception:
			print('Could not create directory ./Models/Torch')

		T.save(self.Q_eval.state_dict(), name)

	def load_model(self, name):
		self.Q_eval.load_state_dict(T.load(name))
