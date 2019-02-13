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
		self.maxp1 = nn.MaxPool2d(3, stride=2)
		self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=0)
		self.maxp2 = nn.MaxPool2d(2, stride=1)
		# self.conv3 = nn.Conv2d(64, 128, 3)
		self.fc1 = nn.Linear(4*4*32, 256)
		self.fc2 = nn.Linear(256, 5)

		self.optimiser = optim.SGD(self.parameters(), lr=ALPHA)
		self.loss = nn.MSELoss()
		# self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
		self.device = T.device('cpu')
		# self.to(self.device)

	def forward(self, observation):
		observation = T.Tensor(observation).to(self.device)
		observation = observation.view(-1, 3, 9, 9) # size of input
		observation = F.relu(self.conv1(observation))
		# observation = F.relu(self.maxp1(observation))
		observation = F.relu(self.conv2(observation))
		observation = self.maxp2(observation)
		# observation = F.relu(self.conv3(observation))
		# print(observation)
		observation = observation.view(-1, 4*4*32) # flatten
		observation = F.leaky_relu(self.fc1(observation))
		
		actions = self.fc2(observation)
		# actions = F.normalize(actions, p=2, dim=1)

		return actions

class Agent(object):
	def __init__(self, gamma, epsilon, alpha, maxMemorySize, epsEnd=0.08, replace=10000, actionSpace=[0,1,2,3,4]):
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
		# self.Q_eval = DeepQNetwork(alpha)
		# self.Q_next = DeepQNetwork(alpha)
		self.model = DeepQNetwork(alpha)
		self.model.train()
		self.actionSpace = actionSpace


	def storeTransition(self, state, action, reward, terminal, state_):
		if self.memCntr < self.memSize:
			self.memory.append([state, action, reward, terminal, state_])
		else:
			self.memory[self.memCntr%self.memSize] = [state, action, reward, terminal, state_]
		self.memCntr += 1


	def chooseAction(self, observation):
		rand = np.random.random()
		# actions = self.Q_eval.forward(observation)
		actions = self.model.forward(observation)
		# print("actions",actions)
		# print("actions[0]",actions[0])
		if rand < 1 - self.EPSILON:
			action = T.argmax(actions[0]).item()
		else:
			action = np.random.choice(self.actionSpace)
		self.steps += 1

		return action

	# Batch optimisation
	# Mitigate being trapped in a local minimum - break correlations between state transitions
	def learn(self, batch_size):
		# self.Q_eval.optimiser.zero_grad() # batch optimisation instead of full optimisation
		
		# if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
		# 	self.Q_next.load_state_dict(self.Q_eval.state_dict()) # not used

		if self.memCntr+batch_size < self.memSize:
			memStart = int(np.random.choice(range(self.memCntr)))
		else:
			memStart = int(np.random.choice(range(self.memSize-batch_size-1)))

		miniBatch = self.memory[memStart:memStart+batch_size]
		# print("miniBatch", miniBatch)
		# print()
		# print()

		memory = np.array(miniBatch)
		# print("memory", list(memory))
		# memory = miniBatch

		# Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
		# Qnext = self.Q_next.forward(list(memory[:, 4][:])).to(self.Q_eval.device)

		Qpred = self.model.forward(list(memory[:, 0][:]))#.to(self.model.device)
		Qnext = self.model.forward(list(memory[:, 4][:]))#.to(self.model.device)

		# print("Qpred", Qpred)
		# print("Qnext", Qnext)

		# maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
		# reward = T.Tensor(list(memory[:,2])).to(self.Q_eval.device)
		maxA = T.argmax(Qnext, dim=1)#.to(self.model.device)
		# print("maxA", maxA)
		reward = T.Tensor(list(memory[:,2]))#.to(self.model.device)
		# reward = list(memory[:,2])#.to(self.model.device)
		# print("reward", reward)
		terminal = memory[:,3]
		# Qtarget = Qpred
		Qtarget = Qpred.clone()
		# Qtarget[:, maxA] = reward + self.GAMMA*T.max(Qnext[1])
		
		for i in range(batch_size):
			if terminal[i]:
				Qtarget[i][maxA[i]] = reward[i]
			else:
				Qtarget[i][maxA[i]] = reward[i] + self.GAMMA*T.max(Qnext[i])

		# print("Qtarget", Qtarget)
		# print()

		# linear decrease of epsilon
		if self.steps > 500:
			if self.EPSILON - 1e-5 > self.EPS_END:
				self.EPSILON -= 1e-5
			else:
				self.EPSILON = self.EPS_END

		# loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device) # calculate loss function
		loss = self.model.loss(Qtarget, Qpred)#.to(self.model.device) # calculate loss function

		# print("loss", loss)

		self.model.optimiser.zero_grad() # Zero gradients

		loss.backward() # back propagate

		# self.Q_eval.optimiser.step()
		self.model.optimiser.step()
		self.learn_step_counter += 1

		return loss

	def save_model(self, name):
		try:
			os.makedirs('./Models/Torch', exist_ok=True)
		except Exception:
			print('Could not create directory ./Models/Torch')

		# T.save(self.Q_eval.state_dict(), name)
		T.save(self.model.state_dict(), name)

	def load_model(self, name):
		# self.Q_eval.load_state_dict(T.load(name))
		self.model.load_state_dict(T.load(name))
