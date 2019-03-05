'''
Simple steveAI Game 

@author: Matthew Reynard
@year: 2018

Purpose: Use Tensorflow to create a RL environment to train this steve to beat the game, i.e. fill the grid
Start Date: 5 March 2018
Due Date: Dec 2018

DEPENDING ON THE WINDOW SIZE
1200-1 number comes from 800/20 = 40; 600/20 = 30; 40*30 = 1200 grid blocks; subtract one for the "head"

BUGS:
To increase the FPS, from 10 to 120, the player is able to press multiple buttons before the steve is updated, mkaing the nsake turn a full 180.

NOTES:
Action space with a tail can be 3 - forward, left, right
Action space without a tail can be 4 - up, down, left, right

This is because with 4 actions and a tail, it is possible to go backwards and over your tail
This could just end the game (but that wont work well.. i think)

Also, when having 3 actions, the game needs to know what forward means, and at this point, with just
a head and a food, its doesn't 

'''

import sys, os
import math # Used for infinity game time
import numpy as np
import contextlib
with contextlib.redirect_stdout(None):
    import pygame # Allows pygame to import without printing the pygame welcome message

# custom imports
from steve import Steve
from food import Food
from obstacle import Obstacle
from lava import Lava
from zombie import Zombie
from utils import createGrid
from math import pi

# import csv
# import pandas as pd
# import matplotlib.pyplot as plt 

class Environment:

    def __init__(self, wrap=False, grid_size=10, local_size=9, rate=100, max_time=math.inf, action_space=5, food_count=1, obstacle_count=0, lava_count=0, zombie_count=0, map_path=None):
        """
        Initialise the Game Environment with default values
        """

        #self.FPS = 120 # NOT USED YET
        self.UPDATE_RATE = rate
        self.SCALE = 20 # Scale of each block 20x20 pixels
        self.GRID_SIZE = grid_size
        self.LOCAL_GRID_SIZE = local_size # Has to be an odd number
        self.ENABLE_WRAP = wrap
        if not self.ENABLE_WRAP: 
            self.GRID_SIZE += 2
        self.ACTION_SPACE = action_space
        self.MAP_PATH = map_path
        
        self.DISPLAY_WIDTH = self.GRID_SIZE * self.SCALE
        self.DISPLAY_HEIGHT = self.GRID_SIZE * self.SCALE

        # Maximum timesteps before an episode is stopped
        self.MAX_TIME_PER_EPISODE = max_time

        # Create and Initialise steve 
        self.steve = Steve()

        # Create Food
        self.NUM_OF_FOOD = food_count
        self.food = Food(self.NUM_OF_FOOD)

        # Create Obstacles
        self.NUM_OF_OBSTACLES = obstacle_count
        self.obstacle = Obstacle(self.NUM_OF_OBSTACLES)

        # Create Lava
        self.NUM_OF_LAVA = lava_count
        self.lava = Lava(self.NUM_OF_LAVA)

        # Create Zombies
        self.NUM_OF_ZOMBIES = zombie_count
        self.zombie = Zombie(self.NUM_OF_ZOMBIES)

        self.score = 0
        self.time = 0
        self.state = None

        self.display = None
        self.bg = None
        self.clock = None
        self.font = None

        self.steps = 0
        self.spawn_new_food = False
        self.last_eaten_food = -1

        # Used putting food and obstacles on the grid
        self.grid = []

        for j in range(self.GRID_SIZE):
            for i in range(self.GRID_SIZE):
                self.grid.append((i*self.SCALE, j*self.SCALE))

        self.maze = [] # created in reset()


    def prerender(self):
        """
        If you want to render the game to the screen, you will have to prerender
        Load textures / images
        """

        pygame.init()
        pygame.key.set_repeat(1, 1)

        self.display = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
        pygame.display.set_caption('Malmo Simulation')
        self.clock = pygame.time.Clock()

        pygame.font.init()
        self.font = pygame.font.SysFont('Default', 32, bold=False)

        # Creates a visual steve 
        self.steve.create(pygame)

        # Creates visual Food 
        self.food.create(pygame)

        # Creates visual Obstacles 
        self.obstacle.create(pygame)

        # Creates visual Lava 
        self.lava.create(pygame)

        # Creates visual Multipliers 
        self.zombie.create(pygame)

        # Creates the grid background
        self.bg = pygame.image.load("./Images/Grid50.png").convert()


    def reset(self):
        """Reset the environment"""

        self.steps = 0

        # Reset the score to 0
        self.score = 0
  
        # Positions on the grid that are not allowed to spawn things
        disallowed = []

        # Create obstacles and lava in the environment
        self.obstacle.array.clear()
        self.obstacle.array_length = 0
        self.lava.array.clear()
        self.lava.array_length = 0

        if self.MAP_PATH != None:
            self.obstacle.reset_map(self.GRID_SIZE, self.MAP_PATH, self.ENABLE_WRAP)
            self.lava.reset_map(self.GRID_SIZE, self.MAP_PATH, self.ENABLE_WRAP)

        if not self.ENABLE_WRAP:
            self.obstacle.create_border(self.GRID_SIZE, self.SCALE)

        # Add all obstacle positions to the disallowed list
        [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]
        [disallowed.append(grid_pos) for grid_pos in self.lava.array]
        
        # Add random obstacles not already on the map
        self.obstacle.reset(self.grid, disallowed)

        # Add all obstacle positions to the disallowed list
        disallowed = []
        [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]
        [disallowed.append(grid_pos) for grid_pos in self.lava.array]

        # Create lava in the environment
        # self.lava.array.clear()
        # self.lava.array_length = 0
        # if self.MAP_PATH != None:
        #     self.lava.reset_map(self.GRID_SIZE, self.MAP_PATH, self.ENABLE_WRAP)

        # [disallowed.append(grid_pos) for grid_pos in self.lava.array]

        # Add random lava not already on the map
        self.lava.reset(self.grid, disallowed)
        disallowed = []
        [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]
        [disallowed.append(grid_pos) for grid_pos in self.lava.array]

        self.maze = createGrid(self.GRID_SIZE, disallowed, self.SCALE)
        # print(self.obstacle.array)
        # print("\n")
        # print(self.maze)

        # Reseting Steve at a random (or specific) position
        self.steve.reset(self.grid, disallowed)

        # Initialise the movement to not moving
        if self.ACTION_SPACE == 5:
            self.steve.dx = 0
            self.steve.dy = 0

        disallowed.append(self.steve.pos)

        # Create a piece of food
        self.food.reset(self.NUM_OF_FOOD, self.grid, disallowed)
        # self.food.make_within_range(self.GRID_SIZE, self.SCALE, self.steve)
        self.spawn_new_food = False

        [disallowed.append(grid_pos) for grid_pos in self.food.array]

        # Spawn in a zombie
        self.zombie.reset(self.grid, disallowed)

        # Fill the state array with the appropriate state representation
        # self.state = self.state_array()
        # self.state = self.state_vector_3D()
        self.state = self.local_state_vector_3D()

        # Reset the time
        self.time = 0

        # A dictionary of information that can be useful
        info = {"time": self.time, "score": self.score}

        return self.state, info


    def render(self):
        """Renders ONLY the CURRENT state"""

        # Mainly to close the window and stop program when it's running
        action = self.controls()

        # Set the window caption to the score
        # pygame.display.set_caption("Score: " + str(self.score))

        # Draw the background, steve and the food
        self.display.blit(self.bg, (0, 0))
        self.obstacle.draw(self.display)
        self.lava.draw(self.display)
        self.food.draw(self.display)
        self.zombie.draw(self.display)
        self.steve.draw(self.display)

        # Text on screen
        text = self.font.render("Score: "+str(int(self.score)), True, (240, 240, 240, 0))
        self.display.blit(text,(10,0))
        # text = self.font.render("Multiplier: "+str(self.steve.score_multiplier)+"x", True, (240, 240, 240, 0))
        # self.display.blit(text,(150,0))

        # Update the pygame display
        pygame.display.update()

        # print(pygame.display.get_surface().get_size())

        # pygame.display.get_surface().lock()
        # p = pygame.PixelArray(pygame.display.get_surface())
        # p = pygame.surfarray.array3d(pygame.display.get_surface())
        # print("PIXELS:", p.shape)
        # pygame.display.get_surface().unlock()

        # print(clock.get_rawtime())
        # clock.tick(self.FPS) # FPS setting

        # Adds a delay to the the game when rendering so that humans can watch
        pygame.time.delay(self.UPDATE_RATE)

        return action


    def end(self):
        """
        Ending the Game -  This has to be at the end of the code
        Clean way to end the pygame env
        with a few backups...
        """

        pygame.quit()
        quit()
        sys.exit(0) # safe backup  


    def wrap(self):
        """ If the steve goes out the screen bounds, wrap it around"""

        if self.steve.x > self.DISPLAY_WIDTH - self.SCALE:
            self.steve.x = 0
        if self.steve.x < 0:
            self.steve.x = self.DISPLAY_WIDTH - self.SCALE
        if self.steve.y > self.DISPLAY_HEIGHT - self.SCALE:
            self.steve.y = 0
        if self.steve.y < 0:
            self.steve.y = self.DISPLAY_HEIGHT - self.SCALE
        self.steve.pos = (self.steve.x, self.steve.y)


    def step(self, action):
        """
        Step through the game, one state at a time.
        Return the reward, the new_state, whether its reached_food or not, and the time
        """

        self.steps += 1

        # Rewards:
        # reward_each_time_step = 1.0
        reward_each_time_step = -0.05
        reward_collecting_diamond = 10.0
        reward_out_of_bounds = -1.0 # not used
        reward_zombie_hit = -10.0
        reward_in_lava = -1.0

        # Increment time step
        self.time += 1

        # If the steve has reached the food
        reached_food = False

        hit_obstacle = False

        in_lava = False

        # If the episode is finished - after a certain amount of timesteps or it crashed
        done = False

        # Initialze to -1 for every time step - to find the fastest route (can be a more negative reward)
        # reward = 0.2
        reward = reward_each_time_step

        # Negetive exponential distance rewards
        # if len(self.zombie.array) > 0:
        #     distance = (math.sqrt((self.steve.x - self.zombie.array[0][0])**2 + (self.steve.y - self.zombie.array[0][1])**2)/self.GRID_SIZE)/2
            # print(distance)
            
            # normal_distance = ((distance/self.GRID_SIZE)*(pi/2))-pi/4
            # normal_distance = ((distance/self.GRID_SIZE)*(1.0))-0.4

            # reward = np.tan(normal_distance)
            # reward = -np.exp(-distance)*10

        # Linear distance reward
        # if len(self.zombie.array) > 0:
        #     reward = (math.sqrt((self.steve.x - self.zombie.array[0][0])**2 + (self.steve.y - self.zombie.array[0][1])**2)/20)/self.GRID_SIZE

        # Exponential distance reward
        # if len(self.zombie.array) > 0:
        #     reward = (((self.steve.x - self.zombie.array[0][0])**2 + (self.steve.y - self.zombie.array[0][1])**2)/20**2)/self.GRID_SIZE

        # Test: if moving, give a reward
        # if self.steve.dx != 0 or self.steve.dy != 0:
        #     reward = -0.01

        # Test for having a barrier around the zombie
        # if reward < 0.143:
        #     reward = -5


        # Update the position of steve 
        self.steve.update(self.SCALE, action, self.ACTION_SPACE)

        if self.ENABLE_WRAP:
            self.wrap()
        else:
            if self.steve.x > self.DISPLAY_WIDTH - self.SCALE:
                reward = reward_out_of_bounds
                done = True 
            if self.steve.x < 0:
                reward = reward_out_of_bounds
                done = True
            if self.steve.y > self.DISPLAY_HEIGHT - self.SCALE:
                reward = reward_out_of_bounds
                done = True
            if self.steve.y < 0:
                reward = reward_out_of_bounds
                done = True

        # Check for obstacle collision
        for i in range(self.obstacle.array_length):
            hit_obstacle = (self.steve.pos == self.obstacle.array[i])

            if hit_obstacle:
                self.steve.x = self.steve.prev_pos[0]
                self.steve.y = self.steve.prev_pos[1]
                self.steve.pos = self.steve.prev_pos
                # done = True
                # reward = -0.1

        # Check for lava collision
        for i in range(self.lava.array_length):
            in_lava = (self.steve.pos == self.lava.array[i])

            if in_lava:
                done = True
                reward = reward_in_lava

        # Update the position of the zombie 
        self.zombie.move(self.maze, self.steve, self.steps)

        # Check if zombie gets steve
        for i in range(self.zombie.amount):
            if self.steve.pos == (self.zombie.array[i][0], self.zombie.array[i][1]):
                zombie_hit = True
            else:
                zombie_hit = False

            if zombie_hit:
                done = True
                reward = reward_zombie_hit
                break

        # Make the most recent history have the most negative rewards
        decay = (1+reward_each_time_step)/(self.steve.history_size-1)
        for i in range(len(self.steve.history) - 1):
            # print(-1*(1-decay*i))
            if ((self.steve.x, self.steve.y) == (self.steve.history[-i-2][0], self.steve.history[-i-2][1])):
                # reward = -1*(1-decay*i)
                break

        # Checking if Steve has reached the food
        reached_food, eaten_food = self.food.eat(self.steve)

        # Reward: Including the distance between them
        # if reward == 0:
        #     reward = ((self.GRID_SIZE**2) / np.sqrt(((self.steve.x/self.SCALE-self.food.x/self.SCALE)**2 + (self.steve.y/self.SCALE-self.food.y/self.SCALE)**2) + 1)**2)/(self.GRID_SIZE**2)
            # print(reward) 

        # ADDED FOR ALLOWING THE MODEL TO HAVE STEVE OVER THE FOOD IN THE STATE
        # if self.spawn_new_food:
        #     disallowed = []
        #     [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]
        #     # [disallowed.append(grid_pos) for grid_pos in self.zombie.array]
        #     # if self.steve.pos not in disallowed:
        #         # disallowed.append(self.steve.pos)
        #     self.food.make(self.grid, disallowed, index = self.last_eaten_food)
        #     self.spawn_new_food = False
        #     reached_food = False
        
        # If Steve reaches the food, increment score
        if reached_food:
            self.score += 1
            self.last_eaten_food = eaten_food

            self.spawn_new_food = False
            # Create a piece of food that is not within Steve
            disallowed = []
            [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]
            # [disallowed.append(grid_pos) for grid_pos in self.zombie.array]
            self.food.make(self.grid, disallowed, index = eaten_food)

            # Only collect 1 food item at a time
            # done = True

            # Reward functions
            reward = reward_collecting_diamond
            # reward = 100 / (np.sqrt((self.steve.x-self.food.x)**2 + (self.steve.y-self.food.y)**2) + 1) # Including the distance between them
            # reward = 1000 * self.score
            # reward = 1000 / self.time # Including the time in the reward function
        else:
            self.spawn_new_food = False

        # For fun
        if self.score == 2:
            self.steve.hasSword = True

        # To make it compatible with malmo
        if self.score == self.NUM_OF_FOOD:
            # reward = 1
            pass
            if self.NUM_OF_FOOD != 0:
                done = True
                pass

        # If the episode takes longer than the max time, it ends
        if self.time == self.MAX_TIME_PER_EPISODE:
            # print("Steve survived :)")
            # reward = -1.0
            done = True

        # Get the new_state
        # new_state = self.state_array()
        # new_state = self.state_vector_3D()
        new_state = self.local_state_vector_3D()

        # print(reward)

        # A dictionary of information that may be useful
        info = {"time": self.time, "score": self.score}

        return new_state, reward, done, info


    def state_index(self, state_array):
        """
        Given the state array, return the index of that state as an integer
        Used for the Qlearning lookup table
        """
        return int((self.GRID_SIZE**3)*state_array[0]+(self.GRID_SIZE**2)*state_array[1]+(self.GRID_SIZE**1)*state_array[2]+(self.GRID_SIZE**0)*state_array[3])


    def sample_action(self):
        """
        Return a random action

        Can't use action space 4 with a tail, else it will have a double chance of doing nothing
        or crash into itself
        """
        return np.random.randint(0, self.ACTION_SPACE) 


    def set_state(self, state):
        """Set the state of the game environment"""
        self.state = state


    def set_map(self, map_path):

        self.MAP_PATH = map_path


    def number_of_states(self):
        """
        Return the number of states with just the steve head and 1 food

        Used for Q Learning look up table
        """
        return (self.GRID_SIZE**2)*((self.GRID_SIZE**2))


    def number_of_actions(self):
        """
        Return the number of possible actions 

        Action space:
        > nothing, up, down, left, right
        """
        return self.ACTION_SPACE


    def state_array(self):
        """
        The state represented as an array or steve positions and food positions

        Used for Q learning
        """
        new_state = np.zeros(4) 

        new_state[0] = int(self.steve.x / self.SCALE)
        new_state[1] = int(self.steve.y / self.SCALE)
        new_state[2] = int(self.food.x / self.SCALE)
        new_state[3] = int(self.food.y / self.SCALE)

        return new_state


    def state_vector(self):
        """
        The state represented as a onehot 1D vector

        Used for the feed forward NN
        """
        # (rows, columns)
        state = np.zeros((self.GRID_SIZE**2, 3))
        
        # This is for STEVE, the FOOD and EMPTY
        for i in range(self.GRID_SIZE): # rows
            for j in range(self.GRID_SIZE): # columns
                if ((self.steve.x/self.SCALE) == j and (self.steve.y/self.SCALE) == i):
                    state[i*self.GRID_SIZE+j] = [1,0,0]
                    # print("steve:", i*self.GRID_SIZE+j)
                elif ((self.food.x/self.SCALE) == j and (self.food.y/self.SCALE) == i):
                    state[i*self.GRID_SIZE+j] = [0,1,0]
                    # print("Food:", i*self.GRID_SIZE+j)
                else:
                    state[i*self.GRID_SIZE+j] = [0,0,1]

        # Flatten the vector to a 1 dimensional vector for the input layer to the NN
        state = state.flatten()

        state = state.reshape(1,(self.GRID_SIZE**2)*3)

        # state = np.transpose(state)

        return state


    def state_vector_3D(self):
        """
        State as a 3D vector of the whole map for the CNN

        Shape = (Layers, GRID_SIZE, GRID_SIZE)
        """

        state = np.zeros((3, self.GRID_SIZE, self.GRID_SIZE))

        state[0, int(self.steve.y/self.SCALE), int(self.steve.x/self.SCALE)] = 1

        state[1, int(self.food.y/self.SCALE), int(self.food.x/self.SCALE)] = 1

        # Obstacles
        for i in range(self.obstacle.array_length):
            state[2, int(self.obstacle.array[i][1]/self.SCALE), int(self.obstacle.array[i][0]/self.SCALE)] = 1

        # Zombies
        # for i in range(len(self.zombie.array)):
        #     state[1, int(self.zombie.array[i][1]/self.SCALE), int(self.zombie.array[i][0]/self.SCALE)] = 1

        return state


    def local_state_vector_3D(self): 
        """
        State as a 3D vector of the local area around the steve

        Shape = (Layers, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE)
        """

        s_pos = 0
        d_pos = 1
        # z_pos = 2
        l_pos = 2
        o_ops = 3
        # h_pos = 3


        #s = steve
        sx = int(self.steve.x/self.SCALE)
        sy = int(self.steve.y/self.SCALE)

        state = np.zeros((4, self.LOCAL_GRID_SIZE, self.LOCAL_GRID_SIZE)) 

        # Agent
        local_pos = int((self.LOCAL_GRID_SIZE-1)/2)
        state[s_pos, local_pos, local_pos] = 1

        # Food
        for i in range(self.food.amount):
            x_prime_food = local_pos+int(self.food.array[i][0]/self.SCALE)-sx
            y_prime_food = local_pos+int(self.food.array[i][1]/self.SCALE)-sy

            if x_prime_food < self.LOCAL_GRID_SIZE and x_prime_food >= 0 and y_prime_food < self.LOCAL_GRID_SIZE and y_prime_food >= 0:
                state[d_pos, y_prime_food, x_prime_food] = 1
                pass

        # Obstacles
        for i in range(self.obstacle.array_length):
            x_prime_obs = local_pos+int(self.obstacle.array[i][0]/self.SCALE)-sx
            y_prime_obs = local_pos+int(self.obstacle.array[i][1]/self.SCALE)-sy

            if x_prime_obs < self.LOCAL_GRID_SIZE and x_prime_obs >= 0 and y_prime_obs < self.LOCAL_GRID_SIZE and y_prime_obs >= 0:
                state[o_ops, y_prime_obs, x_prime_obs] = 1
                pass

        # Out of bounds
        for j in range(0, self.LOCAL_GRID_SIZE):
            for i in range(0, self.LOCAL_GRID_SIZE):

                x_prime_wall = local_pos-sx
                y_prime_wall = local_pos-sy

                if i < x_prime_wall or j < y_prime_wall:
                    state[o_ops, j, i] = 1
                    pass

                x_prime_wall = local_pos+(self.GRID_SIZE-sx)-1
                y_prime_wall = local_pos+(self.GRID_SIZE-sy)-1

                if i > x_prime_wall or j > y_prime_wall:
                    state[o_ops, j, i] = 1
                    pass

        # Zombies
        for i in range(len(self.zombie.array)):
            x_prime_zom = local_pos+int(self.zombie.array[i][0]/self.SCALE)-sx
            y_prime_zom = local_pos+int(self.zombie.array[i][1]/self.SCALE)-sy

            if x_prime_zom < self.LOCAL_GRID_SIZE and x_prime_zom >= 0 and y_prime_zom < self.LOCAL_GRID_SIZE and y_prime_zom >= 0:
                # state[z_pos, y_prime_zom, x_prime_zom] = 1
                pass

        # Lava
        for i in range(self.lava.array_length):
            x_prime_lava = local_pos+int(self.lava.array[i][0]/self.SCALE)-sx
            y_prime_lava = local_pos+int(self.lava.array[i][1]/self.SCALE)-sy

            if x_prime_lava < self.LOCAL_GRID_SIZE and x_prime_lava >= 0 and y_prime_lava < self.LOCAL_GRID_SIZE and y_prime_lava >= 0:
                state[l_pos, y_prime_lava, x_prime_lava] = 1
                pass

        # History
        decay = 1/self.steve.history_size

        for i in range(len(self.steve.history)-1):
            x_prime = local_pos+int(self.steve.history[-i-2][0]/self.SCALE)-sx
            y_prime = local_pos+int(self.steve.history[-i-2][1]/self.SCALE)-sy

            if x_prime < self.LOCAL_GRID_SIZE and x_prime >= 0 and y_prime < self.LOCAL_GRID_SIZE and y_prime >= 0:
                # if 1-decay*i >= 0 and state[h_pos, y_prime, x_prime] == 0:
                #     state[h_pos, y_prime, x_prime] = 1-decay*i
                pass
                # else:
                    # state[h_pos, y_prime, x_prime] = 0

        return state


    def get_pixels(self): 
        """
        Returns the pixels in a (GRID*20, GRID*20, 3) size array/
    
        Unfortunatly it has to render in order to gather the pixels
        """
        return pygame.surfarray.array3d(pygame.display.get_surface())


    def controls(self):
        """
        Defines all the players controls during the game
        """

        action = 0 # Do nothing as default

        for event in pygame.event.get():
            # print(event) # DEBUGGING

            if event.type == pygame.QUIT:
                self.end()

            if event.type == pygame.KEYDOWN:
                # print(event.key) #DEBUGGING

                # In order to quit easily
                if (event.key == pygame.K_q):
                    self.end()

                # Moving up
                if (event.key == pygame.K_UP or event.key == pygame.K_w):

                    if self.ACTION_SPACE == 5:
                        action = 1 # up

                # Moving down
                elif (event.key == pygame.K_DOWN or event.key == pygame.K_s):

                    if self.ACTION_SPACE == 5:
                        action = 2 # down

                # Moving left
                elif (event.key == pygame.K_LEFT or event.key == pygame.K_a):

                    if self.ACTION_SPACE == 5:
                        action = 3 # left


                # Moving right
                elif (event.key == pygame.K_RIGHT or event.key == pygame.K_d):

                    if self.ACTION_SPACE == 5:
                        action = 4 # right

        return action


    def play(self):
        """ 
        Lets you simply play the game

        Useful for debugging and testing out the environment
        """

        GAME_OVER = False

        self.prerender()

        # self.reset()

        for i in range(10):

            MAP_NUMBER = np.random.randint(10)

            MAP_PATH = "./Maps/Grid10/map{}.txt".format(MAP_NUMBER)

            self.set_map(MAP_PATH)

            self.reset()

            GAME_OVER = False

            while not GAME_OVER:

                # print(self.steve.history)
                action = self.controls()
                # action = self.render()

                # print(self.get_pixels().shape)

                s, r, GAME_OVER, i = self.step(action)
                
                # print("\n\n\n") # DEBUGGING
                # print(self.local_state_vector_3D()) # DEBUGGING
                # print(self.state_vector_3D()) # DEBUGGING
                
                print(r)

                self.render()

                if GAME_OVER:
                    print("Game Over: time:", i["time"])

        self.end()


# If I run this file by accident :P
if __name__ == "__main__":

    print("This file does not have a main method")
