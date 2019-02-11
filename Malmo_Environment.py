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

import numpy as np
import contextlib
with contextlib.redirect_stdout(None):
    import pygame # Allows pygame to import without printing the pygame welcome message
from steve import Steve
from food import Food
from obstacle import Obstacle
from lava import Lava
from zombie import Zombie
import sys
import math # Used for infinity game time
from utils import createGrid

# import csv
# import pandas as pd
# import matplotlib.pyplot as plt 

class Environment:

    def __init__(self, wrap = False, grid_size = 10, rate = 100, max_time = math.inf, action_space = 5, food_count = 1, obstacle_count = 0, lava_count = 0, zombie_count = 0, map_path = None):
        """
        Initialise the Game Environment with default values
        """

        #self.FPS = 120 # NOT USED YET
        self.UPDATE_RATE = rate
        self.SCALE = 20 # Scale of each block 20x20 pixels
        self.GRID_SIZE = grid_size
        self.LOCAL_GRID_SIZE = 9 # Has to be an odd number
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
        self.state = self.state_vector_3D()
        # self.state = self.local_state_vector_3D()

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
        reward_out_of_bounds = -10
        reward_hitting_tail = -10
        reward_each_time_step = -0.01
        reward_reaching_food = 10

        # Increment time step
        self.time += 1

        # If the steve has reached the food
        reached_food = False

        hit_obstacle = False

        in_lava = False

        # If the episode is finished - after a certain amount of timesteps or it crashed
        done = False

        # Initialze to -1 for every time step - to find the fastest route (can be a more negative reward)
        # reward = -1
        reward = -0.1
        # reward = 0.3

        # Test: if moving, give a reward
        # if self.steve.dx != 0 or self.steve.dy != 0:
        #     reward = -0.01


        # Update the position of steve 
        self.steve.update(self.SCALE, action, self.ACTION_SPACE)

        if self.ENABLE_WRAP:
            self.wrap()
        else:
            if self.steve.x > self.DISPLAY_WIDTH - self.SCALE:
                reward = -1 # very negative reward, to ensure that it never crashes into the side
                done = True 
            if self.steve.x < 0:
                reward = -1
                done = True
            if self.steve.y > self.DISPLAY_HEIGHT - self.SCALE:
                reward = -1
                done = True
            if self.steve.y < 0:
                reward = -1
                done = True

        # Check for obstacle collision
        for i in range(self.obstacle.array_length):
            hit_obstacle = (self.steve.pos == self.obstacle.array[i])

            if hit_obstacle:
                self.steve.x = self.steve.prev_pos[0]
                self.steve.y = self.steve.prev_pos[1]
                self.steve.pos = self.steve.prev_pos
                # done = True
                reward = -0.15

        # Check for lava collision
        for i in range(self.lava.array_length):
            in_lava = (self.steve.pos == self.lava.array[i])

            if in_lava:
                done = True
                reward = -0.8

        # Update the position of the zombie 
        self.zombie.move(self.maze, self.steve, self.steps)

        # Check if zombie gets steve
        for i in range(self.zombie.amount):
            # if self.steve.pos == (self.zombie.array[i][0]+1*20, self.zombie.array[i][1]) \
            # or self.steve.pos == (self.zombie.array[i][0]-1*20, self.zombie.array[i][1]) \
            # or self.steve.pos == (self.zombie.array[i][0], self.zombie.array[i][1]+1*20) \
            # or self.steve.pos == (self.zombie.array[i][0], self.zombie.array[i][1]-1*20):
            if self.steve.pos == (self.zombie.array[i][0], self.zombie.array[i][1]):
                zombie_hit = True
            else:
                zombie_hit = False
            # print(zombie_hit)
            if zombie_hit:
                done = True
                reward = -1.0
                break


        # Make the most recent history have the most negative rewards
        # decay = (1+reward_each_time_step)/(self.steve.history_size-1)
        # for i in range(len(self.steve.history) - 1):
        #     # print(-1*(1-decay*i))
        #     if ((self.steve.x, self.steve.y) == (self.steve.history[-i-2][0], self.steve.history[-i-2][1])):
        #         reward = -1*(1-decay*i)
        #         break

        # Checking if Steve has reached the food
        reached_food, eaten_food = self.food.eat(self.steve)

        # Checking if the steve has reached the multiplier
        # reached_multiplier, eaten_multiplier = self.zombie.eat(self.steve)

        # if reached_multiplier:
        #     # After every 3 multipliers are gathered, increase the score multiplier
        #     if self.multiplier.amount_eaten % 3 == 0 and self.multiplier.amount_eaten != 0:
        #         self.steve.score_multiplier += 1

        #     disallowed = []
        #     [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]
        #     [disallowed.append(grid_pos) for grid_pos in self.food.array]
        #     self.multiplier.make(self.grid, disallowed, index = eaten_multiplier)


        # Reward: Including the distance between them
        # if reward == 0:
        #     reward = ((self.GRID_SIZE**2) / np.sqrt(((self.steve.x/self.SCALE-self.food.x/self.SCALE)**2 + (self.steve.y/self.SCALE-self.food.y/self.SCALE)**2) + 1)**2)/(self.GRID_SIZE**2)
            # print(reward) 

        if self.spawn_new_food:
            disallowed = []
            [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]
            # [disallowed.append(grid_pos) for grid_pos in self.zombie.array]
            # if self.steve.pos not in disallowed:
            #     disallowed.append(self.steve.pos)
            self.food.make(self.grid, disallowed, index = self.last_eaten_food)
            self.spawn_new_food = False
            reached_food = False
        
        # If Steve reaches the food, increment score
        if reached_food:
            self.score += 1
            self.last_eaten_food = eaten_food

            self.spawn_new_food = True
            # Create a piece of food that is not within Steve
            # disallowed = []
            # [disallowed.append(grid_pos) for grid_pos in self.obstacle.array]
            # [disallowed.append(grid_pos) for grid_pos in self.zombie.array]
            # self.food.make(self.grid, disallowed, index = eaten_food)

            # Test for one food item at a time
            done = True

            # Reward functions
            reward = 10
            # reward = 100 / (np.sqrt((self.steve.x-self.food.x)**2 + (self.steve.y-self.food.y)**2) + 1) # Including the distance between them
            # reward = 1000 * self.score
            # reward = 1000 / self.time # Including the time in the reward function
        else:
            self.spawn_new_food = False

        # To make it compatible with malmo
        if self.score == self.NUM_OF_FOOD:
            reward = 100
            if self.NUM_OF_FOOD != 0:
                done = True

        # If the episode takes longer than the max time, it ends
        if self.time == self.MAX_TIME_PER_EPISODE:
            # reward = 10
            done = True

        # Get the new_state
        # new_state = self.state_array()
        new_state = self.state_vector_3D()
        # new_state = self.local_state_vector_3D()

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
        """
        # print(int(self.steve.y/self.SCALE), int(self.steve.x/self.SCALE))

        state = np.zeros((3, self.GRID_SIZE, self.GRID_SIZE))

        state[0, int(self.steve.y/self.SCALE), int(self.steve.x/self.SCALE)] = 1

        state[1, int(self.food.y/self.SCALE), int(self.food.x/self.SCALE)] = 1

        for i in range(self.obstacle.array_length):
            state[2, int(self.obstacle.array[i][0]/self.SCALE), int(self.obstacle.array[i][1]/self.SCALE)] = 1

        return state


    def local_state_vector_3D(self): 
        """
        State as a 3D vector of the local area around the steve

        Shape = (3,9,9) w/out history
        Shape = (4,9,9) w/ history
        """

        #s = steve
        sx = int(self.steve.x/self.SCALE)
        sy = int(self.steve.y/self.SCALE)

        # state = np.zeros((3, self.LOCAL_GRID_SIZE, self.LOCAL_GRID_SIZE))
        state = np.zeros((4, self.LOCAL_GRID_SIZE, self.LOCAL_GRID_SIZE)) 

        # Agent
        local_pos = int((self.LOCAL_GRID_SIZE-1)/2)
        state[0, local_pos, local_pos] = 1

        # History
        # decay = 1/self.steve.history_size

        # for i in range(len(self.steve.history)-1):
        #     x_prime = local_pos+int(self.steve.history[-i-2][0]/self.SCALE)-int(self.steve.x/self.SCALE)
        #     y_prime = local_pos+int(self.steve.history[-i-2][1]/self.SCALE)-int(self.steve.y/self.SCALE)

        #     if x_prime < self.LOCAL_GRID_SIZE and x_prime >= 0 and y_prime < self.LOCAL_GRID_SIZE and y_prime >= 0:
        #         if 1-decay*i >= 0 and state[3, y_prime, x_prime] == 0:
        #             state[3, y_prime, x_prime] = 1-decay*i
        #         # else:
        #             # state[3, y_prime, x_prime] = 0

        # Food
        for i in range(self.food.amount):
            x_prime_food = local_pos+int(self.food.array[i][0]/self.SCALE)-int(self.steve.x/self.SCALE)
            y_prime_food = local_pos+int(self.food.array[i][1]/self.SCALE)-int(self.steve.y/self.SCALE)

            if x_prime_food < self.LOCAL_GRID_SIZE and x_prime_food >= 0 and y_prime_food < self.LOCAL_GRID_SIZE and y_prime_food >= 0:
                # state[1, y_prime_food, x_prime_food] = 1
                pass


        # Obstacles
        for i in range(self.obstacle.array_length):
            x_prime_obs = local_pos+int(self.obstacle.array[i][0]/self.SCALE)-int(self.steve.x/self.SCALE)
            y_prime_obs = local_pos+int(self.obstacle.array[i][1]/self.SCALE)-int(self.steve.y/self.SCALE)

            if x_prime_obs < self.LOCAL_GRID_SIZE and x_prime_obs >= 0 and y_prime_obs < self.LOCAL_GRID_SIZE and y_prime_obs >= 0:
                state[2, y_prime_obs, x_prime_obs] = 1

        # Out of bounds
        for j in range(0, self.LOCAL_GRID_SIZE):
            for i in range(0, self.LOCAL_GRID_SIZE):

                x_prime_wall = local_pos-sx
                y_prime_wall = local_pos-sy

                if i < x_prime_wall or j < y_prime_wall:
                    state[2, j, i] = 1

                x_prime_wall = local_pos+(self.GRID_SIZE-sx)-1
                y_prime_wall = local_pos+(self.GRID_SIZE-sy)-1

                if i > x_prime_wall or j > y_prime_wall:
                    state[2, j, i] = 1

        # Zombies
        for i in range(len(self.zombie.array)):
            x_prime_zom = local_pos+int(self.zombie.array[i][0]/self.SCALE)-int(self.steve.x/self.SCALE)
            y_prime_zom = local_pos+int(self.zombie.array[i][1]/self.SCALE)-int(self.steve.y/self.SCALE)

            if x_prime_zom < self.LOCAL_GRID_SIZE and x_prime_zom >= 0 and y_prime_zom < self.LOCAL_GRID_SIZE and y_prime_zom >= 0:
                state[1, y_prime_zom, x_prime_zom] = 1
                pass

        # Lava
        for i in range(self.lava.array_length):
            x_prime_lava = local_pos+int(self.lava.array[i][0]/self.SCALE)-int(self.steve.x/self.SCALE)
            y_prime_lava = local_pos+int(self.lava.array[i][1]/self.SCALE)-int(self.steve.y/self.SCALE)

            if x_prime_lava < self.LOCAL_GRID_SIZE and x_prime_lava >= 0 and y_prime_lava < self.LOCAL_GRID_SIZE and y_prime_lava >= 0:
                state[3, y_prime_lava, x_prime_lava] = 1

        return state


    def pixels(self): 
        """
        Returns the pixels in a (GRID*20, GRID*20, 3) size array/
    
        Unfortunatly it has to render in order to gather the pixels
        """
        return pygame.surfarray.array3d(pygame.display.get_surface())


    def controls(self):
        """Defines all the players controls during the game"""

        GAME_OVER = False # NOT IMPLEMENTED YET

        action = 0 # Do nothing as default

        for event in pygame.event.get():
            # print(event) # DEBUGGING

            if event.type == pygame.QUIT:
                self.end()

            if event.type == pygame.KEYDOWN:
                # print(event.key) #DEBUGGING

                # In order to stop training and still save the Q txt file
                if (event.key == pygame.K_q):
                    self.end()

                # Moving up
                if (event.key == pygame.K_UP or event.key == pygame.K_w) and GAME_OVER == False:

                    if self.ACTION_SPACE == 5:
                        action = 1 # up

                # Moving down
                elif (event.key == pygame.K_DOWN or event.key == pygame.K_s) and GAME_OVER == False:

                    if self.ACTION_SPACE == 5:
                        action = 2 # down

                # Moving left
                elif (event.key == pygame.K_LEFT or event.key == pygame.K_a) and GAME_OVER == False:

                    if self.ACTION_SPACE == 5:
                        action = 3 # left


                # Moving right
                elif (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and GAME_OVER == False:

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

            self.reset()

            GAME_OVER = False

            while not GAME_OVER:

                # print(self.steve.history)

                action = self.render()

                # print(self.pixels().shape)

                # When the steve touches the food, game ends
                # action_space has to be 3 for the players controls, 
                # because they know that the steve can't go backwards
                s, r, GAME_OVER, i = self.step(action)
                # print("\n\n\n") # DEBUGGING
                # if r != -0.05:
                #     print("Reward: ",r) # DEBUGGING
                # print(self.local_state_vector_3D()) # DEBUGGING
                # print(self.state_vector_3D()) # DEBUGGING
                
                # print(r)
                # For the steve to look like it ate the food, render needs to be last
                # Next piece of code if very BAD programming

                # self.zombie.move(self.maze, self.steve, self.steps)

                # print("")

                if GAME_OVER:
                    print("Game Over")
                    # self.render()

        self.end()


# If I run this file by accident :P
if __name__ == "__main__":

    print("This file does not have a main method")
