import numpy as np

class Zombie:

    def __init__(self, number = 1):
        self.x = 0
        self.y = 0
        self.pos = (self.x, self.y)

        self.zombie_img = None

        self.array = [self.pos]
        self.amount = number

        self.amount_eaten = 0

    # Create the Pygame sections of the food to render it
    def create(self, pygame):

        # PYGAME STUFF

        white = (255,255,255)
        self.zombie_img = pygame.image.load("./Images/zombie_head.png").convert()
        # self.zombie_img.set_colorkey(white)

        # If the image isn't 20x20 pixels
        self.zombie_img = pygame.transform.scale(self.zombie_img, (20, 20))


    def reset(self, grid, disallowed):
        self.array.clear()
        self.amount_eaten = 0

        # Make a copy of the grid
        allowed = grid[:]

        # Remove all the disallowed positions from the allowed list
        [allowed.remove(pos) for pos in disallowed]

        # If you want the food to only spawn in 3 different locations
        # if self.amount == 1:
            # allowed = [(2*20,7*20),(6*20,6*20),(7*20,1*20)]

        for i in range(self.amount):
            new_pos = allowed[np.random.choice(len(allowed))]
            self.array.append(new_pos)
            allowed.remove(new_pos)

        # To add backwards compatibility to my old functions and algorithms
        if self.amount == 1:
            self.x = self.array[0][0]
            self.y = self.array[0][1]


    # Load a food item into the screen at a random location
    def make(self, grid, disallowed, index = 0):
        
        # Make a copy of the grid
        allowed = grid[:]

        # Add the others foods positions (including this previous one) to the the disallowed list
        [disallowed.append(grid_pos) for grid_pos in self.array]

        # Show if theres a duplicate in the disallowed list
        # print(set([x for x in disallowed if disallowed.count(x) > 1])) # DEBUGGING

        # Remove the disallowed positions from the allowed grid positions
        [allowed.remove(pos) for pos in disallowed]

        # If you want the food to only spawn in 3 different locations
        # if self.amount == 1:
            # allowed = [(2*20,6*20),(5*20,5*20),(6*20,1*20)]

        # Choose a random allowed position to be the new food position
        self.array[index] = allowed[np.random.choice(len(allowed))]

        # To add backwards compatibility to my old functions and algorithms
        if self.amount == 1:
            self.x = self.array[0][0]
            self.y = self.array[0][1]

    # make a piece of food within the local grid
    def make_within_range(self, grid_size, scale, snake, local_grid_size = 3):
        made = False
        rows = grid_size
        cols = grid_size

        while not made:
            myRow = np.random.randint(0, rows-1)
            myCol = np.random.randint(0, cols-1)

            self.pos = (myCol * scale, myRow * scale) # multiplying by scale

            for i in range(0, snake.tail_length + 1):
                # print("making food")
                # Need to change this to the whole body of the snake
                if self.pos == snake.box[i]:
                    made = False # the food IS within the snakes body
                    break
                elif abs(snake.x - self.pos[0])/scale <= local_grid_size and abs(snake.y - self.pos[1])/scale <= local_grid_size:
                    self.x = myCol * scale
                    self.y = myRow * scale
                    made = True # the food IS NOT within the snakes body and within range


    def eat(self, snake):

        reached = False
        index = 0

        for i in range(self.amount):
            increase_multiplier = ((snake.x, snake.y) == (self.array[i][0], self.array[i][1]))
            if increase_multiplier:
                self.amount_eaten += 1
                reached = True
                index = i
                break

        return reached, index

    #Draw the food
    def draw(self, display):
        for i in range(self.amount):
            display.blit(self.zombie_img, self.array[i])
