import numpy as np
from utils import Node

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


    def move(self, maze, steve, steps):
        """After a certain amount of charater steps, the zombie takes a step"""
        if self.amount > 0:
            start = tuple([int(x/20) for x in self.array[0]])
            end = tuple([int(y/20) for y in steve.pos])

            path = self.astar(maze, start, end)
            # print(path)

            # 1st zombie moves according to the a* algorithm
            star_steps = 7
            self.array[0] = list(self.array[0])
            if path != None:
                if len(path) > 1 and steps%star_steps == 0:
                    # print("move")
                    self.array[0][0] += (path[1][0] - path[0][0])*20
                    self.array[0][1] += (path[1][1] - path[0][1])*20

                    # self.pos = (self.x, self.y)

                    # self.array[0] = self.pos
            self.array[0] = tuple(self.array[0])

            # 2nd zombie moves randomly
            random_steps = 3
            # print(self.array)
            if self.amount > 1 and steps%random_steps == 0:
                self.array[1] = list(self.array[1])
                random_move = np.random.randint(0,4)
                if random_move == 0 and maze[int(self.array[1][0]/20)+1][int(self.array[1][1]/20)] == 0:
                    self.array[1][0] += 1*20
                if random_move == 1 and maze[int(self.array[1][0]/20)-1][int(self.array[1][1]/20)] == 0:
                    self.array[1][0] -= 1*20
                if random_move == 2 and maze[int(self.array[1][0]/20)][int(self.array[1][1]/20)+1] == 0:
                    self.array[1][1] += 1*20
                if random_move == 3 and maze[int(self.array[1][0]/20)][int(self.array[1][1]/20)-1] == 0:
                    self.array[1][1] -= 1*20

                self.array[1] = tuple(self.array[1])


    #Draw the food
    def draw(self, display):
        """Draw the zombies on the display"""
        for i in range(self.amount):
            display.blit(self.zombie_img, self.array[i])


    def astar(self, maze, start, end):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""

        # Create start and end node
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Add the start node
        open_list.append(start_node)

        count = 0

        # Loop until you find the end
        while len(open_list) > 0:
            count += 1

            # Get the current node
            current_node = open_list[0]
            current_index = 0
            for index, item in enumerate(open_list):
                if item.f < current_node.f:
                    current_node = item
                    current_index = index

            # Pop current off open list, add to closed list
            open_list.pop(current_index)
            closed_list.append(current_node)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1] # Return reversed path

            # Generate children
            children = []
            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0), (1, 1), (-1, 1), (1, -1)]: # Adjacent squares
            # for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                    continue

                # Make sure walkable terrain
                if maze[node_position[0]][node_position[1]] != 0:
                    continue

                # Create new node
                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:

                # Child is on the closed list
                for closed_child in closed_list:
                    if child == closed_child:
                        continue

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                for open_node in open_list:
                    if child == open_node and child.g > open_node.g:
                        continue

                # Add the child to the open list
                open_list.append(child)

            if count > 100:
                # print(count)
                break
