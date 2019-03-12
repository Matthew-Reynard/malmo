import numpy as np

class Steve:

    def __init__(self, history_len):
        self.x = 0
        self.y = 0
        self.dx = 0
        self.dy = 0

        self.pos = (self.x, self.y)
        self.prev_pos = (self.x, self.y)
        
        self.steve_img = None
        self.steve_img2 = None

        # Used for the CNN local network input to see where the snake has been recently
        self.history_size = history_len # Needs to be proportional to the grid size
        self.history = []

        # Not used
        self.score_multiplier = 1

        self.hasSword = False


    # Load steve image for better visuals 
    def create(self, pygame):

        # PYGAME STUFF

        white = (255,255,255)
        self.steve_img = pygame.image.load("./Images/steve_head.png").convert()
        self.steve_img2 = pygame.image.load("./Images/steve_sword.png").convert()
        # self.steve_img.set_colorkey(white) # sets white to alpha

        # self.steve_img = pygame.transform.flip(self.steve_img, False, True) #
        # self.steve_img = pygame.transform.rotate(self.steve_img, 90) # Start facing right

        # If the images arent 20x20 pixels, scales down or up
        self.steve_img = pygame.transform.scale(self.steve_img, (20, 20))
        self.steve_img2 = pygame.transform.scale(self.steve_img2, (20, 20))


    # Reset steve at a specific/random location in env
    def reset(self, grid, disallowed):

        allowed = grid[:]

        for pos in disallowed:
            try:
                allowed.remove(pos)
            except:
                print("ERROR CAUGHT => ValueError: list.remove(x): x not in list (s)")
        # [allowed.remove(pos) for pos in disallowed]

        # Only spawn in one location
        # allowed = [(2*20, 2*20)]

        self.pos = allowed[np.random.choice(len(allowed))]

        self.x = self.pos[0]
        self.y = self.pos[1]

        self.history.clear()
        self.history.append((self.x, self.y))

        self.hasSword = False


    # Update steve position
    def update(self, scale, action, action_space):

        # ACTION SPACE OF 5 - Nothing, Up, Down, Left, Right

        # Human AI controls, 5 controls allowed, but limited movement
        if action_space == 5:
            # Nothing
            if action == 0:
                # pass
                self.dy = 0 # stop
                self.dx = 0
            # Up
            elif action == 1:
                self.dy = -1 # move up
                self.dx = 0
                self.prev_pos = (self.x, self.y)

            # Down
            elif action == 2:
                self.dy = 1 # move down
                self.dx = 0
                self.prev_pos = (self.x, self.y)

            # Left
            elif action == 3:
                self.dx = -1 # move left
                self.dy = 0
                self.prev_pos = (self.x, self.y)

            # Right
            elif action == 4:
                self.dx = 1 # move right
                self.dy = 0
                self.prev_pos = (self.x, self.y)

            
        # Updating positions using velocity
        self.x += self.dx * scale
        self.y += self.dy * scale

        self.pos = (self.x, self.y)

        self.history.append(self.pos)

        if len(self.history) > self.history_size:
            self.history.pop(0)


    # Draw steve
    def draw(self, display):

        if self.hasSword:
            display.blit(self.steve_img2, self.pos)
        else:
            display.blit(self.steve_img, self.pos)
        