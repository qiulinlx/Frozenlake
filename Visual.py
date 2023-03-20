import Gridworld
import pygame
import jax.numpy as jnp
 
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (137, 207, 240)
RED = (255, 0, 0)
DBLUE= (25, 25, 112)
# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 95
HEIGHT = 95
 
# This sets the margin between each cell
MARGIN = 5
 
# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
grid=jnp.array([[ 0, 0, 0, 0], [ 0, -1, 0, -1], [ 0, 0, 0, -1], [-1, 0, 0, 1]])
#grid=Gridworld.State.grid
elfpos=(0,0)
# Set row 1, cell 5 to one. (Remember rows and
# column numbers start at zero.)
def render(grid):
    #grid=jnp.array(Gridworld.State.grid)
    # Initialize pygame
    pygame.init()
    #elfpos=jnp.array(Gridworld.State.elf_position)
    # Set the HEIGHT and WIDTH of the screen
    WINDOW_SIZE = [400, 400]
    screen = pygame.display.set_mode(WINDOW_SIZE)
    
    # Set title of screen
    pygame.display.set_caption("Frozen_Lake")
    
    # Loop until the user clicks the close button.
    done = False
    
    
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
    
    # # -------- Main Program Loop -----------
    while not done:

        # Set the screen background
        screen.fill(BLACK)
        
        # Draw the grid
        for row in range(4):
            for column in range(4):
                color = BLUE
                if grid[row][column] == 1:
                    color = RED
                if grid[row][column] == -1:
                    color= DBLUE
                pygame.draw.rect(screen,
                                color,
                                [(MARGIN + WIDTH) * column + MARGIN,
                                (MARGIN + HEIGHT) * row + MARGIN,
                                WIDTH,
                                HEIGHT])
        
        elf=pygame.image.load('elf.png')
        elf = pygame.transform.scale(elf, (80, 80))
        screen.blit(elf, (elfpos[0]*105,elfpos[1]*105))
    
        # Limit to 60 frames per second
        clock.tick(10)
    
        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
    
    # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()

render(grid)