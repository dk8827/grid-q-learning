import numpy as np
import random
import pygame
import time

# Define the grid environment
class GridEnvironment:
    def __init__(self):
        self.size = 4  # Size of the grid (4x4)
        self.start = (0, 0)  # Starting position of the agent
        self.goal = (3, 3)  # Goal position of the agent
        self.reset()  # Initialize the agent's position

    def reset(self):
        self.agent_pos = self.start  # Reset the agent's position to the start
        return self.agent_pos  # Return the initial position

    def step(self, action):
        # Get the current position of the agent
        x, y = self.agent_pos
        # Update the position based on the action taken
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.size - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.size - 1)

        self.agent_pos = (x, y)  # Update the agent's position
        # Determine the reward and whether the goal is reached
        reward = 1 if self.agent_pos == self.goal else 0
        done = self.agent_pos == self.goal
        return self.agent_pos, reward, done  # Return the new position, reward, and done status

# Initialize Pygame
pygame.init()
screen_size = 400  # Screen size in pixels
cell_size = screen_size // 4  # Size of each cell in the grid
screen = pygame.display.set_mode((screen_size, screen_size))  # Set up the display
pygame.display.set_caption("Q-learning Grid")  # Set the window title

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.3  # Exploration rate
num_episodes = 1000  # Number of episodes for training

# Initialize Q-table
q_table = np.zeros((4, 4, 4))  # 4x4 grid with 4 possible actions per state

env = GridEnvironment()  # Create the grid environment
font = pygame.font.Font(None, 24)  # Set the font for displaying text

# Q-learning algorithm with Pygame visualization
for episode in range(num_episodes):
    state = env.reset()  # Reset the environment for each episode
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1, 2, 3])  # Explore: choose a random action
        else:
            action = np.argmax(q_table[state[0], state[1]])  # Exploit: choose the best action

        # Take action and observe the result
        next_state, reward, done = env.step(action)
        old_value = q_table[state[0], state[1], action]  # Current Q-value
        next_max = np.max(q_table[next_state[0], next_state[1]])  # Max Q-value for next state

        # Update Q-value using the Q-learning formula
        q_table[state[0], state[1], action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state  # Move to the next state

        # Draw the grid
        screen.fill(WHITE)
        for i in range(4):
            for j in range(4):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, BLACK, rect, 1)
                
                # Color cells based on max Q-value
                max_q = np.max(q_table[i, j])
                color_intensity = min(max(int(max_q * 255), 0), 255)
                cell_color = (255 - color_intensity, 255, 255 - color_intensity)
                pygame.draw.rect(screen, cell_color, rect)
                
                # Draw Q-values
                q_values = q_table[i, j]
                text_surface = font.render(f'U:{q_values[0]:.2f}', True, BLACK)
                screen.blit(text_surface, (j * cell_size + 5, i * cell_size + 5))
                text_surface = font.render(f'D:{q_values[1]:.2f}', True, BLACK)
                screen.blit(text_surface, (j * cell_size + 5, i * cell_size + 20))
                text_surface = font.render(f'L:{q_values[2]:.2f}', True, BLACK)
                screen.blit(text_surface, (j * cell_size + 5, i * cell_size + 35))
                text_surface = font.render(f'R:{q_values[3]:.2f}', True, BLACK)
                screen.blit(text_surface, (j * cell_size + 5, i * cell_size + 50))

                # Highlight the goal cell
                if (i, j) == env.goal:
                    pygame.draw.rect(screen, GREEN, rect)
                # Highlight the agent's current position
                if (i, j) == state:
                    pygame.draw.rect(screen, BLUE, rect)

        pygame.display.flip()
        #time.sleep(0.01)  # Uncomment to slow down the visualization

# Display the Q-table after training
print("Q-table after training:")
print(q_table)
pygame.quit()
