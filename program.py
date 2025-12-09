import numpy as np
import random
import pygame
import time

# Constants for grid dimensions
NUM_ROWS = 6
NUM_COLS = 6

# Define the grid environment
class GridEnvironment:
    def __init__(self, num_landmines):
        self.num_rows = NUM_ROWS
        self.num_cols = NUM_COLS
        self.start = (0, 0)  # Starting position of the agent
        self.goal = (NUM_ROWS - 1, NUM_COLS - 1)  # Goal position of the agent
        self.num_landmines = num_landmines
        self.landmines = []
        self.agent_pos = self.start  # Initialize the agent's position
        self.place_landmines()  # Place landmines in random positions

    def reset(self):
        self.agent_pos = self.start  # Reset the agent's position to the start
        return self.agent_pos  # Return the initial position

    def place_landmines(self):
        self.landmines = []
        available_positions = [(x, y) for x in range(self.num_rows) for y in range(self.num_cols)
                               if (x, y) != self.start and (x, y) != self.goal]
        self.landmines = random.sample(available_positions, self.num_landmines)

    def step(self, action):
        # Get the current position of the agent
        x, y = self.agent_pos
        # Update the position based on the action taken
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.num_rows - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.num_cols - 1)

        self.agent_pos = (x, y)  # Update the agent's position
        # Determine the reward and whether the goal is reached or a landmine is hit
        if self.agent_pos in self.landmines:
            reward = -10  # Penalty for hitting a landmine
            done = True
        elif self.agent_pos == self.goal:
            reward = 10  # Reward for reaching the goal
            done = True
        else:
            reward = -1  # Small penalty for each move to encourage efficiency
            done = False
        return self.agent_pos, reward, done  # Return the new position, reward, and done status

# Initialize Pygame
pygame.init()
screen_size = 400  # Screen size in pixels
cell_width = screen_size // NUM_COLS  # Width of each cell in the grid
cell_height = screen_size // NUM_ROWS  # Height of each cell in the grid
screen = pygame.display.set_mode((screen_size, screen_size + 100))  # Set up the display with extra space for UI
pygame.display.set_caption("Q-learning Grid with Landmines")  # Set the window title

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
initial_epsilon = 1.0  # Initial exploration rate
min_epsilon = 0.01  # Minimum exploration rate
epsilon_decay = 0.915  # Exponential decay factor per episode (reaches 0.7 at episode 4)
num_episodes = 1000  # Number of episodes for training

font = pygame.font.Font(None, 24)  # Set the font for displaying text

# Create a dropdown for selecting the number of landmines
class Dropdown:
    def __init__(self, x, y, w, h, options):
        self.rect = pygame.Rect(x, y, w, h)
        self.options = options
        self.selected = options[0]
        self.dropdown_open = False

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        text = font.render(str(self.selected), True, BLACK)
        surface.blit(text, (self.rect.x + 5, self.rect.y + 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dropdown_open = not self.dropdown_open
            elif self.dropdown_open:
                for i, option in enumerate(self.options):
                    option_rect = pygame.Rect(self.rect.x, self.rect.y + (i+1)*self.rect.height, self.rect.width, self.rect.height)
                    if option_rect.collidepoint(event.pos):
                        self.selected = option
                        self.dropdown_open = False
                        return True
        return False

    def draw_options(self, surface):
        if self.dropdown_open:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.rect.x, self.rect.y + (i+1)*self.rect.height, self.rect.width, self.rect.height)
                pygame.draw.rect(surface, WHITE, option_rect)
                pygame.draw.rect(surface, BLACK, option_rect, 2)
                text = font.render(str(option), True, BLACK)
                surface.blit(text, (option_rect.x + 5, option_rect.y + 5))

# Create a button for starting the simulation
class Button:
    def __init__(self, x, y, w, h, text, selected=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.selected = selected

    def draw(self, surface):
        bg_color = BLUE if self.selected else WHITE
        pygame.draw.rect(surface, bg_color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        text_color = WHITE if self.selected else BLACK
        text = font.render(self.text, True, text_color)
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


# Create UI elements
landmine_dropdown = Dropdown(10, screen_size/2, 100, 30, [0, 1, 2, 3, 4])
start_button = Button(120, screen_size/2, 100, 30, "Start")
speed_1x_button = Button(10, screen_size + 60, 60, 30, "1x", selected=True)
speed_5x_button = Button(80, screen_size + 60, 60, 30, "5x")
speed_20x_button = Button(150, screen_size + 60, 60, 30, "20x")
speed_multiplier = 1  # Default to 1x speed
base_sleep_time = 0.5  # Base sleep time in seconds for 1x speed

# Main game loop
running = True
simulation_started = False
env = None
epsilon = initial_epsilon  # Initialize epsilon

# Create a surface for the episode counter
episode_surface = pygame.Surface((screen_size, 50))
episode_surface.fill(WHITE)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if not simulation_started:
            if landmine_dropdown.handle_event(event):
                pass
            if start_button.handle_event(event):
                simulation_started = True
                env = GridEnvironment(landmine_dropdown.selected)
                q_table = np.zeros((NUM_ROWS, NUM_COLS, 4))  # Reset Q-table for new simulation
                epsilon = initial_epsilon  # Reset epsilon for new simulation

    screen.fill(WHITE)

    if not simulation_started:
        landmine_label = font.render("Number of landmines:", True, BLACK)
        screen.blit(landmine_label, (10, screen_size/2 - 25))
        landmine_dropdown.draw(screen)
        landmine_dropdown.draw_options(screen)
        start_button.draw(screen)
    else:
        # Q-learning algorithm with Pygame visualization
        for episode in range(num_episodes):
            # Decay epsilon exponentially (faster decay)
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
            
            state = env.reset()  # Reset the environment for each episode
            done = False

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        done = True
                        break
                    if speed_1x_button.handle_event(event):
                        speed_1x_button.selected = True
                        speed_5x_button.selected = False
                        speed_20x_button.selected = False
                        speed_multiplier = 1
                    elif speed_5x_button.handle_event(event):
                        speed_1x_button.selected = False
                        speed_5x_button.selected = True
                        speed_20x_button.selected = False
                        speed_multiplier = 5
                    elif speed_20x_button.handle_event(event):
                        speed_1x_button.selected = False
                        speed_5x_button.selected = False
                        speed_20x_button.selected = True
                        speed_multiplier = 20

                if not running:
                    break

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

                def get_color(value, min_value, max_value):
                    threshold = 0.01  # Values within this range of 0 are considered neutral (white)
                    if abs(value) < threshold:
                        return (255, 255, 255)  # White for neutral values near 0
                    
                    max_abs = max(abs(min_value), abs(max_value))
                    if max_abs < threshold:
                        return (255, 255, 255)  # White if all values are near zero
                    
                    normalized = value / max_abs  # Normalize relative to zero
                    
                    if normalized > 0:
                        # Positive values: white to green
                        intensity = min(1.0, normalized)
                        return (int(255 * (1 - intensity)), 255, int(255 * (1 - intensity)))
                    else:
                        # Negative values: white to red
                        intensity = min(1.0, abs(normalized))
                        return (255, int(255 * (1 - intensity)), int(255 * (1 - intensity)))

                # Find the min and max Q-values across the entire grid
                min_q = np.min(q_table)
                max_q = np.max(q_table)
                
                # Draw the grid
                for i in range(NUM_ROWS):
                    for j in range(NUM_COLS):
                        rect = pygame.Rect(j * cell_width, i * cell_height, cell_width, cell_height)
                        pygame.draw.rect(screen, BLACK, rect, 1)
                        
                        # Color cells based on max Q-value, relative to overall min and max
                        max_q_cell = np.max(q_table[i, j])
                        cell_color = get_color(max_q_cell, min_q, max_q)
                        pygame.draw.rect(screen, cell_color, rect)
                        
                        # Draw Q-values
                        q_values = q_table[i, j]
                        for idx, direction in enumerate(['U', 'D', 'L', 'R']):
                            text_color = BLACK if sum(cell_color) > 382 else WHITE  # Use black text on light backgrounds, white on dark
                            text_surface = font.render(f'{direction}:{q_values[idx]:.2f}', True, text_color)
                            screen.blit(text_surface, (j * cell_width + 5, i * cell_height + 5 + idx * 15))

                        # Highlight the goal cell
                        if (i, j) == env.goal:
                            pygame.draw.rect(screen, YELLOW, rect, 3)
                        # Highlight the landmines
                        if (i, j) in env.landmines:
                            pygame.draw.rect(screen, BLACK, rect, 3)
                        # Highlight the agent's current position
                        if (i, j) == state:
                            pygame.draw.circle(screen, BLUE, (j * cell_width + cell_width // 2, i * cell_height + cell_height // 2), cell_width // 4)

                # Clear the episode surface and redraw the episode number and epsilon
                episode_surface.fill(WHITE)
                episode_text = font.render(f"Episode: {episode + 1}/{num_episodes}", True, BLACK)
                episode_surface.blit(episode_text, (10, 10))
                epsilon_text = font.render(f"Epsilon: {epsilon:.2f}", True, BLACK)
                episode_surface.blit(epsilon_text, (10, 30))
                screen.blit(episode_surface, (0, screen_size))
                
                speed_1x_button.draw(screen)
                speed_5x_button.draw(screen)
                speed_20x_button.draw(screen)

                pygame.display.flip()
                time.sleep(base_sleep_time / speed_multiplier)

            if not running:
                break

        if running:
            # Display "Simulation Complete" message
            screen.fill(WHITE)
            complete_text = font.render("Simulation Complete", True, BLACK)
            screen.blit(complete_text, (screen_size // 2 - complete_text.get_width() // 2, screen_size // 2))
            pygame.display.flip()

            # Wait for user to close the window
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                        running = False

    pygame.display.flip()

pygame.quit()
