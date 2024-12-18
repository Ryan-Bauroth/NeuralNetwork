import pygame
import numpy as np

from nn import NeuralNetwork

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 280  # 10x scaling for 28x28 grid
GRID_SIZE = 28
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
SIDE_BUFFER = 50  # Added buffer area
TOP_BUFFER = 50
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)  # Full red, full green, no blue

# Initialize screen
screen_width = WINDOW_SIZE + 200 + 2 * SIDE_BUFFER
screen_height = WINDOW_SIZE + 2 * TOP_BUFFER
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Draw a Number")

# Canvas for drawing
drawing_canvas = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

def draw_grid():
    """Draws the grid lines on the canvas."""
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x + SIDE_BUFFER, TOP_BUFFER), (x + SIDE_BUFFER, WINDOW_SIZE + TOP_BUFFER))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (SIDE_BUFFER, y + TOP_BUFFER), (WINDOW_SIZE + SIDE_BUFFER, y + TOP_BUFFER))

def update_predictions(classification_array):
    """Placeholder for updating predictions."""
    # Clear the predictions area
    pygame.draw.rect(screen, BLACK, (WINDOW_SIZE + SIDE_BUFFER, TOP_BUFFER, 200, WINDOW_SIZE))

    # Display placeholder predictions
    font = pygame.font.SysFont(None, 36)
    predicted = np.argmax(classification_array)
    classification_array[classification_array < 0] = 0
    total = np.sum(classification_array)
    for i in range(len(classification_array)):
        text_color = WHITE
        if i == predicted:
            text_color = YELLOW
        text = font.render(f"{i} : {max(0, classification_array[i] / total):6.2f}", True, text_color)
        screen.blit(text, (WINDOW_SIZE + SIDE_BUFFER + 10, TOP_BUFFER + i * 30))

def main():
    nn = NeuralNetwork([784, 200, 10], 100, .1)
    nn.load_model("mnist_model.json")

    running = True
    while running:
        screen.fill(BLACK)

        # Draw grid and canvas
        draw_grid()
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if drawing_canvas[y, x] == 1:
                    rect = pygame.Rect(x * CELL_SIZE + SIDE_BUFFER, y * CELL_SIZE + TOP_BUFFER, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, WHITE, rect)
                elif drawing_canvas[y, x] == .5:
                    rect = pygame.Rect(x * CELL_SIZE + SIDE_BUFFER, y * CELL_SIZE + TOP_BUFFER, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, GRAY, rect)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle drawing
        if pygame.mouse.get_pressed()[0]:  # Left mouse button
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if SIDE_BUFFER <= mouse_x < SIDE_BUFFER + WINDOW_SIZE and TOP_BUFFER <= mouse_y < TOP_BUFFER + WINDOW_SIZE:
                grid_x = (mouse_x - SIDE_BUFFER) // CELL_SIZE
                grid_y = (mouse_y - TOP_BUFFER) // CELL_SIZE
                drawing_canvas[grid_y, grid_x] = 1.0  # Mark the cell as drawn
                if grid_x + 1 <= 27:
                    drawing_canvas[grid_y, grid_x + 1] = 1
                if grid_y + 1 <= 27:
                    drawing_canvas[grid_y + 1, grid_x] = 1
                if grid_y + 1 <= 27 and grid_x <= 27:
                    drawing_canvas[grid_y + 1, grid_x + 1] = 1

        # Handle clearing
        keys = pygame.key.get_pressed()
        if keys[pygame.K_c]:
            drawing_canvas.fill(0)  # Clear the canvas

        # Update predictions
        classification = nn.classify(drawing_canvas.flatten())
        update_predictions(classification)

        # Update the display
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
