"""
Block Drop Game - Juego de atrapar bloques que caen.
Mueve la plataforma para atrapar los bloques y evitar que caigan al suelo.

Controles:
  Flecha IZQUIERDA / A = mover izquierda
  Flecha DERECHA / D   = mover derecha
  ESC = salir
"""

import pygame
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 100, 220)
YELLOW = (240, 200, 40)
ORANGE = (240, 130, 40)
GRAY = (180, 180, 180)
DARK_BG = (30, 30, 45)

BLOCK_COLORS = [RED, GREEN, BLUE, YELLOW, ORANGE]

# Game settings
SCREEN_W = 400
SCREEN_H = 600
PADDLE_W = 80
PADDLE_H = 15
BLOCK_SIZE = 30
PADDLE_SPEED = 8
BLOCK_FALL_SPEED = 3
MAX_BLOCKS = 5
FPS = 60


class BlockDropGame:
    """Juego de atrapar bloques - funciona como entorno Gymnasium y como juego manual."""

    def __init__(self, render_mode=None, difficulty_increase=True):
        self.render_mode = render_mode
        self.difficulty_increase = difficulty_increase
        self.screen = None
        self.clock = None
        self.font = None

        # Gymnasium spaces
        # State: [paddle_x, block1_x, block1_y, block2_x, block2_y, ..., blockN_x, blockN_y]
        # Normalized to [0, 1]
        obs_size = 1 + MAX_BLOCKS * 2  # paddle_x + (x, y) per block
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=left, 1=stay, 2=right

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.paddle_x = SCREEN_W // 2 - PADDLE_W // 2
        self.blocks = []
        self.score = 0
        self.lives = 3
        self.spawn_timer = 0
        self.spawn_interval = 60  # frames between spawns
        self.fall_speed = BLOCK_FALL_SPEED
        self.done = False
        self.frame_count = 0

        self._spawn_block()
        return self._get_obs(), {}

    def _spawn_block(self):
        if len(self.blocks) < MAX_BLOCKS:
            x = random.randint(0, SCREEN_W - BLOCK_SIZE)
            color = random.choice(BLOCK_COLORS)
            self.blocks.append({'x': x, 'y': -BLOCK_SIZE, 'color': color})

    def _get_obs(self):
        obs = np.zeros(1 + MAX_BLOCKS * 2, dtype=np.float32)
        obs[0] = self.paddle_x / SCREEN_W

        for i, block in enumerate(self.blocks[:MAX_BLOCKS]):
            obs[1 + i * 2] = block['x'] / SCREEN_W
            obs[1 + i * 2 + 1] = block['y'] / SCREEN_H

        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        reward = 0
        self.frame_count += 1

        # Move paddle
        if action == 0:
            self.paddle_x = max(0, self.paddle_x - PADDLE_SPEED)
        elif action == 2:
            self.paddle_x = min(SCREEN_W - PADDLE_W, self.paddle_x + PADDLE_SPEED)

        # Move blocks
        blocks_to_remove = []
        for i, block in enumerate(self.blocks):
            block['y'] += self.fall_speed

            # Check catch
            if (block['y'] + BLOCK_SIZE >= SCREEN_H - PADDLE_H - 10 and
                block['y'] + BLOCK_SIZE <= SCREEN_H - PADDLE_H + self.fall_speed + 10 and
                block['x'] + BLOCK_SIZE > self.paddle_x and
                block['x'] < self.paddle_x + PADDLE_W):
                self.score += 1
                reward = 1.0
                blocks_to_remove.append(i)

            # Check miss
            elif block['y'] > SCREEN_H:
                self.lives -= 1
                reward = -1.0
                blocks_to_remove.append(i)

        for i in sorted(blocks_to_remove, reverse=True):
            self.blocks.pop(i)

        # Spawn new blocks
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self._spawn_block()
            self.spawn_timer = 0

        # Increase difficulty
        if self.difficulty_increase and self.score > 0 and self.score % 5 == 0:
            self.fall_speed = min(BLOCK_FALL_SPEED + self.score // 5, 10)
            self.spawn_interval = max(20, 60 - self.score // 3)

        # Check game over
        if self.lives <= 0:
            self.done = True
            reward = -2.0

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), reward, self.done, False, {"score": self.score}

    def _init_render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Block Drop - Atrapa los bloques!")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 18)

    def _render_human(self):
        self._init_render()
        self.screen.fill(DARK_BG)

        # Draw blocks
        for block in self.blocks:
            pygame.draw.rect(self.screen, block['color'],
                           (block['x'], block['y'], BLOCK_SIZE, BLOCK_SIZE),
                           border_radius=5)
            pygame.draw.rect(self.screen, WHITE,
                           (block['x'], block['y'], BLOCK_SIZE, BLOCK_SIZE),
                           width=2, border_radius=5)

        # Draw paddle
        paddle_y = SCREEN_H - PADDLE_H - 10
        pygame.draw.rect(self.screen, WHITE,
                        (self.paddle_x, paddle_y, PADDLE_W, PADDLE_H),
                        border_radius=8)

        # Draw HUD
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        lives_text = self.font.render(f"Lives: {'❤ ' * self.lives}", True, RED)
        speed_text = self.font_small.render(f"Speed: {self.fall_speed}", True, GRAY)

        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (10, 40))
        self.screen.blit(speed_text, (SCREEN_W - 100, 10))

        if self.done:
            go_text = self.font.render("GAME OVER!", True, RED)
            restart_text = self.font_small.render("Press SPACE to restart, ESC to quit", True, WHITE)
            self.screen.blit(go_text, (SCREEN_W // 2 - go_text.get_width() // 2, SCREEN_H // 2 - 30))
            self.screen.blit(restart_text, (SCREEN_W // 2 - restart_text.get_width() // 2, SCREEN_H // 2 + 10))

        pygame.display.flip()
        self.clock.tick(FPS)

    def render(self):
        if self.render_mode == "rgb_array":
            self._init_render()
            self._render_human()
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        elif self.render_mode == "human":
            self._render_human()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


def play():
    """Jugar manualmente."""
    game = BlockDropGame(render_mode="human")
    game._init_render()
    state, _ = game.reset()

    print("=== BLOCK DROP ===")
    print("← / A = izquierda")
    print("→ / D = derecha")
    print("SPACE  = reiniciar")
    print("ESC    = salir")
    print("==================")

    running = True
    while running:
        action = 1  # stay by default

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and game.done:
                    state, _ = game.reset()

        if not running:
            break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = 0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = 2

        if not game.done:
            state, reward, done, _, info = game.step(action)

    game.close()


if __name__ == "__main__":
    play()
