"""
Jugar Pong manualmente con las teclas de flecha.
Arriba = mover arriba
Abajo = mover abajo
ESC = salir
"""

import gymnasium as gym
import pygame
import numpy as np

env = gym.make("ALE/Pong-v5", render_mode="human")
state, _ = env.reset()

print("=== PONG MANUAL ===")
print("Flecha ARRIBA  = mover arriba")
print("Flecha ABAJO   = mover abajo")
print("Sin tecla      = no hacer nada")
print("ESC            = salir")
print("=" * 20)

running = True
total_reward = 0

while running:
    action = 0  # NOOP por defecto

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    if not running:
        break

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        action = 2  # RIGHT = UP in Pong
    elif keys[pygame.K_DOWN]:
        action = 3  # LEFT = DOWN in Pong

    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Game Over! Score: {total_reward:.0f}")
        total_reward = 0
        state, _ = env.reset()

env.close()
print("Bye!")
