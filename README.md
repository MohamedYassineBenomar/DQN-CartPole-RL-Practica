# DQN Agent for Atari Pong

**Alumno:** Mohamed Yassine BenOmar  
**Asignatura:** OPT-ML (Deep Learning & Reinforcement Learning)  
**Fecha:** 05/04/2026

## Descripcion

Agente DQN (Deep Q-Network) con arquitectura CNN (Nature DQN 2015) entrenado para jugar al clasico juego de Pong de Atari, utilizando observaciones de pixeles como entrada.

### Caracteristicas
- **Entorno:** ALE/Pong-v5 (Gymnasium Atari)
- **Arquitectura:** CNN con 3 capas convolucionales + 2 capas fully connected (~1.7M parametros)
- **Double DQN:** Reduce sobreestimacion de Q-valores
- **Experience Replay:** Buffer de 100K transiciones (uint8 para eficiencia de memoria)
- **Target Network:** Actualizacion cada 1000 pasos
- **Preprocessing:** Grayscale, resize 84x84, frame skip 4, frame stack 4

## Estructura del Proyecto

| Archivo | Descripcion |
|---|---|
| `BenOmar_MohamedYassine_RL_Practica.ipynb` | Notebook principal con 7 secciones |
| `agent.py` | Agente DQN con CNN (QNetworkCNN, DQNAgent, DQNConfig) |
| `environment.py` | Entorno Pong con preprocessing pipeline |
| `utils.py` | ReplayBuffer, funciones de plotting y creacion de GIF |
| `train.py` | Script de entrenamiento con CLI args |
| `watch.py` | Ver al agente entrenado jugar en vivo |
| `play_manual.py` | Jugar a Pong manualmente con teclado |
| `models/dqn_pong.pth` | Modelo entrenado (~547K pasos) |

## Instalacion

```bash
pip install torch gymnasium "gymnasium[atari]" ale-py matplotlib numpy imageio pygame opencv-python-headless
```

## Uso

### Entrenar el agente
```bash
python3 train.py                    # Entrenar sin visualizacion (rapido)
python3 train.py --render           # Entrenar con visualizacion (lento)
python3 train.py --total-steps 1000000  # Entrenar 1M pasos
```

### Ver al agente entrenado jugar
```bash
python3 watch.py
```

### Jugar manualmente
```bash
python3 play_manual.py
```

### Ejecutar el notebook
Abrir `BenOmar_MohamedYassine_RL_Practica.ipynb` en VS Code o Jupyter y ejecutar todas las celdas.

## Resultados

El agente mejora progresivamente desde -21 (perder siempre) hasta aproximadamente -16 tras 547K pasos de entrenamiento, demostrando aprendizaje significativo comparado con el baseline aleatorio (-20.7).

## Bonificaciones

- **Double DQN (+5%):** Implementado y comparado con DQN estandar
- **Analisis de hiperparametros (+5%):** Comparativa de learning rates (5e-5, 1e-4, 5e-4)
- **Visualizacion del agente (+5%):** GIF del agente jugando Pong
