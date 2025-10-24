# Deep Q-Learning y Q-Learning 

Este proyecto implementa y compara **Q-Learning clásico** y **Deep Q-Network (DQN)**, dos algoritmos fundamentales del Aprendizaje por Refuerzo (*Reinforcement Learning*).  
El objetivo principal es **evaluar la estabilidad, velocidad de convergencia y desempeño** de ambos métodos en entornos como *FrozenLake-v1*, *CartPole-v1* o *MinAtar/Breakout-v0*, explorando además los efectos de técnicas clave como:

- Uso de *target networks*  
- *Experience replay buffer*  
- Estrategias de exploración *ε-greedy* con decaimiento exponencial  

---

## Estructura general del proyecto

```
code/
  ├── dqn/
  │    ├── dqn.py              → implementación del agente DQN, configuración y funciones de entrenamiento
  │    ├── experiments.py      → experimentos: target vs no-target, replay vs no-replay, DQN vs REINFORCE
  │    ├── replayBuffer.py     → implementación del buffer de experiencia FIFO
  
  ├── q_learning/
  │    ├── q_learning.py       → implementación del agente Q-Learning tabular
  │    ├── experiments.py      → experimentos: decay vs no-decay, hiperparámetros, entornos personalizados
  
  ├── arquitectures.py         → definiciones de redes neuronales (MLP y CNN)
  ├── environments.py          → entornos personalizados de prueba (ConstantRewardEnv, RandomObsBinaryRewardEnv, TwoStepDelayedRewardEnv, SimpleFrameStack para MinAtar)
  ├── plots/                   → graficos resultantes de probar q_learning en los entornos de prueba
  ├── TP2 DQN.pdf              → consigna del trabajo práctico
  └── README.md                → este archivo

```

---

## Descripción de los algoritmos

### Q-Learning (tabular)
Q-Learning es un algoritmo *off-policy* que aprende una función de valor de acción \( Q(s,a) \) mediante actualizaciones iterativas:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

- Se utiliza una tabla Q discreta para representar los valores.
- Se aplica exploración *ε-greedy* y opcionalmente un decaimiento exponencial de ε.
- Ideal para entornos pequeños o discretos (como *FrozenLake-v1*).

### Deep Q-Network (DQN)
DQN reemplaza la tabla Q por una **red neuronal** que aproxima la función \( Q(s,a;\theta) \).  
Introduce dos mecanismos para estabilizar el aprendizaje:

1. **Target Network:** una copia periódicamente actualizada de la red principal para romper correlaciones en las actualizaciones.
2. **Replay Buffer:** almacena transiciones pasadas \((s,a,r,s')\) y permite muestreo aleatorio para entrenar con batches de datos no correlacionados.

La pérdida usada es la **Huber Loss**:
$$
L(\theta) \;=\; \mathbb{E}\Big[\operatorname{Huber}\big(r + \gamma \max_{a'} Q'(s',a';\theta^-) - Q(s,a;\theta)\big)\Big]
$$


---

## Instalación y dependencias

Se usa Python 3.10+ y se deben tener las siguientes librerías instaladas:

```bash
pip install torch gymnasium matplotlib numpy
```

Si se utilizan entornos *MinAtar*, también instalar:

```bash
pip install minatar
```

---

## Ejemplos de ejecución

### Experimentos Q-Learning
```bash
python code/q_learning/experiments.py
```
Los distintos experimentos pueden habilitarse/deshabilitarse dentro del archivo `experiments.py`.

---

### Experimentos DQN 
```bash
python code/dqn/experiments.py
```
Los distintos experimentos pueden habilitarse/deshabilitarse dentro del archivo `experiments.py`.

---

## Resultados esperados

Durante el entrenamiento se registran y grafican las siguientes métricas:

- **Reward por episodio**
- **Media móvil de 100 episodios (modificable)**
- **Pérdida TD (Huber loss)**
- **Epsilon a lo largo del entrenamiento**

Los graficos se guardan automáticamente en el directorio `plots/` dentro de su corrrespondiente carpeta (q_learning o dqn) y también pueden visualizarse en **TensorBoard**.

- En los entornos personalizados, se verifica la convergencia exacta de los valores Q esperados (por ejemplo, \( Q(0,0)=1 \) en `ConstantRewardEnv`). Los resultados de ejecutar q_learning para los entornos 
custom se encuentran en la carpeta `plots/` de mas alto nivel.
---

## Experimentos incluidos

| Experimento | Descripción | Archivo |
|--------------|--------------|----------|
| **Target vs No Target** | Evalúa la estabilidad del entrenamiento al usar una red objetivo. | `dqn/experiments.py` |
| **Replay vs No Replay** | Compara el uso del replay buffer vs actualizaciones online. | `dqn/experiments.py` |
| **DQN vs REINFORCE** | Compara el desempeño en CartPole de un agente de DQN vs un agente de Reinforce. | `dqn/experiments.py` |
| **Decay vs No Decay** | Evalúa el impacto del decaimiento de ε sobre Q-Learning. | `q_learning/experiments.py` |
| **Tuning de Hiperparámetros** | Compara distintas combinaciones de α y ε en FrozenLake. | `q_learning/experiments.py` |
| **Entornos personalizados** | Prueba propiedades del algoritmo con dinámicas controladas. | `q_learning/experiments.py` |