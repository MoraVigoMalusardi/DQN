import numpy as np
import gymnasium as gym

class ConstantRewardEnv(gym.Env):
    """
    Acción única, sin observación, un solo paso, recompensa constante
        ● Acciones disponibles: 1 (única acción)
        ● Observaciones: constante 0
        ● Duración: 1 paso de tiempo (episodio de una sola transición)
        ● Recompensa: +1 en cada episodio
    """
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=int) # Aunque la observación es constante, devolver un vector np.array([0]) es más estándar si después vamos a usar redes neuronales, porque muchas esperan vectores como entrada.
        """
        - low (Union[SupportsFloat, np.ndarray]): Lower bounds of the intervals.
        - high (Union[SupportsFloat, np.ndarray]): Upper bounds of the intervals.
        - shape (Optional[Sequence[int]]): This only needs to be specified if both low and high are scalars and determines the shape of the space. Otherwise, the shape is inferred from the shape of low or high.
        - dtype: The dtype of the elements of the space. If this is an integer type, the Box is essentially a discrete space.
        """
        self.action_space = gym.spaces.Discrete(1) # Una sola accion posible: 0
        self.rewards = {0: 1} # Recompensa constante +1 para la unica accion posible 0
    
    # Internal function for current observation.
    def _get_obs(self):
        """
        Converts internal state to observation format.
    
        Returns
            np array: current observation. Always [0]
        """
        return np.array([0], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """ Resets the environment to an initial state and returns an initial observation. """
        super().reset(seed=seed)
        return self._get_obs(), {} # el diccionario es de info

    def step(self, action):
        """ 
        Runs one timestep of the environment's dynamics. 
        When end of episode is reached, you are responsible 
        for calling reset() to reset this environment's state.

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after taking action passed as parameter
            terminated (bool): whether the episode has ended
            truncated (bool): whether the episode was truncated (due to a time/steps limit)
            info (dict): contains auxiliary information
        """
        
        assert self.action_space.contains(action), "Acción inválida"
        reward = self.rewards[action] # De todas formas sera una recompensa constante +1
        terminated = True # Siempre termina en un solo paso
        return self._get_obs(), reward, terminated, False, {} # El false es de truncated.
    
    def close(self):
        pass

class RandomObsBinaryRewardEnv(gym.Env):
    """
    Acción única, observación aleatoria, un solo paso, recompensa dependiente de
    la observación
        ● Acciones disponibles: 1 (única acción)
        ● Observaciones: aleatorias, con valor +1 o -1
        ● Duración: 1 paso de tiempo
        ● Recompensa: coincide con la observación (+1 o -1)
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=int) 
        self.action_space = gym.spaces.Discrete(1) # Una sola accion posible: 0
        self.rewards = {-1: -1, 1: 1} # Recompensa +1 si la observacion es +1 y -1 si la observacion es -1
        self.state = None 

    def _get_obs(self):
        """
        Converts internal state to observation format.

        Returns
            np array: current observation. 
        """
        return np.array([self.state], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.choice([-1, 1]) # Estado inicial aleatorio entre -1 y 1
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"
        reward = self.rewards[self.state] # Recompensa depende del estado
        terminated = True # Siempre termina en un solo paso
        return self._get_obs(), reward, terminated, False, {}
    
    def close(self):
        pass

class TwoStepDelayedRewardEnv(gym.env):
    """
    Acción única, observación determinista, dos pasos, recompensa diferida
        ● Acciones disponibles: 1 (única acción)
        ● Observaciones: en el primer paso se observa 0; en el segundo paso se observa 1
        ● Duración: 2 pasos por episodio
        ● Recompensa: 0 en el primer paso, +1 al final del episodio
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=int) 
        self.action_space = gym.spaces.Discrete(1) # Una sola accion posible: 0
        self.rewards = {0: 0, 1: 1} # Recompensa +1 si la observacion es +1 y -1 si la observacion es -1
        self.state = None 
        self.step_count = 0
    
    def _get_obs(self):
        """
        Convert internal state to observation format.

        Returns:
            np.ndarray: current observation. Either [0] or [1]
        """
        return np.array([self.state], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0 # Estado inicial es 0
        self.step_count = 0
        return self._get_obs(), {} 
    
    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"
        
        reward = self.rewards[self.state] # Primero observo el reward, desps avanzo
        self.step_count += 1
        
        if self.step_count == 1:
            self.state = 1 # Cambia a estado 1 en el segundo paso
            terminated = False
        else:
            terminated = True # Termina después del segundo paso

        return self._get_obs(), reward, terminated, False, {}
    
    def close(self):
        pass