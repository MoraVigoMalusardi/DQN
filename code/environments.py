import numpy as np
import gymnasium as gym
import collections

class ConstantRewardEnv(gym.Env):
    """
    Acción única, sin observación, un solo paso, recompensa constante
        ● Acciones disponibles: 1 (única acción)
        ● Observaciones: constante 0
        ● Duración: 1 paso de tiempo (episodio de una sola transición)
        ● Recompensa: +1 en cada episodio

    Q learning deberia dar como resultado Q(0,0) = 1
    """
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)        # Una sola accion posible: 0
        self._state = 0.0
    
    # Internal function for current observation.
    def _get_obs(self):
        return np.array([self._state], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """ Resets the environment to an initial state and returns an initial observation. """
        super().reset(seed=seed)
        self._state = 0.0
        return self._get_obs(), {}

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
        reward = 1.0
        terminated = True  # episodio de 1 paso
        return self._get_obs(), float(reward), terminated, False, {}
    
    def close(self):
        pass

class RandomObsBinaryRewardEnv(gym.Env):
    """
    Acción única, observación aleatoria, un solo paso, recompensa dependiente de
    la observación
        ● Acciones disponibles: 1 (única acción)
        ● Observaciones: aleatorias, con valor +1 o -1 --> las cambie a 0 y 1
        ● Duración: 1 paso de tiempo
        ● Recompensa: coincide con la observación (+1 o -1) --> puse -1 para 0 y +1 para 1

    Q learning deberia dar como resultado Q(0,0) = -1 y Q(1,0) = +1
    """

    def __init__(self):
        super().__init__()
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)
        self.rewards = {0: -1, 1: 1} # Recompensa +1 si la observacion es +1 y -1 si la observacion es -1
        self._state = None 

    def _get_obs(self):
        return np.array([float(self._state)], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = int(np.random.randint(0, 2))  # 0 o 1
        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"
        reward = self.rewards[self._state] # Recompensa depende del estado
        terminated = True
        return self._get_obs(), float(reward), terminated, False, {}
    
    def close(self):
        pass

class TwoStepDelayedRewardEnv(gym.Env):
    """
    Acción única, observación determinista, dos pasos, recompensa diferida
        ● Acciones disponibles: 1 (única acción)
        ● Observaciones: en el primer paso se observa 0; en el segundo paso se observa 1
        ● Duración: 2 pasos por episodio
        ● Recompensa: 0 en el primer paso, +1 al final del episodio
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(1)
        self._state = 0
        self._t = 0
        self.rewards = {0: 0, 1: 1} # Recompensa +1 si la observacion es +1 y -1 si la observacion es -1
    
    def _get_obs(self):
        return np.array([float(self._state)], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = 0 # Estado inicial en 0
        self._t = 0
        return self._get_obs(), {}
    
    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"
        
        reward = self.rewards[self._state] # Primero observo el reward, desps avanzo
        self._t += 1
        
        if self._t == 1:
            self._state = 1 # Cambia a estado 1 en el segundo paso
            terminated = False
        else:
            terminated = True # Termina después del segundo paso

        return self._get_obs(), reward, terminated, False, {}
    
    def close(self):
        pass

# Wrapper para apilar k frames de MinAtar
class SimpleFrameStack:
    def __init__(self, env, k: int = 4):
        self.env = env
        self.k = k
        self.frames = collections.deque(maxlen=k)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._preprocess(obs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        next_obs, r, terminated, truncated, info = self.env.step(action)
        next_obs = self._preprocess(next_obs)
        self.frames.append(next_obs)
        return self._get_obs(), r, terminated, truncated, info

    def _get_obs(self):
        # Apila a lo largo del canal: (C*k, H, W)
        return np.concatenate(list(self.frames), axis=0)

    def _preprocess(self, obs):
        # pasamos de (H, W, C) a (C, H, W) 
        if obs.ndim == 3:
            obs = np.transpose(obs, (2, 0, 1))
        obs = obs.astype(np.float32)
        return obs / 1.0

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()