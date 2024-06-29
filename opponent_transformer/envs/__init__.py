from absl import flags

from .mpe.environment import MultiAgentEnv
from .starcraft2.StarCraft2_Env import StarCraft2Env
from .wrappers import DummyVecEnv, ShareDummyVecEnv, ShareSubprocVecEnv, SubprocVecEnv

FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])
