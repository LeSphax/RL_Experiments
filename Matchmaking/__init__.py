from abc import ABC, abstractmethod, abstractproperty
from types import SimpleNamespace


class EnvConfiguration(ABC):

    @abstractmethod
    def create_model(self, name, input_shape, reuse=False):
        """
        Creates the model to use for the policy and value function of this environment
        """
        pass

    @property
    def parameters(self):
        parameters = SimpleNamespace(**self._parameters())
        parameters.total_batches = parameters.total_timesteps // parameters.batch_size
        return parameters

    @abstractmethod
    def _parameters(self):
        """
        A dictionary containing the ppo parameters
        """
        pass

    @abstractproperty
    def env_name(self):
        """
        The name of the gym environment (i.e CartPole-v1)
        """
        pass
