# An abstract class defining a SuperSystem
# Daniel Graham

from abc import ABC, abstractmethod

class SuperSystem(ABC):

    @abstractmethod
    def init_density(self):
        pass

    @abstractmethod
    def get_supersystem_energy(self):
        pass

    @abstractmethod
    def env_in_env_energy(self):
        pass

    @abstractmethod
    def get_hl_energy(self):
        pass

    @abstractmethod
    def update_proj_pot(self):
        pass

    @abstractmethod
    def update_fock(self):
        pass

    @abstractmethod
    def save_chkfile(self):
        pass

    @abstractmethod
    def read_chkfile(self):
        pass

    @abstractmethod
    def freeze_and_thaw(self):
        pass
