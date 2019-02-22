# An abstract class defining the subsystem class
# Daniel Graham

from abc import ABC, abstractmethod

class SubSystem(ABC):

    @abstractmethod
    def init_density(self):
        pass

    @abstractmethod
    def get_env_energy(self):
        pass

    @abstractmethod
    def update_proj_pot(self, new_POp):
        pass

    @abstractmethod
    def update_emb_pot(self, new_emb_pot):
        pass

    @abstractmethod
    def update_emb_fock(self):
        pass

    @abstractmethod
    def update_density(self, new_den):
        pass

    @abstractmethod
    def save_orbitals(self):
        pass
