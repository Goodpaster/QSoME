# A method to define all cluster supsystem objects
# Daniel Graham

from qsome import subsystem
from pyscf import gto, scf, dft
import os
class ClusterEnvSubSystem(subsystem.SubSystem):

    def __init__(self, mol, env_method, filename=None, smearsigma=0, damp=0, 
                 shift=0, subcycles=1, freeze=False, initguess=None,
                 grid_level=4, verbose=4, analysis=False, debug=False):

        self.mol = mol
        self.mol.basis = self.mol._basis # Always save basis as internal pyscf format
        self.env_method = env_method

        #Check if none
        if filename == None:
            filename = os.getcwd() + '/temp.inp'
        self.filename = filename

        self.smearsigma = smearsigma
        self.damp = damp
        self.shift = shift

        # diagonalization subcycles
        self.subcycles = subcycles 

        self.freeze = freeze #Whether to freeze during freeze and thaw or not.

        self.initguess = initguess

        self.grid_level = grid_level
        self.verbose = verbose
        self.analysis = analysis
        self.debug = debug
        self.init_env_scf()
        self.dmat = [None, None] # alpha and beta dmat

        self.env_mo_coeff = None
        self.env_mo_occ = None
        self.env_mo_energy = None

    def init_env_scf(self):

        if self.env_method[0] == 'u':
            if self.env_method[1:] == 'hf':
                scf_obj = scf.UHF(self.mol) 
            else:
                scf_obj = scf.UKS(self.mol)
                scf_obj.xc = self.env_method[1:]
                scf_obj.small_rho_cutoff = 1e-20 #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
        elif self.env_method[:2] == 'ro':
            if self.env_method[2:] == 'hf':
                scf_obj = scf.ROHF(self.mol) 
            else:
                scf_obj = scf.ROKS(self.mol)
                scf_obj.xc = self.env_method[2:]
                scf_obj.small_rho_cutoff = 1e-20 #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
        else:
            if self.env_method == 'hf' or self.env_method[1:] == 'hf':
               scf_obj = scf.RHF(self.mol) 
            else:
                scf_obj = scf.RKS(self.mol)
                scf_obj.xc = self.env_method
                if self.env_method[0] == 'r':
                    scf_obj.xc = self.env_method[1:]
                scf_obj.small_rho_cutoff = 1e-20 #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)

        self.env_scf = scf_obj

    def init_density(self):
        pass
    def get_env_energy(self):
        pass
    def update_proj_op(self, new_POp):
        pass
    def update_embedding_pot(self, new_emb_pot):
        pass
    def update_fock(self):
        pass
    def update_density(self, new_den):
        pass
    def save_chkfile(self):
        pass
    def save_orbitals(self):
        pass

class ClusterActiveSubSystem(ClusterEnvSubSystem):

    def __init__(self, mol, env_method, active_method, localize_orbitals=False, active_orbs=None,
                 active_conv=1e-8, active_grad=1e-8, active_cycles=100, 
                 active_damp=0, active_shift=0, **kwargs):

        self.active_method = active_method
        self.localize_orbitals = localize_orbitals
        self.active_orbs = active_orbs
        self.active_conv = active_conv
        self.active_grad = active_grad
        self.active_cycles = active_cycles
        self.active_damp = active_damp
        self.active_shift = active_shift

        self.active_mo_coeff = None
        self.active_mo_occ = None
        self.active_mo_energy = None
 
        super().__init__(mol, env_method, **kwargs)

class ClusterExcitedSubSystem(ClusterActiveSubSystem):

    def __init__(self):
        super().__init__()

