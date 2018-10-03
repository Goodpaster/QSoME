# A method to define all cluster supsystem objects
# Daniel Graham

from qsome import subsystem
from pyscf import gto
import os
class ClusterEnvSubSystem(subsystem.SubSystem):

    def __init__(self, mol, env_method, filename=None, smearsigma=0, damp=0, 
                 shift=0, subcycles=1, freeze=False, initguess='minao',
                 grid=4, verbose=4, analysis=False, debug=False):

        self.mol = mol
        self.load_basis()
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

        self.grid = grid
        self.verbose = verbose
        self.analysis = analysis
        self.debug = debug

    def load_basis(self):

        # a function to turn a basis string ('3-21g') into a basis dictionary for use in concat mols.
        if isinstance(self.mol.basis, str):
            new_basis = {}
            nghost = 0
            for i in range(self.mol.natm):
                if 'ghost.' in self.mol.atom[i][0] or 'gh.' in self.mol.atom[i][0]: 
                    nghost += 1
                    atom_name = self.mol.atom[i][0].split('.')[1]
                    ghost_name = f'ghost:{nghost}'
                    self.mol.atom[i][0] = ghost_name
                    new_basis.update({ghost_name: gto.basis.load(self.mol.basis, atom_name)})
                else:
                    atom_name = self.mol.atom[i][0]
                    new_basis.update({atom_name: gto.basis.load(self.mol.basis, atom_name)})
            self.mol.basis = new_basis
            self.mol.build(dump_input=False)
        else:
            return True
        

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
 
        super().__init__(mol, env_method, **kwargs)

class ClusterExcitedSubSystem(ClusterActiveSubSystem):

    def __init__(self):
        super().__init__()

