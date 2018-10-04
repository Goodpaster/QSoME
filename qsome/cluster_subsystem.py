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
        self.hcore = self.env_scf.get_hcore()
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
    def get_env_elec_energy(self):
        hcore_e = np.einsum('ij, ji', self.emb_hcore, self.dmat).real
        e_coul = np.einsum('ij,ji', self.emb_vA, self.dmat).real * 0.5 
        e_coul = np.einsum('ij,ji', self.emb_vA, self.dmat).real * 0.5 
        e_emb = np.einsum('ij,ji', self.emb_pot, self.dmat).real * 0.5 #Testing. I think this needs *0.5 though.
        return hcore_e + e_coul

    def get_env_energy(self):
        return self.get_env_elec_energy() + self.env_scf.energy_nuc()

    def update_proj_op(self, new_POp):
        self.proj_pot = new_POp

    def update_emb_pot(self, new_emb_pot):
        self.emb_pot = new_emb_pot

    def update_fock(self):
        self.env_hcore = self.env_scf.get_hcore() #could probably just use the saved value
        if self.env_method[0] == 'u' or self.env_method[:2] == 'ro':
            self.env_V = self.env_scf.get_veff(dm=self.dmat)
        else:
            Va = self.env_scf.get_veff(dm=self.dmat[0])
            Vb = self.env_scf.get_veff(dm=self.dmat[1])
            self.env_V = [Va, Vb]

        self.fock = [self.env_hcore + self.env_V[0], self.env_hcore + self.env_V[1]]

    def update_density(self, new_den):
        self.dmat = new_den

    def save_chkfile(self):
        pass
    def save_orbitals(self):
        pass

    def diagonalize(self):
        #Smat may not be necessary. May be able to get from the mol object.
        # finish this method
        nA = self.fock.shape[0]
        N = np.zeros((nA))
        N[:self.mol.nelectron//2] = 2.

        if self.env_method[0] == 'u' or self.env_method[:2] == 'ro':
            emb_fock = [None, None]
            emb_fock[0] = self.fock[0] + self.emb_pot[0] + self.proj_pot[0]
            #This is the costly part. I think.
            E, C = sp.linalg.eigh(emb_fock[0], self.env_scf.get_ovlp())
            self.mo_energy[0] = E
            self.mo_coeff[0] = C
            emb_fock[1] = self.fock[1] + self.emb_pot[1] + self.proj_pot[1]
            #This is the costly part. I think.
            E, C = sp.linalg.eigh(emb_fock[1], self.env_scf.get_ovlp())
            self.mo_energy[1] = E
            self.mo_coeff[1] = C
        else:
            #This is the costly part. I think.
            emb_fock = self.fock[0] + self.emb_pot[0] + self.proj_pot[0]
            E, C = sp.linalg.eigh(emb_fock, self.env_scf.get_ovlp())
            self.mo_energy[0] = E
            self.mo_coeff[0] = C
            self.mo_energy[1] = E
            self.mo_coeff[1] = C
        
        # get fermi energy
        nocc_orbs = mol.nelectron // 2
        e_sorted = np.sort(E)
        if (len(e_sorted) > nocc_orbs):
            fermi = (e_sorted[nocc_orbs] + e_sorted[nocc_orbs -1]) / 2.
        else:
            fermi = 0.    #Minimal basis

        if smear_sigma > 0.:
            mo_occ = ( E - fermi ) / smear_sigma
            ie = np.where( mo_occ < 1000 )
            i0 = np.where( mo_occ >= 1000 )
            mo_occ[ie] = 2. / ( np.exp( mo_occ[ie] ) + 1. )
            mo_occ[i0] = 0.

        else:
            mo_occ = np.zeros_like(E)
            if (len(e_sorted) > nocc_orbs):
                mo_occ[E<fermi] = 2.
            else:
                mo_occ[:] = 2.

        Dnew = np.dot((C * mo_occ), C.transpose().conjugate())


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

