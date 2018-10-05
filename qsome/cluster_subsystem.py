# A method to define all cluster supsystem objects
# Daniel Graham

from qsome import subsystem
from pyscf import gto, scf, dft
import os

import numpy as np
import scipy as sp

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
        self.env_hcore = self.env_scf.get_hcore()
        self.env_mo_coeff = [np.zeros_like(self.env_hcore), np.zeros_like(self.env_hcore)]
        self.env_mo_occ = [np.zeros_like(self.env_hcore[0]), np.zeros_like(self.env_hcore[0])]
        self.env_mo_energy = self.env_mo_occ.copy()
        self.dmat = self.env_mo_coeff.copy() # alpha and beta dmat

    def init_density(self, dmat):
        self.dmat = dmat
        self.update_fock()
        self.emb_pot = [np.zeros_like(self.env_hcore), np.zeros_like(self.env_hcore)]

    def get_env_elec_energy(self):
        hcore_e = np.einsum('ij, ji', self.env_hcore, (self.dmat[0] + self.dmat[1])).real
        if self.env_method[0] == 'u' or self.env_method[:2] == 'ro':
            e_coul = (np.einsum('ij,ji', self.env_V[0], self.dmat[0]) + 
                      np.einsum('ij,ji', self.env_V[1], self.dmat[1])).real * 0.5
            e_emb = (np.einsum('ij,ji', self.emb_pot[0], self.dmat[0]) + 
                     np.einsum('ij,ji', self.emb_pot[1], self.dmat[1])).real * 0.5
        else:
            e_coul = (np.einsum('ij,ji', self.env_V[0], (self.dmat[0] + self.dmat[1])).real * 0.5)
            e_emb = (np.einsum('ij,ji', (self.emb_pot[0] + self.emb_pot[1])/2., (self.dmat[0] + self.dmat[1])).real * 0.5)
        #There is some kind of double counting happening where e_coul is too large. Can't just add electronic of each to get the total energy.
        return hcore_e + e_coul + e_emb

    def get_env_energy(self):
        return self.get_env_elec_energy() + self.env_scf.energy_nuc()

    def update_proj_pot(self, new_POp):
        self.proj_pot = new_POp

    def update_emb_pot(self, new_emb_pot):
        self.emb_pot = new_emb_pot 

    def update_fock(self):
        if self.env_method[0] == 'u' or self.env_method[:2] == 'ro':
            self.env_V = self.env_scf.get_veff(dm=self.dmat)
        else:
            V = self.env_scf.get_veff(dm=(self.dmat[0] + self.dmat[1]))
            self.env_V = [V, V]

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
        nA_a = self.fock[0].shape[0]
        nA_b = self.fock[1].shape[0]
        N = [np.zeros((nA_a)), np.zeros((nA_b))]
        N[0][:self.mol.nelec[0]] = 1.
        N[1][:self.mol.nelec[0]] = 1.

        #Need to include subcycle possibility.

        if self.env_method[0] == 'u' or self.env_method[:2] == 'ro':
            emb_fock = [None, None]
            emb_fock[0] = self.fock[0] + self.emb_pot[0] + self.proj_pot[0]
            #This is the costly part. I think.
            E_a, C_a = sp.linalg.eigh(emb_fock[0], self.env_scf.get_ovlp())
            emb_fock[1] = self.fock[1] + self.emb_pot[1] + self.proj_pot[1]
            #This is the costly part. I think.
            E_b, C_b = sp.linalg.eigh(emb_fock[1], self.env_scf.get_ovlp())
            self.env_mo_energy = [E_a, E_b]
            self.env_mo_coeff = [C_a, C_b]
        else:
            #This is the costly part. I think.
            emb_fock = self.fock[0] + self.emb_pot[0] + self.proj_pot[0]
            E, C = sp.linalg.eigh(emb_fock, self.env_scf.get_ovlp())
            self.env_mo_energy = [E, E]
            self.env_mo_coeff = [C, C]
        
        # get fermi energy
        nocc_orbs = [self.mol.nelec[0], self.mol.nelec[1]]
        e_sorted = [np.sort(self.env_mo_energy[0]), np.sort(self.env_mo_energy[1])]
        fermi = [None, None]
        if (len(e_sorted[0]) > nocc_orbs[0]):
            fermi[0] = (e_sorted[0][nocc_orbs[0]] + e_sorted[0][nocc_orbs[0] -1]) / 2.
        else:
            fermi[0] = 0.    #Minimal basis
        if (len(e_sorted[1]) > nocc_orbs[1]):
            fermi[1] = (e_sorted[1][nocc_orbs[1]] + e_sorted[1][nocc_orbs[1] -1]) / 2.
        else:
            fermi[1] = 0.    #Minimal basis

        #Smear sigma may not be right for single elctron
        mo_occ = [np.zeros_like(self.env_mo_energy[0]), np.zeros_like(self.env_mo_energy[1])]
        if self.smearsigma > 0.:
            mo_occ[0] = ( self.env_mo_energy[0] - fermi[0] ) / self.smearsigma
            ie = np.where( mo_occ[0] < 1000 )
            i0 = np.where( mo_occ[0] >= 1000 )
            mo_occ[0][ie] = 1. / ( np.exp( mo_occ[0][ie] ) + 1. )
            mo_occ[0][i0] = 0.

            mo_occ[1] = ( self.env_mo_energy[1] - fermi[1] ) / self.smearsigma
            ie = np.where( mo_occ[1] < 1000 )
            i0 = np.where( mo_occ[1] >= 1000 )
            mo_occ[1][ie] = 1. / ( np.exp( mo_occ[1][ie] ) + 1. )
            mo_occ[1][i0] = 0.

        else:
            if (len(e_sorted[0]) > nocc_orbs[0]):
                mo_occ[0][self.env_mo_energy[0]<fermi[0]] = 1.
            else:
                mo_occ[0][:] = 1.

            if (len(e_sorted[1]) > nocc_orbs[1]):
                mo_occ[1][self.env_mo_energy[1]<fermi[1]] = 1.
            else:
                mo_occ[1][:] = 1.

        self.env_mo_occ = mo_occ
        self.fermi = fermi
        self.dmat[0] = np.dot((self.env_mo_coeff[0] * self.env_mo_occ[0]), self.env_mo_coeff[0].transpose().conjugate())
        self.dmat[1] = np.dot((self.env_mo_coeff[1] * self.env_mo_occ[1]), self.env_mo_coeff[1].transpose().conjugate())



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

