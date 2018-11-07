# A method to define all cluster supsystem objects
# Daniel Graham

import re
from qsome import subsystem, custom_pyscf_methods
from pyscf import gto, scf, dft, cc
from pyscf.cc import ccsd_t, ccsd_t_rdm_slow, ccsd_t_lambda_slow
from pyscf.scf import diis as scf_diis
from pyscf.lib import diis as lib_diis
import os

from functools import reduce

import numpy as np
import scipy as sp

#Custom PYSCF method for the active subsystem.
from copy import deepcopy as copy


class ClusterEnvSubSystem(subsystem.SubSystem):

    def __init__(self, mol, env_method, filename=None, smearsigma=0, damp=0, 
                 shift=0, subcycles=1, diis=0, freeze=False, initguess=None,
                 grid_level=4, verbose=3, analysis=False, debug=False, rhocutoff=1e-7):

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

        self.rho_cutoff = rhocutoff

        self.grid_level = grid_level
        self.verbose = verbose
        self.analysis = analysis
        self.debug = debug
        self.init_env_scf()
        self.diis_num = diis
        if diis == 1:
            #Use subtractive diis. Most simple
            self.diis = lib_diis.DIIS()
        elif diis == 2:
            self.diis = scf_diis.CDIIS(self.env_scf)
        elif diis == 3:
            self.diis = scf_diis.EDIIS()
        elif diis == 4:
            self.diis = scf.diis.ADIIS()
        else:
            self.diis = None 


    def init_env_scf(self):

        if self.env_method[0] == 'u':
            if self.env_method[1:] == 'hf':
                scf_obj = scf.UHF(self.mol) 
            else:
                scf_obj = scf.UKS(self.mol)
                scf_obj.xc = self.env_method[1:]
                scf_obj.small_rho_cutoff = self.rho_cutoff #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
                self.env_xc = 0.0
        elif self.env_method[:2] == 'ro':
            if self.env_method[2:] == 'hf':
                scf_obj = scf.ROHF(self.mol) 
            else:
                scf_obj = scf.ROKS(self.mol)
                scf_obj.xc = self.env_method[2:]
                scf_obj.small_rho_cutoff = self.rho_cutoff #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
                self.env_xc = 0.0
        else:
            if self.env_method == 'hf' or self.env_method[1:] == 'hf':
               scf_obj = scf.RHF(self.mol) 
            else:
                scf_obj = scf.RKS(self.mol)
                scf_obj.xc = self.env_method
                if self.env_method[0] == 'r':
                    scf_obj.xc = self.env_method[1:]
                scf_obj.small_rho_cutoff = self.rho_cutoff #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
                self.env_xc = 0.0

        self.env_scf = scf_obj

        # basic scf settings
        self.env_scf.verbose = self.verbose
        self.env_scf.damp = self.damp
        self.env_scf.level_shift = self.shift

        self.env_hcore = self.env_scf.get_hcore()
        self.env_mo_coeff = [np.zeros_like(self.env_hcore), np.zeros_like(self.env_hcore)]
        self.env_mo_occ = [np.zeros_like(self.env_hcore[0]), np.zeros_like(self.env_hcore[0])]
        self.env_mo_energy = self.env_mo_occ.copy()
        self.dmat = self.env_mo_coeff.copy() # alpha and beta dmat
        self.env_energy = 0.0

    def init_density(self, dmat):
        self.dmat = dmat
        self.update_fock()
        self.emb_pot = [np.zeros_like(self.env_hcore), np.zeros_like(self.env_hcore)]

    def get_env_elec_energy(self):
        e_emb = 0.0
        subsys_e = np.einsum('ij,ji', self.env_hcore, (self.dmat[0] + self.dmat[1])).real
        if self.env_method[0] == 'u' or self.env_method[:2] == 'ro':
            e_proj = (np.einsum('ij,ji', self.proj_pot[0], self.dmat[0]) + 
                      np.einsum('ij,ji', self.proj_pot[1], self.dmat[1])).real
            subsys_e = self.env_scf.energy_elec(dm=self.dmat)
            e_emb = (np.einsum('ij,ji', self.emb_pot[0], self.dmat[0]) + 
                     np.einsum('ij,ji', self.emb_pot[1], self.dmat[1])).real * 0.5

        else:
            e_proj = (np.einsum('ij,ji', (self.proj_pot[0] + self.proj_pot[1])/2., (self.dmat[0] + self.dmat[1])).real)
            e_emb = (np.einsum('ij,ji', (self.emb_pot[0] + self.emb_pot[1])/2., (self.dmat[0] + self.dmat[1])).real) #* .5
            #e_emb = 0.0
            subsys_e += self.env_scf.energy_elec(dm=(self.dmat[0] + self.dmat[1]))[1]

        return subsys_e + e_proj + e_emb

    def get_env_energy(self):
        self.env_energy = self.get_env_elec_energy() + self.mol.energy_nuc()
        return self.env_energy

    def update_proj_pot(self, new_POp):
        self.proj_pot = new_POp
    def get_env_proj_e(self):

        if self.env_method[0] == 'u' or self.env_method[:2] == 'ro':
            e_proj = (np.einsum('ij,ji', self.proj_pot[0], self.dmat[0]) + 
                      np.einsum('ij,ji', self.proj_pot[1], self.dmat[1])).real
        else:
            e_proj = (np.einsum('ij,ji', (self.proj_pot[0] + self.proj_pot[1])/2., (self.dmat[0] + self.dmat[1])).real)

        return e_proj 

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

    def save_orbitals(self):
        pass

    def diagonalize(self, run_diis=True):

        # finish this method
        nA_a = self.fock[0].shape[0]
        nA_b = self.fock[1].shape[0]
        N = [np.zeros((nA_a)), np.zeros((nA_b))]
        N[0][:self.mol.nelec[0]] = 1.
        N[1][:self.mol.nelec[1]] = 1.

        #Need to include subcycle possibility.
        for i in range(self.subcycles):
            if i > 0:
                self.update_fock()
                #print ('update')

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
                emb_fock += self.fock[1] + self.emb_pot[1] + self.proj_pot[1]

                #Errors abound here. Doesn't converge to correct value.
                if not self.diis is None and run_diis:
                    if self.diis_num == 1:
                        emb_fock = self.diis.update(emb_fock)
                    else:
                        s1e = self.env_scf.get_ovlp()
                        dm = self.dmat[0] + self.dmat[1]
                        f = emb_fock
                        mf = self.env_scf
                        h1e = self.env_hcore + \
                              (self.emb_pot[0] + self.emb_pot[1])/2. + \
                              (self.proj_pot[0] + self.proj_pot[1])/2.
                        vhf = self.env_V[0]
                        emb_fock = self.diis.update(s1e, dm, f, mf, h1e, vhf)

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
                 active_conv=1e-9, active_grad=None, active_cycles=100, 
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

    def active_proj_energy(self):
        #trace of 1p den mat with proj operator
        return np.trace(self.active_dmat, self.proj_pot)

           
    def active_in_env_energy(self):
        self.active_energy = 0.0
        if self.active_method[0] == 'u': 
            if self.active_method[1:] == 'hf':
                self.active_scf = scf.UHF(self.mol)
                self.active_scf.conv_tol = self.active_conv
                self.active_scf.conv_tol_grad = self.active_grad
                self.active_scf.max_cycle = self.active_cycles
                self.active_scf.level_shift = self.active_shift
                self.active_scf.damp = self.active_damp
                self.active_scf.get_hcore = lambda *args, **kwargs: self.env_hcore
                self.active_scf.get_fock = lambda *args, **kwargs: custom_pyscf_methods.uhf_get_fock(self.active_scf, self.emb_pot, *args, **kwargs)
                self.active_scf.energy_elec = lambda *args, **kwargs: custom_pyscf_methods.uhf_energy_elec(self.active_scf, self.emb_pot, *args, **kwargs)
                self.active_scf.kernel()

            elif self.active_method[1:] == 'ccsd' or self.active_method[1:] == 'ccsd(t)':
                pass
                if self.active_method[1:] == 'ccsd(t)':
                    pass
            elif re.match(re.compile('cas(pt2)?\[.*\].*'), inp.active_method[1:]):
                pass
            elif self.active_method[1:] == 'fci':
                pass
            else: 
                pass
        elif self.active_method[:2] == 'ro': 
            pass
        else: 
            if self.active_method[0] == 'r':
                self.active_method = self.active_method[1:]
            if self.active_method == 'hf':
                self.active_scf = scf.RHF(self.mol)
                self.active_scf.conv_tol = self.active_conv
                self.active_scf.conv_tol_grad = self.active_grad
                self.active_scf.max_cycle = self.active_cycles
                self.active_scf.level_shift = self.active_shift
                self.active_scf.damp = self.active_damp
                self.active_scf.get_hcore = lambda *args, **kwargs: self.env_hcore
                self.active_scf.get_fock = lambda *args, **kwargs: custom_pyscf_methods.rhf_get_fock(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2.,(self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)
                self.active_scf.energy_elec = lambda *args, **kwargs: custom_pyscf_methods.rhf_energy_elec(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2., (self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)

                self.active_energy = self.active_scf.kernel(dm0=(self.dmat[0] + self.dmat[1])) #Includes the nuclear
                # this slows down execution.
                #self.active_dmat = self.active_scf.make_rdm1()

            elif self.active_method == 'ccsd' or self.active_method == 'ccsd(t)':
                self.active_scf = scf.RHF(self.mol)
                self.active_scf.conv_tol = self.active_conv
                self.active_scf.conv_tol_grad = self.active_grad
                self.active_scf.max_cycle = self.active_cycles
                self.active_scf.level_shift = self.active_shift
                self.active_scf.damp = self.active_damp
                self.active_scf.get_hcore = lambda *args, **kwargs: self.env_hcore
                self.active_scf.get_fock = lambda *args, **kwargs: custom_pyscf_methods.rhf_get_fock(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2.,(self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)
                self.active_scf.energy_elec = lambda *args, **kwargs: custom_pyscf_methods.rhf_energy_elec(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2., (self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)
                self.active_energy = self.active_scf.kernel(dm0=(self.dmat[0] + self.dmat[1]))
                 
                self.active_cc = cc.CCSD(self.active_scf)
                self.active_cc.max_cycle = self.active_cycles
                new_eris = self.active_cc.ao2mo()
                new_eris.fock = reduce(np.dot, (self.active_scf.mo_coeff.conj().T, self.active_scf.get_fock(), self.active_scf.mo_coeff))
                ecc, t1, t2 = self.active_cc.kernel(eris=new_eris)
                self.active_energy += ecc
                if self.active_method == 'ccsd(t)':
                    ecc_t = ccsd_t.kernel(self.active_cc, new_eris)
                    self.active_energy += ecc_t
                    #l1, l2 = ccsd_t_lambda_slow.kernel(self.active_cc, new_eris, t1, t2,)[1:]
                    # this slows down execution.
                    #self.active_dmat = ccsd_t_rdm_slow.make_rdm1(self.active_cc, t1, t2, l1, l2, eris=new_eris)
                else:
                    pass
                    # this slows down execution.
                    #self.active_dmat = self.active_cc.make_rdm1()

                # Convert to AO form
                #temp_dmat = copy(self.active_dmat)
                #ao_dmat = reduce (np.dot, (self.active_cc.mo_coeff, np.dot(temp_dmat, self.active_cc.mo_coeff.T)))
                #print ("DMATS")
                #print (self.active_dmat)
                #print (ao_dmat)
                #print (self.dmat[0] + self.dmat[1])
                #self.active_dmat = ao_dmat
            else: #DFT
                self.active_scf = scf.RKS(self.mol)
                self.active_scf.xc = self.active_method
                self.active_scf.grids = self.env_scf.grids
                self.active_scf.small_rho_cutoff = self.rho_cutoff
                self.active_scf.conv_tol = self.active_conv
                self.active_scf.conv_tol_grad = self.active_grad
                self.active_scf.max_cycle = self.active_cycles
                self.active_scf.level_shift = self.active_shift
                self.active_scf.damp = self.active_damp
                self.active_scf.get_hcore = lambda *args, **kwargs: self.env_hcore
                self.active_scf.get_fock = lambda *args, **kwargs: custom_pyscf_methods.rks_get_fock(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2.,(self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)
                self.active_scf.energy_elec = lambda *args, **kwargs: custom_pyscf_methods.rks_energy_elec(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2., (self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)
                self.active_energy = self.active_scf.kernel(dm0=(self.dmat[0] + self.dmat[1]))
                #Slows down execution
                #self.active_dmat = self.active_scf.make_rdm1()

            #temp_dmat = copy(self.active_dmat)
            #self.active_dmat = [temp_dmat/2., temp_dmat/2.]
           
        return self.active_energy
 
class ClusterExcitedSubSystem(ClusterActiveSubSystem):

    def __init__(self):
        super().__init__()

