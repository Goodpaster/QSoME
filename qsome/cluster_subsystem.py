# A method to define all cluster supsystem objects
# Daniel Graham

import re
from qsome import subsystem
from pyscf import gto, scf, dft, cc
from pyscf.cc import ccsd_t, ccsd_t_rdm_slow, ccsd_t_lambda_slow
import os

from functools import reduce

import numpy as np
import scipy as sp

#Custom PYSCF method for the active subsystem.
from copy import deepcopy as copy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf,rohf,uhf
from pyscf.scf import jk
from pyscf.dft import rks, roks, uks
import numpy as np

#RHF Methods
def rhf_get_fock(mf, emb_pot=None, proj_pot=None, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    '''F = h^{core} + V^{HF}
    Special treatment (damping, DIIS, or level shift) will be applied to the
    Fock matrix if diis and cycle is specified (The two parameters are passed
    to get_fock function during the SCF iteration)
    Kwargs:
        h1e : 2D ndarray
            Core hamiltonian
        s1e : 2D ndarray
            Overlap matrix, for DIIS
        vhf : 2D ndarray
            HF potential matrix
        dm : 2D ndarray
            Density matrix, for DIIS
        cycle : int
            Then present SCF iteration step, for DIIS
        diis : an object of :attr:`SCF.DIIS` class
            DIIS object to hold intermediate Fock and error vectors
        diis_start_cycle : int
            The step to start DIIS.  Default is 0.
        level_shift_factor : float or int
            Level shift (in AU) for virtual space.  Default is 0.
    '''
    if emb_pot is None: emb_pot = 0.0
    if proj_pot is None: proj_pot = 0.0
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(dm=dm)
    f = h1e + vhf + emb_pot #Added embedding potential to fock

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = self.make_rdm1()

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
        f = damping(s1e, dm*.5, f, damp_factor)
    if diis is not None and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    if abs(level_shift_factor) > 1e-4:
        f = level_shift(s1e, dm*.5, f, level_shift_factor)
    return f

def rhf_energy_elec(mf, emb_pot=None, proj_pot=None, dm=None, h1e=None, vhf=None):

    if emb_pot is None: emb_pot = 0.0
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    h1e = copy(h1e + emb_pot + proj_pot) #Add embedding potential to the core ham
    e1 = np.einsum('ij,ji', h1e, dm).real
    e_coul = np.einsum('ij,ji', vhf, dm).real * .5
    logger.debug(mf, 'E_coul = %.15g', e_coul)
    return e1+e_coul, e_coul

#ROHF Methods
def rohf_get_fock(mf, emb_pot=None, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
    '''Build fock matrix based on Roothaan's effective fock.
    See also :func:`get_roothaan_fock`
    '''
    if emb_pot is None: emb_pot = [0.0, 0.0]
    if h1e is None: h1e = mf.get_hcore()
    if s1e is None: s1e = mf.get_ovlp()
    if vhf is None: vhf = mf.get_veff(dm=dm)
    if dm is None: dm = mf.make_rdm1()
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
# To Get orbital energy in get_occ, we saved alpha and beta fock, because
# Roothaan effective Fock cannot provide correct orbital energy with `eig`
# TODO, check other treatment  J. Chem. Phys. 133, 141102
    focka = h1e + vhf[0] + emb_pot[0] #Add embedding potential
    fockb = h1e + vhf[1] + emb_pot[1] #Add embedding potential
    f = rohf.get_roothaan_fock((focka,fockb), dm, s1e)
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    dm_tot = dm[0] + dm[1]
    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4:
        raise NotImplementedError('ROHF Fock-damping')
    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm_tot, f, mf, h1e, vhf)
    if abs(level_shift_factor) > 1e-4:
        f = hf.level_shift(s1e, dm_tot*.5, f, level_shift_factor)
    f = lib.tag_array(f, focka=focka, fockb=fockb)
    return f

def rohf_energy_elec(mf, emb_pot=None, dm=None, h1e=None, vhf=None):

    if emb_pot is None: emb_pot = [0.0, 0.0]
    if dm is None: dm = mf.make_rdm1()
    elif isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
    ee, ecoul = uhf_energy_elec(mf, emb_pot, dm, h1e, vhf)
    logger.debug(mf, 'Ecoul = %.15g', ecoul)
    return ee, ecoul


#UHF Methods
def uhf_get_fock(mf, emb_pot=None, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None):


    if emb_pot is None: emb_pot = [0.0, 0.0]
    if h1e is None: h1e = mf.get_hcore()
    #if vhf is None: vhf = mf.get_veff(dm=dm)
    #For some reason the vhf being passed is wrong I believe.
    vhf = mf.get_veff(dm=dm)
    f = h1e + vhf 
    if f.ndim == 2:
        f = (f, f)
    f[0] = f[0] + emb_pot[0] #Add embedding potential
    f[1] = f[1] + emb_pot[1] #Add embedding potential

    #print ("vhf")
    #print (vhf[0])

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()
    if dm is None: dm = self.make_rdm1()

    if isinstance(level_shift_factor, (tuple, list, np.ndarray)):
        shifta, shiftb = level_shift_factor
    else:
        shifta = shiftb = level_shift_factor
    if isinstance(damp_factor, (tuple, list, np.ndarray)):
        dampa, dampb = damp_factor
    else:
        dampa = dampb = damp_factor

    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = [dm*.5] * 2
    if 0 <= cycle < diis_start_cycle-1 and abs(dampa)+abs(dampb) > 1e-4:
        f = (hf.damping(s1e, dm[0], f[0], dampa),
             hf.damping(s1e, dm[1], f[1], dampb))
    if diis and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf)
    if abs(shifta)+abs(shiftb) > 1e-4:
        f = (hf.level_shift(s1e, dm[0], f[0], shifta),
             hf.level_shift(s1e, dm[1], f[1], shiftb))
    return np.array(f)

def uhf_energy_elec(mf, emb_pot=None, dm=None, h1e=None, vhf=None):
    '''Electronic energy of Unrestricted Hartree-Fock
    Returns:
        Hartree-Fock electronic energy and the 2-electron part contribution
    '''
    if emb_pot is None: emb_pot = [0.0, 0.0]
    if dm is None: dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    e1 = np.einsum('ij,ij', h1e.conj(), dm[0]+dm[1])

    vhf[0] = vhf[0] + 2.*emb_pot[0] #May need to multiply emb_pot by 2
    vhf[1] = vhf[1] + 2.*emb_pot[1]

    e_coul =(np.einsum('ij,ji', vhf[0], dm[0]) +
             np.einsum('ij,ji', vhf[1], dm[1])).real * .5
    return e1+e_coul, e_coul

#RKS Methods
def rks_energy_elec(ks, emb_pot=None, dm=None, h1e=None, vhf=None):
    r'''Electronic part of RKS energy.
    Args:
        ks : an instance of DFT class
        dm : 2D ndarray
            one-partical density matrix
        h1e : 2D ndarray
            Core hamiltonian
    Returns:
        RKS electronic energy and the 2-electron part contribution
    '''
    if emb_pot is None: emb_pot = 0.0
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)

    h1e = h1e + emb_pot
    e1 = np.einsum('ij,ji', h1e, dm).real
    tot_e = e1 + vhf.ecoul + vhf.exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', vhf.ecoul, vhf.exc)
    return tot_e, vhf.ecoul+vhf.exc

rks_get_fock = rhf_get_fock


#UKS Methods
def uks_energy_elec(ks, emb_pot=None, dm=None, h1e=None, vhf=None):
    if emb_pot is None: emb_pot = [0.0,0.0]
    if dm is None: dm = ks.make_rdm1()
    if h1e is None: h1e = ks.get_hcore()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = ks.get_veff(ks.mol, dm)
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
    emb_h1e = [None,None] 
    emb_h1e[0] = h1e + emb_pot[0]
    emb_h1e[1] = h1e + emb_pot[1]
    e1 = np.einsum('ij,ji', emb_h1e[0], dm[0]) + np.einsum('ij,ji', emb_h1e[1], dm[1])
    tot_e = e1.real + vhf.ecoul + vhf.exc
    logger.debug(ks, 'Ecoul = %s  Exc = %s', vhf.ecoul, vhf.exc)
    return tot_e, vhf.ecoul+vhf.exc

uks_get_fock = uhf_get_fock

#ROKS Methods
roks_energy_elec = uks_energy_elec
roks_get_fock = rohf_get_fock

class ClusterEnvSubSystem(subsystem.SubSystem):

    def __init__(self, mol, env_method, filename=None, smearsigma=0, damp=0, 
                 shift=0, subcycles=1, freeze=False, initguess=None,
                 grid_level=4, verbose=3, analysis=False, debug=False, rhocutoff=1e-20):

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



    def init_env_scf(self):

        if self.env_method[0] == 'u':
            if self.env_method[1:] == 'hf':
                scf_obj = scf.UHF(self.mol) 
            else:
                scf_obj = scf.UKS(self.mol)
                scf_obj.xc = self.env_method[1:]
                scf_obj.small_rho_cutoff = self.rho_cutoff #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
        elif self.env_method[:2] == 'ro':
            if self.env_method[2:] == 'hf':
                scf_obj = scf.ROHF(self.mol) 
            else:
                scf_obj = scf.ROKS(self.mol)
                scf_obj.xc = self.env_method[2:]
                scf_obj.small_rho_cutoff = self.rho_cutoff #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)
        else:
            if self.env_method == 'hf' or self.env_method[1:] == 'hf':
               scf_obj = scf.RHF(self.mol) 
            else:
                scf_obj = scf.RKS(self.mol)
                scf_obj.xc = self.env_method
                if self.env_method[0] == 'r':
                    scf_obj.xc = self.env_method[1:]
                scf_obj.small_rho_cutoff = self.rho_cutoff #this prevents pruning. Also slows down code. Can probably remove and use default in pyscf (1e-7)

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
        #This is not correct for dft methods.
        hcore_e = np.einsum('ij, ji', self.env_hcore, (self.dmat[0] + self.dmat[1])).real
        if self.env_method[0] == 'u' or self.env_method[:2] == 'ro':
            e_coul = (np.einsum('ij,ji', self.env_V[0], self.dmat[0]) + 
                      np.einsum('ij,ji', self.env_V[1], self.dmat[1])).real * 0.5
            e_emb = (np.einsum('ij,ji', self.emb_pot[0], self.dmat[0]) + 
                     np.einsum('ij,ji', self.emb_pot[1], self.dmat[1])).real * 0.5
        else:
            e_coul = np.einsum('ij,ji', self.env_V[0] + self.emb_pot[0], (self.dmat[0] + self.dmat[1])).real * 0.5
            #e_emb = (np.einsum('ij,ji', self.emb_pot[0], (self.dmat[0] + self.dmat[1])).real * 0.5)
            #print (self.env_V[1] + self.emb_pot[1])
            # Error is in the evaluation of the energy. All potentials and density matrices are the same. How the energy is evaluated is wrong.
        return hcore_e + e_coul

    def get_env_energy(self):
        return self.env_energy
        #return self.get_env_elec_energy() + self.env_scf.energy_nuc()

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
    def get_env_proj_energy(self):
        pass

    def diagonalize(self):

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
                print ('update')
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

    def get_active_proj_energy(self):
         #trace of 1p den mat with proj operator
         return np.trace(self.active_dmat, self.proj_pot)

    def get_active_in_env_energy(self):
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
                self.active_scf.get_fock = lambda *args, **kwargs: uhf_get_fock(self.active_scf, self.emb_pot, *args, **kwargs)
                self.active_scf.energy_elec = lambda *args, **kwargs: uhf_energy_elec(self.active_scf, self.emb_pot, *args, **kwargs)
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
                self.active_scf.get_fock = lambda *args, **kwargs: rhf_get_fock(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2.,(self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)
                self.active_scf.energy_elec = lambda *args, **kwargs: rhf_energy_elec(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2., (self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)

                self.active_energy = self.active_scf.kernel()
                # this slows down execution.
                # self.active_dmat = self.active_scf.make_rdm1()

            elif self.active_method == 'ccsd' or self.active_method == 'ccsd(t)':
                self.active_scf = scf.RHF(self.mol)
                self.active_scf.conv_tol = self.active_conv
                self.active_scf.conv_tol_grad = self.active_grad
                self.active_scf.max_cycle = self.active_cycles
                self.active_scf.level_shift = self.active_shift
                self.active_scf.damp = self.active_damp
                self.active_scf.get_hcore = lambda *args, **kwargs: self.env_hcore
                self.active_scf.get_fock = lambda *args, **kwargs: rhf_get_fock(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2.,(self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)
                self.active_scf.energy_elec = lambda *args, **kwargs: rhf_energy_elec(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2., (self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)
                self.active_energy = self.active_scf.kernel()
                 
                mCCSD = cc.CCSD(self.active_scf)
                mCCSD.max_cycle = self.active_cycles
                new_eris = mCCSD.ao2mo()
                new_eris.fock = reduce(np.dot, (self.active_scf.mo_coeff.conj().T, self.active_scf.get_fock(), self.active_scf.mo_coeff))
                ecc, t1, t2 = mCCSD.kernel(eris=new_eris)
                self.active_energy += ecc
                if self.active_method == 'ccsd(t)':
                    ecc_t = ccsd_t.kernel(mCCSD, mCCSD.ao2mo())
                    self.active_energy += ecc_t
                    l1, l2 = ccsd_t_lambda_slow.kernel(mCCSD, new_eris, t1, t2,)[1:]
                    # this slows down execution.
                    # self.active_dmat = ccsd_t_rdm_slow.make_rdm1(mCCSD, t1, t2, l1, l2, eris=new_eris)
                else:
                    # this slows down execution.
                    # self.active_dmat = mCCSD.make_rdm1()
                    pass
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
                self.active_scf.get_fock = lambda *args, **kwargs: rks_get_fock(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2.,(self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)
                self.active_scf.energy_elec = lambda *args, **kwargs: rks_energy_elec(self.active_scf, (self.emb_pot[0] + self.emb_pot[1])/2., (self.proj_pot[0] + self.proj_pot[1])/2., *args, **kwargs)
                self.active_energy = self.active_scf.kernel()
                #Slows down execution
                #self.active_dmat = self.active_scf.make_rdm1()
            
        return self.active_energy
 
class ClusterExcitedSubSystem(ClusterActiveSubSystem):

    def __init__(self):
        super().__init__()

