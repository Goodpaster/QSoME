# A method to define all cluster supsystem objects
# Daniel Graham

import re
from qsome import subsystem, custom_pyscf_methods, molpro_calc, comb_diis
from pyscf import gto, scf, dft, cc, tools
from pyscf.cc import ccsd_t, uccsd_t, ccsd_t_rdm_slow, ccsd_t_lambda_slow
from pyscf.scf import diis as scf_diis
from pyscf.lib import diis as lib_diis
import os

from functools import reduce

import numpy as np
import scipy as sp

#Custom PYSCF method for the active subsystem.
from copy import deepcopy as copy


class ClusterEnvSubSystem(subsystem.SubSystem):
    """
    A base subsystem object for use in projection embedding.

    Attributes
    ----------
    mol : Mole
        The pyscf Mole object specifying the geometry and basis
    env_method : str
        Defines the method to use for environment calculations.
    env_scf : SCF
        The pyscf SCF object of the subsystem.
    dmat : np.float64
        A numpy array of electron density matrix, compatible with pyscf.
    env_hcore : np.float64 
        A numpy array of core hamiltonian matrix, compatible with pyscf.
    env_mo_coeff : np.float64 
        A numpy array of mo coefficients, compatible with pyscf.
    env_mo_occ : np.float
        A numpy array of mo occupations, compatible with psycf
    env_mo_energy : np.float
        A numpy array of mo energies, compatible with psycf
    env_energy : float
        The total energy of this subsystem.
    filename : str
        The path to the input file being read.
    smearsigma : float
        Sigma value for fermi smearing.
    damp : float
        Damping percentage. Mixeas a percent of previous density into
        each new density. 
    shift : float
        How much to level shift orbitals.
    subcycles : int
        Number of diagonalization cycles.
    diis : DIIS
        pyscf DIIS accelerator to use for diagonalization.
    freeze : bool
        Whether to freeze the subsystem density.
    initguess : str
        Which method to use for the initial density guess.
    grid_level : int
        Specifies pyscf Grids size. 
    rhocutoff : float
        Numerical integration rho cutoff.
    verbose : int
        Specifies level of verbose output.
    analysis : bool
        Analysis flag.
    debug : bool
        Debug flag.
    nproc : int
        Number of processors provided for calculation.
    pmem : int
        Memory per processor available in MB.
    scr_dir : str
        Path to the directory used for scratch.

    Methods
    ------- 
    init_env_scf()
       Initializes the pyscf SCF object. 
    init_density()
        Sets the initial subsystem density matrix.
    get_env_elec_energy()
        Get the electronic energy for the subsystem.
    get_env_energy()
        Get the total energy for the subsystem.
    update_proj_pot(new_POp)
        Update the projection operator potential.    
    get_env_proj_e()
        Get the energy of the peojection potential.
    update_emb_pot(new_emb_pot)
        Update the potential the subsystem is embedded into.
    update_fock()
        Update the subsystem fock matrix.
    update_density(new_den)
        Update the subsystem density matrix.
    save_orbitals()
        Save the subsystem orbitals.
    diagonalize(run_diis=True)
        Diagonailze the subsystem fock matrix and generate new density.
    """


    def __init__(self, mol, env_method, unrestricted=False, filename=None, 
                 smearsigma=0., damp=0., shift=0., subcycles=1, diis=0, 
                 freeze=False, initguess=None, grid_level=4, rhocutoff=1e-7, 
                 verbose=3, analysis=False, debug=False, nproc=None, pmem=None,
                 scr_dir=None, save_orbs=False, save_density=False):
        """
        Parameters
        ----------
        mol : Mole
            The pyscf Mole object specifying the geometry and basis
        env_method : str
            Defines the method to use for environment calculations.
        filename : str, optional
            The path to the input file being read. (default is None)
        smearsigma : float, optional
            Sigma value for fermi smearing, (default is 0.)
        damp : float, optional
            Damping percentage. Mixeas a percent of previous density into
            each new density. (default is 0.)
        shift : float, optional
            How much to level shift orbitals. (default is 0.)
        subcycles : int, optional
            Number of diagonalization cycles. (default is 1)
        diis : int, optional
            Specifies DIIS method to use. (default is 0)
        freeze : bool, optional
            Whether to freeze the subsystem density or not.
            (default is False)
        initguess : str, optional
            Which method to use for the initial density guess.
            (default is None)
        grid_level : int, optional
            Specifies pyscf Grids size. (default is 4)
        rhocutoff : float, optional
            Numerical integration rho cutoff. (default is 1e-7) 
        verbose : int, optional
            Specifies level of verbose output. (default is 3)
        analysis : bool, optional
            Analysis flag. (default is False)
        debug : bool, optional
            Debug flag. (default is False)
        nproc : int, optional
            Number of processors provided for calculation. (default is None)
        pmem : int, optional
            Memory per processor available in MB. (default is None)
        scr_dir : str, optional
            Path to the directory used for scratch. (default is None) 
        """

        self.mol = mol
        self.env_method = env_method
        self.unrestricted = unrestricted

        self.initguess = initguess
        self.smearsigma = smearsigma
        self.rho_cutoff = rhocutoff
        self.grid_level = grid_level
        self.damp = damp
        self.shift = shift

        self.freeze = freeze
        self.subcycles = subcycles 

        self.verbose = verbose
        self.analysis = analysis
        self.debug = debug
        self.nproc = nproc
        self.pmem = pmem
        self.scr_dir = scr_dir
        self.save_orbs = save_orbs
        self.save_density = save_density
        if filename == None:
            filename = os.getcwd() + '/temp.inp'
        self.filename = filename
        self.fermi = [0., 0.]

        self.env_scf = self.init_env_scf()
        self.dmat = self.init_density()
        self.env_hcore = self.env_scf.get_hcore()
        self.emb_pot = [np.zeros_like(self.env_hcore), 
                        np.zeros_like(self.env_hcore)]
        self.proj_pot = [np.zeros_like(self.env_hcore), 
                        np.zeros_like(self.env_hcore)]
        self.fock = copy(self.proj_pot)
        self.emb_fock = None
        self.env_mo_coeff = [np.zeros_like(self.env_hcore), 
                             np.zeros_like(self.env_hcore)]
        self.env_mo_occ = [np.zeros_like(self.env_hcore[0]), 
                           np.zeros_like(self.env_hcore[0])]
        self.env_mo_energy = self.env_mo_occ.copy()
        self.env_energy = 0.0
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
        elif diis == 5:
            self.diis = comb_diis.EDIIS_DIIS(self.env_scf)
        elif diis == 6:
            self.diis = comb_diis.ADIIS_DIIS(self.env_scf)
        else:
            self.diis = None 

    def init_env_scf(self, mol=None, env_method=None, rho_cutoff=None, 
                     verbose=None, damp=None, shift=None):
        """Initializes the environment pyscf scf object.
        
        Parameters
        ----------
        mol : Mole, optional
            Mole object containing geometry and basis (default is None).
        method : str, optional
            Subsystem method for calculation (default is None). 
        rho_cutoff : float, optional
            DFT rho cutoff parameter (default is None).
        verbose : int, optional
            Verbosity parameter (default is None).
        damp : float, optional
            Damping parameter (default is None).
        shift : float, optional
            Level shift parameter (default is None).
        """

        if mol is None:
            mol = self.mol
        if env_method is None:
            env_method = self.env_method
        if rho_cutoff is None:
            rho_cutoff = self.rho_cutoff
        if verbose is None:
            verbose = self.verbose
        if damp is None:
            damp = self.damp
        if shift is None:
            shift = self.shift

        if self.pmem:
            mol.max_memory = self.pmem

        if self.unrestricted:
            if env_method == 'hf':
                scf_obj = scf.UHF(mol) 
            else:
                scf_obj = scf.UKS(mol)
                scf_obj.xc = env_method
                scf_obj.small_rho_cutoff = rho_cutoff

        elif mol.spin != 0:
            if env_method == 'hf':
                scf_obj = scf.ROHF(mol) 
            else:
                scf_obj = scf.ROKS(mol)
                scf_obj.xc = env_method
                scf_obj.small_rho_cutoff = rho_cutoff
        else:
            if env_method == 'hf':
               scf_obj = scf.RHF(mol) 
            else:
                scf_obj = scf.RKS(mol)
                scf_obj.xc = env_method
                scf_obj.small_rho_cutoff = rho_cutoff

        env_scf = scf_obj
        env_scf.verbose = verbose
        env_scf.damp = damp
        env_scf.level_shift = shift
        return env_scf

    def init_density(self, in_dmat=None, scf_obj=None, env_method=None, 
                     initguess=None):
        """Initializes the subsystem density..
        
        Parameters
        ----------
        in_dmat : numpy.float64
            New subsystem density matrix (default is None).
        scf_obj : SCF, optional
            Subsystem SCF object (default is None). 
        env_method : str, optional
            Subsystem energy method (default is None).
        initguess : str, optional
            Subsystem density guess method (default is None).
        """
        if not in_dmat is None:
            return in_dmat

        else:
            if scf_obj is None:
                scf_obj = self.env_scf
            if env_method is None:
                env_method = self.env_method
            if initguess is None:
                initguess = self.initguess

            if self.unrestricted:
                if initguess in ['atom', '1e', 'minao']:
                    dmat = scf_obj.get_init_guess(self.initguess)
                else:
                    dmat = scf_obj.get_init_guess()
            elif self.mol.spin != 0:
                if initguess in ['atom', '1e', 'minao']:
                    dmat = scf_obj.get_init_guess(initguess)
                else:
                    dmat = scf_obj.get_init_guess()
            else:
                if initguess in ['atom', '1e', 'minao']:
                    t_dmat = scf_obj.get_init_guess(initguess)
                else:
                    t_dmat = scf_obj.get_init_guess()
                dmat = [t_dmat/2., t_dmat/2.]

            return dmat

    def get_env_elec_energy(self, env_method=None, fock=None,  dmat=None, 
                            env_hcore=None, proj_pot=None, emb_pot=None):
        """Returns the electronic energy of the subsystem
        
        Parameters
        ----------
        env_method : str, optional
            Subsystem low level method (default is None).
        env_scf : np.float64, optional
            Subsystem fock matrix (default is None).
        dmat : np.float64, optional
            Subsystem density matrix (default is None).
        env_hcore : np.float64, optional
            Subsystem core hamiltonian (default is None).
        proj_pot : np.float64, optional
            Projection potential matrix (default is None).
        emb_pot : np.float64, optional
            Embedding potential matrix (default is None).
        """

        #Need to use embedding fock for freeze and thaw, and not for energies.
        if env_method is None:
            env_method = self.env_method
        if dmat is None:
            dmat = self.dmat
        if fock is None:
            if self.unrestricted or self.mol.spin != 0:
                self.fock = (self.env_scf.get_hcore() 
                             + self.env_scf.get_veff(dm=dmat))
                fock = self.fock
            else:
                self.fock = (self.env_scf.get_hcore() 
                             + self.env_scf.get_veff(dm=(dmat[0] + dmat[1])))
                fock = self.fock
        if env_hcore is None:
            env_hcore = self.env_hcore
        if proj_pot is None:
            proj_pot = self.proj_pot
        if emb_pot is None:
            if self.emb_fock is None:
                emb_pot = [np.zeros_like(dmat[0]), np.zeros_like(dmat[1])]
            else:
                if self.unrestricted or self.mol.spin != 0:
                    emb_pot = [self.emb_fock[0] - fock[0], 
                               self.emb_fock[1] - fock[1]]
                else:
                    emb_pot = [self.emb_fock[0] - fock, 
                               self.emb_fock[1] - fock]


        e_emb = 0.0
        #subsys_e = np.einsum('ij,ji', env_hcore, (dmat[0] + dmat[1])).real
        if self.unrestricted or self.mol.spin != 0:
            e_proj = (np.einsum('ij,ji', proj_pot[0], dmat[0]) + 
                      np.einsum('ij,ji', proj_pot[1], dmat[1])).real
            e_emb = (np.einsum('ij,ji', emb_pot[0], dmat[0]) + 
                     np.einsum('ij,ji', emb_pot[1], dmat[1])).real
            subsys_e = self.env_scf.energy_elec(dm=dmat)[0]
        else:
            e_proj = (np.einsum('ij,ji', (proj_pot[0] + proj_pot[1])/2., 
                                (dmat[0] + dmat[1])).real)
            e_emb = (np.einsum('ij,ji', (emb_pot[0] + emb_pot[1])/2., 
                     (dmat[0] + dmat[1])).real)
            subsys_e = self.env_scf.energy_elec(dm=(dmat[0] + dmat[1]))[0]
        return subsys_e + e_emb + e_proj

    def get_env_energy(self, mol=None):
        """Return the total subsystem energy

        Parameters
        ----------
        mol : Mole, optional
            Subsystem Mole object (default is None).
        """

        if mol is None:
            mol = self.mol
        self.env_energy = self.get_env_elec_energy() + mol.energy_nuc()
        return self.env_energy

    def update_emb_pot(self, new_emb_pot):
        self.emb_pot = new_emb_pot 

    def update_proj_pot(self, new_POp):
        self.proj_pot = new_POp

    def get_env_proj_e(self, env_method=None, proj_pot=None, dmat=None):
        """Gets the projection operator energy

        Parameters
        ----------
        env_method : str, optional
            Subsystem low level method string (default is None).
        proj_pot : numpy.float64, optional
            Projection potential matrix (default is None).
        dmat : numpy.float64, optional
            Subsystem density matrix (default is None).
        """


        if env_method is None:
            env_method = self.env_method
        if proj_pot is None:
            proj_pot = self.proj_pot
        if dmat is None:
            dmat = self.dmat

        if self.unrestricted or self.mol.spin != 0:
            e_proj = (np.einsum('ij,ji', proj_pot[0], dmat[0]) + 
                      np.einsum('ij,ji', proj_pot[1], dmat[1])).real
        else:
            e_proj = (np.einsum('ij,ji', (proj_pot[0] + proj_pot[1])/2., 
                      (dmat[0] + dmat[1])).real)

        return e_proj 

    def update_emb_fock(self, new_fock):
        self.emb_fock = new_fock

    def update_density(self, new_den):
        self.dmat = new_den
        return self.dmat

    #TODO
    def save_orbitals(self):
        pass

    def diagonalize(self, scf=None, subcycles=None, env_method=None,
                     fock=None, env_hcore=None, emb_pot=None, proj_pot=None, 
                     dmat=None, diis=None):
        """Diagonalizes the subsystem fock matrix and returns updated density.

        Parameters
        ----------
        scf : SCF, optional
            pyscf SCF object to diagonalize (default is None)
        subcycles : int, optional
            Number of diagonalization subcycles (default is None).
        env_method : str, optional
            Which method to calculate properties (default is None).
        fock : numpy.float64, optional
            Subsystem fock matrix (default is None).
        emb_pot : numpy.float64, optional
            Potential which the subsystem is embedded into (default is None).
        proj_pot : numpy.float64, optional
            Projectioon potential of the subsystem (default is None).
        dmat : numpy.float64, optional
            Density matrix of subsystem (default is None).
        diis : DIIS or int, optional  
            Which DIIS to use for diagonalization. 
            Passing a negative value turns of DIIS (default is None).
        """

        if scf is None:
            scf = self.env_scf
        if subcycles is None:
            subcycles = self.subcycles
        if env_method is None:
            env_method = self.env_method
        if dmat is None:
            dmat = self.dmat
        if fock is None:
            if self.emb_fock is None:
                if self.unrestricted:
                    fock = scf.get_fock(dm=dmat)
                elif scf.mol.spin != 0:
                    #RO case
                    pass
                else:
                    single_fock = scf.get_fock(dm=(dmat[0] + dmat[1]))
                    fock = [single_fock, single_fock]
            else:
                fock = self.emb_fock
        if env_hcore is None:
            env_hcore = self.env_hcore
        if proj_pot is None:
            proj_pot = self.proj_pot
        if diis is None:
            diis = self.diis

        mol = scf.mol

        nA_a = fock[0].shape[0]
        nA_b = fock[1].shape[0]
        N = [np.zeros((nA_a)), np.zeros((nA_b))]
        N[0][:mol.nelec[0]] = 1.
        N[1][:mol.nelec[1]] = 1.

        for i in range(subcycles):
            #TODO This doesn't work.
            if i > 0:
                fock = self.update_fock()

            if self.unrestricted:
                emb_proj_fock = [None, None]
                emb_proj_fock[0] = fock[0] + proj_pot[0]
                emb_proj_fock[1] = fock[1] + proj_pot[1]
                #This is the costly part. I think.
                E, C = scf.eig(emb_proj_fock, scf.get_ovlp())
                env_mo_energy = [E[0], E[1]]
                env_mo_coeff = [C[0], C[1]]
            elif mol.spin != 0:
                #Do ROKS Diagonalize
                pass
            else:
                emb_proj_fock = fock[0] + proj_pot[0]
                emb_proj_fock += fock[1] + proj_pot[1]
                emb_proj_fock = emb_proj_fock / 2.
                # Errors abound here. Doesn't converge to correct value.
                # Need DIIS with a projection component.
                if ( not((type(diis) is int) and diis < 0) 
                    and not (diis is None)):
                    if self.diis_num == 1:
                        emb_fock = diis.update(emb_proj_fock)
                    else:
                        s1e = scf.get_ovlp()
                        dm = dmat[0] + dmat[1]
                        f = emb_fock
                        mf = scf
                        h1e = (env_hcore
                               + (emb_pot[0] + emb_pot[1])/2
                               + (proj_pot[0] + proj_pot[1])/2.)
                        vhf = scf.get_veff(dm=(self.dmat[0] + self.dmat[1]))
                        emb_fock = diis.update(s1e, dm, f, mf, h1e, vhf)

                E, C = scf.eig(emb_proj_fock, scf.get_ovlp())
                env_mo_energy = [E, E]
                env_mo_coeff = [C, C]
        
            nocc_orbs = [mol.nelec[0], mol.nelec[1]]
            e_sorted = [np.sort(env_mo_energy[0]), np.sort(env_mo_energy[1])]
            fermi = [None, None]
            if (len(e_sorted[0]) > nocc_orbs[0]):
                fermi[0] = ((e_sorted[0][nocc_orbs[0]] 
                            + e_sorted[0][nocc_orbs[0] -1]) / 2.)
            else:
                fermi[0] = 0.    #Minimal basis
            if (len(e_sorted[1]) > nocc_orbs[1]):
                fermi[1] = ((e_sorted[1][nocc_orbs[1]] 
                            + e_sorted[1][nocc_orbs[1] -1]) / 2.)
            else:
                fermi[1] = 0.    #Minimal basis

            #Smear sigma may not be right for single elctron
            mo_occ = [np.zeros_like(env_mo_energy[0]), 
                      np.zeros_like(env_mo_energy[1])]
            if self.smearsigma > 0.:
                mo_occ[0] = ((env_mo_energy[0] 
                              - fermi[0]) / smearsigma)
                ie = np.where( mo_occ[0] < 1000 )
                i0 = np.where( mo_occ[0] >= 1000 )
                mo_occ[0][ie] = 1. / ( np.exp( mo_occ[0][ie] ) + 1. )
                mo_occ[0][i0] = 0.

                mo_occ[1] = (env_mo_energy[1] - fermi[1] ) / self.smearsigma
                ie = np.where( mo_occ[1] < 1000 )
                i0 = np.where( mo_occ[1] >= 1000 )
                mo_occ[1][ie] = 1. / ( np.exp( mo_occ[1][ie] ) + 1. )
                mo_occ[1][i0] = 0.

            else:
                if (len(e_sorted[0]) > nocc_orbs[0]):
                    for i in range(nocc_orbs[0]): 
                        mo_occ[0][i] = 1
                else:
                    mo_occ[0][:] = 1.

                if (len(e_sorted[1]) > nocc_orbs[1]):
                    for i in range(nocc_orbs[1]): 
                        mo_occ[1][i] = 1
                else:
                    mo_occ[1][:] = 1.

            self.env_mo_energy = env_mo_energy
            self.env_mo_coeff = env_mo_coeff
            self.env_mo_occ = mo_occ
            self.fermi = fermi
            self.dmat[0] = np.dot((env_mo_coeff[0] * mo_occ[0]), 
                                   env_mo_coeff[0].transpose().conjugate())
            self.dmat[1] = np.dot((env_mo_coeff[1] * mo_occ[1]), 
                                   env_mo_coeff[1].transpose().conjugate())

            return self.dmat

class ClusterActiveSubSystem(ClusterEnvSubSystem):
    """
    Extends ClusterEnvSubSystem to calculate higher level methods.

    Attributes
    ----------
    active_method : str
        Which method to use for high level calculation.
    localize_orbitals : bool
        Whether or not to localize orbitals.
    active_orbs : list
        List of active orbitals for multireference methods.
    active_conv : float
        Convergence parameter for active method.
    active_grad : float
        Gradient parameter for active method.
    active_cycles : int
        Number of active cycles.
    active_damp : float
        Damping parameter.
    active_shift : float
        Level shift parameter.
    use_molprol : bool
        Whether or not to use molpro.
    writeorbs : bool
        Whether or not to write active orbs.
    active_mo_coeff : numpy.float64
        Array of mo coefficients.
    active_mo_occ : list
        List of occupied orbitals.
    active_mo_energy : list
        List of mo orbital energies.

    Methods
    -------
    active_proj_energy(dmat, proj_pot)
        Get the projection potential for the density of the active system.
    active_in_env_energy(mol, dmat, env_hcore, emb_pot, proj_pot, active_method)
        Get the high level energy embedded into the total system.
    """ 

    def __init__(self, mol, env_method, active_method, 
                 active_unrestricted=False, localize_orbitals=False,
                 active_orbs=None, avas=None, active_conv=1e-9, active_grad=None, 
                 active_cycles=100, use_molpro=False, active_damp=0, 
                 active_shift=0, active_initguess='ft', active_save_orbs=False,
                 active_save_density=False, **kwargs):
        """
        Parameters
        ----------
        mol : Mole
            The pyscf Mole object specifitying geometry and basis.
        env_method : str
            Defines the method for use in env calculations.
        active_method : str
            Defines the high level method for the calculations.
        localize_orbitals : bool, optional
            Whether to localize the orbitals first (default is False).
        active_orbs : list, optional
            A list of active orbitals (default is None).
        active_conv : float, optional
            Active SCF convergence parameters (default is 1e-9).
        active_grad : float, optional
            Active SCF gradient parameters (default is None).
        active_cycles : int, optional
            Number of active cycles for SCF (default is 100).
        use_molpro : bool, optional
            Whether or not to use molpro for high level calculations
            (default is False).
        writeorbs : bool, optional
            Whether or not to write the orbitals out (default is False).
        active_damp : int, optional
            SCF damping parameters (default is 0).
        active_shift : int, optional
            SCF level shift parameters (default is 0).
        """

        super().__init__(mol, env_method, **kwargs)
        self.active_method = active_method
        self.active_unrestricted = active_unrestricted
        self.localize_orbitals = localize_orbitals
        self.active_orbs = active_orbs
        self.avas = avas
        self.active_conv = active_conv
        self.active_grad = active_grad
        self.active_cycles = active_cycles
        self.active_damp = active_damp
        self.active_shift = active_shift
        self.active_initguess = active_initguess
        self.use_molpro = use_molpro
        self.active_save_orbs = active_save_orbs
        self.active_save_density = active_save_density

        self.active_mo_coeff = None
        self.active_mo_occ = None
        self.active_mo_energy = None
 

    def active_proj_energy(self, dmat=None, proj_pot=None):
        """Return the projection energy

        Parameters
        ----------
        dmat : numpy.float64, optional
            The active subsystem density matrix (default is None).
        proj_pot : numpy.float64, optional
            The projection potential (default is None).
        """

        if dmat is None:
            dmat = self.active_dmat
        if proj_pot is None:
            proj_pot = self.proj_pot
        return np.trace(dmat, proj_pot)
           
    def active_in_env_energy(self, mol=None, dmat=None, emb_pot=None, 
                             proj_pot=None, active_method=None, 
                             active_conv=None, active_grad=None, 
                             active_shift=None, active_damp=None, 
                             active_cycles=None, active_initguess=None):
        """Returns the active subsystem energy as embedded in the full system.

        Parameters
        ----------
        mol : Mole, optional
            pyscf Mole object (default is None).
        dmat : numpy.float64, optional
            Subsystem density matrix (default is None).
        env_hcore : numpy.float64, optional
            Subsystem core hamiltonian matrix (default is None).
        emb_pot : numpy.float64, optional
            Subsystem embedding potential matrix (default is None).
        proj_pot : numpy.float64, optional
            Subsystem projection potential matrix (default is None).
        active_method : str, optional
            Active method string (default is None).
        """

        if mol is None:
            mol = self.mol
        if dmat is None:
            dmat = self.dmat
        if emb_pot is None:
            if self.emb_fock is None:
                emb_pot = [np.zeros_like(dmat[0]), np.zeros_like(dmat[1])]
            else:
                if self.unrestricted:
                    self.fock = (self.env_scf.get_hcore() 
                                 + self.env_scf.get_veff(dm=dmat))
                    fock = self.fock
                    emb_pot = (self.emb_fock[0] - fock[0], 
                               self.emb_fock[1] - fock[1])
                elif mol.spin != 0:
                    #RO
                    pass
                else:
                    self.fock = (self.env_scf.get_hcore() 
                        + self.env_scf.get_veff(dm=(dmat[0] + dmat[1])))
                    fock = self.fock
                    emb_pot = (self.emb_fock[0] - fock, 
                               self.emb_fock[1] - fock)
        if proj_pot is None:
            proj_pot = self.proj_pot 
        if active_method is None:
            active_method = self.active_method
        if active_conv is None:
            active_conv = self.active_conv
        if active_grad is None:
            active_grad = self.active_grad
        if active_shift is None:
            active_shift = self.active_shift
        if active_damp is None:
            active_damp = self.active_damp
        if active_cycles is None:
            active_cycles = self.active_cycles
        if active_initguess is None:
            active_initguess = self.active_initguess

        active_energy = 0.0
        if self.use_molpro:
            mod_hcore = (self.env_scf.get_hcore() 
                         + ((emb_pot[0] + emb_pot[1])/2.
                         + (proj_pot[0] + proj_pot[1])/2.))
            if active_method[0] == 'r':
                active_method = active_method[1:]
            if active_method == 'hf':
                energy = molpro_calc.molpro_energy(
                          mol, mod_hcore, active_method, self.filename, 
                          self.active_save_orbs, scr_dir=self.scr_dir, 
                          nproc=self.nproc, pmem=self.pmem)
                active_energy = energy[0]
            elif active_method == 'ccsd' or active_method == 'ccsd(t)':
                energy = molpro_calc.molpro_energy(
                          mol, mod_hcore, active_method, self.filename, 
                          self.active_save_orbs, scr_dir=self.scr_dir, 
                          nproc=self.nproc, pmem=self.pmem)
                active_energy = energy[0]
            elif re.match(re.compile('cas(pt2)?\[.*\].*'), active_method):
                energy = molpro_calc.molpro_energy(
                          mol, mod_hcore, active_method, self.filename, 
                          self.active_save_orbs, self.active_orbs, self.avas,
                          self.localize_orbitals, scr_dir=self.scr_dir, 
                          nproc=self.nproc, pmem=self.pmem)
                active_energy = energy[0]
            elif active_method == 'fcidump':
                energy = molpro_calc.molpro_energy(
                          mol, mod_hcore, active_method, self.filename, 
                          self.active_save_orbs, scr_dir=self.scr_dir, 
                          nproc=self.nproc, pmem=self.pmem)
                active_energy = energy[0]
            else:
                pass

        else:
            if self.active_unrestricted: 
                if (active_method == 'hf' or active_method == 'ccsd' or 
                    active_method == 'uccsd' or active_method == 'ccsd(t)' or 
                    active_method == 'uccsd(t)' or 
                    re.match(re.compile('cas(pt2)?\[.*\].*'), 
                                           active_method[1:])):
                    active_scf = scf.UHF(mol)
                    active_scf.conv_tol = active_conv
                    active_scf.conv_tol_grad = active_grad
                    active_scf.max_cycle = active_cycles
                    active_scf.level_shift = active_shift
                    active_scf.damp = active_damp
                    active_scf.get_fock = lambda *args, **kwargs: (
                        custom_pyscf_methods.uhf_get_fock(active_scf, 
                        emb_pot, proj_pot, *args, **kwargs))
                    active_scf.energy_elec = lambda *args, **kwargs: (
                        custom_pyscf_methods.uhf_energy_elec(active_scf, 
                        emb_pot, proj_pot, *args, **kwargs))
                    if active_initguess != 'ft':
                        dmat = active_scf.get_init_guess(key=active_initguess)

                    active_energy = active_scf.kernel(dm0=dmat)

                    self.active_scf = active_scf
                    if 'ccsd' in active_method:
                        active_cc = cc.UCCSD(active_scf)
                        active_cc.max_cycle = active_cycles
                        active_cc.conv_tol = active_conv
                        #active_cc.conv_tol_normt = 1e-6
                        ecc, t1, t2 = active_cc.kernel()
                        active_energy += ecc
                        if (active_method == 'ccsd(t)' 
                            or active_method == 'uccsd(t)'):
                            eris = active_cc.ao2mo(active_scf.mo_coeff)
                            ecc_t = uccsd_t.kernel(active_cc, eris)
                            active_energy += ecc_t
                elif re.match(re.compile('cas(pt2)?\[.*\].*'), 
                                         active_method[1:]):
                    pass
                elif active_method[1:] == 'fci':
                    pass
                else: 
                    pass
            elif mol.spin != 0: 
                #RO
                pass
            elif active_method == 'fcidump':
                active_scf = scf.RHF(mol)
                active_scf.conv_tol = active_conv
                active_scf.conv_tol_grad = active_grad
                active_scf.max_cycle = active_cycles
                active_scf.level_shift = active_shift
                active_scf.damp = active_damp
                mod_hcore = ((self.env_scf.get_hcore() 
                             + (emb_pot[0] + emb_pot[1])/2. 
                             + (proj_pot[0] + proj_pot[1])/2.))
                #active_scf.get_hcore = lambda *args, **kwargs: mod_hcore
                active_energy = active_scf.kernel(dm0=(dmat[0] + dmat[1]))
                fcidump_filename = (os.path.splitext(self.filename)[0] 
                                    + '.fcidump')
                print (f"FCIDUMP GENERATED AT {fcidump_filename}")
                tools.fcidump.from_scf(active_scf, (
                    os.path.splitext(self.filename)[0] + '.fcidump'), 
                    tol=1e-200)
            else: 
                if active_method[0] == 'r':
                    active_method = active_method[1:]
                if (active_method == 'hf' or active_method == 'ccsd' or 
                    active_method == 'ccsd(t)' or  
                    re.match(re.compile('cas(pt2)?\[.*\].*'), 
                                           active_method[1:])):
                    active_scf = scf.RHF(mol)
                    active_scf.conv_tol = active_conv
                    active_scf.conv_tol_grad = active_grad
                    active_scf.max_cycle = active_cycles
                    active_scf.level_shift = active_shift
                    active_scf.damp = active_damp
                    active_scf.get_fock = lambda *args, **kwargs: (
                        custom_pyscf_methods.rhf_get_fock(active_scf, 
                        (emb_pot[0] + emb_pot[1])/2.,
                        (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
                    active_scf.energy_elec = lambda *args, **kwargs: (
                        custom_pyscf_methods.rhf_energy_elec(active_scf, 
                        (emb_pot[0] + emb_pot[1])/2., 
                        (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
                    if active_initguess == 'ft':
                        init_dmat = dmat[0] + dmat[1]
                    else:
                        init_dmat = active_scf.get_init_guess(key=active_initguess)
                    active_energy = active_scf.kernel(dm0=init_dmat)
                    self.active_scf = active_scf
                    # this slows down execution.
                    #self.active_dmat = self.active_scf.make_rdm1()
                    if 'ccsd' in active_method:
                        active_cc = cc.CCSD(active_scf)
                        active_cc.max_cycle = active_cycles
                        active_cc.conv_tol = active_conv
                        #active_cc.conv_tol_normt = 1e-6
                        ecc, t1, t2 = active_cc.kernel()
                        eris = active_cc.ao2mo()
                        active_energy += ecc
                        self.active_scf = active_cc
                        if active_method == 'ccsd(t)':
                            ecc_t = ccsd_t.kernel(active_cc, eris)
                            active_energy += ecc_t
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
                    active_scf = scf.RKS(mol)
                    active_scf.xc = active_method
                    active_scf.grids = self.env_scf.grids
                    active_scf.small_rho_cutoff = self.rho_cutoff
                    active_scf.conv_tol = active_conv
                    active_scf.conv_tol_grad = active_grad
                    active_scf.max_cycle = active_cycles
                    active_scf.level_shift = active_shift
                    active_scf.damp = active_damp
                    #active_scf.get_hcore = lambda *args, **kwargs: env_hcore
                    active_scf.get_fock = lambda *args, **kwargs: (
                        custom_pyscf_methods.rks_get_fock(active_scf, 
                        (emb_pot[0] + emb_pot[1])/2.,
                        (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
                    active_scf.energy_elec = lambda *args, **kwargs: (
                        custom_pyscf_methods.rks_energy_elec(active_scf, 
                        (emb_pot[0] + emb_pot[1])/2., 
                        (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
                    active_energy = active_scf.kernel(dm0=(dmat[0] + dmat[1]))
                    #Slows down execution
                    #self.active_dmat = self.active_scf.make_rdm1()

                #temp_dmat = copy(self.active_dmat)
                #self.active_dmat = [temp_dmat/2., temp_dmat/2.]
        self.active_energy = active_energy 

        if self.active_save_density:
            pass
        if self.active_save_orbs:
            pass
        return self.active_energy
 
class ClusterExcitedSubSystem(ClusterActiveSubSystem):

    def __init__(self):
        super().__init__()

