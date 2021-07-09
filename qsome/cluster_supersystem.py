""" A method to define a cluster supersystem
Daniel Graham
Dhabih V. Chulhai"""

import os
import copy as copy
import numpy as np
import h5py

from pyscf import scf, dft, lib, lo
from pyscf.tools import cubegen, molden

from qsome import custom_diis
from qsome.helpers import concat_mols
from qsome.utilities import time_method


class ClusterSuperSystem:
    """
    Defines a molecular system ready for embedding.

    Attributes
    ----------
    subsystems : list
        List of SubSystem objects.
    fs_method : str
        Defines the supersystem method.
    proj_oper : str
        Which projection operator to use.
    nproc : int
        Number of processors available.
    pmem : float
        Per processor memory available in MB.
    scr_dir : str
        Path to scratch directory.
    filename : str
        Path to input file. General location for file operations.
    chk_filename : str
        Path to chkpoint file.
    ft_cycles : int
        Max number of freeze and thaw cycles.
    ft_conv : float
        Freeze and thaw convergence parameter
    ft_grad : float
        Freeze and thaw gradient parameter
    ft_writeorbs : bool
        Whether to write cube file of subsystem densities.
    ft_setfermi : float
        For huzfermi operator. The fermi level for the projector.
    ft_initguess : str
        Iniital density guess for freeze and thaw cycles.
    ft_updatefock : int
        How often to update the full system fock matrix during F&T cycles.
    cycles : int
        Number of pyscf cycles for supersystem calculation.
    conv : float
        Convergence parameter for pyscf supersystem calculation.
    grad : float
        Gradient parameter for pyscf supersysttem calculation.
    rho_cutoff : float
        Small rho cutoff value for pyscf dft grid.
    damp : float
        Damping parameter for supersystem pyscf.
    shift : float
        Level shift parameter for supersystem pyscf.
    smearsigma : float
        Electron smearing sigma value for supersystem pyscf.
    initguess : str
        Initial density guess for supersystem calculation.
    grid_level : int
        Grid size for pyscf Grids object.
    verbose : int
        Verbosity for pyscf settings.
    analysis : bool
        Whether to provide results analysis or not.
    debug : bool
        Whether to provide debug results or not.
    is_ft_conv : bool
        Whether the freeze and thaw cycles have converged.
    mol : Mole
        Mole object for the full system.
    sub2sup : np.float64
        Matrix for converting between supersystem and subsystem.
    fs_scf : SCF
        Full system pyscf SCF object.
    smat : np.float64
        Full system overlap matrix.
    mo_coeff : np.float64
        Full system molecular orbnital coefficients.
    mo_occ : np.float
        Full system molecular orbital occupations.
    mo_energy : np.float
        Full system molecular orbital energies..
    fock : np.float64
        Full system fock matrix.
    hcore : np.float64
        Full system core hamiltonian matrix.
    proj_pot : list
        Projection potential matrices for each subsystem.
    fs_energy : float
        Full system energy.
    dmat : np.float64
        Full system density matrix.
    ft_diis : DIIS
        DIIS method to use for fock generation during F&T cycles.
    ft_fermi : list
        List of subsystem fermi energies.

    Methods
    -------
    gen_sub2sup(mol=None, subsystems=None)
    init_fs_scf(mol=None, fs_method=None, verbose=None, damp=None, shift=None)
    init_density(mol=None, fs_scf=None, subsystems=None)
    concat_mols(subsys_list=None)
    get_supersystem_energy()
    get_emb_subsys_elec_energy()
    correct_env_energy()
    get_active_energy()
    correct_active_energy()
    env_in_env_energy()
    update_fock()
    update_proj_pot()
    read_chkfile()
    save_chkfile()
    freeze_and_thaw()
    """


    def __init__(self, subsystems, fs_method, env_order=1., fs_smearsigma=0.,
                 fs_initguess=None, fs_conv=None, fs_grad=None, fs_cycles=None,
                 fs_excited=False, fs_excited_dict=None, fs_damp=0.,
                 fs_shift=0., fs_diis=1, fs_grid_level=None, fs_rhocutoff=None,
                 fs_verbose=None, fs_unrestricted=False,
                 fs_density_fitting=False, compare_density=False, chkfile_index=0,
                 fs_save_orbs=False, fs_save_density=False, fs_save_spin_density=False,
                 ft_cycles=100, ft_excited_relax=False, ft_excited_dict=None,
                 ft_basis_tau=1., ft_conv=1e-8, ft_grad=None, ft_damp=0,
                 ft_diis=1, ft_setfermi=None, ft_updatefock=0, ft_updateproj=1,
                 ft_initguess=None, ft_unrestricted=False, ft_save_orbs=False,
                 ft_save_density=False, ft_save_spin_density=False,
                 ft_proj_oper='huz', filename=None, scr_dir=None, nproc=None,
                 pmem=None):

        """
        Parameters
        ----------
        subsystems : list
            List of SubSystem objects.
        fs_method : str
            Defines the supersystem method.
        proj_oper : str, optional
            Which projection operator to use. (default is huz)
        nproc : int
            Number of processors available.
        pmem : float
            Per processor memory available in MB.
        scr_dir : str
            Path to scratch directory.
        filename : str
            Path to input file. General location for file operations.
        chk_filename : str
            Path to chkpoint file.
        ft_cycles : int
            Max number of freeze and thaw cycles.
        ft_conv : float
            Freeze and thaw convergence parameter
        ft_grad : float
            Freeze and thaw gradient parameter
        ft_writeorbs : bool
            Whether to write cube file of subsystem densities.
        ft_setfermi : float
            For huzfermi operator. The fermi level for the projector.
        ft_initguess : str
            Iniital density guess for freeze and thaw cycles.
        ft_updatefock : int
            How often to update the full system fock matrix during F&T cycles.
        cycles : int
            Number of pyscf cycles for supersystem calculation.
        conv : float
            Convergence parameter for pyscf supersystem calculation.
        grad : float
            Gradient parameter for pyscf supersysttem calculation.
        rho_cutoff : float
            Small rho cutoff value for pyscf dft grid.
        damp : float
            Damping parameter for supersystem pyscf.
        shift : float
            Level shift parameter for supersystem pyscf.
        smearsigma : float
            Electron smearing sigma value for supersystem pyscf.
        initguess : str
            Initial density guess for supersystem calculation.
        grid_level : int
            Grid size for pyscf Grids object.
        verbose : int
            Verbosity for pyscf settings.
        analysis : bool
            Whether to provide results analysis or not.
        debug : bool
            Whether to provide debug results or not.
        is_ft_conv : bool
            Whether the freeze and thaw cycles have converged.
        mol : Mole
            Mole object for the full system.
        sub2sup : np.float64
            Matrix for converting between supersystem and subsystem.
        fs_scf : SCF
            Full system pyscf SCF object.
        smat : np.float64
            Full system overlap matrix.
        mo_coeff : np.float64
            Full system molecular orbnital coefficients.
        mo_occ : np.float
            Full system molecular orbital occupations.
        mo_energy : np.float
            Full system molecular orbital energies..
        fock : np.float64
            Full system fock matrix.
        hcore : np.float64
            Full system core hamiltonian matrix.
        proj_pot : list
            Projection potential matrices for each subsystem.
        fs_energy : float
            Full system energy.
        dmat : np.float64
            Full system density matrix.
        ft_diis : DIIS
            DIIS method to use for fock generation during F&T cycles.
        ft_fermi : list
            List of subsystem fermi energies.
        """

        self.subsystems = subsystems
        self.env_order = env_order
        self.set_chkfile_index(chkfile_index)

        self.fs_method = fs_method
        self.fs_smearsigma = fs_smearsigma
        self.fs_initguess = fs_initguess
        self.fs_excited = fs_excited
        self.fs_excited_dict = {}
        if fs_excited_dict:
            self.fs_excited_dict = fs_excited_dict
            self.fs_excited_nroots = fs_excited_dict.get('nroots')
        self.fs_conv = fs_conv
        self.fs_grad = fs_grad
        self.fs_cycles = fs_cycles
        self.fs_damp = fs_damp
        self.fs_shift = fs_shift
        self.fs_diis_num = fs_diis
        self.grid_level = fs_grid_level
        self.rho_cutoff = fs_rhocutoff
        self.fs_verbose = fs_verbose

        self.fs_unrestricted = fs_unrestricted
        self.fs_density_fitting = fs_density_fitting
        self.compare_density = compare_density
        self.fs_save_orbs = fs_save_orbs
        self.fs_save_density = fs_save_density
        self.fs_save_spin_density = fs_save_spin_density


        # freeze and thaw settings
        self.ft_excited_relax = ft_excited_relax
        self.ft_excited_dict = {}
        if ft_excited_dict:
            self.ft_excited_dict = ft_excited_dict
            self.ft_excited_nroots = ft_excited_dict.get('nroots')
        self.ft_cycles = ft_cycles
        self.ft_basis_tau = ft_basis_tau
        self.ft_conv = ft_conv
        self.ft_grad = ft_grad
        self.ft_damp = ft_damp
        self.ft_diis_num = ft_diis
        self.ft_setfermi = ft_setfermi
        self.ft_updatefock = ft_updatefock
        self.ft_updateproj = ft_updateproj
        self.ft_initguess = ft_initguess
        self.ft_unrestricted = ft_unrestricted
        self.ft_save_orbs = ft_save_orbs
        self.ft_save_density = ft_save_density
        self.ft_save_spin_density = ft_save_spin_density
        self.proj_oper = ft_proj_oper

        self.nproc = nproc
        self.pmem = pmem
        self.scr_dir = scr_dir
        self.filename = filename

        # Densities are stored separately to allow for alpha and beta.
        self.is_ft_conv = False
        self.ext_pot = np.array([0., 0.])
        mol_list = [x.mol for x in self.subsystems]
        self.mol = concat_mols(mol_list)
        self.fs_scf = None
        self.__gen_sub2sup()
        self.__init_scf()

        self.smat = self.fs_scf.get_ovlp()
        self.mo_coeff = np.array([np.zeros_like(self.smat), np.zeros_like(self.smat)])
        self.local_mo_coeff = np.array([None, None])
        self.mo_occ = np.array([np.zeros_like(self.smat[0]),
                                np.zeros_like(self.smat[0])])
        self.mo_energy = self.mo_occ.copy()
        self.hcore = self.fs_scf.get_hcore()
        self.fock = [None, None]
        self.emb_vhf = None
        self.proj_pot = [np.array([0.0, 0.0]) for sub in self.subsystems]
        #DIIS Stuff
        self.sub_diis = [np.array([lib.diis.DIIS(), lib.diis.DIIS()]) for sub in self.subsystems]
        self.fs_energy = None
        self.env_in_env_energy = None
        self.fs_nuc_grad = None

        self.fs_dmat = None
        self.emb_dmat = None

        self.ft_diis = None
        if ft_diis == 1:
            self.ft_diis = lib.diis.DIIS()
            self.ft_diis.space = 20
        elif ft_diis == 2:
            self.ft_diis = lib.diis.DIIS()
            self.ft_diis.space = 15
        elif ft_diis == 3:
            self.ft_diis = lib.diis.DIIS()
            self.ft_diis.space = 15
            self.ft_diis_2 = lib.diis.DIIS()
            self.ft_diis.space = 15

        self.ft_fermi = [np.array([0., 0.]) for sub in subsystems]

    def __gen_sub2sup(self, mol=None, subsystems=None):
        #There should be a better way to do this.
        """Generate the translation matrix between subsystem to supersystem.

        Parameters
        ----------
        mol : Mole
            Full system Mole object.
        subsystems : list
            List of subsystems which comprise the full system.
        """

        if mol is None:
            mol = self.mol
        if subsystems is None:
            subsystems = self.subsystems

        nao = np.array([sub.mol.nao_nr() for sub in subsystems])
        nssl = [None for i in range(len(subsystems))]

        for i, sub in enumerate(subsystems):
            nssl[i] = np.zeros(sub.mol.natm, dtype=int)
            for j in range(sub.mol.natm):
                ib_t = np.where(sub.mol._bas.transpose()[0] == j)[0]
                i_b = ib_t.min()
                ie_t = np.where(sub.mol._bas.transpose()[0] == j)[0]
                i_e = ie_t.max()
                i_r = sub.mol.nao_nr_range(i_b, i_e + 1)
                i_r = i_r[1] - i_r[0]
                nssl[i][j] = i_r

            assert nssl[i].sum() == sub.mol.nao_nr(), "naos not equal!"

        nsl = np.zeros(mol.natm, dtype=int)
        for i in range(mol.natm):
            i_b = np.where(mol._bas.transpose()[0] == i)[0].min()
            i_e = np.where(mol._bas.transpose()[0] == i)[0].max()
            i_r = mol.nao_nr_range(i_b, i_e + 1)
            i_r = i_r[1] - i_r[0]
            nsl[i] = i_r

        assert nsl.sum() == mol.nao_nr(), "naos not equal!"

        sub2sup = [None for i in range(len(subsystems))]
        for i, sub in enumerate(subsystems):
            sub2sup[i] = np.zeros(nao[i], dtype=int)
            for j in range(sub.mol.natm):
                match = False
                c_1 = sub.mol.atom_coord(j)
                for k in range(mol.natm):
                    c_2 = mol.atom_coord(k)
                    dist = np.dot(c_1 - c_2, c_1 - c_2)
                    if dist < 0.0001:
                        match = True
                        i_a = nssl[i][0:j].sum()
                        j_a = i_a + nssl[i][j]
                        #ja = ia + nsl[b]
                        i_b = nsl[0:k].sum()
                        j_b = i_b + nsl[k]
                        #jb = ib + nssl[i][a]
                        sub2sup[i][i_a:j_a] = range(i_b, j_b)

                assert match, 'no atom match!'
        self.sub2sup = sub2sup
        return True

    def __init_scf(self, mol=None, fs_method=None, verbose=None, damp=None,
                   shift=None):
        """Initialize the supersystem pyscf SCF object using given settings.

        Parameters
        ----------
        mol : Mole
            Full system Mole object.
        fs_method : str
            String defining which SCF method to use for full system.
        verbose : int
            Level of verbosity for output.
        damp : float
            Damping parameter for scf convergence.
        shift : float
            Level shift parameter.
        """

        if mol is None:
            mol = self.mol
        if fs_method is None:
            fs_method = self.fs_method
        if verbose is None:
            verbose = self.fs_verbose
        if damp is None:
            damp = self.fs_damp
        if shift is None:
            shift = self.fs_shift

        if self.pmem:
            self.mol.max_memory = self.pmem

        self.fs_num_attempt = 0
        if self.fs_unrestricted:
            if fs_method == 'hf':
                scf_obj = scf.UHF(mol)
                u_scf_obj = scf_obj
            else:
                scf_obj = scf.UKS(mol)
                scf_obj.xc = fs_method
                if self.rho_cutoff is not None:
                    scf_obj.small_rho_cutoff = self.rho_cutoff
                u_scf_obj = scf_obj

        elif mol.spin != 0:
            if fs_method == 'hf':
                scf_obj = scf.ROHF(mol)
                u_scf_obj = scf.UHF(mol)
            else:
                scf_obj = scf.ROKS(mol)
                u_scf_obj = scf.UKS(mol)
                scf_obj.xc = fs_method
                u_scf_obj.xc = fs_method
                if self.rho_cutoff is not None:
                    scf_obj.small_rho_cutoff = self.rho_cutoff
                    u_scf_obj.small_rho_cutoff = self.rho_cutoff
        else:
            if fs_method == 'hf':
                scf_obj = scf.RHF(mol)
                u_scf_obj = scf.UHF(mol)
            else:
                scf_obj = scf.RKS(mol)
                u_scf_obj = scf.UKS(mol)
                scf_obj.xc = fs_method
                u_scf_obj.xc = fs_method
                if self.rho_cutoff is not None:
                    scf_obj.small_rho_cutoff = self.rho_cutoff
                    u_scf_obj.small_rho_cutoff = self.rho_cutoff

        fs_scf = scf_obj
        if self.fs_cycles is not None:
            fs_scf.max_cycle = self.fs_cycles
        if self.fs_conv is not None:
            fs_scf.conv_tol = self.fs_conv
        if self.fs_grad is not None:
            fs_scf.conv_tol_grad = self.fs_grad
        #fs_scf.damp = self.fs_damp
        #fs_scf.level_shift = self.fs_shift
        if self.fs_verbose is not None:
            fs_scf.verbose = self.fs_verbose

        grids = dft.gen_grid.Grids(mol)
        if self.grid_level is not None:
            grids.level = self.grid_level

        grids.build()
        fs_scf.grids = grids
        u_scf_obj.grids = grids
        if self.fs_density_fitting:
            fs_scf = fs_scf.density_fit()
            u_scf_obj = u_scf_obj.density_fit()

        if self.fs_diis_num == 2:
            fs_scf.DIIS = scf.diis.ADIIS

        if self.fs_diis_num == 3:
            fs_scf.DIIS = scf.diis.EDIIS

        self.fs_scf = fs_scf
        self.os_scf = u_scf_obj
        return True


    @time_method("Initialize Densities")
    def init_density(self, fs_scf=None, subsystems=None):
        """Initializes subsystem densities and returns full system density.

        Parameters
        ----------
        mol : Mole
            Full system Mole object.
        fs_scf : SCF
            Full system SCF object.
        subsystems : list
            List of subsystems which comprise the full system.
        """
        if fs_scf is None:
            fs_scf = self.fs_scf
        if subsystems is None:
            subsystems = self.subsystems

        print("".center(80, '*'))
        print("  Generate Initial System Densities  ".center(80))
        print("".center(80, '*'))
        # If supersystem dft should be read from chkfile.
        super_chk = (self.fs_initguess == 'readchk')
        s2s = self.sub2sup

        # Initialize supersystem density.
        if self.ft_initguess == 'supmol':
            self.get_supersystem_energy(readchk=super_chk)

        elif self.ft_initguess == 'rosupmol':
            rodmat = self.get_init_ro_supersystem_dmat()

        for i, subsystem in enumerate(subsystems):
            sub_dmat = [0., 0.]
            subsystem.fullsys_cs = not (self.fs_unrestricted or self.mol.spin != 0)
            # Ensure same gridpoints and rho_cutoff for all systems
            subsystem.env_scf.grids = fs_scf.grids
            if not 'hf' in subsystem.env_method:
                subsystem.env_scf.small_rho_cutoff = fs_scf.small_rho_cutoff

            sub_guess = subsystem.env_initguess
            if sub_guess is None:
                sub_guess = self.ft_initguess
            if sub_guess is None:
                sub_guess = 'readchk'
            subsystem.filename = self.filename
            subsystem.init_density(initguess=sub_guess)
            if sub_guess != 'supmol':
                sub_guess = subsystem.env_initguess
            if sub_guess == 'supmol':
                self.get_supersystem_energy(readchk=super_chk)
                sub_dmat[0] = self.fs_dmat[0][np.ix_(s2s[i], s2s[i])]
                sub_dmat[1] = self.fs_dmat[1][np.ix_(s2s[i], s2s[i])]
                temp_smat = self.smat
                temp_sm = temp_smat[np.ix_(s2s[i], s2s[i])]
                #Normalize for num of electrons in each subsystem.
                num_e_a = np.trace(np.dot(sub_dmat[0], temp_sm))
                num_e_b = np.trace(np.dot(sub_dmat[1], temp_sm))
                sub_dmat[0] *= subsystem.mol.nelec[0]/num_e_a
                sub_dmat[1] *= subsystem.mol.nelec[1]/num_e_b
                subsystem.env_dmat = sub_dmat
            elif sub_guess == 'rosupmol':
                sub_dmat[0] = rodmat[0][np.ix_(s2s[i], s2s[i])]
                sub_dmat[1] = rodmat[1][np.ix_(s2s[i], s2s[i])]
                temp_smat = self.smat
                temp_sm = temp_smat[np.ix_(s2s[i], s2s[i])]
                #Normalize for num of electrons in each subsystem.
                num_e_a = np.trace(np.dot(sub_dmat[0], temp_sm))
                num_e_b = np.trace(np.dot(sub_dmat[1], temp_sm))
                sub_dmat[0] *= subsystem.mol.nelec[0]/num_e_a
                sub_dmat[1] *= subsystem.mol.nelec[1]/num_e_b
            elif sub_guess == 'localsup': # Localize supermolecular density.
                pass

        self.update_fock(diis=False)
        self.update_proj_pot()
        print("".center(80, '*'))

    def get_emb_dmat(self):
        """Dmat info
        """
        num_rank = self.mol.nao_nr()
        dm_env = [np.zeros((num_rank, num_rank)), np.zeros((num_rank, num_rank))]
        for i, sub in enumerate(self.subsystems):
            dm_env[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += sub.env_dmat[0]
            dm_env[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += sub.env_dmat[1]
        if not (self.fs_unrestricted or self.mol.spin != 0):
            dm_env = dm_env[0] + dm_env[1]
        return dm_env

    def get_emb_ext_pot(self):
        """Returns the potential of the higher order subsystem
        """
        self.update_fock(diis=False)
        froz_veff = np.array([0., 0.])
        s2s = self.sub2sup
        for i, sub in enumerate(self.subsystems):
            sub_fock = sub.update_subsys_fock()
            if (not sub.unrestricted) and sub.mol.spin == 0:
                sub_fock = [sub_fock, sub_fock]
            if sub.env_order > self.env_order:
                fock_aa = [None, None]
                fock_aa[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
                fock_aa[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
                froz_veff[0] = (fock_aa[0] - sub_fock[0])
                froz_veff[1] = (fock_aa[1] - sub_fock[1])

        return froz_veff

    @time_method("Supersystem Energy")
    def get_supersystem_energy(self, scf_obj=None, fs_method=None,
                               readchk=False, local_orbs=False):
        """Calculate full system energy and save density matrix.

        Parameters
        ----------
        scf_obj : SCF
            SCF object for the full system.
        fs_method : str
            Which SCF method to use for calculation.
        readchk : bool
            Whether to read initial density from checkpoint file.
        """

        if self.fs_energy is None or not self.fs_scf.converged:
            if scf_obj is None:
                scf_obj = self.fs_scf
            if fs_method is None:
                fs_method = self.fs_method
            mol = scf_obj.mol

            print("".center(80, '*'))
            print("  Supersystem Calculation  ".center(80))
            print("".center(80, '*'))
            if self.is_ft_conv:
                num_rank = mol.nao_nr()
                ft_dmat = [np.zeros((num_rank, num_rank)), np.zeros((num_rank, num_rank))]
                s2s = self.sub2sup
                for i in range(len(self.subsystems)):
                    subsystem = self.subsystems[i]
                    ft_dmat[0][np.ix_(s2s[i], s2s[i])] += subsystem.env_dmat[0]
                    ft_dmat[1][np.ix_(s2s[i], s2s[i])] += subsystem.env_dmat[1]
                if self.fs_unrestricted or scf_obj.mol.spin != 0:
                    if self.fs_num_attempt == 0:
                        scf_obj.scf(dm0=ft_dmat)
                        self.fs_num_attempt += 1
                    elif self.fs_num_attempt == 1:
                        self.fs_scf = scf.fast_newton(scf_obj)
                        self.fs_num_attempt += 1
                    elif self.fs_num_attempt == 2:
                        self.fs_scf = scf.newton(scf_obj)
                        self.fs_num_attempt += 1
                    else:
                        print ('FS SCF NOT CONVERGED')
                else:
                    if self.fs_num_attempt == 0:
                        scf_obj.scf(dm0=(ft_dmat[0] + ft_dmat[1]))
                        self.fs_num_attempt += 1
                    elif self.fs_num_attempt == 1:
                        self.fs_scf = scf.fast_newton(scf_obj)
                        self.fs_num_attempt += 1
                    elif self.fs_num_attempt == 2:
                        self.fs_scf = scf.newton(scf_obj)
                        self.fs_num_attempt += 1
                    else:
                        print ('FS SCF NOT CONVERGED')
            elif readchk:
                if self.fs_unrestricted or scf_obj.mol.spin != 0:
                    init_guess = scf_obj.make_rdm1(mo_coeff=self.mo_coeff,
                                                   mo_occ=self.mo_occ)
                else:
                    init_guess = scf_obj.make_rdm1(mo_coeff=self.mo_coeff[0],
                                                   mo_occ=(self.mo_occ[0] +
                                                           self.mo_occ[1]))

                scf_obj.scf(dm0=(init_guess))
            else:
                if self.fs_num_attempt == 0:
                    scf_obj.scf()
                    self.fs_num_attempt += 1
                elif self.fs_num_attempt == 1:
                    self.fs_scf = scf.fast_newton(scf_obj)
                    self.fs_num_attempt += 1
                elif self.fs_num_attempt == 2:
                    self.fs_scf = scf.newton(scf_obj)
                    self.fs_num_attempt += 1
                else:
                    print ('FS SCF NOT CONVERGED')
            scf_obj = self.fs_scf
            self.fs_dmat = scf_obj.make_rdm1()

            if self.fs_unrestricted:
                rho_grid = self.fs_scf.grids
                alpha_dmat = self.fs_dmat[0]
                alpha_rho = scf_obj._numint.get_rho(self.mol, alpha_dmat, rho_grid,
                                                    self.fs_scf.max_memory)
                #alpha_den = alpha_rho * rho_grid.weights
                alpha_den = alpha_rho
                beta_dmat = self.fs_dmat[1]
                beta_rho = scf_obj._numint.get_rho(self.mol, beta_dmat, rho_grid,
                                                    self.fs_scf.max_memory)

                #beta_den = beta_rho * rho_grid.weights
                beta_den = beta_rho
                spin_diff = np.subtract(alpha_den,beta_den)
                neg_spin_diff = spin_diff.clip(max=0)
                spin_contam = np.dot(neg_spin_diff, rho_grid.weights)
                alpha,beta = self.mol.nelec
                s = (alpha-beta)/2.
                ss = s*(s+1) - spin_contam
                print(f"approx <S^2>:{ss:>67.8f}")

            if self.fs_dmat.ndim == 2:
                t_d = [self.fs_dmat.copy()/2., self.fs_dmat.copy()/2.]
                self.fs_dmat = t_d
                self.mo_coeff = [self.fs_scf.mo_coeff, self.fs_scf.mo_coeff]
                self.mo_occ = [self.fs_scf.mo_occ/2., self.fs_scf.mo_occ/2.]
                self.mo_energy = [self.fs_scf.mo_energy, self.fs_scf.mo_energy]

            self.save_chkfile()
            self.fs_energy = scf_obj.energy_tot()
            print("".center(80, '*'))
            if self.fs_save_density:
                self.save_fs_density_file()

            if self.fs_save_spin_density:
                self.save_fs_spin_density_file()

            if self.fs_save_orbs:
                self.save_fs_orbital_file()

            self.fs_scf = scf_obj
        print(f"KS-DFT  Energy:{self.fs_energy:>65.8f}  ")
        if local_orbs and self.local_mo_coeff[0] is None:
            #Can also use ER and Boys
            if self.fs_unrestricted or scf_obj.mol.spin != 0:
                nelec_a = scf_obj.mol.nelec[0]
                self.local_mo_coeff[0] = lo.ER(scf_obj.mol, self.mo_coeff[0][:, :nelec_a]).kernel()
                nelec_b = scf_obj.mol.nelec[1]
                self.local_mo_coeff[1] = lo.ER(scf_obj.mol, self.mo_coeff[1][:, :nelec_b]).kernel()
            else:
                nelec_a = scf_obj.mol.nelec[0]
                self.local_mo_coeff[0] = lo.ER(scf_obj.mol, self.mo_coeff[0][:, :nelec_a]).kernel()
                self.local_mo_coeff[1] = self.local_mo_coeff[0]

        if self.fs_excited:
            #Run the Excited state calculation on the self.fs_scf object.  
            pass

        return self.fs_energy

    @time_method("Supersystem RO DMAT generation")
    def get_init_ro_supersystem_dmat(self):
        mol = self.mol
        fs_method = self.fs_method
        if self.pmem:
            mol.max_memory = self.pmem
        if fs_method == 'hf':
            ro_scf = scf.ROHF(mol)
            if self.fs_diis_num == 2:
                ro_scf.DIIS = scf.diis.ADIIS
            ro_scf.kernel()
            ro_dmat = ro_scf.make_rdm1()
        else:
            ro_scf = scf.ROKS(mol)
            ro_scf.xc = fs_method
            grids = dft.gen_grid.Grids(mol)
            if self.grid_level is not None:
                grids.level = self.grid_level
            grids.build()
            if self.fs_diis_num == 2:
                ro_scf.DIIS = scf.diis.ADIIS
            ro_scf.kernel()
            ro_dmat = ro_scf.make_rdm1() 
        return ro_dmat

    def save_fs_density_file(self, filename=None, density=None):
        """Save the electron density of the full system KS-DFT
        """
        if filename is None:
            filename = self.filename
        if density is None:
            density = self.fs_dmat

        print('Writing Full System Density'.center(80))
        if self.mol.spin != 0 or self.fs_unrestricted:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_fs_alpha.cube'
            cubegen.density(self.mol, cubegen_fn, density[0])
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_fs_beta.cube'
            cubegen.density(self.mol, cubegen_fn, density[1])
        else:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_fs.cube'
            cubegen.density(self.mol, cubegen_fn, (density[0] + density[1]))

    def save_fs_spin_density_file(self, filename=None, density=None):
        """Save the spin density of the full system KS-DFT
        """
        if filename is None:
            filename = self.filename
        if density is None:
            density = self.fs_dmat

        if self.mol.spin != 0 or self.fs_unrestricted:
            print('Writing Full System Spin Density'.center(80))
            cubegen_fn = (os.path.splitext(filename)[0] + '_' +
                          self.chkfile_index + '_fs_spinden.cube')
            cubegen.density(self.mol, cubegen_fn, np.subtract(density[0], density[1]))
        else:
            print('Cannot write spin density for a closed shell system.'.center(80))

    def save_fs_orbital_file(self, filename=None, scf_obj=None, mo_occ=None,
                             mo_coeff=None, mo_energy=None):
        """Save the full system KS-DFT electron density as a molden file.
        """
        if filename is None:
            filename = self.filename
        if scf_obj is None:
            scf_obj = self.fs_scf
        if mo_occ is None:
            mo_occ = scf_obj.mo_occ
        if mo_coeff is None:
            mo_coeff = scf_obj.mo_coeff
        if mo_energy is None:
            mo_energy = scf_obj.mo_energy
        if not self.fs_unrestricted:
            molden_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_fs.molden'
            print('Writing Full System Orbitals'.center(80))
            with open(molden_fn, 'w') as fin:
                molden.header(scf_obj.mol, fin)
                molden.orbital_coeff(self.mol, fin, mo_coeff, ene=mo_energy, occ=mo_occ)
        else:
            molden_fn_a = (os.path.splitext(filename)[0] + '_' +
                           self.chkfile_index + '_fs_alpha.molden')
            print('Writing Full System Orbitals'.center(80))
            with open(molden_fn_a, 'w') as fin:
                molden.header(scf_obj.mol, fin)
                molden.orbital_coeff(self.mol, fin, mo_coeff[0],
                                     spin='Alpha', ene=mo_energy[0],
                                     occ=mo_occ[0])
            molden_fn_b = (os.path.splitext(filename)[0] + '_' +
                           self.chkfile_index + '_fs_beta.molden')
            print('Writing Full System Orbitals'.center(80))
            with open(molden_fn_b, 'w') as fin:
                molden.header(scf_obj.mol, fin)
                molden.orbital_coeff(self.mol, fin, mo_coeff[1], spin='Beta',
                                     ene=mo_energy[1], occ=mo_occ[1])


    @time_method("Subsystem Energies")
    def get_emb_subsys_elec_energy(self):
        """Calculates subsystem energy
        """

        s2s = self.sub2sup
        for i, subsystem in enumerate(self.subsystems):
            subsystem.update_fock()
            fock_aa = [None, None]
            fock_aa[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
            fock_aa[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
            froz_veff = [None, None]
            froz_veff[0] = (fock_aa[0] - subsystem.env_hcore - subsystem.env_V[0])
            froz_veff[1] = (fock_aa[1] - subsystem.env_hcore - subsystem.env_V[1])
            subsystem.update_emb_pot(froz_veff)
            subsystem.get_env_energy()
            print(f"Uncorrected Energy:{subsystem.env_energy:>61.8f}")

    @time_method("Active Energy")
    def get_hl_energy(self):
        """Determines the hl energy.
        """
        #This is crude. Later iterations should account for more than 2 subsystems.
        print("".center(80, '*'))
        print("  HL Subsystems Calculation  ".center(80))
        print("".center(80, '*'))
        s2s = self.sub2sup
        #ONLY DO FOR THE RO SYSTEM. THIS IS KIND OF A TEST.
        if self.mol.spin != 0 and not self.fs_unrestricted:
            self.update_ro_fock()
        for i, sub in enumerate(self.subsystems):
            #Check if subsystem is HLSubSystem but rightnow it is being difficult.
            if i == 0:
                fock_aa = [None, None]
                fock_aa[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
                fock_aa[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
                sub.emb_fock = fock_aa
                sub.get_hl_in_env_energy()
                act_e = sub.hl_energy
                print(f"Energy:{act_e:>73.8f}")
                print("".center(80, '*'))
                if sub.hl_save_density:
                    pass
                if sub.hl_save_spin_density:
                    pass
                if sub.hl_save_orbs:
                    pass

    @time_method("Env Energy")
    def get_env_energy(self):
        """Determines the subsystem env energy
        """
        print("".center(80, '*'))
        print("  Env Subsystem Calculation  ".center(80))
        print("".center(80, '*'))
        for i in range(len(self.subsystems)):
            sub = self.subsystems[i]
            #sub.update_subsys_fock()
            s2s = self.sub2sup
            fock_aa = np.array([None, None])
            fock_aa[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
            fock_aa[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
            sub.emb_fock = fock_aa
            sub.get_env_energy()
            env_e = sub.env_energy
            print(f"Energy Subsystem {i}:{env_e:>61.8f}")
        print("".center(80, '*'))

    @time_method("Env. in Env. Energy")
    def get_env_in_env_energy(self):
        """Calculates the energy of dft-in-dft.

        This is unnecessary for the Total Embedding energy, however it
        may be a useful diagnostic tool.

        """
        print("".center(80, '*'))
        print("  Env-in-Env Calculation  ".center(80))
        print("".center(80, '*'))
        dm_env = self.get_emb_dmat()
        self.env_in_env_energy = self.fs_scf.energy_tot(dm=dm_env, h1e=self.hcore, vhf=self.emb_vhf)
        proj_e = 0.
        for i, sub in enumerate(self.subsystems):
            proj_e += (np.einsum('ij,ji', self.proj_pot[i][0], sub.env_dmat[0]) +
                       np.einsum('ij,ji', self.proj_pot[i][1], sub.env_dmat[1]))
        self.env_in_env_energy += proj_e
        print(f"Env-in-Env Energy:{self.env_in_env_energy:>62.8f}")
        #Approx spin from this paper: https://aip-scitation-org.ezp3.lib.umn.edu/doi/10.1063/1.468585
        if self.fs_unrestricted:
            rho_grid = self.fs_scf.grids
            alpha_dmat = dm_env[0]
            alpha_rho = self.fs_scf._numint.get_rho(self.mol, alpha_dmat, rho_grid,
                                                self.fs_scf.max_memory)
            #alpha_den = alpha_rho * rho_grid.weights
            alpha_den = alpha_rho
            beta_dmat = dm_env[1]
            beta_rho = self.fs_scf._numint.get_rho(self.mol, beta_dmat, rho_grid,
                                                self.fs_scf.max_memory)

            #beta_den = beta_rho * rho_grid.weights
            beta_den = beta_rho
            spin_diff = np.subtract(alpha_den,beta_den)
            neg_spin_diff = spin_diff.clip(max=0)
            spin_contam = np.dot(neg_spin_diff, rho_grid.weights)
            alpha,beta = self.mol.nelec
            s = (alpha-beta)/2.
            ss = s*(s+1) - spin_contam
            print(f"approx <S^2>:{ss:>67.8f}")
        print("".center(80, '*'))
        return self.env_in_env_energy

    def update_ro_fock(self):
        """Updates the full system fock  for restricted open shell matrix.

        Parameters
        ----------
        diis : bool
            Whether to use the diis method.
        """
        # Optimize: Rather than recalc. the full V, only calc. the V for changed densities.
        # get 2e matrix
        num_rank = self.mol.nao_nr()
        dmat = [np.zeros((num_rank, num_rank)), np.zeros((num_rank, num_rank))]
        s2s = self.sub2sup
        for i, sub in enumerate(self.subsystems):
            dmat[0][np.ix_(s2s[i], s2s[i])] += (sub.env_dmat[0])
            dmat[1][np.ix_(s2s[i], s2s[i])] += (sub.env_dmat[1])
        temp_fock = self.fs_scf.get_fock(h1e=self.hcore, dm=dmat)
        self.fock = [temp_fock, temp_fock]

        for i, sub in enumerate(self.subsystems):
            sub_fock_ro = self.fock[0][np.ix_(s2s[i], s2s[i])]
            sub_fock_0 = self.fock[0].focka[np.ix_(s2s[i], s2s[i])]
            sub_fock_1 = self.fock[0].fockb[np.ix_(s2s[i], s2s[i])]
            sub.emb_ro_fock = [sub_fock_ro, sub_fock_0, sub_fock_1]

        return True

    def update_fock(self, diis=True):
        """Updates the full system fock matrix.

        Parameters
        ----------
        diis : bool
            Whether to use the diis method.
        """

        # Optimize: Rather than recalc. the full V, only calc. the V for changed densities.
        # get 2e matrix
        num_rank = self.mol.nao_nr()
        dmat = [np.zeros((num_rank, num_rank)), np.zeros((num_rank, num_rank))]
        sub_unrestricted = False
        s2s = self.sub2sup
        for i, sub in enumerate(self.subsystems):
            if sub.unrestricted:
                sub_unrestricted = True
            dmat[0][np.ix_(s2s[i], s2s[i])] += (sub.env_dmat[0])
            dmat[1][np.ix_(s2s[i], s2s[i])] += (sub.env_dmat[1])

        if self.fs_unrestricted or sub_unrestricted:
            self.emb_vhf = self.os_scf.get_veff(self.mol, dmat)
            self.fock = self.os_scf.get_fock(h1e=self.hcore, vhf=self.emb_vhf, dm=dmat)
        elif self.mol.spin != 0:
            self.emb_vhf = self.fs_scf.get_veff(self.mol, dmat)
            temp_fock = self.fs_scf.get_fock(h1e=self.hcore, vhf=self.emb_vhf, dm=dmat)
            self.fock = [temp_fock, temp_fock]
        else:
            dmat = dmat[0] + dmat[1]
            self.emb_vhf = self.fs_scf.get_veff(self.mol, dmat)
            temp_fock = self.fs_scf.get_fock(h1e=self.hcore, vhf=self.emb_vhf, dm=dmat)
            self.fock = [temp_fock, temp_fock]

        if (not self.ft_diis is None) and diis:
            if self.fs_unrestricted or sub_unrestricted:
                new_fock = self.ft_diis.update(self.fock)
                self.fock[0] = new_fock[0]
                self.fock[1] = new_fock[1]
            else:
                new_fock = self.ft_diis.update(self.fock[0])
                self.fock[0] = new_fock
                self.fock[1] = new_fock

        #Add the external potential to each fock.
        self.fock[0] += self.ext_pot[0]
        self.fock[1] += self.ext_pot[1]
        for i, sub in enumerate(self.subsystems):
            sub_fock_0 = self.fock[0][np.ix_(s2s[i], s2s[i])]
            sub_fock_1 = self.fock[1][np.ix_(s2s[i], s2s[i])]
            sub.emb_fock = [sub_fock_0, sub_fock_1]

        return True

    def update_proj_pot(self):
        # currently updates all at once. Can mod. to update one subsystem; likely won't matter.
        """Updates the projection potential of the system.
        """
        s2s = self.sub2sup
        for i, sub_a in enumerate(self.subsystems):
            num_rank_a = sub_a.mol.nao_nr()
            proj_op = [np.zeros((num_rank_a, num_rank_a)), np.zeros((num_rank_a, num_rank_a))]

            # cycle over all other subsystems
            for j, sub_b in enumerate(self.subsystems):
                if j == i:
                    continue

                sub_b_dmat = sub_b.env_dmat
                smat_ab = self.smat[np.ix_(s2s[i], s2s[j])]
                smat_ba = self.smat[np.ix_(s2s[j], s2s[i])]

                # get mu-parameter projection operator
                if isinstance(self.proj_oper, (int, float)):
                    proj_op[0] += self.proj_oper * np.dot(smat_ab, np.dot(sub_b_dmat[0], smat_ba))
                    proj_op[1] += self.proj_oper * np.dot(smat_ab, np.dot(sub_b_dmat[1], smat_ba))

                elif self.proj_oper in ('huzinaga', 'huz'):
                    fock_ab = [None, None]
                    fock_ab[0] = self.fock[0][np.ix_(s2s[i], s2s[j])]
                    fock_ab[1] = self.fock[1][np.ix_(s2s[i], s2s[j])]
                    fock_den_smat = [None, None]
                    fock_den_smat[0] = np.dot(fock_ab[0], np.dot(sub_b_dmat[0], smat_ba))
                    fock_den_smat[1] = np.dot(fock_ab[1], np.dot(sub_b_dmat[1], smat_ba))
                    proj_op[0] += -1. * (fock_den_smat[0] + fock_den_smat[0].transpose())
                    proj_op[1] += -1. * (fock_den_smat[1] + fock_den_smat[1].transpose())

                elif self.proj_oper in ('huzinagafermi', 'huzfermi'):
                    fock_ab = [None, None]
                    fock_ab[0] = self.fock[0][np.ix_(s2s[i], s2s[j])]
                    fock_ab[1] = self.fock[1][np.ix_(s2s[i], s2s[j])]
                    #The max of the fermi energy
                    efermi = [None, None]
                    if self.ft_setfermi is None:
                        efermi[0] = max([fermi[0] for fermi in self.ft_fermi])
                        efermi[1] = max([fermi[1] for fermi in self.ft_fermi])
                    else:
                        efermi[0] = self.ft_setfermi
                        efermi[1] = self.ft_setfermi

                    fock_ab[0] -= smat_ab * efermi[0]
                    fock_ab[1] -= smat_ab * efermi[1]

                    fock_den_smat = [None, None]
                    fock_den_smat[0] = np.dot(fock_ab[0], np.dot(sub_b_dmat[0], smat_ba))
                    fock_den_smat[1] = np.dot(fock_ab[1], np.dot(sub_b_dmat[1], smat_ba))
                    proj_op[0] += -1. * (fock_den_smat[0] + fock_den_smat[0].transpose())
                    proj_op[1] += -1. * (fock_den_smat[1] + fock_den_smat[1].transpose())

            self.proj_pot[i] = proj_op.copy()

        return True

    def update_fock_proj_diis(self, iter_num):
        """Updates the fock matrix and the projection potential together
           using a diis algorithm. Then subdivided into subsystems for density
           relaxation. This only works in the absolutely localized basis."""

        fock = copy.copy(self.fock)
        fock = np.array(fock)
        num_rank = self.mol.nao_nr()
        dmat = [np.zeros((num_rank, num_rank)), np.zeros((num_rank, num_rank))]
        s2s = self.sub2sup
        sub_unrestricted = False
        diis_start_cycle = 10
        proj_energy = 0.
        for i, sub in enumerate(self.subsystems):
            if sub.unrestricted:
                sub_unrestricted = True
            fock[0][np.ix_(s2s[i], s2s[i])] += self.proj_pot[i][0]
            fock[1][np.ix_(s2s[i], s2s[i])] += self.proj_pot[i][1]
            dmat[0][np.ix_(s2s[i], s2s[i])] += (sub.env_dmat[0])
            dmat[1][np.ix_(s2s[i], s2s[i])] += (sub.env_dmat[1])
            proj_energy += (np.einsum('ij,ji', self.proj_pot[i][0], sub.env_dmat[0]) +
                            np.einsum('ij,ji', self.proj_pot[i][1], sub.env_dmat[1]))
                            
        #remove off diagonal elements of fock matrix
        #new_fock = np.zeros_like(fock)
        #for i, sub in enumerate(self.subsystems):
        #    new_fock[0][np.ix_(s2s[i], s2s[i])] = fock[0][np.ix_(s2s[i], s2s[i])]
        #    new_fock[1][np.ix_(s2s[i], s2s[i])] = fock[1][np.ix_(s2s[i], s2s[i])]

        #fock = new_fock

        if self.mol.spin == 0 and not(sub_unrestricted or self.fs_unrestricted):
            elec_dmat = dmat[0] + dmat[1]
        else:
            elec_dmat = copy.copy(dmat)

        
        elec_energy = self.fs_scf.energy_elec(dm=elec_dmat, h1e=self.hcore, vhf=self.emb_vhf)[0]
        elec_proj_energy = elec_energy + proj_energy

        if not(sub_unrestricted or self.fs_unrestricted):
            fock = (fock[0] + fock[1]) / 2.
            dmat = dmat[0] + dmat[1]

        if self.ft_diis_num == 2:
            fock = self.ft_diis.update(fock)
        elif self.ft_diis_num == 3:
            fock = self.ft_diis_2.update(fock)
            #temp_fock = self.ft_diis.update(self.smat, dmat, fock)
            #if iter_num > diis_start_cycle:
            #    fock = temp_fock
        #elif self.ft_diis_num > 3:
        #    temp_fock = self.ft_diis.update(self.smat, dmat, fock, elec_proj_energy)
        #    if iter_num > diis_start_cycle:
        #        fock = temp_fock

        if not(sub_unrestricted or self.fs_unrestricted):
            fock = [fock, fock]

        for i, sub in enumerate(self.subsystems):
            sub_fock_0 = fock[0][np.ix_(s2s[i], s2s[i])]
            sub_fock_1 = fock[1][np.ix_(s2s[i], s2s[i])]
            sub.emb_proj_fock = [sub_fock_0, sub_fock_1]

    def set_chkfile_index(self, index):
        """Sets the index of the checkfile
        """
        self.chkfile_index = str(index)
        for i in range(len(self.subsystems)):
            sub = self.subsystems[i]
            sub.chkfile_index = str(index) + "_" + str(i)

    #These should catch exceptions.
    def read_chkfile(self, filename=None):
        """Read the associated checkfile
        """
    # Need to make more robust. Handle errors and such.

        if filename is None:
            if self.filename is None:
                return False
            filename = os.path.splitext(self.filename)[0] + '.hdf5'
        assert(self.chkfile_index is not None), 'Need to set chkfile_index'

        chk_index = self.chkfile_index
        if os.path.isfile(filename):
            try:
                with h5py.File(filename, 'r') as h5_file:
                    supsys_coeff = h5_file[f'supersystem:{chk_index}/mo_coeff']
                    if self.mol.nao == supsys_coeff.shape[1]:
                        self.mo_coeff = supsys_coeff[:]
                        supsys_occ = h5_file[f'supersystem:{chk_index}/mo_occ']
                        self.mo_occ = supsys_occ[:]
                        supsys_energy = h5_file[f'supersystem:{chk_index}/mo_energy']
                        self.mo_energy = supsys_energy[:]
                        return True
                    print("chkfile improperly formatted".center(80))
                    return False
            except TypeError:
                print("chkfile improperly formatted".center(80))
                return False
        print("chkfile NOT found".center(80))
        return False

    def save_chkfile(self, filename=None):
        """Save the chkfile of the supersystem including subsystem elements.
        """
        # current plan is to save mo_coefficients, occupation vector, and energies.
        # becasue of how h5py works we need to check if none and save as the correct filetype (f)
        #Need to make more robust. Handle errors and such.

        # check if file exists.
        if filename is None:
            if self.filename is None:
                return False
            filename = os.path.splitext(self.filename)[0] + '.hdf5'
        assert(self.chkfile_index is not None), 'Need to set chkfile_index'

        chk_index = self.chkfile_index
        if os.path.isfile(filename):
            try:
                with h5py.File(filename, 'r+') as h5_file:
                    supsys_coeff = h5_file[f'supersystem:{chk_index}/mo_coeff']
                    supsys_coeff[...] = self.mo_coeff
                    supsys_occ = h5_file[f'supersystem:{chk_index}/mo_occ']
                    supsys_occ[...] = self.mo_occ
                    supsys_energy = h5_file[f'supersystem:{chk_index}/mo_energy']
                    supsys_energy[...] = self.mo_energy
            except TypeError:
                print("Overwriting existing chkfile".center(80))
                with h5py.File(filename, 'w') as h5_file:
                    sup_mol = h5_file.create_group(f'supersystem:{chk_index}')
                    sup_mol.create_dataset('mo_coeff', data=self.mo_coeff)
                    sup_mol.create_dataset('mo_occ', data=self.mo_occ)
                    sup_mol.create_dataset('mo_energy', data=self.mo_energy)
            except KeyError:
                print("Updating existing chkfile".center(80))
                with h5py.File(filename, 'a') as h5_file:
                    sup_mol = h5_file.create_group(f'supersystem:{chk_index}')
                    sup_mol.create_dataset('mo_coeff', data=self.mo_coeff)
                    sup_mol.create_dataset('mo_occ', data=self.mo_occ)
                    sup_mol.create_dataset('mo_energy', data=self.mo_energy)

        else:
            with h5py.File(filename, 'w') as h5_file:
                sup_mol = h5_file.create_group(f'supersystem:{chk_index}')
                sup_mol.create_dataset('mo_coeff', data=self.mo_coeff)
                sup_mol.create_dataset('mo_occ', data=self.mo_occ)
                sup_mol.create_dataset('mo_energy', data=self.mo_energy)

        return True


    def save_ft_density_file(self, filename=None, density=None):
        """Save the density of the freeze and thaw cycles
        """
        if filename is None:
            filename = self.filename
        if density is None:
            mat_rank = self.mol.nao_nr()
            dmat = [np.zeros((mat_rank, mat_rank)), np.zeros((mat_rank, mat_rank))]
            s2s = self.sub2sup
            for i, sub in enumerate(self.subsystems):
                if self.subsystems[i].unrestricted or self.subsystems[i].mol.spin != 0:
                    dmat[0][np.ix_(s2s[i], s2s[i])] += self.subsystems[i].get_dmat()[0]
                    dmat[1][np.ix_(s2s[i], s2s[i])] += self.subsystems[i].get_dmat()[1]
                else:
                    dmat[0][np.ix_(s2s[i], s2s[i])] += (self.subsystems[i].get_dmat()/2.)
                    dmat[1][np.ix_(s2s[i], s2s[i])] += (self.subsystems[i].get_dmat()/2.)
        print('Writing DFT-in-DFT Density'.center(80))
        if self.mol.spin != 0 or self.ft_unrestricted or self.fs_unrestricted:
            cubegen_fn = (os.path.splitext(filename)[0] + '_' +
                          self.chkfile_index +  '_ft_alpha.cube')
            cubegen.density(self.mol, cubegen_fn, dmat[0])
            cubegen_fn = (os.path.splitext(filename)[0] + '_' +
                          self.chkfile_index +  '_ft_beta.cube')
            cubegen.density(self.mol, cubegen_fn, dmat[1])
        else:
            cubegen_fn = (os.path.splitext(filename)[0] + '_' +
                          self.chkfile_index +  '_ft.cube')
            cubegen.density(self.mol, cubegen_fn, (dmat[0] + dmat[1]))

    def save_ft_spin_density_file(self, filename=None, density=None):
        """Saves the spin density of the freeze and thaw system

        Parameters
        ----------
        filename : str
            Density save file.
        density : array
            Density array.
        """

        if filename is None:
            filename = self.filename
        if density is None:
            mat_rank = self.mol.nao_nr()
            dmat = [np.zeros((mat_rank, mat_rank)), np.zeros((mat_rank, mat_rank))]
            s2s = self.sub2sup
            for i, sub in enumerate(self.subsystems):
                if sub.unrestricted or sub.mol.spin != 0:
                    dmat[0][np.ix_(s2s[i], s2s[i])] += sub.get_dmat()[0]
                    dmat[1][np.ix_(s2s[i], s2s[i])] += sub.get_dmat()[1]
                else:
                    dmat[0][np.ix_(s2s[i], s2s[i])] += (sub.get_dmat()/2.)
                    dmat[1][np.ix_(s2s[i], s2s[i])] += (sub.get_dmat()/2.)
        if self.mol.spin != 0 or self.ft_unrestricted or self.fs_unrestricted:
            print('Writing DFT-in-DFT Density'.center(80))
            cubegen_fn = (os.path.splitext(filename)[0] + '_' +
                          self.chkfile_index +  '_ft_spinden.cube')
            cubegen.density(self.mol, cubegen_fn, np.subtract(dmat[0], dmat[1]))
        else:
            print('Cannot write spin density for a closed shell system.'.center(80))


    @time_method("Freeze and Thaw")
    def freeze_and_thaw(self):
        """Performs the freeze and thaw subsystem density optimization
        """
        # Opt.: rather than recalc. vA use the existing fock and subtract out blocks dble counted.

        print("".center(80, '*'))
        print("Freeze-and-Thaw".center(80))
        print("".center(80, '*'))

        ft_err = 1.
        ft_iter = 0
        swap_diis = False
        while((ft_err > self.ft_conv) and (ft_iter < self.ft_cycles)):
            # cycle over subsystems
            ft_err = 0
            ft_iter += 1
            #Correct for DIIS
            # If fock only updates after cycling, then use python multiprocess todo simultaneously. python multiprocess may not be able to do what I want here. 
            if self.ft_diis_num == 2 or (self.ft_diis_num == 3 and swap_diis):
                self.update_fock(diis=False)
            else:
                self.update_fock()
            self.update_proj_pot()
            if self.ft_diis_num == 3 and swap_diis:
                self.update_fock_proj_diis(ft_iter)
            elif self.ft_diis_num == 2:
                self.update_fock_proj_diis(ft_iter)

            for i, sub in enumerate(self.subsystems):
                sub.proj_pot = self.proj_pot[i]
                ddm = sub.relax_sub_dmat(damp_param=self.ft_damp)
                proj_e = sub.get_env_proj_e() #When DIIS acceleration, this is not actually true because it is bound up in the acceleration.
                # print output to console.
                #print(f"iter:{ft_iter:>3d}:{i:<2d}  |ddm|:{ddm:12.6e}  |Tr[DP]|:{proj_e:12.6e}  |Fermi|:[{sub.fermi[0]},{sub.fermi[1]}]")
                print(f"iter:{ft_iter:>3d}:{i:<2d}               |ddm|:{ddm:12.6e}               |Tr[DP]|:{proj_e:12.6e}")
                ft_err += ddm
                self.ft_fermi[i] = sub.fermi

                #Check if need to update fock or proj pot.
                #if self.ft_updatefock > 0 and ((i + 1) % self.ft_updatefock) == 0:
                #    self.update_fock()
                #if self.ft_updateproj > 0 and ((i + 1) % self.ft_updateproj) == 0:
                #    self.update_proj_pot()

            if (ft_err < 1e-2):
                swap_diis = True
            
            self.save_chkfile()

        print("".center(80))
        self.is_ft_conv = True
        if ft_err > self.ft_conv:
            print("".center(80))
            print("Freeze-and-Thaw NOT converged".center(80))

        self.update_fock(diis=False)
        self.update_proj_pot()
        # print subsystem energies
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            subsystem.get_env_energy()
            print(f"Subsystem {i} Energy:{subsystem.env_energy:>61.8f}")
        print("".center(80))
        print("".center(80, '*'))

        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            if subsystem.save_density:
                print(f'Writing Subsystem {i} Env Density'.center(80))
                subsystem.save_density_file()

            if subsystem.save_spin_density:
                print(f'Writing Subsystem {i} Env Spin Density'.center(80))
                subsystem.save_spin_density_file()

            if subsystem.save_orbs:
                print(f'Writing Subsystem {i} Env orbitals'.center(80))
                subsystem.save_orbital_file()

        if self.ft_save_density:
            print('Writing Supersystem FT density'.center(80))
            self.save_ft_density_file()

        if self.ft_save_spin_density:
            print('Writing Supersystem FT spin density'.center(80))
            self.save_ft_spin_density_file()
