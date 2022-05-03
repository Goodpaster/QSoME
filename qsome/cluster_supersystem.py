""" A method to define a cluster supersystem
Daniel Graham
Dhabih V. Chulhai"""

import os
import time
import copy as copy
import numpy as np
import h5py

from functools import reduce

from pyscf import scf, dft, lib, lo, hessian
from pyscf.tools import cubegen, molden

from qsome import custom_diis, helpers
from qsome import custom_pyscf_methods
from qsome.helpers import concat_mols
from qsome.utilities import time_method
#TEMP
from qsome import cluster_subsystem


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
    ft_init_guess : str
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
    init_guess : str
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


    def __init__(self, subsystems, env_method, fs_scf_obj, env_order=1,
                 max_cycle=200, subcycles=1, conv_tol=1e-9, damp=0., 
                 diis_num=2, update_fock=1, update_proj=1, init_guess='supmol',
                 unrestricted=False, proj_oper='huz', excited_relax=False,
                 filename=None, chkfile_index=0, **kwargs):

        """
        Parameters
        ----------
        subsystems : list
            List of SubSystem objects.
        fs_method : str
            Defines the supersystem method.
        env_order : int 
            Orders the supersystems within the full embedding (default is 1)
        fs_param_dict : dict
            Defines parameters for full system environment correction calculation
        emb_param_dict : dict
            Defines parameters for embedding procedure
        filename : str
            Path to input file. General location for file operations.
            Full system core hamiltonian matrix.
        """

        self.subsystems = subsystems
        self.env_order = env_order
        self.env_method = env_method
        self.fs_scf_obj = fs_scf_obj
        self.max_cycle = max_cycle
        self.subcycles = subcycles
        self.conv_tol = conv_tol
        self.damp = damp
        self.diis_num = diis_num
        self.fock_subcycles = update_fock
        self.proj_subcycles = update_proj
        self.init_guess = init_guess
        if self.fs_scf_obj.init_guess == 'submol':
            self.init_guess = 'submol'
        self.unrestricted = unrestricted
        if hasattr(fs_scf_obj, 'unrestricted'):
            self.unrestricted = fs_scf_obj.unrestricted
        self.proj_oper = proj_oper
        self.excited_relax = excited_relax
        self.filename = filename

        # freeze and thaw settings
        if self.excited_relax:
            self.excited_dict = {}
            if excited_dict in kwargs:
                self.excited_dict = kwargs.pop('excited_dict')

        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.set_chkfile_index(chkfile_index)
        # Densities are stored separately to allow for alpha and beta.
        self.is_ft_conv = False
        self.ext_pot = np.array([0., 0.])
        self.mol = self.fs_scf_obj.mol
        self.env_in_env_scf = None
        self.__gen_sub2sup()
        self.__atm_sub2sup()
        self.__init_env_in_env_scf()

        self.smat = self.env_in_env_scf.get_ovlp()
        self.mo_coeff = np.array([np.zeros_like(self.smat), np.zeros_like(self.smat)])
        self.local_mo_coeff = np.array([None, None])
        self.mo_occ = np.array([np.zeros_like(self.smat[0]),
                                np.zeros_like(self.smat[0])])
        self.mo_energy = self.mo_occ.copy()
        self.hcore = self.env_in_env_scf.get_hcore()
        self.fock = [None, None]
        self.emb_vhf = None
        self.proj_pot = [np.array([np.zeros_like(sub.env_scf.get_hcore()), np.zeros_like(sub.env_scf.get_hcore())]) for sub in self.subsystems]
        #DIIS Stuff
        self.sub_diis = [np.array([lib.diis.DIIS(), lib.diis.DIIS()]) for sub in self.subsystems]
        self.fs_energy = None
        self.env_in_env_energy = None
        self.fs_nuc_grad = None

        self.fs_dmat = None
        self.emb_dmat = None

        self.emb_diis = None
        if self.diis_num == 1:
            self.emb_diis = lib.diis.DIIS()
        elif self.diis_num == 2:
            self.emb_diis = lib.diis.DIIS()
        elif self.diis_num == 3:
            self.emb_diis = lib.diis.DIIS()
            self.emb_diis_2 = lib.diis.DIIS()
        elif self.diis_num == 4:
            self.emb_diis = custom_diis.ADIIS()
            self.emb_diis_2 = lib.diis.DIIS()
        elif self.diis_num == 5:
            self.emb_diis = custom_diis.ADIIS_CDIIS()
            self.emb_diis_2 = lib.diis.DIIS()
        elif self.diis_num == 6:
            self.emb_diis = custom_diis.ADIIS_DIIS()
            self.emb_diis_2 = lib.diis.DIIS()

        if self.emb_diis:
            self.emb_diis.space = 15

        self.ft_fermi = [np.array([0., 0.]) for sub in subsystems]
        self.base_xterms = None
        self.b_terms = None
        self.ao_grad = None
        self.fa1emb_ao = None

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

        sub2sup = [None for i in enumerate(subsystems)]
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

    def __atm_sub2sup(self):
        """Creates a list relating the index of sub atoms to sup atoms"""
        atm_sub2sup = [None for i in enumerate(self.subsystems)]
        for i, sub in enumerate(self.subsystems):
            atm_sub2sup[i] = []
            for j in range(sub.mol.natm):
                match = False
                c_1 = sub.mol.atom_coord(j)
                for k in range(self.mol.natm):
                    c_2 = self.mol.atom_coord(k)
                    dist = np.dot(c_1 - c_2, c_1 - c_2)
                    if dist < 0.0001:
                        match = True
                        atm_sub2sup[i].append(k)
                assert match, 'no atom match!'
        self.atm_sub2sup = atm_sub2sup
        return True


    def __init_env_in_env_scf(self):
        """Initialize the supersystem pyscf SCF object using given settings."""

        mol = self.mol

        if self.unrestricted:
            if 'hf' in self.env_method:
                scf_obj = scf.UHF(mol)
                u_scf_obj = scf_obj
            else:
                scf_obj = scf.UKS(mol)
                scf_obj.xc = self.env_method
                u_scf_obj = scf_obj

        elif mol.spin != 0:
            if 'hf' in self.env_method:
                scf_obj = scf.ROHF(mol)
                u_scf_obj = scf.UHF(mol)
            else:
                scf_obj = scf.ROKS(mol)
                u_scf_obj = scf.UKS(mol)
                scf_obj.xc = self.env_method
                u_scf_obj.xc = self.env_method
        else:
            if 'hf' in self.env_method:
                scf_obj = scf.RHF(mol)
                u_scf_obj = scf.UHF(mol)
            else:
                scf_obj = scf.RKS(mol)
                u_scf_obj = scf.UKS(mol)
                scf_obj.xc = self.env_method
                u_scf_obj.xc = self.env_method

        env_in_env_scf = scf_obj
        if 'hf' not in self.env_method:
            scf_obj.grids = copy.copy(self.fs_scf_obj.grids)
            u_scf_obj.grids = copy.copy(self.fs_scf_obj.grids)
            scf_obj.small_rho_cutoff = self.fs_scf_obj.small_rho_cutoff
            u_scf_obj.small_rho_cutoff = self.fs_scf_obj.small_rho_cutoff
            scf_obj.grids = copy.copy(self.fs_scf_obj.grids)
            u_scf_obj.grids = copy.copy(self.fs_scf_obj.grids)

        self.env_in_env_scf = scf_obj
        self.os_env_in_env_scf = u_scf_obj
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
            fs_scf = self.fs_scf_obj
        if subsystems is None:
            subsystems = self.subsystems

        print("".center(80, '*'))
        print("  Generate Initial System Densities  ".center(80))
        print("".center(80, '*'))
        # If supersystem dft should be read from chkfile.
        super_chk = (fs_scf.init_guess == 'chk')
        s2s = self.sub2sup

        # Initialize supersystem density.
        if self.init_guess == 'supmol':
            self.get_supersystem_energy(readchk=super_chk)

        elif self.init_guess == 'rosupmol':
            rodmat = self.get_init_ro_supersystem_dmat()

        for i, subsystem in enumerate(subsystems):
            sub_dmat = [0., 0.]
            fs_unrestricted = (hasattr(fs_scf, 'unrestricted') and fs_scf.unrestricted)
            subsystem.fullsys_cs = not (fs_unrestricted or self.mol.spin != 0)
            # Ensure same gridpoints and rho_cutoff for all systems
            # This grid causes issues with gradient.
            if not 'hf' in subsystem.env_method:
                subsystem.env_scf.grids = fs_scf.grids
                subsystem.env_scf.small_rho_cutoff = fs_scf.small_rho_cutoff
                subsystem.env_scf.grids = fs_scf.grids

            sub_guess = subsystem.env_init_guess
            if sub_guess is None:
                sub_guess = self.init_guess
            if sub_guess is None:
                sub_guess = 'chk'
            subsystem.filename = self.filename
            subsystem.init_density(init_guess=sub_guess)
            if sub_guess != 'supmol':
                sub_guess = subsystem.env_init_guess
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
        if not (self.unrestricted or self.mol.spin != 0):
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
    def get_supersystem_energy(self, scf_obj=None, readchk=False,
                               local_orbs=False):
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
                scf_obj = self.fs_scf_obj
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
                if (hasattr(scf_obj, 'unrestricted') and scf_obj.unrestricted) or scf_obj.mol.spin != 0:
                    if hasattr(scf_obj, 'use_fast_newton') and scf_obj.use_fast_newton:
                        scf_obj.fast_newton(dm0=ft_dmat)
                    else:
                        scf_obj.scf(dm0=ft_dmat)
                else:
                    if hasattr(scf_obj, 'use_fast_newton') and scf_obj.use_fast_newton:
                        scf_obj.fast_newwton(dm0=(ft_dmat[0] + ft_dmat[1]))
                    else:
                        scf_obj.scf(dm0=(ft_dmat[0] + ft_dmat[1]))
            elif readchk:
                if (hasattr(scf_obj, 'unrestricted') and scf_obj.unrestricted) or scf_obj.mol.spin != 0:
                    init_guess = scf_obj.make_rdm1(mo_coeff=self.mo_coeff,
                                                   mo_occ=self.mo_occ)
                else:
                    init_guess = scf_obj.make_rdm1(mo_coeff=self.mo_coeff[0],
                                                   mo_occ=(self.mo_occ[0] +
                                                           self.mo_occ[1]))

                scf_obj.scf(dm0=(init_guess))
            else:
                scf_obj.scf()

            #DO stability analysis
            if hasattr(scf_obj, 'stability_analysis'):
                if scf_obj.stability_analysis == 'external':
                    new_mos = scf_obj.stability(external=True)[0]
                else:
                    new_mos = scf_obj.stability()[0]
                    if not scf_obj.converged:
                        print ("FS SCF NOT converged. Performing stability analysis.")
                        new_dm = scf_obj.make_rdm1(new_mos, scf_obj.mo_occ)
                        scf_obj.scf(dm0=new_dm)
                        scf_obj.stability()

            self.fs_dmat = scf_obj.make_rdm1()

            if hasattr(scf_obj, 'unrestricted') and scf_obj.unrestricted:
                rho_grid = self.fs_scf_obj.grids
                alpha_dmat = self.fs_dmat[0]
                alpha_rho = scf_obj._numint.get_rho(self.mol, alpha_dmat, rho_grid,
                                                    self.fs_scf_obj.max_memory)
                #alpha_den = alpha_rho * rho_grid.weights
                alpha_den = alpha_rho
                beta_dmat = self.fs_dmat[1]
                beta_rho = scf_obj._numint.get_rho(self.mol, beta_dmat, rho_grid,
                                                    self.fs_scf_obj.max_memory)

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
                self.mo_coeff = [self.fs_scf_obj.mo_coeff, self.fs_scf_obj.mo_coeff]
                self.mo_occ = [self.fs_scf_obj.mo_occ/2., self.fs_scf_obj.mo_occ/2.]
                self.mo_energy = [self.fs_scf_obj.mo_energy, self.fs_scf_obj.mo_energy]

            self.save_chkfile()
            self.fs_energy = scf_obj.energy_tot()
            print("".center(80, '*'))
            if (hasattr(self.fs_scf_obj, 'save_density') and
                    self.fs_scf_obj.save_density):
                self.save_fs_density_file()

            if (hasattr(self.fs_scf_obj, 'save_spin_density') and
                    self.fs_scf_obj.save_spin_density):
                self.save_fs_spin_density_file()

            if (hasattr(self.fs_scf_obj, 'save_orbs') and 
                    self.fs_scf_obj.save_orbs):
                self.save_fs_orbital_file()

            self.fs_scf = scf_obj
        print(f"KS-DFT  Energy:{self.fs_energy:>65.8f}  ")
        if local_orbs and self.local_mo_coeff[0] is None:
            #Can also use ER and Boys
            if ((hasattr(scf_obj, 'unrestricted') and scf_obj.unrestricted)
                or scf_obj.mol.spin != 0):
                nelec_a = scf_obj.mol.nelec[0]
                self.local_mo_coeff[0] = lo.ER(scf_obj.mol, self.mo_coeff[0][:, :nelec_a]).kernel()
                nelec_b = scf_obj.mol.nelec[1]
                self.local_mo_coeff[1] = lo.ER(scf_obj.mol, self.mo_coeff[1][:, :nelec_b]).kernel()
            else:
                nelec_a = scf_obj.mol.nelec[0]
                self.local_mo_coeff[0] = lo.ER(scf_obj.mol, self.mo_coeff[0][:, :nelec_a]).kernel()
                self.local_mo_coeff[1] = self.local_mo_coeff[0]


        #if self.fs_excited:
            #Run the Excited state calculation on the self.fs_scf object.  
        #    pass

        return self.fs_energy

    def get_supersystem_nuc_grad(self):
        """Calculates the nuclear gradient of the supersystem KS-DFT calculation"""

        if self.fs_energy is None or not self.fs_scf.converged:
            self.get_supersystem_energy()
            if not self.fs_scf.converged:
                print(f"KS-DFT NOT CONVERGED. Cannot calculate gradients")
        self.fs_nuc_grad_obj = self.fs_scf.nuc_grad_method()
        self.fs_nuc_grad = self.fs_nuc_grad_obj.kernel()

        #true_coulomb = self.fs_nuc_grad_obj.get_jk()[0]

        #grad_2e_int = self.mol.intor('int2e_ip1')
        #grad_2e_int2 = self.mol.intor('int2e_ip2')
        #fs_dmat = self.fs_dmat[0] + self.fs_dmat[1]
        #test_coulomb = np.einsum('xijkl,lk->xij', grad_2e_int, fs_dmat) * -1
        #test_coulomb_2 = np.einsum('xijkl,lk->xij', grad_2e_int2, fs_dmat) * -1
        #test_coulomb_2[0] = test_coulomb_2[0].transpose()
        #test_coulomb_2[1] = test_coulomb_2[1].transpose()
        #test_coulomb_2[2] = test_coulomb_2[2].transpose()

        #aoslices = self.mol.aoslice_by_atom()
        #p0,p1 = aoslices [0,2:]
        #x_grad = np.einsum('xij,ij->x', test_coulomb[:,p0:p1], fs_dmat[p0:p1])
        #x_grad_2 = np.einsum('xij,ij->x', test_coulomb_2[:,p0:p1], fs_dmat.transpose()[p0:p1])
        #print (x_grad)
        #print (x_grad_2)


        #bas_start, bas_end, ao_start, ao_end = self.mol.aoslice_by_atom()[0]
        #eri1 = self.mol.intor('int2e_ip1_sph', shls_slice=(bas_start, bas_end,
        #                                      0, self.mol.nbas,
        #                                      0, self.mol.nbas,
        #                                      0, self.mol.nbas))

        #print (true_coulomb.shape)
        ##https://github.com/pyscf/pyscf/blob/master/examples/gto/20-ao_integrals.py
        #print (eri1.shape)



        #veff = self.fs_scf.get_veff(self.mol, fs_dmat)
        #hcore_e = np.einsum('ij,ji->', self.hcore, fs_dmat)

        #hcore_deriv = self.fs_nuc_grad_obj.hcore_generator(self.mol)

        #from pyscf import hessian
        #hess_obj = hessian.rhf.Hessian(self.fs_scf)
        #mo_coeff = self.fs_scf.mo_coeff
        #mo_occ = self.fs_scf.mo_occ
        #mo_en = self.fs_scf.mo_energy
        #h1ao = hess_obj.make_h1(mo_coeff, mo_occ)
        #mo_coeff1, mo_en1 = hess_obj.solve_mo1(mo_en, mo_coeff, mo_occ, h1ao)

        #mocc = self.fs_scf.mo_coeff[:,self.fs_scf.mo_occ>0]

        #s1 = self.fs_nuc_grad_obj.get_ovlp(self.mol)
        #dme0 = self.fs_nuc_grad_obj.make_rdm1e(self.fs_scf.mo_energy, self.fs_scf.mo_coeff, self.fs_scf.mo_occ)

        #aoslices = self.mol.aoslice_by_atom()
        #print ('fs dmat')
        #print (fs_dmat)
        #hcore_weighted = np.dot(np.dot(fs_dmat.T, self.fs_scf_obj.get_hcore()), fs_dmat) * 0.5
        #hcore_en = np.dot(np.dot(fs_dmat.T, self.fs_scf_obj.get_hcore()), fs_dmat)
        #hcore_emol = np.diag(hcore_en)
        #hcore_dme0 = self.fs_nuc_grad_obj.make_rdm1e(hcore_emol, self.fs_scf.mo_coeff, self.fs_scf.mo_occ)
        #veff_weighted = np.dot(np.dot(fs_dmat.T, self.fs_scf_obj.get_veff()), fs_dmat) * 0.5
        #for i in range(self.mol.natm):
        #    print (self.mol._atom[i])
        #    p0,p1 = aoslices [i,2:]
        #    hcore_grad_pot = hcore_deriv(i)
        #    hcore_grad = np.einsum('xij,ij->x', hcore_grad_pot, fs_dmat)
        #    hcore_grad_2 = np.einsum('xij,ij->x', hcore_grad_pot, fs_dmat)
        #    dm1 = np.einsum('ypi,qi->ypq', mo_coeff1[i], mocc) * 4.
        #    hcore_grad += np.einsum('xij,ij->x', dm1, self.hcore)
        #    print (hcore_grad)
        #    hcore_grad_weighted = np.einsum('xij,ij->x', s1[:,p0:p1], hcore_dme0[p0:p1]) * -2.
        #    hcore_grad_weighted += hcore_grad_2


        #print ('weighted_densityMat')
        #print (dme0)
        #print (np.dot(np.dot(fs_dmat.T, self.fs_scf_obj.get_fock()), fs_dmat) * 0.5)
        #inv_fock = np.linalg.inv(np.dot(np.dot(self.fs_scf_obj.mo_coeff.T, self.fs_scf_obj.get_fock()), self.fs_scf_obj.mo_coeff))
        #ddm = np.dot(inv_fock, np.dot(s1[2], dme0) * -2.)
        #print ('ddm')
        #print (ddm)
        #print ('true ddm')
        #print (dm1)
        #print ('hcore_grad')
        #print (hcore_grad)
        #print ('hcore_en')
        #print (np.einsum('ij,ij->', fs_dmat, self.fs_scf_obj.get_hcore()))
        ##print (hcore_grad_weighted)

        ##print ('weighted_den')
        ##hcore_en = np.dot(np.dot(self.fs_scf_obj.mo_coeff.T, self.hcore), self.fs_scf_obj.mo_coeff)
        ##print (hcore_en)
        ##veff_en = np.dot(np.dot(self.fs_scf_obj.mo_coeff.T, self.fs_scf_obj.get_veff()), self.fs_scf_obj.mo_coeff)
        ##print (veff_en)
        ##print (hcore_en + veff_en)

        ##print (np.dot(np.dot(self.fs_scf_obj.get_fock(), self.fs_scf_obj.mo_coeff), self.smat))

        #print (ADS)
        # 


        ##mocc = mo_coeff[:,mo_occ>0]
        ##dm1_mo = np.einsum('ypi,qi->ypq', mo_coeff1[2], mocc)
        ##dm1 = dm1_mo[2]
        ##print ("FS DMAT")
        ##print (self.fs_scf.mo_coeff)
        ##print (mo_coeff1[2])
        ##new_fs_dmat = np.dot(fs_dmat, self.smat)
        ##print (new_fs_dmat[0,0])
        ##print (new_fs_dmat[0,1])
        ##print (new_fs_dmat[0,3])
        ##print (new_fs_dmat[0,4])
        ##print (new_fs_dmat[0,5])
        ##print (new_fs_dmat[0,6])
        ##print (new_fs_dmat[0,7])
        ##print (new_fs_dmat[0,8])
        ##print (new_fs_dmat[-1,-1])
        ##print ("DM GRAD")
        ##print (dm1[0,0])
        ##print (dm1[0,1])
        ##print (dm1[0,3])
        ##print (dm1[0,4])
        ##print (dm1[0,5])
        ##print (dm1[0,6])
        ##print (dm1[0,7])
        ##print (dm1[0,8])
        ##print (dm1[-1,-1])
        return self.fs_nuc_grad

    def save_fs_density_file(self, filename=None, density=None):
        """Save the electron density of the full system KS-DFT
        """
        if filename is None:
            filename = self.filename
        if density is None:
            density = self.fs_dmat

        print('Writing Full System Density'.center(80))
        if self.mol.spin != 0 or (hasattr(self.fs_scf_obj, 'unrestricted') and self.fs_scf_obj.unrestricted):
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

        if self.mol.spin != 0 or (hasattr(self.fs_scf_obj, 'unrestricted') and self.fs_scf_obj.unrestricted):
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
            scf_obj = self.fs_scf_obj
        if mo_occ is None:
            mo_occ = scf_obj.mo_occ
        if mo_coeff is None:
            mo_coeff = scf_obj.mo_coeff
        if mo_energy is None:
            mo_energy = scf_obj.mo_energy
        if not (hasattr(self.fs_scf_obj, 'unrestricted') and self.fs_scf_obj.unrestricted):
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
        if self.mol.spin != 0 and not ((hasattr(self.fs_scf_obj, 'unrestricted') and self.fs_scf_obj.unrestricted)):
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
        self.env_in_env_energy = self.env_in_env_scf.energy_tot(dm=dm_env, h1e=self.hcore, vhf=self.emb_vhf)
        proj_e = 0.
        for i, sub in enumerate(self.subsystems):
            proj_e += (np.einsum('ij,ji', self.proj_pot[i][0], sub.env_dmat[0]) +
                       np.einsum('ij,ji', self.proj_pot[i][1], sub.env_dmat[1]))
        self.env_in_env_energy += proj_e
        print(f"Env-in-Env Energy:{self.env_in_env_energy:>62.8f}")
        #Approx spin from this paper: https://aip-scitation-org.ezp3.lib.umn.edu/doi/10.1063/1.468585
        if (hasattr(self.fs_scf_obj, 'unrestricted') and self.fs_scf_obj.unrestricted):
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
        temp_fock = self.env_in_env_scf.get_fock(h1e=self.hcore, dm=dmat)
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

        if self.unrestricted or sub_unrestricted:
            self.emb_vhf = self.os_env_in_env_scf.get_veff(self.mol, dmat)
            self.fock = self.os_env_in_env_scf.get_fock(h1e=self.hcore, vhf=self.emb_vhf, dm=dmat)
        elif self.mol.spin != 0:
            self.emb_vhf = self.env_in_env_scf.get_veff(self.mol, dmat)
            temp_fock = self.env_in_env_scf.get_fock(h1e=self.hcore, vhf=self.emb_vhf, dm=dmat)
            self.fock = [temp_fock, temp_fock]
        else:
            dmat = dmat[0] + dmat[1]
            self.emb_vhf = self.env_in_env_scf.get_veff(self.mol, dmat)
            temp_fock = self.env_in_env_scf.get_fock(h1e=self.hcore, vhf=self.emb_vhf, dm=dmat)
            #TEMP
            #temp_fock = copy.copy(self.hcore) + copy.copy(self.env_in_env_scf.get_jk(self.mol, dmat)[0])
            #for i, sub in enumerate(self.subsystems):
            #    temp_fock[np.ix_(s2s[i], s2s[i])] += sub.env_scf.get_veff(sub.mol, sub.get_dmat())
            self.fock = [temp_fock, temp_fock]

        if self.emb_diis and diis:
            if self.unrestricted or sub_unrestricted:
                new_fock = self.emb_diis.update(self.fock)
                self.fock[0] = new_fock[0]
                self.fock[1] = new_fock[1]
            else:
                new_fock = self.emb_diis.update(self.fock[0])
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
                        efermi[0] = self.set_fermi
                        efermi[1] = self.set_fermi

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
                            
        if self.mol.spin == 0 and not(sub_unrestricted or self.unrestricted):
            elec_dmat = dmat[0] + dmat[1]
        else:
            elec_dmat = copy.copy(dmat)

        
        elec_energy = self.env_in_env_scf.energy_elec(dm=elec_dmat, h1e=self.hcore, vhf=self.emb_vhf)[0]
        elec_proj_energy = elec_energy + proj_energy

        if not(sub_unrestricted or self.unrestricted):
            fock = (fock[0] + fock[1]) / 2.
            dmat = dmat[0] + dmat[1]

        if self.diis_num == 2:
            fock = self.emb_diis.update(fock)
        elif self.diis_num == 3:
            fock = self.emb_diis_2.update(fock)
            #temp_fock = self.ft_diis.update(self.smat, dmat, fock)
            #if iter_num > diis_start_cycle:
            #    fock = temp_fock
        elif self.diis_num == 4:
            fock = self.emb_diis.update(self.smat, dmat, fock, elec_proj_energy)
        #    if iter_num > diis_start_cycle:
        #        fock = temp_fock

        elif self.diis_num > 4:
            fock = self.emb_diis.update(self.smat, dmat, fock, elec_proj_energy, s2s)

        if not(sub_unrestricted or self.unrestricted):
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
        if self.mol.spin != 0 or self.unrestricted:
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
        if self.mol.spin != 0 or self.unrestricted:
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
        while((ft_err > self.conv_tol) and (ft_iter < self.max_cycle)):
            # cycle over subsystems
            ft_err = 0
            ft_iter += 1
            #Correct for DIIS
            # If fock only updates after cycling, then use python multiprocess todo simultaneously. python multiprocess may not be able to do what I want here. 
            if self.diis_num == 2 or (self.diis_num == 3 and swap_diis) or self.diis_num > 3:
                self.update_fock(diis=False)
            else:
                self.update_fock()
            self.update_proj_pot()
            if self.diis_num == 3 and swap_diis:
                self.update_fock_proj_diis(ft_iter)
            elif self.diis_num >= 2:
                self.update_fock_proj_diis(ft_iter)

            for i, sub in enumerate(self.subsystems):
                sub.proj_pot = self.proj_pot[i]
                ddm = sub.relax_sub_dmat(damp_param=self.damp)
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
        if ft_err > self.conv_tol:
            print("".center(80))
            print("Freeze-and-Thaw NOT converged".center(80))

        self.update_fock(diis=False)
        self.update_proj_pot()
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

        if hasattr(self, 'save_density') and self.save_density:
            print('Writing Supersystem FT density'.center(80))
            self.save_ft_density_file()

        if hasattr(self, 'save_spin_density') and self.save_spin_density:
            print('Writing Supersystem FT spin density'.center(80))
            self.save_ft_spin_density_file()

    def gen_proj_response(self, sub_index, alt_sub_index):
        s2s = self.sub2sup
        hermi = 0
        if isinstance(self.fs_scf_obj, (dft.rks.RKS, dft.roks.ROKS, dft.uks.UKS)):
            pass
        else:
            if (self.unrestricted or self.mol.spin != 0):
                def vind(dm1):
                    v1 = np.zeros_like(dm1)
                    dm1_s = np.zeros_like(self.get_emb_dmat())
                    dm1_s[0][np.ix_(s2s[sub_index], s2s[alt_sub_index])] += np.dot(dm1[0], self.smat[np.ix_(s2s[sub_index], s2s[alt_sub_index])])
                    dm1_s[1][np.ix_(s2s[sub_index], s2s[alt_sub_index])] += np.dot(dm1[1], self.smat[np.ix_(s2s[sub_index], s2s[alt_sub_index])])
                    vj, vk = self.fs_scf_obj.get_jk(self.mol, dm1_s, hermi=hermi)
                    v_full = vj[0] + vj[1] - vk
                    v1[0] += v_full[0][np.ix_(s2s[sub_index], s2s[sub_index])]
                    v1[1] += v_full[0][np.ix_(s2s[sub_index], s2s[sub_index])]
                    return v1

            else:
                sub = self.subsystems[sub_index]
                alt_sub = self.subsystems[alt_sub_index]
                if sub_index == alt_sub_index:
                    def proj_response(dm1):
                        v1 = np.zeros_like(dm1)
                        for k, diff_sub in enumerate(self.subsystems):
                            if k != alt_sub_index:
                                dm2 = diff_sub.get_dmat()
                                dm1_s_dm2 = np.zeros_like(self.get_emb_dmat())
                                dm1_s_dm2[np.ix_(s2s[alt_sub_index], s2s[k])] += np.dot(dm1, np.dot(self.smat[np.ix_(s2s[alt_sub_index], s2s[k])], dm2))
                                v_full = self.fs_scf_obj.get_veff(self.mol, dm1_s_dm2, hermi=0)#, hermi=hermi)
                                v_full_sub = v_full[np.ix_(s2s[sub_index], s2s[sub_index])]
                                v1 += v_full_sub + v_full_sub.T
                                v1 *= 0.5
                        return v1
                else:
                    def proj_response(dm1):
                        v1 = np.zeros_like(sub.get_dmat())
                        for k, diff_sub in enumerate(self.subsystems):
                            if k != alt_sub_index:
                                dm2 = diff_sub.get_dmat()
                                dm1_s_dm2 = np.zeros_like(self.get_emb_dmat())
                                dm1_s_dm2[np.ix_(s2s[alt_sub_index], s2s[k])] += np.dot(dm1, np.dot(self.smat[np.ix_(s2s[alt_sub_index], s2s[k])], dm2))
                                v_full = self.fs_scf_obj.get_veff(self.mol, dm1_s_dm2, hermi=0)#, hermi=hermi)
                                v_full_sub = v_full[np.ix_(s2s[sub_index], s2s[sub_index])]
                                v1 += v_full_sub + v_full_sub.T
                                v1 *= 0.5
                        fds_1 = np.dot(self.smat[np.ix_(s2s[sub_index], s2s[alt_sub_index])], np.dot(dm1, self.fock[0][np.ix_(s2s[alt_sub_index], s2s[sub_index])]))
                        v1 += (fds_1 + fds_1.T) * 0.5
                        return v1

        return proj_response

    def gen_sub_response(self, sub_index, alt_sub_index):
        #Generate the response matrix for each subsystem.
        proj_response = self.gen_proj_response(sub_index, alt_sub_index)
        s2s = self.sub2sup
        if isinstance(self.fs_scf_obj, (dft.rks.RKS, dft.roks.ROKS, dft.uks.UKS)):
            if (self.unrestricted or self.mol.spin != 0):
                pass
            else:
                pass
        else:
            if (self.unrestricted or self.mol.spin != 0):
                pass
            else:
                sub = self.subsystems[sub_index]
                alt_sub = self.subsystems[alt_sub_index]
                if sub_index == alt_sub_index:
                    sub_vresp_term = sub.env_scf.gen_response(sub.env_mo_coeff[0], sub.env_mo_occ[0]+sub.env_mo_occ[1], hermi=1)
                else:
                    fs_emb_mo_coeff = np.zeros_like(self.get_emb_dmat())
                    fs_emb_mo_occ = np.zeros((self.get_emb_dmat().shape[0]))
                    fs_emb_mo_coeff[np.ix_(s2s[alt_sub_index],s2s[alt_sub_index])] += alt_sub.env_mo_coeff[0]
                    fs_emb_mo_occ[np.ix_(s2s[alt_sub_index])] += alt_sub.env_mo_occ[0]
                    fs_emb_mo_occ[np.ix_(s2s[alt_sub_index])] += alt_sub.env_mo_occ[1]
                    sup_vresp_term = self.fs_scf_obj.gen_response(fs_emb_mo_coeff, fs_emb_mo_occ, hermi=0)
                    def sub_term(z1):
                        z_full = np.zeros_like(self.get_emb_dmat())
                        z_full[np.ix_(s2s[alt_sub_index],s2s[alt_sub_index])] += z1
                        subsys_term = sup_vresp_term(z_full)[np.ix_(s2s[sub_index], s2s[sub_index])]
                        return subsys_term
                    sub_vresp_term = sub_term
                def sub_emb_vresp_term(z1):
                    return sub_vresp_term(z1) - proj_response(z1)

        return sub_emb_vresp_term

                

    def gen_sub_vind(self, sub_index, alt_sub_index):
        #Generate the induced potential for each subsystem
        sub_response = self.gen_sub_response(sub_index, alt_sub_index)
        if isinstance(self.fs_scf_obj, (dft.rks.RKS, dft.roks.ROKS, dft.uks.UKS)):
            if (self.unrestricted or self.mol.spin != 0):
                pass
            else:
                pass
        else:
            if (self.unrestricted or self.mol.spin != 0):
                pass
            else:
                sub = self.subsystems[sub_index]
                alt_sub = self.subsystems[alt_sub_index]
                nao, nmo = sub.env_mo_coeff[0].shape
                alt_nao, alt_nmo = alt_sub.env_mo_coeff[0].shape
                mocc = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]>0]
                alt_mocc = alt_sub.env_mo_coeff[0][:,alt_sub.env_mo_occ[0]>0]
                vir = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]==0]
                alt_vir = alt_sub.env_mo_coeff[0][:,alt_sub.env_mo_occ[0]==0]
                nocc = mocc.shape[1]
                alt_nocc = alt_mocc.shape[1]
                nvir = vir.shape[1]
                alt_nvir = alt_vir.shape[1]
                def fx(mo1):
                    mo1 = mo1.reshape(alt_nvir, alt_nocc)
                    dm1 = np.empty((alt_nao,alt_nao))
                    dm = np.dot(alt_vir, np.dot(mo1*2., alt_mocc.T))
                    dm1 = dm + dm.T
                    v1 = sub_response(dm1)
                    v1vo = np.dot(vir.T, np.dot(v1, mocc))
                    return v1vo
        return fx

               
    def solve_zvec(self):
        zvec_0 = []
        zvec_1 = []
        s2s = self.sub2sup
        x_terms = []
        if (self.unrestricted or self.mol.spin != 0):
            for i, sub in enumerate(self.subsystems):
                occidxa = sub.env_mo_occ[0] > 0
                occidxb = sub.env_mo_occ[0] > 0
                viridxa = ~occidxa
                viridxb = ~occidxb
                nocca = np.count_nonzero(occidxa)
                noccb = np.count_nonzero(occidxb)
                nmoa, nmob = sub.env_mo_occ[0].size, sub.env_mo_occ[1].size
                eai_a = sub.env_mo_energy[0][viridxa,None] - sub.env_mo_energy[0][occidxa]
                eai_b = sub.env_mo_energy[1][viridxb,None] - sub.env_mo_energy[1][occidxb]

                eai_a = 1./ eai_a
                eai_b = 1./ eai_b

                def vind_vo(z_vec):
                    pass

        else:
            for i, sub in enumerate(self.subsystems):

                nao, nmo = sub.env_mo_coeff[0].shape
                mocc = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]>0]
                vir = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]==0]
                nocc = mocc.shape[1]
                nvir = vir.shape[1]
                zvec_0.append(np.zeros((nvir,nocc)))
                zvec_1.append(np.ones((nvir,nocc)))

        z_diff = [100] * len(self.subsystems)
        while (np.max(np.abs(np.array(z_diff))) > 1e-8):
            x_terms = self.get_x_terms(zvec_0)
            for i,sub in enumerate(self.subsystems):
                occidx = sub.env_mo_occ[0] > 0
                viridx = sub.env_mo_occ[0] == 0
                e_ai = sub.env_mo_energy[0][viridx,None] - sub.env_mo_energy[0][occidx]
                nocc = np.count_nonzero(occidx)
                nvir = sub.env_mo_occ[0].size - nocc
                e_ai = 1. / e_ai
                def vind_vo(zvec):
                    v = self.gen_sub_vind(i,i)(zvec)
                    v *= e_ai
                    return v.ravel()
                zvec_1_long = lib.krylov(vind_vo, x_terms[i].ravel())
                zvec_1[i] = copy.copy(zvec_1_long.reshape(nvir,nocc))
                z_diff[i] = np.max(np.abs(zvec_1[i] - zvec_0[i]))
            zvec_0 = copy.copy(zvec_1)
        self.zvec = zvec_0

    def make_rhf_fa1emb(self):
        mol = self.mol
        atmlst = range(mol.natm)

        dm0 = self.get_emb_dmat()
        hcore_deriv = self.fs_nuc_grad_obj.hcore_generator(mol)

        aoslices = mol.aoslice_by_atom()
        fa1emb_ao = [None] * mol.natm
        for i0, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            shls_slice = (shl0, shl1) + (0, mol.nbas)*3
            vj1, vj2, vk1, vk2 = hessian.rhf._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                          'lk->s1ij', -dm0         ,  # vj2
                                          'li->s1kj', -dm0[:,p0:p1],  # vk1
                                          'jk->s1il', -dm0         ], # vk2
                                         shls_slice=shls_slice)
            vhf = vj1 - vk1*.5
            vhf[:,p0:p1] += vj2 - vk2*.5
            f1 = vhf + vhf.transpose(0,2,1)
            f1 += hcore_deriv(ia)

            fa1emb_ao[ia] = f1

        self.fa1emb_ao = fa1emb_ao
        return self.fa1emb_ao
    
    #def make_rks_fa1emb(self):
    #    mol = self.mol
    #    atmlst = range(mol.natm)

    #    dm0 = self.get_emb_dmat()
    #    hcore_deriv = self.fs_nuc_grad_obj.hcore_generator(mol)

    #    mf = self.fs_scf_obj
    #    ni = mf._numint
    #    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    #    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)

    #    mem_now = lib.current_memory()[0]
    #    max_memory = max(2000, mf.max_memory*.9-mem_now)
    #    h1ao = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    #    aoslices = mol.aoslice_by_atom()
    #    for i0, ia in enumerate(atmlst):
    #        shl0, shl1, p0, p1 = aoslices[ia]
    #        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
    #        if abs(hyb) > 1e-10:
    #            vj1, vj2, vk1, vk2 = \
    #                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
    #                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
    #                                      'lk->s1ij', -dm0         ,  # vj2
    #                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
    #                                      'jk->s1il', -dm0         ], # vk2
    #                                     shls_slice=shls_slice)
    #            veff = vj1 - hyb * .5 * vk1
    #            veff[:,p0:p1] += vj2 - hyb * .5 * vk2
    #            if abs(omega) > 1e-10:
    #                with mol.with_range_coulomb(omega):
    #                    vk1, vk2 = \
    #                        rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
    #                                         ['li->s1kj', -dm0[:,p0:p1],  # vk1
    #                                          'jk->s1il', -dm0         ], # vk2
    #                                         shls_slice=shls_slice)
    #                veff -= (alpha-hyb) * .5 * vk1
    #                veff[:,p0:p1] -= (alpha-hyb) * .5 * vk2
    #        else:
    #            vj1, vj2 = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
    #                                        ['ji->s2kl', -dm0[:,p0:p1],  # vj1
    #                                         'lk->s1ij', -dm0         ], # vj2
    #                                        shls_slice=shls_slice)
    #            veff = vj1
    #            veff[:,p0:p1] += vj2

    #        h1ao[ia] += veff + veff.transpose(0,2,1)
    #        h1ao[ia] += hcore_deriv(ia)

    #    self.fa1emb_ao = fa1emb_ao
    #    return self.fa1emb_ao

    def get_b_terms(self, atm_index):
        '''Gets the b terms to solve for U. '''
        if isinstance(self.fs_scf_obj, dft.rks.RKS):
            if self.fa1emb_ao is None:
                s2s = self.sub2sup
                full_mo_coeff = np.zeros_like(self.get_emb_dmat())
                full_mo_coeff[np.ix_(s2s[0], s2s[0])] += self.subsystems[0].env_mo_coeff[0]
                full_mo_coeff[np.ix_(s2s[1], s2s[1])] += self.subsystems[1].env_mo_coeff[0]

                full_mo_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
                full_mo_occ[np.ix_(s2s[0])] += self.subsystems[0].env_mo_occ[0]*2.
                full_mo_occ[np.ix_(s2s[1])] += self.subsystems[1].env_mo_occ[0]*2.
                hess_obj = hessian.rks.Hessian(self.fs_scf_obj)
                self.fa1emb_ao = hess_obj.make_h1(full_mo_coeff, full_mo_occ)
        else:
            if self.fa1emb_ao is None:
                s2s = self.sub2sup
                full_mo_coeff = np.zeros_like(self.get_emb_dmat())
                full_mo_coeff[np.ix_(s2s[0], s2s[0])] += self.subsystems[0].env_mo_coeff[0]
                full_mo_coeff[np.ix_(s2s[1], s2s[1])] += self.subsystems[1].env_mo_coeff[0]

                full_mo_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
                full_mo_occ[np.ix_(s2s[0])] += self.subsystems[0].env_mo_occ[0]*2.
                full_mo_occ[np.ix_(s2s[1])] += self.subsystems[1].env_mo_occ[0]*2.
                hess_obj = hessian.rhf.Hessian(self.fs_scf_obj)
                self.fa1emb_ao = hess_obj.make_h1(full_mo_coeff, full_mo_occ)

        sub1 = self.subsystems[0]
        occidx_sub1 = sub1.env_mo_occ[0] > 0
        viridx_sub1 = sub1.env_mo_occ[0] == 0
        nocc_sub1 = np.count_nonzero(occidx_sub1)
        nvir_sub1 = sub1.env_mo_occ[0].size - nocc_sub1
        nao_sub1 = sub1.env_mo_coeff[0].shape[0]
        mocc_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]>0]
        vir_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]==0]
        e_a_s1 = sub1.env_mo_energy[0][viridx_sub1]
        e_i_s1 = sub1.env_mo_energy[0][occidx_sub1]
        e_ai_s1 = 1. / lib.direct_sum('a-i->ai', e_a_s1, e_i_s1)
        sub2 = self.subsystems[1]
        occidx_sub2 = sub2.env_mo_occ[0] > 0
        viridx_sub2 = sub2.env_mo_occ[0] == 0
        nocc_sub2 = np.count_nonzero(occidx_sub2)
        nvir_sub2 = sub2.env_mo_occ[0].size - nocc_sub2
        nao_sub2 = sub2.env_mo_coeff[0].shape[0]
        mocc_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]>0]
        vir_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]==0]
        e_a_s2 = sub2.env_mo_energy[0][viridx_sub2]
        e_i_s2 = sub2.env_mo_energy[0][occidx_sub2]
        e_ai_s2 = 1. / lib.direct_sum('a-i->ai', e_a_s2, e_i_s2)

        fa1emb_ao = self.fa1emb_ao[atm_index]
        aoslices = self.mol.aoslice_by_atom()
        p0,p1 = aoslices [atm_index, 2:]
        s1 = self.fs_nuc_grad_obj.get_ovlp(self.mol)
        full_s1ao = np.zeros_like(fa1emb_ao)
        full_s1ao[:,p0:p1] += s1[:,p0:p1]
        full_s1ao[:,:,p0:p1] += s1[:,p0:p1].transpose(0,2,1)

        atm_s2s = self.atm_sub2sup
        s2s = self.sub2sup
        atmlist = range(self.mol.natm)
        xyz = [0,1,2]
        block_s1ao = np.zeros_like(fa1emb_ao)
        #block diagonalize the s matrix.
        for i, sub in enumerate(self.subsystems):
            block_s1ao[np.ix_(xyz, s2s[i], s2s[i])] += full_s1ao[np.ix_(xyz, s2s[i], s2s[i])]
        bterm_subsys = []
        full_mo_coeff = np.zeros_like(self.get_emb_dmat())
        full_mo_coeff[np.ix_(s2s[0], s2s[0])] += self.subsystems[0].env_mo_coeff[0]
        full_mo_coeff[np.ix_(s2s[1], s2s[1])] += self.subsystems[1].env_mo_coeff[0]
        full_mo_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
        full_mo_occ[np.ix_(s2s[0])] += self.subsystems[0].env_mo_occ[0]*2.
        full_mo_occ[np.ix_(s2s[1])] += self.subsystems[1].env_mo_occ[0]*2.
        full_response = self.fs_scf_obj.gen_response(full_mo_coeff, full_mo_occ)
        dm_s1_dm = np.einsum('mi,xmn,nj->xij', self.get_emb_dmat(), block_s1ao, self.get_emb_dmat()) * 0.5
        full_resp_mat = full_response(dm_s1_dm)
        for i, sub in enumerate(self.subsystems):
            mocc = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]>0]
            vir = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]==0]
            proj1ao = np.zeros_like(fa1emb_ao)[np.ix_(xyz, s2s[i],s2s[i])]
            proj_resp = np.zeros_like(fa1emb_ao)[np.ix_(xyz, s2s[i],s2s[i])]
            s1_vo = np.einsum('ma,xmn,ni->xai', vir, block_s1ao[np.ix_(xyz, s2s[i],s2s[i])], mocc)
            s1_vo = np.einsum('xai,i->xai', s1_vo, sub.env_mo_energy[0][sub.env_mo_occ[0]>0])
            veff_term = full_resp_mat[np.ix_(xyz, s2s[i], s2s[i])]
            s1_vo += np.einsum('ma,xmn,ni->xai', vir, veff_term, mocc)

            for j, alt_sub in enumerate(self.subsystems):
                if j != i :
                    alt_dmat = alt_sub.get_dmat()
                    fock_grad_ab = fa1emb_ao[np.ix_(xyz, s2s[i], s2s[j])]
                    smat_ba = self.smat[np.ix_(s2s[j], s2s[i])]
                    fock_ab = self.fock[0][np.ix_(s2s[i], s2s[j])]
                    smat_grad_ba = full_s1ao[np.ix_(xyz, s2s[j], s2s[i])]
                    proj1ao[0] += np.dot(fock_grad_ab[0], np.dot(alt_dmat, smat_ba))
                    proj1ao[0] += np.dot(fock_ab, np.dot(alt_dmat, smat_grad_ba[0]))
                    proj1ao[1] += np.dot(fock_grad_ab[1], np.dot(alt_dmat, smat_ba))
                    proj1ao[1] += np.dot(fock_ab, np.dot(alt_dmat, smat_grad_ba[1]))
                    proj1ao[2] += np.dot(fock_grad_ab[2], np.dot(alt_dmat, smat_ba))
                    proj1ao[2] += np.dot(fock_ab, np.dot(alt_dmat, smat_grad_ba[2]))
                    proj1ao += proj1ao.transpose(0,2,1)
                    sub_proj_term = np.einsum('xmn,nl,lp->xmp', full_resp_mat[np.ix_(xyz, s2s[i], s2s[j])], alt_sub.get_dmat(), self.smat[np.ix_(s2s[j], s2s[i])])
                    proj_resp += sub_proj_term + sub_proj_term.transpose(0,2,1)
                    fds_term = np.einsum('mn,xnl,lp->xmp', self.fock[0][np.ix_(s2s[i], s2s[j])], dm_s1_dm[np.ix_(xyz, s2s[j],s2s[j])], self.smat[np.ix_(s2s[j], s2s[i])])
                    fds_term += fds_term.transpose(0,2,1)
                    s1_vo -= np.einsum('ma,xmn,ni->xai', vir, (fds_term*0.5), mocc)

            s1_vo -= np.einsum('ma,xmn,ni->xai', vir, (proj_resp*0.5), mocc)

            f1ao_emb = fa1emb_ao[np.ix_(xyz, s2s[i], s2s[i])] - (proj1ao * 0.5)

            f1vo_emb = np.einsum('ma,xmn,ni->xai', vir, f1ao_emb, mocc)

            #if atm_index in atm_s2s[i]:
            #    s1_vo = np.einsum('ma,xmn,ni->xai', vir, s1ao[np.ix_(xyz, s2s[i],s2s[i])], mocc)
            #    s1_vo = np.einsum('xai,i->xai', s1_vo, sub.env_mo_energy[0][sub.env_mo_occ[0]>0])

            #    dm_s1_dm = np.einsum('mi,xmn,nj->xij', sub.get_dmat(), s1ao[np.ix_(xyz, s2s[i], s2s[i])], sub.get_dmat()) * 0.5
            #    full_dm_s1_dm = np.zeros_like(fa1emb_ao)
            #    full_dm_s1_dm[np.ix_(xyz, s2s[i], s2s[i])] += dm_s1_dm

            #    #veff_term = np.zeros_like(f1ao_emb) 
            #    #veff_term[0] += sub.env_scf.get_veff(sub.mol, dm_s1_dm[0], hermi=0)
            #    #veff_term[1] += sub.env_scf.get_veff(sub.mol, dm_s1_dm[1], hermi=0)
            #    #veff_term[2] += sub.env_scf.get_veff(sub.mol, dm_s1_dm[2], hermi=0)
            #    veff_term = sub.env_scf.gen_response(sub.env_mo_coeff[0], sub.env_mo_occ[0]*2.)(dm_s1_dm)
            #    s1_vo += np.einsum('ma,xmn,ni->xai', vir, veff_term, mocc)

            #    full_mo_coeff = np.zeros_like(self.get_emb_dmat())
            #    full_mo_coeff[np.ix_(s2s[0], s2s[0])] += self.subsystems[0].env_mo_coeff[0]
            #    full_mo_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
            #    full_mo_occ[np.ix_(s2s[0])] += self.subsystems[0].env_mo_occ[0]*2.
            #    proj_veff = self.fs_scf_obj.gen_response(full_mo_coeff, full_mo_occ)(full_dm_s1_dm)
            #    #proj_veff = np.zeros_like(fa1emb_ao)
            #    #proj_veff[0] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[0], hermi=0)
            #    #proj_veff[1] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[1], hermi=0)
            #    #proj_veff[2] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[2], hermi=0)
            #    proj_term = np.zeros_like(fa1emb_ao)[np.ix_(xyz, s2s[i],s2s[i])]
            #    for j, alt_sub in enumerate(self.subsystems):
            #        if j != i :
            #            sub_proj_term = np.einsum('xmn,nl,lp->xmp', proj_veff[np.ix_(xyz, s2s[i], s2s[j])], alt_sub.get_dmat(), self.smat[np.ix_(s2s[j], s2s[i])])
            #            proj_term += sub_proj_term + sub_proj_term.transpose(0,2,1)
            #    s1_vo -= np.einsum('ma,xmn,ni->xai', vir, (proj_term*0.5), mocc)

            #else:
            #    s1_vo = np.zeros_like(f1vo_emb)
            #    for j, alt_sub in enumerate(self.subsystems):
            #        if atm_index in atm_s2s[j]:
            #            dm_s1_dm = np.einsum('mi,xmn,nj->xij', alt_sub.get_dmat(), s1ao[np.ix_(xyz, s2s[j], s2s[j])], alt_sub.get_dmat()) * 0.5
            #            full_dm_s1_dm = np.zeros_like(fa1emb_ao)
            #            full_dm_s1_dm[np.ix_(xyz, s2s[j], s2s[j])] += dm_s1_dm

            #            full_mo_coeff = np.zeros_like(self.get_emb_dmat())
            #            full_mo_coeff[np.ix_(s2s[1], s2s[1])] += self.subsystems[1].env_mo_coeff[0]
            #            full_mo_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
            #            full_mo_occ[np.ix_(s2s[1])] += self.subsystems[1].env_mo_occ[0]*2.
            #            full_veff_term = self.fs_scf_obj.gen_response(full_mo_coeff, full_mo_occ)(full_dm_s1_dm)

            #            #full_veff_term = np.zeros_like(fa1emb_ao) 
            #            #full_veff_term[0] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[0], hermi=0)
            #            #full_veff_term[1] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[1], hermi=0)
            #            #full_veff_term[2] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[2], hermi=0)
            #            veff_term = full_veff_term[np.ix_(xyz, s2s[i], s2s[i])]
            #            s1_vo += np.einsum('ma,xmn,ni->xai', vir, veff_term, mocc)

            #            proj_veff = full_veff_term
            #            #proj_veff = np.zeros_like(fa1emb_ao)
            #            #proj_veff[0] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[0], hermi=0)
            #            #proj_veff[1] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[1], hermi=0)
            #            #proj_veff[2] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[2], hermi=0)
            #            sub_proj_term = np.einsum('xmn,nl,lp->xmp', proj_veff[np.ix_(xyz, s2s[i], s2s[j])], alt_sub.get_dmat(), self.smat[np.ix_(s2s[j], s2s[i])])
            #            proj_term = sub_proj_term + sub_proj_term.transpose(0,2,1)
            #            s1_vo -= np.einsum('ma,xmn,ni->xai', vir, (proj_term*0.5), mocc)

            #            #last proj term
            #            fds_term = np.einsum('mn,xnl,lp->xmp', self.fock[0][np.ix_(s2s[i], s2s[j])], dm_s1_dm, self.smat[np.ix_(s2s[j], s2s[i])])
            #            fds_term += fds_term.transpose(0,2,1)
            #            s1_vo -= np.einsum('ma,xmn,ni->xai', vir, (fds_term*0.5), mocc)

            bterm_subsys.append(f1vo_emb - s1_vo)

        bterm_subsys[0] *= -e_ai_s1
        bterm_subsys[1] *= -e_ai_s2
        bterm = np.concatenate((bterm_subsys[0].reshape(-1, (nvir_sub1*nocc_sub1)), bterm_subsys[1].reshape(-1, (nvir_sub2*nocc_sub2))), axis=1)
        
        return bterm

        #else:
        #    print ('here rhf')
        #    if self.fa1emb_ao is None:
        #        s2s = self.sub2sup
        #        full_mo_coeff = np.zeros_like(self.get_emb_dmat())
        #        full_mo_coeff[np.ix_(s2s[0], s2s[0])] += self.subsystems[0].env_mo_coeff[0]
        #        full_mo_coeff[np.ix_(s2s[1], s2s[1])] += self.subsystems[1].env_mo_coeff[0]

        #        full_mo_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
        #        full_mo_occ[np.ix_(s2s[0])] += self.subsystems[0].env_mo_occ[0]*2.
        #        full_mo_occ[np.ix_(s2s[1])] += self.subsystems[1].env_mo_occ[0]*2.
        #        hess_obj = hessian.rhf.Hessian(self.fs_scf_obj)
        #        self.fa1emb_ao = hess_obj.make_h1(full_mo_coeff, full_mo_occ)

        #    sub1 = self.subsystems[0]
        #    occidx_sub1 = sub1.env_mo_occ[0] > 0
        #    viridx_sub1 = sub1.env_mo_occ[0] == 0
        #    nocc_sub1 = np.count_nonzero(occidx_sub1)
        #    nvir_sub1 = sub1.env_mo_occ[0].size - nocc_sub1
        #    nao_sub1 = sub1.env_mo_coeff[0].shape[0]
        #    mocc_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]>0]
        #    vir_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]==0]
        #    e_a_s1 = sub1.env_mo_energy[0][viridx_sub1]
        #    e_i_s1 = sub1.env_mo_energy[0][occidx_sub1]
        #    e_ai_s1 = 1. / lib.direct_sum('a-i->ai', e_a_s1, e_i_s1)
        #    sub2 = self.subsystems[1]
        #    occidx_sub2 = sub2.env_mo_occ[0] > 0
        #    viridx_sub2 = sub2.env_mo_occ[0] == 0
        #    nocc_sub2 = np.count_nonzero(occidx_sub2)
        #    nvir_sub2 = sub2.env_mo_occ[0].size - nocc_sub2
        #    nao_sub2 = sub2.env_mo_coeff[0].shape[0]
        #    mocc_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]>0]
        #    vir_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]==0]
        #    e_a_s2 = sub2.env_mo_energy[0][viridx_sub2]
        #    e_i_s2 = sub2.env_mo_energy[0][occidx_sub2]
        #    e_ai_s2 = 1. / lib.direct_sum('a-i->ai', e_a_s2, e_i_s2)

        #    hcore_deriv = self.fs_nuc_grad_obj.hcore_generator(self.mol)
        #    h1ao = hcore_deriv(atm_index)
        #    aoslices = self.mol.aoslice_by_atom()
        #    p0,p1 = aoslices [atm_index, 2:]
        #    s1 = self.fs_nuc_grad_obj.get_ovlp(self.mol)
        #    s1ao = np.zeros_like(h1ao)
        #    s1ao[:,p0:p1] += s1[:,p0:p1]
        #    s1ao[:,:,p0:p1] += s1[:,p0:p1].transpose(0,2,1)
        #    if (self.ao_grad is None):
        #        self.ao_grad = self.mol.intor('int2e_ip1')
        #    ao_grad_atm = np.zeros_like(self.ao_grad)
        #    ao_grad_atm[:,p0:p1] += self.ao_grad[:,p0:p1]
        #    ao_grad_atm += ao_grad_atm.transpose(0,2,1,3,4)
        #    ao_grad_atm += ao_grad_atm.transpose(0,3,4,1,2)
        #    ao_grad_atm *= -1
        #    v1ao = np.einsum('xijkl,kl->xij', ao_grad_atm, self.get_emb_dmat())
        #    v1ao -= np.einsum('xikjl,kl->xij', ao_grad_atm, self.get_emb_dmat()) * 0.5
        #    atm_s2s = self.atm_sub2sup
        #    s2s = self.sub2sup
        #    atmlist = range(self.mol.natm)
        #    xyz = [0,1,2]
        #    bterm_subsys = []
        #    for i, sub in enumerate(self.subsystems):
        #        mocc = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]>0]
        #        vir = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]==0]
        #        proj1ao = np.zeros_like(h1ao)[np.ix_(xyz, s2s[i],s2s[i])]
        #        for j, alt_sub in enumerate(self.subsystems):
        #            if j != i :
        #                alt_dmat = alt_sub.get_dmat()
        #                fock_grad_ab = (h1ao + v1ao)[np.ix_(xyz, s2s[i], s2s[j])]
        #                smat_ba = self.smat[np.ix_(s2s[j], s2s[i])]
        #                fock_ab = self.fock[0][np.ix_(s2s[i], s2s[j])]
        #                smat_grad_ba = s1ao[np.ix_(xyz, s2s[j], s2s[i])]
        #                proj1ao[0] += np.dot(fock_grad_ab[0], np.dot(alt_dmat, smat_ba))
        #                proj1ao[0] += np.dot(fock_ab, np.dot(alt_dmat, smat_grad_ba[0]))
        #                proj1ao[1] += np.dot(fock_grad_ab[1], np.dot(alt_dmat, smat_ba))
        #                proj1ao[1] += np.dot(fock_ab, np.dot(alt_dmat, smat_grad_ba[1]))
        #                proj1ao[2] += np.dot(fock_grad_ab[2], np.dot(alt_dmat, smat_ba))
        #                proj1ao[2] += np.dot(fock_ab, np.dot(alt_dmat, smat_grad_ba[2]))
        #                proj1ao += proj1ao.transpose(0,2,1)

        #        f1ao_emb = (h1ao + v1ao)[np.ix_(xyz, s2s[i], s2s[i])] - (proj1ao * 0.5)

        #        f1vo_emb = np.einsum('ma,xmn,ni->xai', vir, f1ao_emb, mocc)
        #        if atm_index in atm_s2s[i]:
        #            s1_vo = np.einsum('ma,xmn,ni->xai', vir, s1ao[np.ix_(xyz, s2s[i],s2s[i])], mocc)
        #            s1_vo = np.einsum('xai,i->xai', s1_vo, sub.env_mo_energy[0][sub.env_mo_occ[0]>0])

        #            dm_s1_dm = np.einsum('mi,xmn,nj->xij', sub.get_dmat(), s1ao[np.ix_(xyz, s2s[i], s2s[i])], sub.get_dmat()) * 0.5
        #            full_dm_s1_dm = np.zeros_like(h1ao)
        #            full_dm_s1_dm[np.ix_(xyz, s2s[i], s2s[i])] += dm_s1_dm

        #            veff_term = np.zeros_like(f1ao_emb) 
        #            veff_term[0] += sub.env_scf.get_veff(sub.mol, dm_s1_dm[0], hermi=0)
        #            veff_term[1] += sub.env_scf.get_veff(sub.mol, dm_s1_dm[1], hermi=0)
        #            veff_term[2] += sub.env_scf.get_veff(sub.mol, dm_s1_dm[2], hermi=0)
        #            s1_vo += np.einsum('ma,xmn,ni->xai', vir, veff_term, mocc)

        #            proj_veff = np.zeros_like(h1ao)
        #            proj_veff[0] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[0], hermi=0)
        #            proj_veff[1] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[1], hermi=0)
        #            proj_veff[2] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[2], hermi=0)
        #            proj_term = np.zeros_like(h1ao)[np.ix_(xyz, s2s[i],s2s[i])]
        #            for j, alt_sub in enumerate(self.subsystems):
        #                if j != i :
        #                    sub_proj_term = np.einsum('xmn,nl,lp->xmp', proj_veff[np.ix_(xyz, s2s[i], s2s[j])], alt_sub.get_dmat(), self.smat[np.ix_(s2s[j], s2s[i])])
        #                    proj_term += sub_proj_term + sub_proj_term.transpose(0,2,1)
        #            s1_vo -= np.einsum('ma,xmn,ni->xai', vir, (proj_term*0.5), mocc)

        #        else:
        #            s1_vo = np.zeros_like(f1vo_emb)
        #            for j, alt_sub in enumerate(self.subsystems):
        #                if atm_index in atm_s2s[j]:
        #                    dm_s1_dm = np.einsum('mi,xmn,nj->xij', alt_sub.get_dmat(), s1ao[np.ix_(xyz, s2s[j], s2s[j])], alt_sub.get_dmat()) * 0.5
        #                    full_dm_s1_dm = np.zeros_like(h1ao)
        #                    full_dm_s1_dm[np.ix_(xyz, s2s[j], s2s[j])] += dm_s1_dm

        #                    full_veff_term = np.zeros_like(h1ao) 
        #                    full_veff_term[0] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[0], hermi=0)
        #                    full_veff_term[1] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[1], hermi=0)
        #                    full_veff_term[2] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[2], hermi=0)
        #                    veff_term = full_veff_term[np.ix_(xyz, s2s[i], s2s[i])]
        #                    s1_vo += np.einsum('ma,xmn,ni->xai', vir, veff_term, mocc)

        #                    proj_veff = np.zeros_like(h1ao)
        #                    proj_veff[0] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[0], hermi=0)
        #                    proj_veff[1] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[1], hermi=0)
        #                    proj_veff[2] += self.fs_scf_obj.get_veff(self.mol, full_dm_s1_dm[2], hermi=0)
        #                    sub_proj_term = np.einsum('xmn,nl,lp->xmp', proj_veff[np.ix_(xyz, s2s[i], s2s[j])], alt_sub.get_dmat(), self.smat[np.ix_(s2s[j], s2s[i])])
        #                    proj_term = sub_proj_term + sub_proj_term.transpose(0,2,1)
        #                    s1_vo -= np.einsum('ma,xmn,ni->xai', vir, (proj_term*0.5), mocc)

        #                    #last proj term
        #                    fds_term = np.einsum('mn,xnl,lp->xmp', self.fock[0][np.ix_(s2s[i], s2s[j])], dm_s1_dm, self.smat[np.ix_(s2s[j], s2s[i])])
        #                    fds_term += fds_term.transpose(0,2,1)
        #                    s1_vo -= np.einsum('ma,xmn,ni->xai', vir, (fds_term*0.5), mocc)

        #        bterm_subsys.append(f1vo_emb - s1_vo)

        #    bterm_subsys[0] *= -e_ai_s1
        #    bterm_subsys[1] *= -e_ai_s2
        #    bterm = np.concatenate((bterm_subsys[0].reshape(-1, (nvir_sub1*nocc_sub1)), bterm_subsys[1].reshape(-1, (nvir_sub2*nocc_sub2))), axis=1)
        #
        #    return bterm

    def get_x_terms(self, zvectors, base_xterms=None):
        '''returns the xterms for rhf'''
        if not base_xterms:
            base_xterms = self.base_xterms

        if not base_xterms:
            s2s = self.sub2sup
            base_xterms = []
            for i, sub in enumerate(self.subsystems):
                mocc = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]>0]
                vir = sub.env_mo_coeff[0][:,sub.env_mo_occ[0]==0]
                x_term_ao = np.zeros_like(sub.env_mo_coeff[0])
                for j, alt_sub in enumerate(self.subsystems):
                    if i != j:
                        s_ij = self.smat[np.ix_(s2s[i],s2s[j])]

                        dm1_s_dm2 = np.dot(sub.get_dmat(), np.dot(s_ij, alt_sub.get_dmat()))
                        full_dmat = np.zeros_like(self.get_emb_dmat())
                        full_dmat[np.ix_(s2s[i], s2s[j])] += dm1_s_dm2
                        full_xterm = self.fs_scf_obj.get_veff(self.mol, full_dmat, hermi=0)
                        sub_xterm = full_xterm[np.ix_(s2s[i], s2s[i])]
                        x_term_ao += sub_xterm + sub_xterm.T

                x_term_mo = np.dot(vir.T, np.dot(x_term_ao, mocc))
                base_xterms.append(x_term_mo)
            self.base_xterms = base_xterms

        full_xterm = []
        for i,sub in enumerate(self.subsystems):
            occidx = sub.env_mo_occ[0] > 0
            viridx = sub.env_mo_occ[0] == 0
            new_x = copy.copy(self.base_xterms[i])
            for j,alt_sub in enumerate(self.subsystems):
                if i != j:
                    new_x -= self.gen_sub_vind(i,j)(zvectors[j])
            e_ai = sub.env_mo_energy[0][viridx,None] - sub.env_mo_energy[0][occidx]
            nocc = np.count_nonzero(occidx)
            nvir = sub.env_mo_occ[0].size - nocc
            e_ai = 1. / e_ai
            nmo = nocc + nvir
            new_x *= -e_ai
            full_xterm.append(new_x)
        return full_xterm

    def gen_full_a(self):

        #RHF
        #if isinstance(self.fs_scf_obj, dft.rks.RKS):
        if False:
            print ('dft a')
            sub1 = self.subsystems[0]
            occidx_sub1 = sub1.env_mo_occ[0] > 0
            viridx_sub1 = sub1.env_mo_occ[0] == 0
            nocc_sub1 = np.count_nonzero(occidx_sub1)
            nvir_sub1 = sub1.env_mo_occ[0].size - nocc_sub1
            nao_sub1 = sub1.env_mo_coeff[0].shape[0]
            mocc_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]>0]
            vir_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]==0]
            e_a_s1 = sub1.env_mo_energy[0][viridx_sub1]
            e_i_s1 = sub1.env_mo_energy[0][occidx_sub1]
            e_ai_s1 = 1. / lib.direct_sum('a-i->ai', e_a_s1, e_i_s1)
            sub2 = self.subsystems[1]
            occidx_sub2 = sub2.env_mo_occ[0] > 0
            viridx_sub2 = sub2.env_mo_occ[0] == 0
            nocc_sub2 = np.count_nonzero(occidx_sub2)
            nvir_sub2 = sub2.env_mo_occ[0].size - nocc_sub2
            nao_sub2 = sub2.env_mo_coeff[0].shape[0]
            mocc_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]>0]
            vir_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]==0]
            e_a_s2 = sub2.env_mo_energy[0][viridx_sub2]
            e_i_s2 = sub2.env_mo_energy[0][occidx_sub2]
            e_ai_s2 = 1. / lib.direct_sum('a-i->ai', e_a_s2, e_i_s2)
            s2s = self.sub2sup
            a_size = (nvir_sub1*nocc_sub1) + (nvir_sub2*nocc_sub2)
            def fx(dm1):
                dm1 = dm1.reshape(-1, a_size)
                dm_sub1 = dm1[:,:(nvir_sub1*nocc_sub1)]
                dm_sub2 = dm1[:,(nvir_sub1*nocc_sub1):]
                dm_sub1 = dm_sub1.reshape(-1, nvir_sub1, nocc_sub1)
                dm_sub2 = dm_sub2.reshape(-1, nvir_sub2, nocc_sub2)
                nset = len(dm1)
                ao_dm_sub1 = np.empty((nset, nao_sub1, nao_sub1))
                for i, x in enumerate(dm_sub1):
                    temp_dm = reduce(np.dot, (vir_sub1, x*2., mocc_sub1.T))
                    ao_dm_sub1[i] = copy.copy(temp_dm + temp_dm.T)

                ao_dm_sub2 = np.empty((nset, nao_sub2, nao_sub2))
                for i, x in enumerate(dm_sub2):
                    temp_dm = reduce(np.dot, (vir_sub2, x*2., mocc_sub2.T))
                    ao_dm_sub2[i] = copy.copy(temp_dm + temp_dm.T)

                #Sub 1 part
                a_I_I_ao = np.zeros((nset, nao_sub1, nao_sub1))
                a_I_II_ao = np.zeros((nset, nao_sub1, nao_sub1))
                a_I_I = np.empty_like(dm_sub1)
                a_I_II = np.empty_like(dm_sub1)
                smat_temp = self.smat[np.ix_(s2s[1], s2s[0])]
                for i in range(nset):
                    a_I_I_ao[i] += sub1.env_scf.get_veff(sub1.mol, ao_dm_sub1[i])
                    temp_dm = np.zeros_like(self.get_emb_dmat())
                    temp_dm[np.ix_(s2s[0], s2s[0])] += ao_dm_sub1[i]
                    proj_temp = self.fs_scf_obj.get_veff(self.mol, temp_dm)
                    proj_term = reduce(np.dot, (proj_temp[np.ix_(s2s[0], s2s[1])], sub2.get_dmat(), smat_temp))
                    a_I_I_ao[i] -= 0.5 * (proj_term + proj_term.T)

                    temp_dm = np.zeros_like(self.get_emb_dmat())
                    temp_dm[np.ix_(s2s[1], s2s[1])] += ao_dm_sub2[i]
                    a_I_II_ao[i] += self.fs_scf_obj.get_veff(self.mol, temp_dm)[np.ix_(s2s[0], s2s[0])]
                    proj_temp = self.fs_scf_obj.get_veff(self.mol, temp_dm)
                    proj_term = reduce(np.dot, (proj_temp[np.ix_(s2s[0], s2s[1])], sub2.get_dmat(), smat_temp))
                    a_I_II_ao[i] -= 0.5 * (proj_term + proj_term.T)
                    proj_term = reduce (np.dot, (self.fock[0][np.ix_(s2s[0], s2s[1])], ao_dm_sub2[i], smat_temp))
                    a_I_II_ao[i] -= 0.5 * (proj_term + proj_term.T)

                    a_I_I[i] = reduce(np.dot, (vir_sub1.T, a_I_I_ao[i], mocc_sub1))
                    a_I_II[i] = reduce(np.dot, (vir_sub1.T, a_I_II_ao[i], mocc_sub1))


                a_I = a_I_I + a_I_II
                a_I *= e_ai_s1

                #Sub 2 part
                a_II_I_ao = np.zeros((nset, nao_sub2, nao_sub2))
                a_II_II_ao = np.zeros((nset, nao_sub2, nao_sub2))
                a_II_I = np.empty_like(dm_sub2)
                a_II_II = np.empty_like(dm_sub2)
                smat_temp = self.smat[np.ix_(s2s[0], s2s[1])]
                for i in range(nset):
                    temp_dm = np.zeros_like(self.get_emb_dmat())
                    temp_dm[np.ix_(s2s[0], s2s[0])] += ao_dm_sub1[i]
                    a_II_I_ao[i] += self.fs_scf_obj.get_veff(self.mol, temp_dm)[np.ix_(s2s[1], s2s[1])]
                    proj_temp = self.fs_scf_obj.get_veff(self.mol, temp_dm)
                    proj_term = reduce(np.dot, (proj_temp[np.ix_(s2s[1], s2s[0])], sub1.get_dmat(), smat_temp))
                    a_II_I_ao[i] -= 0.5 * (proj_term + proj_term.T)
                    proj_term = reduce (np.dot, (self.fock[0][np.ix_(s2s[1], s2s[0])], ao_dm_sub1[i], smat_temp))
                    a_II_I_ao[i] -= 0.5 * (proj_term + proj_term.T)

                    a_II_II_ao[i] += sub2.env_scf.get_veff(sub2.mol, ao_dm_sub2[i])

                    temp_dm = np.zeros_like(self.get_emb_dmat())
                    temp_dm[np.ix_(s2s[1], s2s[1])] += ao_dm_sub2[i]
                    proj_temp = self.fs_scf_obj.get_veff(self.mol, temp_dm)
                    proj_term = reduce(np.dot, (proj_temp[np.ix_(s2s[1], s2s[0])], sub1.get_dmat(), smat_temp))
                    a_II_II_ao[i] -= 0.5 * (proj_term + proj_term.T)

                    a_II_I[i] = reduce(np.dot, (vir_sub2.T, a_II_I_ao[i], mocc_sub2))
                    a_II_II[i] = reduce(np.dot, (vir_sub2.T, a_II_II_ao[i], mocc_sub2))

                a_II = a_II_I + a_II_II
                a_II *= e_ai_s2

                total_a_mat = np.concatenate((a_I.reshape(-1, (nvir_sub1*nocc_sub1)), a_II.reshape(-1, (nvir_sub2*nocc_sub2))),axis=1)
                return total_a_mat.ravel()

        else:
            sub1 = self.subsystems[0]
            occidx_sub1 = sub1.env_mo_occ[0] > 0
            viridx_sub1 = sub1.env_mo_occ[0] == 0
            nocc_sub1 = np.count_nonzero(occidx_sub1)
            nvir_sub1 = sub1.env_mo_occ[0].size - nocc_sub1
            nao_sub1 = sub1.env_mo_coeff[0].shape[0]
            mocc_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]>0]
            vir_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]==0]
            e_a_s1 = sub1.env_mo_energy[0][viridx_sub1]
            e_i_s1 = sub1.env_mo_energy[0][occidx_sub1]
            e_ai_s1 = 1. / lib.direct_sum('a-i->ai', e_a_s1, e_i_s1)
            sub2 = self.subsystems[1]
            occidx_sub2 = sub2.env_mo_occ[0] > 0
            viridx_sub2 = sub2.env_mo_occ[0] == 0
            nocc_sub2 = np.count_nonzero(occidx_sub2)
            nvir_sub2 = sub2.env_mo_occ[0].size - nocc_sub2
            nao_sub2 = sub2.env_mo_coeff[0].shape[0]
            mocc_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]>0]
            vir_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]==0]
            e_a_s2 = sub2.env_mo_energy[0][viridx_sub2]
            e_i_s2 = sub2.env_mo_energy[0][occidx_sub2]
            e_ai_s2 = 1. / lib.direct_sum('a-i->ai', e_a_s2, e_i_s2)
            s2s = self.sub2sup
            a_size = (nvir_sub1*nocc_sub1) + (nvir_sub2*nocc_sub2)
            def fx(dm1):
                dm1 = dm1.reshape(-1, a_size)
                dm_sub1 = dm1[:,:(nvir_sub1*nocc_sub1)]
                dm_sub2 = dm1[:,(nvir_sub1*nocc_sub1):]
                dm_sub1 = dm_sub1.reshape(-1, nvir_sub1, nocc_sub1)
                dm_sub2 = dm_sub2.reshape(-1, nvir_sub2, nocc_sub2)
                nset = len(dm1)
                ao_dm_sub1 = np.empty((nset, nao_sub1, nao_sub1))
                for i, x in enumerate(dm_sub1):
                    temp_dm = reduce(np.dot, (vir_sub1, x*2., mocc_sub1.T))
                    ao_dm_sub1[i] = copy.copy(temp_dm + temp_dm.T)

                ao_dm_sub2 = np.empty((nset, nao_sub2, nao_sub2))
                for i, x in enumerate(dm_sub2):
                    temp_dm = reduce(np.dot, (vir_sub2, x*2., mocc_sub2.T))
                    ao_dm_sub2[i] = copy.copy(temp_dm + temp_dm.T)

                a_I_ao = np.zeros((nset, nao_sub1, nao_sub1))
                a_I = np.empty_like(dm_sub1)
                a_II_ao = np.zeros((nset, nao_sub2, nao_sub2))
                a_II = np.empty_like(dm_sub2)
                full_emb_coeff = np.zeros_like(self.get_emb_dmat())
                full_emb_coeff[np.ix_(s2s[0], s2s[0])] += sub1.env_mo_coeff[0]
                full_emb_coeff[np.ix_(s2s[1], s2s[1])] += sub2.env_mo_coeff[0]
                full_emb_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
                full_emb_occ[np.ix_(s2s[0])] += sub1.env_mo_occ[0] * 2.
                full_emb_occ[np.ix_(s2s[1])] += sub2.env_mo_occ[0] * 2.
                full_response = self.fs_scf_obj.gen_response(full_emb_coeff, full_emb_occ)
                for i in range(nset):
                    full_ao_dm = np.zeros_like(self.get_emb_dmat())
                    full_ao_dm[np.ix_(s2s[0], s2s[0])] += ao_dm_sub1[i]
                    full_ao_dm[np.ix_(s2s[1], s2s[1])] += ao_dm_sub2[i]
                    full_resp_mat = full_response(full_ao_dm)
                    a_I_ao[i] += full_resp_mat[np.ix_(s2s[0], s2s[0])]
                    smat_ba = self.smat[np.ix_(s2s[1], s2s[0])]
                    proj_term = reduce(np.dot, (full_resp_mat[np.ix_(s2s[0], s2s[1])], sub2.get_dmat(), smat_ba))
                    a_I_ao[i] -= 0.5 * (proj_term + proj_term.T)
                    proj_term = reduce (np.dot, (self.fock[0][np.ix_(s2s[0], s2s[1])], ao_dm_sub2[i], smat_ba))
                    a_I_ao[i] -= 0.5 * (proj_term + proj_term.T)

                    a_II_ao[i] += full_resp_mat[np.ix_(s2s[1], s2s[1])]
                    smat_ab = self.smat[np.ix_(s2s[0], s2s[1])]
                    proj_term = reduce(np.dot, (full_resp_mat[np.ix_(s2s[1], s2s[0])], sub1.get_dmat(), smat_ab))
                    a_II_ao[i] -= 0.5 * (proj_term + proj_term.T)
                    proj_term = reduce (np.dot, (self.fock[0][np.ix_(s2s[1], s2s[0])], ao_dm_sub1[i], smat_ab))
                    a_II_ao[i] -= 0.5 * (proj_term + proj_term.T)

                    a_I[i] = reduce(np.dot, (vir_sub1.T, a_I_ao[i], mocc_sub1))
                    a_II[i] = reduce(np.dot, (vir_sub2.T, a_II_ao[i], mocc_sub2))
                a_I *= e_ai_s1
                a_II *= e_ai_s2
                total_a_mat = np.concatenate((a_I.reshape(-1, (nvir_sub1*nocc_sub1)), a_II.reshape(-1, (nvir_sub2*nocc_sub2))),axis=1)
                return total_a_mat.ravel()

                ##Sub 1 part
                #a_I_I_ao = np.zeros((nset, nao_sub1, nao_sub1))
                #a_I_II_ao = np.zeros((nset, nao_sub1, nao_sub1))
                #a_I_I = np.empty_like(dm_sub1)
                #a_I_II = np.empty_like(dm_sub1)
                #smat_temp = self.smat[np.ix_(s2s[1], s2s[0])]
                #for i in range(nset):
                #    #a_I_I_ao[i] += sub1.env_scf.get_veff(sub1.mol, ao_dm_sub1[i])
                #    a_I_I_ao[i] += sub1.env_scf.gen_response(sub1.env_mo_coeff[0], sub1.env_mo_occ[0]*2.)(ao_dm_sub1[i])
                #    temp_dm = np.zeros_like(self.get_emb_dmat())
                #    temp_dm[np.ix_(s2s[0], s2s[0])] += ao_dm_sub1[i]
                #    temp_coeff = np.zeros_like(self.get_emb_dmat())
                #    temp_coeff[np.ix_(s2s[0], s2s[0])] += sub1.env_mo_coeff[0]
                #    temp_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
                #    temp_occ[np.ix_(s2s[0])] += sub1.env_mo_occ[0] * 2.
                #    #proj_temp = self.fs_scf_obj.get_veff(self.mol, temp_dm)
                #    proj_temp = self.fs_scf_obj.gen_response(temp_coeff, temp_occ)(temp_dm)
                #    proj_term = reduce(np.dot, (proj_temp[np.ix_(s2s[0], s2s[1])], sub2.get_dmat(), smat_temp))
                #    a_I_I_ao[i] -= 0.5 * (proj_term + proj_term.T)

                #    temp_dm = np.zeros_like(self.get_emb_dmat())
                #    temp_dm[np.ix_(s2s[1], s2s[1])] += ao_dm_sub2[i]
                #    temp_coeff = np.zeros_like(self.get_emb_dmat())
                #    temp_coeff[np.ix_(s2s[1], s2s[1])] += sub2.env_mo_coeff[0]
                #    temp_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
                #    temp_occ[np.ix_(s2s[1])] += sub2.env_mo_occ[0] * 2.
                #    resp_term = self.fs_scf_obj.gen_response(temp_coeff, temp_occ)(temp_dm)
                #    a_I_II_ao[i] += resp_term[np.ix_(s2s[0], s2s[0])]
                #    proj_temp = resp_term
                #    proj_term = reduce(np.dot, (proj_temp[np.ix_(s2s[0], s2s[1])], sub2.get_dmat(), smat_temp))
                #    a_I_II_ao[i] -= 0.5 * (proj_term + proj_term.T)
                #    proj_term = reduce (np.dot, (self.fock[0][np.ix_(s2s[0], s2s[1])], ao_dm_sub2[i], smat_temp))
                #    a_I_II_ao[i] -= 0.5 * (proj_term + proj_term.T)

                #    a_I_I[i] = reduce(np.dot, (vir_sub1.T, a_I_I_ao[i], mocc_sub1))
                #    a_I_II[i] = reduce(np.dot, (vir_sub1.T, a_I_II_ao[i], mocc_sub1))


                #a_I = a_I_I + a_I_II
                #a_I *= e_ai_s1

                ##Sub 2 part
                #a_II_I_ao = np.zeros((nset, nao_sub2, nao_sub2))
                #a_II_II_ao = np.zeros((nset, nao_sub2, nao_sub2))
                #a_II_I = np.empty_like(dm_sub2)
                #a_II_II = np.empty_like(dm_sub2)
                #smat_temp = self.smat[np.ix_(s2s[0], s2s[1])]
                #for i in range(nset):
                #    temp_dm = np.zeros_like(self.get_emb_dmat())
                #    temp_dm[np.ix_(s2s[0], s2s[0])] += ao_dm_sub1[i]
                #    temp_coeff = np.zeros_like(self.get_emb_dmat())
                #    temp_coeff[np.ix_(s2s[0], s2s[0])] += sub1.env_mo_coeff[0]
                #    temp_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
                #    temp_occ[np.ix_(s2s[0])] += sub1.env_mo_occ[0] * 2.
                #    resp_term = self.fs_scf_obj.gen_response(temp_coeff, temp_occ)(temp_dm)
                #    #a_II_I_ao[i] += self.fs_scf_obj.get_veff(self.mol, temp_dm)[np.ix_(s2s[1], s2s[1])]
                #    a_II_I_ao[i] += resp_term[np.ix_(s2s[1], s2s[1])]
                #    #proj_temp = self.fs_scf_obj.get_veff(self.mol, temp_dm)
                #    proj_temp = resp_term
                #    proj_term = reduce(np.dot, (proj_temp[np.ix_(s2s[1], s2s[0])], sub1.get_dmat(), smat_temp))
                #    a_II_I_ao[i] -= 0.5 * (proj_term + proj_term.T)
                #    proj_term = reduce (np.dot, (self.fock[0][np.ix_(s2s[1], s2s[0])], ao_dm_sub1[i], smat_temp))
                #    a_II_I_ao[i] -= 0.5 * (proj_term + proj_term.T)

                #    #a_II_II_ao[i] += sub2.env_scf.get_veff(sub2.mol, ao_dm_sub2[i])
                #    a_II_II_ao[i] += sub2.env_scf.gen_response(sub2.env_mo_coeff[0], sub2.env_mo_occ[0]*2.)(ao_dm_sub2[i])

                #    temp_dm = np.zeros_like(self.get_emb_dmat())
                #    temp_dm[np.ix_(s2s[1], s2s[1])] += ao_dm_sub2[i]
                #    #proj_temp = self.fs_scf_obj.get_veff(self.mol, temp_dm)
                #    temp_coeff = np.zeros_like(self.get_emb_dmat())
                #    temp_coeff[np.ix_(s2s[1], s2s[1])] += sub2.env_mo_coeff[0]
                #    temp_occ = np.zeros_like(self.fs_scf_obj.mo_occ)
                #    temp_occ[np.ix_(s2s[1])] += sub2.env_mo_occ[0] * 2.
                #    proj_temp = self.fs_scf_obj.gen_response(temp_coeff, temp_occ)(temp_dm)
                #    proj_term = reduce(np.dot, (proj_temp[np.ix_(s2s[1], s2s[0])], sub1.get_dmat(), smat_temp))
                #    a_II_II_ao[i] -= 0.5 * (proj_term + proj_term.T)

                #    a_II_I[i] = reduce(np.dot, (vir_sub2.T, a_II_I_ao[i], mocc_sub2))
                #    a_II_II[i] = reduce(np.dot, (vir_sub2.T, a_II_II_ao[i], mocc_sub2))

                #a_II = a_II_I + a_II_II
                #a_II *= e_ai_s2

                #total_a_mat = np.concatenate((a_I.reshape(-1, (nvir_sub1*nocc_sub1)), a_II.reshape(-1, (nvir_sub2*nocc_sub2))),axis=1)
                #return total_a_mat.ravel()
        return fx


    def solve_u(self):
        '''solves for full system U, Currently only for one dimension and one atom.'''

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, self.fs_scf.max_memory*0.9-mem_now)
        u_list = []
        #RHF ONLY
        if True:
            vind_vo = self.gen_full_a()
            sub1 = self.subsystems[0]
            occidx_sub1 = sub1.env_mo_occ[0] > 0
            viridx_sub1 = sub1.env_mo_occ[0] == 0
            nocc_sub1 = np.count_nonzero(occidx_sub1)
            nvir_sub1 = sub1.env_mo_occ[0].size - nocc_sub1
            nao_sub1 = sub1.env_mo_coeff[0].shape[0]
            mocc_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]>0]
            vir_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]==0]
            e_a_s1 = sub1.env_mo_energy[0][viridx_sub1]
            e_i_s1 = sub1.env_mo_energy[0][occidx_sub1]
            e_ai_s1 = 1. / lib.direct_sum('a-i->ai', e_a_s1, e_i_s1)
            sub2 = self.subsystems[1]
            occidx_sub2 = sub2.env_mo_occ[0] > 0
            viridx_sub2 = sub2.env_mo_occ[0] == 0
            nocc_sub2 = np.count_nonzero(occidx_sub2)
            nvir_sub2 = sub2.env_mo_occ[0].size - nocc_sub2
            nao_sub2 = sub2.env_mo_coeff[0].shape[0]
            mocc_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]>0]
            vir_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]==0]
            e_a_s2 = sub2.env_mo_energy[0][viridx_sub2]
            e_i_s2 = sub2.env_mo_energy[0][occidx_sub2]
            e_ai_s2 = 1. / lib.direct_sum('a-i->ai', e_a_s2, e_i_s2)
            atmlist = range(self.mol.natm)
            blksize = max(2, int(max_memory*1e6/8 / (((nvir_sub1*nocc_sub1) + (nvir_sub2*nocc_sub2))*3*6)))
            total_u = [[None]*self.mol.natm,[None]*self.mol.natm]
            for ia0, ia1 in lib.prange(0, len(atmlist), blksize):
                b_terms = []
                for i0 in range(ia0, ia1):
                    atm_index = atmlist[i0]
                    b_terms.append(self.get_b_terms(atm_index))
                b_terms = np.vstack(b_terms)
                print ('pre krylov')
                u_temp = lib.krylov(vind_vo, b_terms.ravel())
                print ('post krylov')
                u_temp = u_temp.reshape(b_terms.shape)
                u_sub1 = u_temp[:,:(nvir_sub1*nocc_sub1)].reshape(-1,3,nvir_sub1, nocc_sub1)
                u_sub2 = u_temp[:,(nvir_sub1*nocc_sub1):].reshape(-1,3,nvir_sub2, nocc_sub2)

                for k in range(ia1-ia0):
                    ia = atmlist[k+ia0]
                    total_u[0][ia] = u_sub1[k]
                    total_u[1][ia] = u_sub2[k]

            self.total_u = total_u
            return True


    def get_sub_den_grad(self):
        """After F&T cycles, calculate the density derivative terms"""

        self.solve_u()
        atmlist = range(self.mol.natm)
        self.ao_dm_grad = [[None] * self.mol.natm, [None]*self.mol.natm]
        atm_s2s = self.atm_sub2sup
        s2s = self.sub2sup

        full_s1 = -self.mol.intor('int1e_ipovlp', comp=3)
        full_aoslices = self.mol.aoslice_by_atom()

        sub1 = self.subsystems[0]
        occidx_sub1 = sub1.env_mo_occ[0] > 0
        viridx_sub1 = sub1.env_mo_occ[0] == 0
        nocc_sub1 = np.count_nonzero(occidx_sub1)
        nvir_sub1 = sub1.env_mo_occ[0].size - nocc_sub1
        nao_sub1 = sub1.env_mo_coeff[0].shape[0]
        mocc_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]>0]
        vir_sub1 = sub1.env_mo_coeff[0][:,sub1.env_mo_occ[0]==0]
        sub2 = self.subsystems[1]
        occidx_sub2 = sub2.env_mo_occ[0] > 0
        viridx_sub2 = sub2.env_mo_occ[0] == 0
        nocc_sub2 = np.count_nonzero(occidx_sub2)
        nvir_sub2 = sub2.env_mo_occ[0].size - nocc_sub2
        nao_sub2 = sub2.env_mo_coeff[0].shape[0]
        mocc_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]>0]
        vir_sub2 = sub2.env_mo_coeff[0][:,sub2.env_mo_occ[0]==0]
        xyz = [0,1,2]
        for atm in atmlist:
            dm_grad_sub1 = np.zeros((3, sub1.env_mo_coeff[0].shape[0], sub1.env_mo_coeff[0].shape[0]))
            dm_grad_sub2 = np.zeros((3, sub2.env_mo_coeff[0].shape[0], sub2.env_mo_coeff[0].shape[0]))

            p0,p1 = full_aoslices[atm, 2:]

            s1_ao_atm = np.zeros_like(full_s1)
            s1_ao_atm[:,p0:p1] += full_s1[:,p0:p1]
            s1_ao_atm[:,:,p0:p1] += full_s1[:,p0:p1].transpose(0,2,1)
            s1_ao_sub1 = copy.copy(s1_ao_atm[np.ix_(xyz, s2s[0], s2s[0])])
            s1_ao_sub2 = copy.copy(s1_ao_atm[np.ix_(xyz, s2s[1], s2s[1])])

            occ_sub1 = np.einsum('mi,xmn,nj->xij', mocc_sub1, s1_ao_sub1, mocc_sub1)
            occ_sub2 = np.einsum('mi,xmn,nj->xij', mocc_sub2, s1_ao_sub2, mocc_sub2)
            occ_sub1 *= -1
            occ_sub2 *= -1
            dm_grad_sub1[:,:nocc_sub1, :nocc_sub1] += occ_sub1
            dm_grad_sub2[:,:nocc_sub2, :nocc_sub2] += occ_sub2

            dm_grad_sub1[:, nocc_sub1:, :nocc_sub1] += self.total_u[0][atm]
            dm_grad_sub1[:, :nocc_sub1, nocc_sub1:] += self.total_u[0][atm].transpose(0,2,1)

            dm_grad_sub2[:,nocc_sub2:, :nocc_sub2] += self.total_u[1][atm]
            dm_grad_sub2[:,:nocc_sub2, nocc_sub2:] += self.total_u[1][atm].transpose(0,2,1)

            self.ao_dm_grad[0][atm] = 2.* np.einsum('mr,xrs,ns->xmn', sub1.env_mo_coeff[0], dm_grad_sub1, sub1.env_mo_coeff[0])
            self.ao_dm_grad[1][atm] = 2.* np.einsum('mr,xrs,ns->xmn', sub2.env_mo_coeff[0], dm_grad_sub2, sub2.env_mo_coeff[0])

        return self.ao_dm_grad




        #full_xterms = self.get_x_terms(self.zvec)
        #u_terms = []
        #dm_terms = []
        ##THIS IS A TEST.
        #full_x = np.zeros((full_xterms[0].size + full_xterms[1].size))
        #full_b = np.zeros((full_xterms[0].size + full_xterms[1].size))
        #full_z = np.zeros((full_xterms[0].size + full_xterms[1].size))
        #print (full_x.shape)
        #start_index = 0
        #for i, sub in enumerate(self.subsystems):
        #    end_index = start_index + full_xterms[i].size
        #    full_x[start_index:end_index] += self.base_xterms[i].ravel()
        #    full_b[start_index:end_index] += self.b_terms[i][0][0].ravel()
        #    full_z[start_index:end_index] += self.zvec[i].ravel()
        #    start_index = end_index
        #inv_x = np.reciprocal(full_x)
        #temp_u = np.dot(inv_x.T, np.dot(full_z.T, full_b))
        #print (temp_u[:self.zvec[0].size])


        #for i, sub in enumerate(self.subsystems):
        #    u_sub = []
        #    dm_sub = []
        #    for atm in range(self.mol.natm):
        #        inv_x = np.linalg.pinv(full_xterms[i].T)
        #        temp_u = np.dot(inv_x, np.dot(self.zvec[i].T, self.b_terms[i][atm][0]))
        #        print (temp_u)
        #        #print(np.einsum('ai,ai,xai->xai', inv_x, self.zvec[i], self.b_terms[i][atm]))
        #        print (x)
        #        u_sub.append(np.einsum('ai,ai,xai->xai', inv_x, self.zvec[i], self.b_terms[i][atm]))
        #        

        #    u_terms.append(u_sub)
        #return u_terms


    def get_emb_nuc_grad(self, den_grad=None):
        """After a F&T embedding convergence, calculates the embedding nuclear gradient. 
        Currently only for 1 subsystem and the HL has to be subsystem 0."""

        #Get numerical components
        s2s = self.sub2sup
        xyz = [0,1,2]
        full_aoslices = self.mol.aoslice_by_atom()
        atm_s2s = self.atm_sub2sup
        self.get_supersystem_nuc_grad()
        #Always the first of the first (HL) subsystem. All of this is in Bohr
        if not den_grad is None:
            num_sub1_den_grad, num_sub2_den_grad = den_grad

       
        full_sub1_den_grad = np.zeros((2,3,self.fs_dmat[0].shape[0], self.fs_dmat[0].shape[1]))
        full_sub1_den_grad[np.ix_([0,1],xyz, s2s[0], s2s[0])] += num_sub1_den_grad
        full_sub2_den_grad = np.zeros((2,3,self.fs_dmat[0].shape[0], self.fs_dmat[0].shape[1]))
        full_sub2_den_grad[np.ix_([0,1],xyz, s2s[1], s2s[1])] += num_sub2_den_grad
        full_emb_den_grad = full_sub1_den_grad + full_sub2_den_grad


        full_sub1_den = np.zeros_like(self.fs_dmat)
        full_sub1_den[np.ix_([0,1], s2s[0], s2s[0])] += self.subsystems[0].env_dmat
        full_sub2_den = np.zeros_like(self.fs_dmat)
        full_sub2_den[np.ix_([0,1], s2s[1], s2s[1])] += self.subsystems[1].env_dmat
        full_emb_den = full_sub1_den + full_sub2_den

        grad_subsys = cluster_subsystem.ClusterEnvSubSystemGrad(self.subsystems[0])
        sub1_env_en_grad = grad_subsys.grad_elec()
        sub_aoslices = grad_subsys.subsys.mol.aoslice_by_atom()

        #Get emb and proj components
        emb_grad = np.zeros_like(sub1_env_en_grad)
        proj_grad = np.zeros_like(sub1_env_en_grad)
        hcore_full = self.fs_nuc_grad_obj.hcore_generator()
        hcore_sub = grad_subsys.grad_obj.hcore_generator()
        ao2int_full = self.mol.intor('int2e')
        ao2int_sub = self.mol.intor('int2e')
        ao2int_grad_full = self.mol.intor('int2e_ip1')
        ao2int_grad_sub = self.mol.intor('int2e_ip1')
        for atm in range(grad_subsys.subsys.mol.natm):
            full_atm = atm_s2s[0][atm]
            h_mat = hcore_full(full_atm)
            hmat_full = h_mat[np.ix_(xyz,s2s[0],s2s[0])]
            hcore_emb_grad = hmat_full - hcore_sub(atm)
            hcore_emb_e = np.einsum('xij,ij->x', hcore_emb_grad, grad_subsys.subsys.env_dmat[0] + grad_subsys.subsys.env_dmat[1])
            p0_sub, p1_sub = sub_aoslices[atm, 2:]
            p0_full, p1_full = full_aoslices[full_atm,2:]
       
            if isinstance(grad_subsys.subsys.env_scf, (dft.rks.RKS, dft.roks.ROKS, dft.uks.UKS)):
                #veff_emb
                #This term is tricky. Need to figure out how to get grad of veff emb.
                atm_index = self.atm_sub2sup[0][0]
                p0, p1 = aoslices [atm_index, 2:]

                #coul term
                atm_ao_2int_grad = np.zeros_like(ao_2int_grad)
                atm_ao_2int_grad[:,p0:p1] += ao_2int_grad[:,p0:p1]
                atm_ao_2int_grad += atm_ao_2int_grad.transpose(0,2,1,3,4) + atm_ao_2int_grad.transpose(0,3,4,1,2) + atm_ao_2int_grad.transpose(0,3,4,2,1)
                atm_ao_2int_grad *= -1.

                ana_coul_emb = np.einsum('xijkl,ji->xkl',atm_ao_2int_grad,(sub_b_dmat[0] + sub_b_dmat[1]))
                ana_coul_emb[0] += np.einsum('ijkl,ji->kl',ao_2int, (sub_b_dmat_grad[0] + sub_b_dmat_grad[1]))

                #exch term
                omega, alph, hyb = self.env_in_env_scf._numint.rsh_and_hybrid_coeff(self.env_in_env_scf.xc, spin=self.mol.spin)
                ana_exc_emb = np.array([np.zeros_like(ana_coul_emb), np.zeros_like(ana_coul_emb)])
                if abs(hyb) >= 1e-10:
                    ana_exc_emb[0] = np.einsum('xijkl,jk->xil', atm_ao_2int_grad, sub_b_dmat[0])
                    ana_exc_emb[1] = np.einsum('xijkl,jk->xil', atm_ao_2int_grad, sub_b_dmat[1])
                    ana_exc_emb[0][0] += np.einsum('ijkl,jk->il', ao_2int, sub_b_dmat_grad[0])
                    ana_exc_emb[1][0] += np.einsum('ijkl,jk->il', ao_2int, sub_b_dmat_grad[1])
                    ana_exc_emb *= hyb

                #fxc term
                #Something is wrong with this term.
                #could be the grid term.
                fxc_emb = self.env_in_env_scf._numint.nr_fxc(self.mol, self.env_in_env_scf.grids, self.env_in_env_scf.xc, sub_b_dmat, sub_b_dmat_grad, spin=self.mol.spin)

                sub_ana_coul_emb_grad = ana_coul_emb[np.ix_(xyz, s2s[0], s2s[0])]
                sub_ana_exc_emb_grad = ana_exc_emb[np.ix_([0,1], xyz, s2s[0], s2s[0])]
                sub_ana_fxc_emb_grad = fxc_emb[np.ix_([0,1], s2s[0], s2s[0])]
            #Using hf for embedding potential
            else:
                #J
                atm_ao2int_grad_full = np.zeros_like(ao2int_grad_full)
                atm_ao2int_grad_full[:,p0_full:p1_full] += ao2int_grad_full[:,p0_full:p1_full]
                atm_ao2int_grad_full += atm_ao2int_grad_full.transpose(0,2,1,3,4) + atm_ao2int_grad_full.transpose(0,3,4,1,2) + atm_ao2int_grad_full.transpose(0,3,4,2,1)
                atm_ao2int_grad_full *= -1.
                sub1_coul_emb = np.einsum('xijkl,ji->xkl', atm_ao2int_grad_full, (full_sub2_den[0] + full_sub2_den[1]))
                sub1_coul_emb += np.einsum('ijkl,xji->xkl', ao2int_full, (full_sub2_den_grad[0] + full_sub2_den_grad[1]))
                #K
                #sub1_exch_emb = [None, None]
                sub1_exch_emb = np.einsum('xijkl,sjk->sxil', atm_ao2int_grad_full, full_sub2_den)
                #sub1_exch_emb[1] = np.einsum('xijkl,sjk->sxil', atm_ao2int_grad_full, full_sub2_den[1])
                sub1_exch_emb += np.einsum('ijkl,sxjk->sxil', ao2int_full, full_sub2_den_grad)

                sub1_veff_emb = sub1_coul_emb[np.ix_(xyz,s2s[0],s2s[0])] - sub1_exch_emb[np.ix_([0,1],xyz,s2s[0],s2s[0])]
                sub1_veff_emb_e = np.einsum('sxij,sij->x', sub1_veff_emb[:,:,p0_sub:p1_sub], np.array(grad_subsys.subsys.env_dmat)[:,p0_sub:p1_sub])

            emb_grad[atm] += hcore_emb_e + sub1_veff_emb_e

            #proj part
            if isinstance(grad_subsys.subsys.env_scf, (dft.rks.RKS, dft.roks.ROKS, dft.uks.UKS)):
                pass
            else:
                #F'DS + FD'S + FDS' + S'DF + SD'F + SDF'
                fock_grad_ab = np.array([h_mat[np.ix_(xyz, s2s[0],s2s[1])], h_mat[np.ix_(xyz,s2s[0],s2s[1])]])
                fock_grad_ab[0] += np.einsum('xijkl,ji->xkl', atm_ao2int_grad_full, (full_emb_den[0] + full_emb_den[1]))[np.ix_(xyz, s2s[0],s2s[1])]
                fock_grad_ab[1] += np.einsum('xijkl,ji->xkl', atm_ao2int_grad_full, (full_emb_den[0] + full_emb_den[1]))[np.ix_(xyz, s2s[0],s2s[1])]
                fock_grad_ab[0] += np.einsum('ijkl,xji->xkl', ao2int_full, (full_emb_den_grad[0] + full_emb_den_grad[1]))[np.ix_(xyz,s2s[0],s2s[1])]
                fock_grad_ab[1] += np.einsum('ijkl,xji->xkl', ao2int_full, (full_emb_den_grad[0] + full_emb_den_grad[1]))[np.ix_(xyz,s2s[0],s2s[1])]
                fock_grad_ab -= np.einsum('xijkl,sjk->sxil', atm_ao2int_grad_full, full_emb_den)[np.ix_([0,1], xyz, s2s[0],s2s[1])]
                fock_grad_ab -= np.einsum('ijkl,sxjk->sxil', ao2int_full, full_emb_den_grad)[np.ix_([0,1], xyz,s2s[0],s2s[1])]
                smat_ba = self.smat[np.ix_(s2s[1],s2s[0])]
                fgradds = np.einsum('sxnj,sji,il->sxnl', fock_grad_ab, self.subsystems[1].env_dmat, smat_ba)
                fock_ab = np.array(self.fock)[np.ix_([0,1],s2s[0],s2s[1])]
                fdgrads = np.einsum('snj,sxji,il->sxnl', fock_ab, num_sub2_den_grad, smat_ba)
                atm_smat_grad = np.zeros_like(self.fs_nuc_grad_obj.get_ovlp())
                atm_smat_grad[:, p0_full:p1_full] += self.fs_nuc_grad_obj.get_ovlp()[:, p0_full:p1_full]
                smat_grad_ab = atm_smat_grad[np.ix_(xyz,s2s[0],s2s[1])]
                smat_grad_ba = smat_grad_ab.transpose(0,2,1)
                fdsgrad = np.einsum('snj,sji,xil->sxnl', fock_ab, self.subsystems[1].env_dmat, smat_grad_ba)

                proj = fgradds + fdgrads + fdsgrad
                proj += proj.transpose(0,1,3,2)
                proj *= -1
                proj_grad[atm] = np.einsum('sxij,sij->x', proj, grad_subsys.subsys.env_dmat)

        return sub1_env_en_grad + emb_grad + proj_grad
        #Hcore num grad
        #num_sub1_h_energy_grad = np.trace(np.dot(subsys1.env_hcore, subsys1.env_dmat[0]+subsys1.env_dmat[1]))
        #num_sub1_h_energy_grad -= np.trace(np.dot(subsys0.env_hcore, subsys0.env_dmat[0]+subsys0.env_dmat[1]))
        #num_sub1_h_energy_grad /= (x_dir_diff*2.)

        #num_sub1_h_pot_grad = (subsys1.env_hcore - subsys0.env_hcore)/(x_dir_diff*2.)
        #num_sub2_h_pot_grad = (env_subsys1.env_hcore - env_subsys0.env_hcore)/(x_dir_diff*2.)

        ##VHF num grad
        #subsys0_grid_weights = custom_pyscf_methods.grids_response(subsys0.env_scf.grids)[1]
        #subsys1_grid_weights = custom_pyscf_methods.grids_response(subsys1.env_scf.grids)[1]
        #num_sub1_grid_weights_grad = (subsys1_grid_weights - subsys0_grid_weights)/(x_dir_diff*2.)

        #subsys0_vhf = subsys0.env_scf.get_veff(dm=subsys0.env_dmat)
        #subsys1_vhf = subsys1.env_scf.get_veff(dm=subsys1.env_dmat)
        #env_subsys0_vhf = env_subsys0.env_scf.get_veff(dm=env_subsys0.env_dmat)
        #env_subsys1_vhf = env_subsys1.env_scf.get_veff(dm=env_subsys1.env_dmat)

        ##num_sub1_exc_grad = (subsys1_vhf.exc - subsys0_vhf.exc)/(x_dir_diff*2.)
        ##num_sub2_exc_grad = (env_subsys1_vhf.exc - env_subsys0_vhf.exc)/(x_dir_diff*2.)
        ###num_sub1_j_energy_grad = (subsys1_vhf.ecoul - subsys0_vhf.ecoul)/(x_dir_diff*2.)
        ###num_sub2_j_energy_grad = (env_subsys1_vhf.ecoul - env_subsys0_vhf.ecoul)/(x_dir_diff*2.)

        ##num_sub1_j_pot_grad = (subsys1_vhf.vj - subsys0_vhf.vj)/(x_dir_diff*2.)
        ##num_sub2_j_pot_grad = (env_subsys1_vhf.vj - env_subsys0_vhf.vj)/(x_dir_diff*2.)
        ##num_sub1_k_pot_grad = (subsys1_vhf.vk - subsys0_vhf.vk)/(x_dir_diff*2.)
        ##num_sub2_k_pot_grad = (env_subsys1_vhf.vk - env_subsys0_vhf.vk)/(x_dir_diff*2.)
        ##num_sub1_vhf_pot_grad = (subsys1_vhf - subsys0_vhf)/(x_dir_diff*2.)
        ##num_sub2_vhf_pot_grad = (env_subsys1_vhf - env_subsys0_vhf)/(x_dir_diff*2.)

        ##num_sub1_xcfun_pot_grad = num_sub1_vhf_pot_grad - num_sub1_j_pot_grad + num_sub1_k_pot_grad
        ##num_sub2_xcfun_pot_grad = num_sub2_vhf_pot_grad - num_sub2_j_pot_grad + num_sub2_k_pot_grad

        ##Proj Pot num grad
        ##num_sub1_p_energy_grad = np.trace(np.dot(subsys1.proj_pot[0], subsys1.env_dmat[0]))
        ##num_sub1_p_energy_grad += np.trace(np.dot(subsys1.proj_pot[1], subsys1.env_dmat[1]))
        ##num_sub1_p_energy_grad -= np.trace(np.dot(subsys0.proj_pot[0], subsys0.env_dmat[0]))
        ##num_sub1_p_energy_grad -= np.trace(np.dot(subsys0.proj_pot[1], subsys0.env_dmat[1]))
        ##num_sub1_p_energy_grad /= (x_dir_diff*2.)

        #num_sub1_p_pot_grad = (np.array(subsys1.proj_pot) - np.array(subsys0.proj_pot))/(x_dir_diff*2.)

        ##Embed Pot num grad
        #emb_pot1 = subsys1.emb_fock - subsys1.subsys_fock
        #emb_pot0 = subsys0.emb_fock - subsys0.subsys_fock
        ##num_sub1_e_energy_grad = np.trace(np.dot(emb_pot1[0], subsys1.env_dmat[0]))
        ##num_sub1_e_energy_grad += np.trace(np.dot(emb_pot1[1], subsys1.env_dmat[1]))
        ##num_sub1_e_energy_grad -= np.trace(np.dot(emb_pot0[0], subsys0.env_dmat[0]))
        ##num_sub1_e_energy_grad -= np.trace(np.dot(emb_pot0[1], subsys0.env_dmat[1]))
        ##num_sub1_e_energy_grad /= (x_dir_diff*2.)

        #emb_hcore1 = supersystem1.hcore[np.ix_(s2s[0],s2s[0])] - subsys1.env_hcore
        #emb_hcore0 = supersystem0.hcore[np.ix_(s2s[0],s2s[0])] - subsys0.env_hcore

        #num_sub1_embhcore_pot_grad = (emb_hcore1 - emb_hcore0)/(x_dir_diff*2.)

        #sub1_emb_pot = subsys1.emb_fock - subsys1.subsys_fock
        #sub0_emb_pot = subsys0.emb_fock - subsys0.subsys_fock
        #env_sub1_emb_pot = env_subsys1.emb_fock - env_subsys1.subsys_fock
        #env_sub0_emb_pot = env_subsys0.emb_fock - env_subsys0.subsys_fock
        #num_sub1_emb_pot_grad = (np.array(sub1_emb_pot) - np.array(sub0_emb_pot))/(x_dir_diff*2.)
        #num_sub2_emb_pot_grad = (np.array(env_sub1_emb_pot) - np.array(env_sub0_emb_pot))/(x_dir_diff*2.)

        #print ('num sub1 energy grad')
        #print (num_sub1_energy_grad)

        ##SUM TO GET NUMERICAL TERMS
        ##hcore
        #grad_subsys = self.subsystems[0]
        #dm = grad_subsys.env_dmat
        ##emb
        #emb_pot = grad_subsys.emb_fock - grad_subsys.subsys_fock
        #num_sub1_emb_energy_grad = np.trace(np.dot(num_sub1_emb_pot_grad[0], dm[0]))
        #num_sub1_emb_energy_grad += np.trace(np.dot(num_sub1_emb_pot_grad[1], dm[1]))
        #num_sub1_emb_energy_grad += np.trace(np.dot(emb_pot[0], num_sub1_den_grad[0]))
        #num_sub1_emb_energy_grad += np.trace(np.dot(emb_pot[1], num_sub1_den_grad[1]))
        ##proj
        #proj_pot = grad_subsys.proj_pot
        #num_sub1_proj_energy_grad = np.trace(np.dot(num_sub1_p_pot_grad[0], dm[0]))
        #num_sub1_proj_energy_grad += np.trace(np.dot(num_sub1_p_pot_grad[1], dm[1]))
        #num_sub1_proj_energy_grad += np.trace(np.dot(proj_pot[0], num_sub1_den_grad[0]))
        #num_sub1_proj_energy_grad += np.trace(np.dot(proj_pot[1], num_sub1_den_grad[1]))

        ##sub1 terms
        ##hcore
        #num_sub1_hcore_energy_grad = np.trace(np.dot(num_sub1_h_pot_grad, dm[0] + dm[1]))
        #num_sub1_hcore_energy_grad += np.trace(np.dot(grad_subsys.env_hcore, num_sub1_den_grad[0] + num_sub1_den_grad[1]))
        ##veff
        ##sub_vhf = grad_subsys.env_scf.get_veff(dm=dm)
        ##num_sub1_coul_energy_grad = np.trace(np.dot(num_sub1_j_pot_grad, dm[0] + dm[1]))
        ##num_sub1_coul_energy_grad += np.trace(np.dot(sub_vhf.vj, num_sub1_den_grad[0] + num_sub1_den_grad[1]))
        ##num_sub1_coul_energy_grad *= 0.5

        ##num_sub1_x_energy_grad = np.trace(np.dot(num_sub1_k_pot_grad[0], dm[0]))
        ##num_sub1_x_energy_grad += np.trace(np.dot(num_sub1_k_pot_grad[1], dm[1]))
        ##num_sub1_x_energy_grad += np.trace(np.dot(sub_vhf.vk[0], num_sub1_den_grad[0]))
        ##num_sub1_x_energy_grad += np.trace(np.dot(sub_vhf.vk[1], num_sub1_den_grad[1]))
        ##num_sub1_x_energy_grad *= 0.5 

        ##num_sub1_xc_energy_grad = np.trace(np.dot(num_sub1_xcfun_pot_grad[0], dm[0]))
        ##num_sub1_xc_energy_grad += np.trace(np.dot(num_sub1_xcfun_pot_grad[1], dm[1]))
        ##num_sub1_xc_energy_grad += np.trace(np.dot(sub_vhf[0] - sub_vhf.vj + sub_vhf.vk[0], num_sub1_den_grad[0]))
        ##num_sub1_xc_energy_grad += np.trace(np.dot(sub_vhf[1] - sub_vhf.vj + sub_vhf.vk[1], num_sub1_den_grad[1]))

        ##SUM TO GET NUMERICAL TERMS

        ##Potential components:
        #num_sub1_emb_deriv = np.trace(np.dot(num_sub1_emb_pot_grad[0], dm[0]))
        #num_sub1_emb_deriv += np.trace(np.dot(num_sub1_emb_pot_grad[1], dm[1]))

        #num_sub1_proj_deriv = np.trace(np.dot(num_sub1_p_pot_grad[0], dm[0]))
        #num_sub1_proj_deriv += np.trace(np.dot(num_sub1_p_pot_grad[1], dm[1]))

        ##Analytical gradients.
        #self.get_supersystem_nuc_grad()
        ##emb
        #grad_subsys_obj = grad_subsys.env_scf.nuc_grad_method()
        #emb_pot = grad_subsys.emb_fock - grad_subsys.subsys_fock

        ##hcore_emb
        #s2s = self.sub2sup
        #aoslices = self.mol.aoslice_by_atom()
        #hcore_full = self.fs_nuc_grad_obj.hcore_generator()
        #hcore_sub = grad_subsys_obj.hcore_generator()

        #sub_b_dmat = np.zeros_like(self.fs_dmat)
        #sub_b_dmat[0][np.ix_(s2s[1],s2s[1])] += self.subsystems[1].env_dmat[0]
        #sub_b_dmat[1][np.ix_(s2s[1],s2s[1])] += self.subsystems[1].env_dmat[1]

        #sub_b_dmat_grad = np.zeros_like(self.fs_dmat)
        #sub_b_dmat_grad[0][np.ix_(s2s[1],s2s[1])] += num_sub2_den_grad[0]
        #sub_b_dmat_grad[1][np.ix_(s2s[1],s2s[1])] += num_sub2_den_grad[1]

        #sub_a_dmat = np.zeros_like(self.fs_dmat)
        #sub_a_dmat[0][np.ix_(s2s[0],s2s[0])] += self.subsystems[0].env_dmat[0]
        #sub_a_dmat[1][np.ix_(s2s[0],s2s[0])] += self.subsystems[0].env_dmat[1]

        #ao_2int_grad = self.mol.intor('int2e_ip1')
        #ao_2int = self.mol.intor('int2e')

        ##veff_grad = self.fs_nuc_grad_obj.get_veff(dm=sub_b_dmat)
        #self.fs_nuc_grad_obj.grid_response = True
        #veff_grad = self.fs_nuc_grad_obj.get_veff(dm=sub_a_dmat)
        #print ('exc relax')
        #print (veff_grad.exc1_grid)

        ##xyz = [0,1,2]
        ##for atm in range(grad_subsys.mol.natm):
        ##    h_mat = hcore_full(atm)
        ##    hmat_full = h_mat[np.ix_(xyz,s2s[0],s2s[0])]
        ##    hcore_emb = hmat_full - hcore_sub(atm)
        ##    hcore_emb_e = np.einsum('xij,ij->x', hcore_emb, grad_subsys.env_dmat[0] + grad_subsys.env_dmat[1])
        ##
        ##    #veff_emb
        ##    #This term is tricky. Need to figure out how to get grad of veff emb.
        ##    atm_index = self.atm_sub2sup[0][0]
        ##    p0, p1 = aoslices [atm_index, 2:]

        ##    #coul term
        ##    atm_ao_2int_grad = np.zeros_like(ao_2int_grad)
        ##    atm_ao_2int_grad[:,p0:p1] += ao_2int_grad[:,p0:p1]
        ##    atm_ao_2int_grad += atm_ao_2int_grad.transpose(0,2,1,3,4) + atm_ao_2int_grad.transpose(0,3,4,1,2) + atm_ao_2int_grad.transpose(0,3,4,2,1)
        ##    atm_ao_2int_grad *= -1.

        ##    ana_coul_emb = np.einsum('xijkl,ji->xkl',atm_ao_2int_grad,(sub_b_dmat[0] + sub_b_dmat[1]))
        ##    ana_coul_emb[0] += np.einsum('ijkl,ji->kl',ao_2int, (sub_b_dmat_grad[0] + sub_b_dmat_grad[1]))

        ##    #exch term
        ##    omega, alph, hyb = self.env_in_env_scf._numint.rsh_and_hybrid_coeff(self.env_in_env_scf.xc, spin=self.mol.spin)
        ##    ana_exc_emb = np.array([np.zeros_like(ana_coul_emb), np.zeros_like(ana_coul_emb)])
        ##    if abs(hyb) >= 1e-10:
        ##        ana_exc_emb[0] = np.einsum('xijkl,jk->xil', atm_ao_2int_grad, sub_b_dmat[0])
        ##        ana_exc_emb[1] = np.einsum('xijkl,jk->xil', atm_ao_2int_grad, sub_b_dmat[1])
        ##        ana_exc_emb[0][0] += np.einsum('ijkl,jk->il', ao_2int, sub_b_dmat_grad[0])
        ##        ana_exc_emb[1][0] += np.einsum('ijkl,jk->il', ao_2int, sub_b_dmat_grad[1])
        ##        ana_exc_emb *= hyb

        ##    #fxc term
        ##    #Something is wrong with this term.
        ##    #could be the grid term.
        ##    fxc_emb = self.env_in_env_scf._numint.nr_fxc(self.mol, self.env_in_env_scf.grids, self.env_in_env_scf.xc, sub_b_dmat, sub_b_dmat_grad, spin=self.mol.spin)

        ##    sub_ana_coul_emb_grad = ana_coul_emb[np.ix_(xyz, s2s[0], s2s[0])]
        ##    sub_ana_exc_emb_grad = ana_exc_emb[np.ix_([0,1], xyz, s2s[0], s2s[0])]
        ##    sub_ana_fxc_emb_grad = fxc_emb[np.ix_([0,1], s2s[0], s2s[0])]
        ##    print ("ana veff terms")
        ##    print (sub_ana_coul_emb_grad[0])
        ##    print (sub_ana_exc_emb_grad[0][0])
        ##    print (sub_ana_fxc_emb_grad[0])
        ##    print (emb_veff_grad_xc[0])
        ##    #print (x)

        ##    veff_emb_e = np.einsum('xij,ij->x', veff_grad[0][:,p0:p1,p0:p1], grad_subsys.env_dmat[0])
        ##    veff_emb_e += np.einsum('xij,ij->x', veff_grad[1][:,p0:p1,p0:p1], grad_subsys.env_dmat[1])

        ##    #Use the fxc and numerical integration with density derivative to get veff emb.


        ##    print ('hcore emb')
        ##    print (hcore_emb_e)
        ##    print ('veff emb')
        ##    print (veff_emb_e)
        ##
        ##    num_sub1_e_grad = np.trace(np.dot(num_sub1_emb_grad[0], dm[0]))
        ##    num_sub1_e_grad += np.trace(np.dot(num_sub1_emb_grad[1], dm[1]))

        ##    print ('sub emb e')
        ##    print (num_sub1_e_grad)

        ##    #proj
        ##    proj_pot = grad_subsys.proj_pot
        ##    num_sub1_p_grad = np.trace(np.dot(num_sub1_proj_grad[0], dm[0]))
        ##    num_sub1_p_grad += np.trace(np.dot(num_sub1_proj_grad[1], dm[1]))

        ##    #analytical portion.
        ##    #FDS + SDF
        ##    f_ab = [None, None]
        ##    f_ba = [None, None]
        ##    f_ab[0] = self.fock[0][np.ix_(s2s[0], s2s[1])]
        ##    f_ab[1] = self.fock[1][np.ix_(s2s[0], s2s[1])]
        ##    f_ba[0] = self.fock[0][np.ix_(s2s[1], s2s[0])]
        ##    f_ba[1] = self.fock[1][np.ix_(s2s[1], s2s[0])]

        ##    d_bb = self.subsystems[1].env_dmat
        ##    s_ba = self.smat[np.ix_(s2s[1], s2s[0])]
        ##    s_ab = self.smat[np.ix_(s2s[0], s2s[1])]

        ##    atm = 0
        ##    vhf_grad = 0.
        ##    d_bb_grad = num_sub2_den_grad

        ##    p0,p1 = aoslices [0,2:]

        ##    s_grad = self.fs_nuc_grad_obj.get_ovlp()[:,p0:p1]
        ##    s_ab_grad = s_grad[0][np.ix_(s2s[0], s2s[1])]
        ##    s_grad = self.fs_nuc_grad_obj.get_ovlp()[:,:,p0:p1]
        ##    s_ba_grad = s_grad[0][np.ix_(s2s[1], s2s[0])]

        ##    f_ab_grad = [None, None]
        ##    f_ab_grad[0] = (hcore_full(atm) + vhf_grad)[0][np.ix_(s2s[0], s2s[1])]
        ##    f_ab_grad[1] = (hcore_full(atm) + vhf_grad)[0][np.ix_(s2s[0], s2s[1])]
        ##    f_ba_grad = [None, None]
        ##    f_ba_grad[0] = (hcore_full(atm) + vhf_grad)[0][np.ix_(s2s[1], s2s[0])]
        ##    f_ba_grad[1] = (hcore_full(atm) + vhf_grad)[0][np.ix_(s2s[1], s2s[0])]


        ##    ana_proj_grad = [None, None]
        ##    ana_proj_grad[0] = np.linalg.multi_dot([f_ab_grad[0],d_bb[0],s_ba])
        ##    ana_proj_grad[1] = np.linalg.multi_dot([f_ab_grad[1],d_bb[1],s_ba])
        ##    ana_proj_grad[0] += np.linalg.multi_dot([f_ab[0],d_bb_grad[0],s_ba])
        ##    ana_proj_grad[1] += np.linalg.multi_dot([f_ab[1],d_bb_grad[1],s_ba])
        ##    ana_proj_grad[0] += np.linalg.multi_dot([f_ab[0],d_bb[0],s_ba_grad])
        ##    ana_proj_grad[1] += np.linalg.multi_dot([f_ab[1],d_bb[1],s_ba_grad])

        ##    print (np.trace(np.dot(grad_subsys.env_dmat[0],ana_proj_grad[0])))
        ##    print (np.trace(np.dot(grad_subsys.env_dmat[1],ana_proj_grad[1])))
        ##    ana_proj_grad[0] += np.linalg.multi_dot([s_ab_grad,d_bb[0],f_ba[0]])
        ##    ana_proj_grad[1] += np.linalg.multi_dot([s_ab_grad,d_bb[1],f_ba[1]])
        ##    ana_proj_grad[0] += np.linalg.multi_dot([s_ab,d_bb_grad[0],f_ba[0]])
        ##    ana_proj_grad[1] += np.linalg.multi_dot([s_ab,d_bb_grad[1],f_ba[1]])
        ##    ana_proj_grad[0] += np.linalg.multi_dot([s_ab,d_bb[0],f_ba_grad[0]])
        ##    ana_proj_grad[1] += np.linalg.multi_dot([s_ab,d_bb[1],f_ba_grad[1]])
        ##    print ("proj_grad")
        ##    print (np.trace(np.dot(grad_subsys.env_dmat[0],ana_proj_grad[0])))
        ##    print (np.trace(np.dot(grad_subsys.env_dmat[1],ana_proj_grad[1])))
        ##    print (num_sub1_p_grad)



        ##This is currently only for one atom in x dimension.
        ##Need to break down this term again into components. I think there is an issue with grids.
        #coords, w0, w1 = custom_pyscf_methods.grids_response(grad_subsys.env_scf.grids)
        #print (np.max(num_sub1_grid_weights_grad - w1[0,:,0]))
        #grad_subsys = cluster_subsystem.ClusterEnvSubSystemGrad(grad_subsys)
        ##grad_subsys_obj.grid_response = True
        ##density gradient terms
        ##ana_sub1_dm0 = np.array(grad_subsys.env_dmat)
        ##numerical terms.
        ##print ('veff num')
        ##print (num_sub1_vhf_grad[0])
        ##print (np.einsum('sij,sij', num_sub1_vhf_grad, ana_sub1_dm0))


        ##analytical terms.
        ##p0,p1 = aoslices[0,2:]
        ##vhf_grad = grad_subsys_obj.get_veff(grad_subsys.mol, grad_subsys.env_dmat)
        ##print (vhf_grad.shape)
        ##print (vhf_grad[0][0]) 
        ##ana_sub1_vhf_grad_terms = np.einsum('sxij,sij->x', vhf_grad[:,:,p0:p1], ana_sub1_dm0[:,p0:p1]) * 2.
        ##print (ana_sub1_vhf_grad_terms)
        ##print (x)

        ##Density grad terms are already small but within a factor of 10^-9
        ##hcore grad terms are already small but within a factor of 10^-12

        ##Need to customize the grid correction term.
        #ana_sub1_en_grad = grad_subsys.grad_elec()

        #print ('ana sub1_en_grad')
        #print (ana_sub1_en_grad)
        #print ("sub1 energy grad")
        #print (ana_sub1_en_grad[0][0] + num_sub1_emb_deriv + num_sub1_proj_deriv)
        ##Grid term accounts for large grid used for the subsystems.
        ##Need to update the grid response term for subsystems with large grids.
        ##num_rank = self.mol.nao_nr()
        ##grid_dmat = [np.zeros((num_rank, num_rank)), np.zeros((num_rank, num_rank))]
        ##grid_dmat[0][np.ix_(s2s[0], s2s[0])] += (grad_subsys.env_dmat[0])
        ##grid_dmat[1][np.ix_(s2s[0], s2s[0])] += (grad_subsys.env_dmat[1])
        ##self.fs_nuc_grad_obj.grid_response = True
        ##vhf_sub = self.fs_nuc_grad_obj.get_veff(dm=grid_dmat)
        ##grid_response = vhf_sub.exc1_grid[0]
        ##print ('grid 1')
        ##print (grid_response)
        ##grid_dmat = [np.zeros((num_rank, num_rank)), np.zeros((num_rank, num_rank))]
        ##grid_dmat[0][np.ix_(s2s[1], s2s[1])] += (self.subsystems[1].env_dmat[0])
        ##grid_dmat[1][np.ix_(s2s[1], s2s[1])] += (self.subsystems[1].env_dmat[1])
        ##self.fs_nuc_grad_obj.grid_response = True
        ##vhf_sub = self.fs_nuc_grad_obj.get_veff(dm=grid_dmat)
        ##grid_response = vhf_sub.exc1_grid[0]
        ##print ('grid 2')
        ##print (grid_response)
        ##print (ana_sub1_en_grad[0][0] + num_sub1_e_grad + num_sub1_p_grad - grid_response[0])
        #print (num_sub1_energy_grad)


        #print (ana_sub1_en_grad[0][0] + num_sub1_e_grad + num_sub1_p_grad-num_sub1_energy_grad)


        #active_atms = self.atm_sub2sup[0]
        #self.emb_nuc_grad = 0
        #return self.emb_nuc_grad
