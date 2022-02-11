""" A method to define a cluster supersystem
Daniel Graham
Dhabih V. Chulhai"""

import os
import copy as copy
import numpy as np
import h5py

from pyscf import scf, dft, lib, lo
from pyscf.tools import cubegen, molden

from qsome import custom_diis, helpers
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
        self.proj_pot = [np.array([0.0, 0.0]) for sub in self.subsystems]
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
        if self.mol.spin != 0 or (hasattr(self.fs_scf_obj, 'unrestricted') and self.fs_scf_obj.unrestricted):
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
                    #TEMP
                    #fock_ab[0] = self.hcore[np.ix_(s2s[i], s2s[j])]
                    #fock_ab[1] = self.hcore[np.ix_(s2s[i], s2s[j])]
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

        if hasattr(self, 'save_density') and self.save_density:
            print('Writing Supersystem FT density'.center(80))
            self.save_ft_density_file()

        if hasattr(self, 'save_spin_density') and self.save_spin_density:
            print('Writing Supersystem FT spin density'.center(80))
            self.save_ft_spin_density_file()

    def get_emb_nuc_grad(self):
        """After a F&T embedding convergence, calculates the embedding nuclear gradient. 
        Currently only for 1 subsystem and the HL has to be subsystem 0."""

        self.get_supersystem_nuc_grad()

        active_atms = self.atm_sub2sup[0]


        #Get each subsystem gradient derivative.
        num_rank = self.mol.nao_nr()

        dm_env = self.get_emb_dmat()
        sub_dmat = np.zeros_like(dm_env)
        sub = self.subsystems[0]
        sub_dmat[np.ix_(self.sub2sup[0], self.sub2sup[0])] += sub.get_dmat()
        full_coulomb = self.fs_nuc_grad_obj.get_jk(self.mol, dm_env)[0] * 4.
        aoslices = self.mol.aoslice_by_atom()
        p0,p1 = aoslices[0, 2:]
        sub_coulomb = self.fs_nuc_grad_obj.get_jk(self.mol, sub_dmat)[0] * 4.
        emb_grad = full_coulomb - sub_coulomb
        emb_de = np.einsum('xij,ij->x', emb_grad[:,p0:p1], sub_dmat[p0:p1])
        sub_coulomb_grad = np.einsum('xij,ij->x', sub_coulomb, sub_dmat)
        print (emb_de)
        print (self.mol.atom[0])
       # for i, sub in enumerate(self.subsystems[1:], 1):
       #     emb_dm_env[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += sub.env_dmat[0]
       #     emb_dm_env[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += sub.env_dmat[1]
       # if not (self.unrestricted or self.mol.spin != 0):
       #     emb_dm_env = emb_dm_env[0] + emb_dm_env[1]

        #emb_env_vhf_grad = self.fs_nuc_grad_obj.get_veff(self.mol, emb_dm_env)
        #emb_env_vhf_grad = self.fs_nuc_grad_obj.get_jk(self.mol, emb_dm_env)[0]
        #print ('env_vhf_grad')
        #print (emb_env_vhf_grad[2])
        emb_hcore_deriv = self.fs_nuc_grad_obj.hcore_generator(self.mol)

        num_rank_a = self.subsystems[0].mol.nao_nr()
        s2s = self.sub2sup

        aoslices = self.mol.aoslice_by_atom()
        #Get electrostatic weighted density term.
        #aoslices = self.mol.aoslice_by_atom()
        #es_wd_de = np.zeros((self.mol.natm, 3))
        #env_s1_grad = self.fs_nuc_grad_obj.get_ovlp(self.mol)
        #sub_0_dmat = [np.zeros((num_rank, num_rank)),
        #              np.zeros((num_rank, num_rank))]
        #sub_0_dmat[0][np.ix_(self.sub2sup[0], self.sub2sup[0])] += self.subsystems[0].env_dmat[0]
        #sub_0_dmat[1][np.ix_(self.sub2sup[0], self.sub2sup[0])] += self.subsystems[0].env_dmat[1]
        #if not (self.unrestricted or self.mol.spin != 0):
        #    sub_0_dmat = sub_0_dmat[0] + sub_0_dmat[1]
        #sub_0_veff = self.fs_scf_obj.get_veff(self.mol, sub_0_dmat)
        #subsystem_dmat = self.subsystems[1].get_dmat()
        #sub_1_dmat = [np.zeros((num_rank, num_rank)),
        #              np.zeros((num_rank, num_rank))]
        #sub_1_dmat[0][np.ix_(self.sub2sup[1], self.sub2sup[1])] += self.subsystems[1].env_dmat[0]
        #sub_1_dmat[1][np.ix_(self.sub2sup[1], self.sub2sup[1])] += self.subsystems[1].env_dmat[1]
        #if not (self.unrestricted or self.mol.spin != 0):
        #    sub_1_dmat = sub_1_dmat[0] + sub_1_dmat[1]
        #vdme = np.dot(np.dot(sub_1_dmat.T, sub_0_veff), sub_1_dmat) * 0.5
        #env_s1 = self.fs_nuc_grad_obj.get_ovlp(self.mol)
        #print (env_s1)
        #print (vdme)
        #for sub_atm, sup_atm in enumerate(active_atms):
        #    p0,p1 = aoslices[sup_atm, 2:]
        #    es_wd_de[sub_atm] -= np.einsum('xij,ij->x', env_s1[:,p0:p1], vdme[p0:p1]) * 2.
        #print ("ES WD DE")
        #print (es_wd_de)


        #Calculate dft-in-dft full system fock matrix derivative.
        active_atms = self.atm_sub2sup[0]
        atom_full_hcore_grad = [None] * len(active_atms)
        atom_emb_vhf_grad = [None] * len(active_atms)
        atom_proj_grad = [None] * len(active_atms)
        atom_s1_grad = [None] * len(active_atms)

        for sub_atm, sup_atm in enumerate(active_atms):
            p0, p1 = aoslices[sup_atm,2:]
            sub_dmat = self.subsystems[0].get_dmat()
            sub_mol = self.subsystems[0].mol
            atom_hcore_grad = emb_hcore_deriv(sup_atm)
            atom_full_hcore_grad[sub_atm] = atom_hcore_grad[:,p0:p1,p0:p1] #This needs to change when doing more than one atom. 
            #atom_emb_vhf_grad[sub_atm] = emb_env_vhf_grad[:,p0:p1,p0:p1] #Again needs to change for more than one atom

            #Get electron-nuclear attraction:
            #nuc_deriv = helpers.nuc_grad_generator(self.fs_nuc_grad_obj)
            #nuc_de = nuc_deriv(sup_atm)
            #elec_nuc_de = np.einsum('xij,ij', nuc_de, emb_dm_env)
            #print ("ELEC NUC DE")
            #print (elec_nuc_de)
            ##sub_dm[p0:p1,p0:p1] += sub_dmat
            ##atom_vhf_grad = np.zeros_like(atom_hcore_grad)
            ##atom_vhf_grad[:,p0:p1] += env_vhf_grad[:,p0:p1]
            ##atom_fock_grad[sub_atm] = atom_hcore_grad + (atom_vhf_grad * 2.)
            ##atom_s1_grad[sub_atm] = np.zeros_like(atom_hcore_grad)
            ##atom_s1_grad[sub_atm][:,p0:p1] += env_s1_grad[:,p0:p1]

        #Get electron-nuclear attraction:


        # Calculate Projection Potential for subsystem 0.
        # Proj Operator is FDS + SDF. 
        # Deriv of Proj Operator is FDS' + F'DS + SDF' + S'DF.
        # Deriv of Proj Operator is FDS' + + FD'S + F'DS + SDF' + SD'F + S'DF.
        #emb_vhf_grad = self.fs_nuc_grad_obj.get_veff(self.mol, full_env_dm) - self.fs_nuc_grad_obj.get_veff(self.mol, sub_dm)
        #sub_b_dm = self.subsystems[1].get_dmat()
        #for sub_atm, sup_atm in enumerate(active_atms):
        #    p0, p1 = aoslices[sup_atm,2:]
        #    vhf_atom = np.zeros_like(atom_full_hcore_grad[sub_atm])
        #    vhf_atom[:,p0:p1] += emb_vhf_grad[:,p0:p1]
        #    temp_vhf = np.zeros((3, num_rank_a, num_rank_a))
        #    temp_hcore = np.zeros((3, num_rank_a, num_rank_a))
        #    for i in range(3):
        #        temp_vhf[i] = vhf_atom[i][np.ix_(s2s[0], s2s[0])]
        #        temp_hcore[i] = atom_full_hcore_grad[sub_atm][i][np.ix_(s2s[0], s2s[0])]

        #    atom_emb_vhf_grad[sub_atm] = temp_vhf
        #    atom_full_hcore_grad[sub_atm] = temp_hcore

        #if isinstance(self.proj_oper, (int, float)):
        #    for sub_atm, sup_atm in enumerate(active_atms):
        #        p0, p1 = aoslices[sup_atm,2:]
        #        sub0_atm = list(range(p0,p1))
        #        total_len = list(range(len(sub0_atm)))
        #        smat_ab = self.smat[np.ix_(sub0_atm, s2s[1])]
        #        smat_ba = self.smat[np.ix_(s2s[1], sub0_atm)]
        #        proj_grad = np.zeros((3, len(sub0_atm), num_rank_a))
        #        for i in range(3):
        #            env_s1_grad_ab = atom_s1_grad[sub_atm][i][np.ix_(total_len, s2s[1])]
        #            env_s1_grad_ba = env_s1_grad_ab.transpose()

        #            SDs = np.dot(smat_ab, np.dot(sub_b_dm, env_s1_grad_ba))
        #            SdS = np.dot(smat_ab, np.dot(sub_b_dm_grad[i], smat_ba))
        #            sDS = np.dot(env_s1_grad_ab, np.dot(sub_b_dm, smat_ba))
        #            proj_grad[i] += self.proj_oper * (SDs + SdS + sDS)
        #        atom_proj_grad[sub_atm] = proj_grad
        #    self.subsystems[0].atom_proj_grad = atom_proj_grad
        #    self.subsystems[0].calc_nuc_grad()
        #    
        #else:
        #    if ((hasattr(self.fs_scf_obj, 'unrestricted')
        #        and self.fs_scf_obj.unrestricted) or self.mol.spin != 0):

        #        sub_b_emb_dm = np.zeros_like(full_env_dm)
        #        sub_b_emb_dm[0][np.ix_(s2s[1], s2s[1])] += sub_b_dm[0]
        #        sub_b_emb_dm[1][np.ix_(s2s[1], s2s[1])] += sub_b_dm[1]
        #        ipnuc = self.mol.intor('int1e_ipnuc', comp=3)
        #        nuc_frozelec_grad = [None, None]
        #        nuc_frozelec_grad[0] = np.einsum('xij,ij->x', -ipnuc, sub_b_emb_dm[0])
        #        nuc_frozelec_grad[1] = np.einsum('xij,ij->x', -ipnuc, sub_b_emb_dm[1])

        #        smat_ba = self.smat[np.ix_(s2s[1], s2s[0])]
        #        temp = np.zeros((3, num_rank_a, num_rank_a))
        #        proj_grad = [copy(temp), copy(temp)]
        #        emb_fock_grad = [copy(temp), copy(temp)]
        #        fock_ab = [None, None]
        #        fock_ab[0] = self.fock[0][np.ix_(s2s[0], s2s[1])]
        #        fock_ab[1] = self.fock[1][np.ix_(s2s[0], s2s[1])]
        #        for i in range(3):
        #            FDs = [None, None]
        #            fDS = [None, None]
        #            env_fock_grad_ab = [None, None]
        #            env_fock_grad_ab[0] = env_fock_grad_ao[0][i][np.ix_(s2s[0], s2s[1])]
        #            env_fock_grad_ab[1] = env_fock_grad_ao[1][i][np.ix_(s2s[0], s2s[1])]
        #            env_s1_grad_ba = env_s1_grad[i][np.ix_(s2s[1], s2s[0])]
        #            FDs[0] = np.dot(fock_ab[0], np.dot(sub_b_dm[0], env_s1_grad_ba))
        #            FDs[1] = np.dot(fock_ab[1], np.dot(sub_b_dm[1], env_s1_grad_ba))

        #            fDS[0] = np.dot(env_fock_grad_ab[0], np.dot(sub_b_dm[0], smat_ba))
        #            fDS[1] = np.dot(env_fock_grad_ab[1], np.dot(sub_b_dm[1], smat_ba))

        #            proj_grad[0][i] += -1 * (FDs[0] + fDS[0] + FDs[0].transpose() + fDS[0].transpose())
        #            proj_grad[1][i] += -1 * (FDs[1] + fDS[1] + FDs[1].transpose() + fDS[1].transpose())

        #            #emb_fock_grad[0][i] += env_fock_grad_ao[0][i][np.ix_(s2s[0], s2s[0])]
        #            #emb_fock_grad[1][i] += env_fock_grad_ao[1][i][np.ix_(s2s[0], s2s[0])]
        #            emb_fock_grad[0][i] += nuc_frozelec_grad[0][i][np.ix_(s2s[0], s2s[0])]
        #            emb_fock_grad[1][i] += nuc_frozelec_grad[1][i][np.ix_(s2s[0], s2s[0])]
        #        
        #    else:
        #        for sub_atm, sup_atm in enumerate(active_atms):
        #            fock_ab = self.fock[0][np.ix_(s2s[0], s2s[1])]
        #            fock_ba = self.fock[0][np.ix_(s2s[1], s2s[0])]
        #            smat_ab = self.smat[np.ix_(s2s[0], s2s[1])]
        #            smat_ba = self.smat[np.ix_(s2s[1], s2s[0])]
        #            proj_grad = np.zeros((3, num_rank_a, num_rank_a))
        #            sub_atm_fock_grad = np.zeros((3, num_rank_a, num_rank_a))
        #            for i in range(3):
        #                env_fock_grad_ab = atom_fock_grad[sub_atm][i][np.ix_(s2s[0], s2s[1])]
        #                env_s1_grad_ab = atom_s1_grad[sub_atm][i][np.ix_(s2s[0], s2s[1])]

        #                env_fock_grad_ba = env_fock_grad_ab.transpose()
        #                env_s1_grad_ba = env_s1_grad_ab.transpose()

        #                FDs = np.dot(fock_ab, np.dot(sub_b_dm, env_s1_grad_ba))
        #                fDS = np.dot(env_fock_grad_ab, np.dot(sub_b_dm, smat_ba))
        #                proj_grad[i] += -0.5 * (FDs + fDS + FDs.transpose() + fDS.transpose())
        #                sub_atm_fock_grad[i] = atom_fock_grad[sub_atm][i][np.ix_(s2s[0], s2s[0])]
        #            atom_proj_grad[sub_atm] = proj_grad

        #        #Calculate Emb Correction Maybe? 
        #        #DOI: 10.1016/S0009-2614(01)00099-9


            self.subsystems[0].atom_proj_grad = atom_proj_grad
            #self.subsystems[0].atom_emb_vhf_grad = atom_emb_vhf_grad
            self.subsystems[0].atom_full_hcore_grad = atom_full_hcore_grad

            self.subsystems[0].calc_nuc_grad()

        self.emb_nuc_grad = 0
        return self.emb_nuc_grad
