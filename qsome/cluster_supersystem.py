# A method to define a cluster supersystem
# Daniel Graham
# Dhabih V. Chulhai

import os
from qsome import supersystem, custom_pyscf_methods, cluster_subsystem
from pyscf import gto, scf, dft, lib, lo

from qsome.helpers import time_method, concat_mols

from pyscf.tools import cubegen, molden

from copy import deepcopy as copy 
import numpy as np
import scipy as sp
import h5py

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
                 fs_damp=0., fs_shift=0., fs_diis=1, fs_grid_level=None, 
                 fs_rhocutoff=None, fs_verbose=None, fs_unrestricted=False, 
                 fs_density_fitting=False, compare_density=False, chkfile_index=0,
                 fs_save_orbs=False, fs_save_density=False, fs_save_spin_density=False,
                 ft_cycles=100, ft_basis_tau=1., ft_conv=1e-8, ft_grad=None, 
                 ft_damp=0, ft_diis=None, ft_setfermi=None, ft_updatefock=0, ft_updateproj=1, ft_initguess=None, ft_unrestricted=False, 
                 ft_save_orbs=False, ft_save_density=False, ft_save_spin_density=False, ft_proj_oper='huz',
                 filename=None, scr_dir=None, nproc=None, pmem=None):

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
        self.pmem= pmem
        self.scr_dir = scr_dir
        self.filename = filename

        # Densities are stored separately to allow for alpha and beta.
        self.is_ft_conv = False
        self.ext_pot = np.array([0., 0.])
        #self.update_hl_basis_tau()
        self.mol = concat_mols(self.subsystems)
        #self.update_hl_basis_tau()
        self.gen_sub2sup()
        self.init_scf()

        # how to include sigmasmear? Currently not in pyscf.
        self.smat = self.fs_scf.get_ovlp()
        self.mo_coeff = np.array([np.zeros_like(self.smat), np.zeros_like(self.smat)])
        self.local_mo_coeff = np.array([None, None])
        self.mo_occ = np.array([np.zeros_like(self.smat[0]), 
                       np.zeros_like(self.smat[0])])
        self.mo_energy = self.mo_occ.copy()
        #self.fock = self.mo_coeff.copy()
        self.hcore = self.fs_scf.get_hcore()
        self.proj_pot = [np.array([0.0, 0.0]) for sub in self.subsystems]
        #DIIS Stuff
        self.sub_diis = [np.array([lib.diis.DIIS(), lib.diis.DIIS()]) for sub in self.subsystems]
        self.fs_energy = None
        self.fs_nuc_grad = None

        self.fs_dmat = None
        self.emb_dmat = None


        # There are other diis methods but these don't work with out method due to subsystem projection.
        if ft_diis is None:
            self.ft_diis = None
        else:
            self.ft_diis = [lib.diis.DIIS(), lib.diis.DIIS()]
            self.ft_diis[0].space = 10
            self.ft_diis[1].space = 10

        self.ft_fermi = [np.array([0., 0.]) for sub in subsystems]

    def update_hl_basis_tau(self, subsystems=None, basis_tau=None):
        """Adds basis functions to the high level subsystem based on the tau overlap parameter"""
        if subsystems is None:
            subsystems = self.subsystems
        if basis_tau is None:
            basis_tau = self.ft_basis_tau

        for i in range(len(self.subsystems)):
            #Needs to check if subsystem is HLSubSsytem but for now just use i==0
            if i == 0:
                hl_sub = self.subsystems[i]
                if len(self.subsystems) <= 2:
                    env_sub = self.subsystems[1]
                    tau_mol = gto.M()
                    tau_mol.unit = 'bohr'
                    tau_mol.atom = []
                    tau_mol.basis = {}
                    for a in range(len(env_sub.mol._atom)):
                        atom_basis = env_sub.mol._basis[env_sub.mol._atom[a][0]]
                        tau_ghost_atom_name = 'X' + str(a)
                        tau_ghost_basis = {}
                        tau_ghost_basis[tau_ghost_atom_name] = []
                        for b in atom_basis:
                            temp_mol = gto.M()
                            temp_mol.atom = [env_sub.mol._atom[a]]
                            temp_mol.basis = {}
                            temp_mol.basis[temp_mol.atom[0][0]] = [b]
                            temp_mol.spin = temp_mol.nelectron
                            temp_mol.build()
                            basis_ovlp = gto.intor_cross('int1e_ovlp', hl_sub.mol, temp_mol)
                            max_ovlp = np.max(np.abs(np.sum(basis_ovlp, axis=0)))
                            if max_ovlp > basis_tau and max_ovlp < 1:
                                tau_ghost_basis[tau_ghost_atom_name].append(b)
                        if len(tau_ghost_basis[tau_ghost_atom_name]) > 0:
                                tau_mol.atom.append([tau_ghost_atom_name, env_sub.mol._atom[a][1]])
                                tau_mol.basis.update(tau_ghost_basis)
                    tau_mol.build()
                    if len(tau_mol.atom) > 0:
                        #Add the tau mol object with ghost functions to the hl subsystem.
                        self.subsystems[i].concat_mol_obj(tau_mol)

                    #ao_slices = gto.aoslice_by_atom(env_sub.mol)
                    #basis_ovlp = gto.intor_cross('int1e_ovlp', hl_sub.mol, env_sub.mol)
                    #if len(uniq_ovlp_coords) > 0:
                    #    new_mol = gto.M()
                    #    new_mol.unit = 'bohr'
                    #    new_mol.atom = []
                    #    new_mol.basis = {}
                    #    prev_basis = 0
                    #    for a in range(len(env_sub.mol._atom)):
                    #        a_coord = env_sub.mol._atom[a]
                    #        a_num_basis = len(env_sub.mol._basis[a_coord[0]])
                    #        print (env_sub.mol._basis)
                    #        prev_basis += a_num_basis
                    #        print (prev_basis)
                else: #Iterate through and concatinate.
                    pass
        #Iterate through active systems.
        #For each active system get the overlap with basis functions.
        #if overlap is greater than the tau parameter, include it into the hl mol object.
        #Include the basis function by 
        # Creating a new mol object with the ghost systems and basis functions to add
        # Concatinate the hl system and the new ghost system
        # recreate the subsystem object with a new mol object. 

    def gen_sub2sup(self, mol=None, subsystems=None):
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

        for i in range(len(subsystems)):
            sub = subsystems[i]
            nssl[i] = np.zeros(sub.mol.natm, dtype=int)
            for j in range(sub.mol.natm):
                ib_t = np.where(sub.mol._bas.transpose()[0] == j)[0]
                ib = ib_t.min()
                ie_t = np.where(sub.mol._bas.transpose()[0] == j)[0]
                ie = ie_t.max()
                ir = sub.mol.nao_nr_range(ib, ie + 1)
                ir = ir[1] - ir[0]
                nssl[i][j] = ir

            assert nssl[i].sum() == sub.mol.nao_nr(),"naos not equal!"

        mAB = mol
        nsl = np.zeros(mAB.natm, dtype=int)
        for i in range(mAB.natm):
            ib = np.where(mAB._bas.transpose()[0] == i)[0].min()
            ie = np.where(mAB._bas.transpose()[0] == i)[0].max()
            ir = mAB.nao_nr_range(ib, ie + 1)
            ir = ir[1] - ir[0]
            nsl[i] = ir

        assert nsl.sum() == mAB.nao_nr(),"naos not equal!"

        sub2sup = [ None for i in range(len(subsystems)) ]
        for i in range(len(subsystems)):
            sub2sup[i] = np.zeros(nao[i], dtype=int)
            for a in range(subsystems[i].mol.natm):
                match = False
                c1 = subsystems[i].mol.atom_coord(a)
                for b in range(mAB.natm):
                    c2 = mAB.atom_coord(b)
                    d = np.dot(c1 - c2, c1 - c2)
                    if d < 0.0001:
                        match = True
                        ia = nssl[i][0:a].sum()
                        ja = ia + nssl[i][a]
                        #ja = ia + nsl[b]
                        ib = nsl[0:b].sum()
                        jb = ib + nsl[b]
                        #jb = ib + nssl[i][a]
                        sub2sup[i][ia:ja] = range(ib, jb)

                assert match,'no atom match!'
        self.sub2sup = sub2sup
        return True

    def init_scf(self, mol=None, fs_method=None, verbose=None, damp=None, 
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

        mol = fs_scf.mol

        print ("".center(80,'*'))
        print ("  Generate Initial System Densities  ".center(80))
        print ("".center(80,'*'))
        dmat = [None, None]
        # If supersystem dft should be read from chkfile.
        super_chk = (self.fs_initguess == 'readchk')
        s2s = self.sub2sup

        # Initialize supersystem density.
        if self.ft_initguess == 'supmol':
            self.get_supersystem_energy(readchk=super_chk)
            dmat = self.fs_dmat

        for i in range(len(subsystems)):
            sub_dmat = [0., 0.]
            subsystem = subsystems[i]
            subsystem.fullsys_cs = not (self.fs_unrestricted or self.mol.spin != 0)
            # Ensure same gridpoints and rho_cutoff for all systems
            subsystem.env_scf.grids = fs_scf.grids
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
            elif sub_guess == 'localsup': # Localize supermolecular density.
                # Incomplete.
                #mo_alpha = lo.PM(mol).kernel(fs_scf.mo_coeff[:, :mol.nelec[0]])
                #mo_beta = lo.PM(mol).kernel(fs_scf.mo_coeff[:, :mol.nelec[1]])
                #temp_smat = np.copy(fs_scf.get_ovlp())
                #sub_dmat = 0.
                #for j in range(self.mol.nelec[0]):
                #    portion = [0 for i in range(len(self.subsystems))]
                #    for n in range(len(self.subsystems)):
                #        temp_smat = np.copy(self.fs_scf.get_ovlp())
                #        temp_smat = temp_smat[np.ix_(s2s[n], s2s[n])]
                #        if n > 0:
                #            sub_nao = self.subsystems[n-1].mol.nao_nr()
                #            dl_n = np.outer(
                #                mo_alpha[sub_nao:,j].T, mo_alpha[sub_nao:,j])
                #            portion[n] = (np.trace(np.dot(dl_n, temp_smat)))
                #        else:
                #            sub_nao = self.subsystems[n].mol.nao_nr()
                #            dl = np.outer(
                #                mo_alpha[:sub_nao,j].T, mo_alpha[:sub_nao,j])
                #            portion[n] = (np.trace(np.dot(dl, temp_smat)))
                #    if portion[i] >= max(portion):
                #        #sub_mo[:,j] += mo_alpha[:,j]
                #        if i > 0:
                #            sub_dmat += (dl_n * 2.)
                #        else:
                #            sub_dmat += (dl * 2.)
                ##print(sub_mo)
                ##temp_dm = self.fs_scf.make_rdm1(sub_mo, self.fs_scf.mo_occ)[np.ix_(s2s[i], s2s[i])]
                ##temp_dm = sub_dmat[np.ix_(s2s[i], s2s[i])]
                #temp_dm = sub_dmat
                ##print ("SEP")    
                ###One way of determining electrons.
                #temp_smat = np.copy(self.fs_scf.get_ovlp())
                #temp_sm = temp_smat[np.ix_(s2s[i], s2s[i])]
                #num_e = np.trace(np.dot(temp_dm, temp_sm))
                #print (f"Subsystem {i} Electrion Number")
                #print (num_e)
                #temp_dm *= subsystem.mol.nelectron / num_e
                #num_e = np.trace(np.dot(temp_dm, temp_sm))
                #print (num_e)
                #print ()

                #loc_dm = [temp_dm/2., temp_dm/2.] 
                #subsystem.update_density(loc_dm)
                s2s = self.sub2sup
                self.get_supersystem_energy(readchk=super_chk, local_orbs=True)
                local_coeff = [0,0]
                local_coeff[0] = self.local_mo_coeff[0].T
                local_coeff[1] = self.local_mo_coeff[1].T
                local_occ = [np.zeros(len(local_coeff[0])), np.zeros(len(local_coeff[1]))]
                occ_order = [None, None]
                temp_co = [0,0]
                temp_dm = [0,0]
                for j in range(len(local_coeff[0])):
                    temp_co[0] = local_coeff[0][j][s2s[i]]
                    temp_co[1] = local_coeff[1][j][s2s[i]]
                    temp_dm[0] = np.outer(temp_co[0], temp_co[0])
                    temp_dm[1] = np.outer(temp_co[1], temp_co[1])
                    local_occ[0][j] = np.trace(np.dot(temp_dm[0], subsystem.env_scf.get_ovlp()))
                    local_occ[1][j] = np.trace(np.dot(temp_dm[1], subsystem.env_scf.get_ovlp()))
                if subsystem.flip_ros:
                    occ_order[0] = np.argsort(local_occ[0])[-1*int(subsystem.mol.nelec[1]):]
                    occ_order[1] = np.argsort(local_occ[1])[-1*int(subsystem.mol.nelec[0]):]
                else:
                    occ_order[0] = np.argsort(local_occ[0])[-1*int(subsystem.mol.nelec[0]):]
                    occ_order[1] = np.argsort(local_occ[1])[-1*int(subsystem.mol.nelec[1]):]
                local_occ = [np.zeros(len(local_coeff[0])), np.zeros(len(local_coeff[1]))]
                local_occ[0][occ_order[0]] = 1
                local_occ[1][occ_order[1]] = 1
                new_dm = np.zeros_like(subsystem.env_dmat)

                #Can likely do this with multiply by the local_occ rather than iterate.
                for k in range(len(local_coeff[0])):
                    if local_occ[0][k]:
                        temp_d = local_coeff[0][k][s2s[i]]
                        temp_d = np.outer(temp_d, temp_d)
                        new_dm[0] += temp_d
                    if local_occ[1][k]:
                        temp_d = local_coeff[1][k][s2s[i]]
                        temp_d = np.outer(temp_d, temp_d)
                        new_dm[1] += temp_d

                subsystem.env_dmat = new_dm

        self.update_fock(diis=False)
        self.update_proj_pot()
        print ("".center(80,'*'))


    def get_emb_dmat(self):
        nS = self.mol.nao_nr()
        dm_env = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        for i in range(len(self.subsystems)):
            dm_env[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].env_dmat[0]
            dm_env[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].env_dmat[1]
        if not (self.fs_unrestricted or self.mol.spin != 0):
            dm_env = dm_env[0] + dm_env[1]
        return dm_env

    def get_emb_ext_pot(self):
        # Returns the potential of higher order environment subsystems, formatted for the next level of environment.
        self.update_fock(diis=False)
        full_fock = self.fock
        froz_veff = np.array([0., 0.])
        s2s = self.sub2sup
        for i in range(len(self.subsystems)):
            sub = self.subsystems[i]
            sub_fock = sub.update_subsys_fock()
            if (not sub.unrestricted) and sub.mol.spin == 0 :
                sub_fock = [sub_fock, sub_fock]
            if sub.env_order > self.env_order:
                FAA = [None, None]
                FAA[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
                FAA[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
                froz_veff[0] = (FAA[0] - sub_fock[0])
                froz_veff[1] = (FAA[1] - sub_fock[1])

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

        if self.fs_energy is None:
            if scf_obj is None:
                scf_obj = self.fs_scf
            if fs_method is None:
                fs_method = self.fs_method
            mol = scf_obj.mol

            print ("".center(80,'*'))
            print("  Supersystem Calculation  ".center(80))
            print ("".center(80,'*'))
            if self.is_ft_conv:  
                nS = mol.nao_nr()
                ft_dmat = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                s2s = self.sub2sup
                for i in range(len(self.subsystems)):
                    subsystem = self.subsystems[i]
                    ft_dmat[0][np.ix_(s2s[i], s2s[i])] += subsystem.env_dmat[0]
                    ft_dmat[1][np.ix_(s2s[i], s2s[i])] += subsystem.env_dmat[1]
                if self.fs_unrestricted or scf_obj.mol.spin != 0:
                    scf_obj.scf(dm0=ft_dmat)
                else:
                    scf_obj.scf(dm0=(ft_dmat[0] + ft_dmat[1]))
            elif readchk:
                if self.fs_unrestricted or scf_obj.mol.spin != 0:
                    init_guess = scf_obj.make_rdm1(mo_coeff=self.mo_coeff, 
                                     mo_occ=self.mo_occ)
                else:
                    init_guess = scf_obj.make_rdm1(mo_coeff=self.mo_coeff[0], 
                                     mo_occ=(self.mo_occ[0] + self.mo_occ[1]))

                scf_obj.scf(dm0=(init_guess))
            else:
                scf_obj.scf()
            self.fs_dmat = scf_obj.make_rdm1()

            if self.fs_dmat.ndim == 2: #Always store as alpha and beta, even if closed shell. Makes calculations easier.
                t_d = [self.fs_dmat.copy()/2., self.fs_dmat.copy()/2.]
                self.fs_dmat = t_d
                self.mo_coeff = [self.fs_scf.mo_coeff, self.fs_scf.mo_coeff]
                self.mo_occ = [self.fs_scf.mo_occ/2., self.fs_scf.mo_occ/2.]
                self.mo_energy = [self.fs_scf.mo_energy, self.fs_scf.mo_energy]
            
            self.save_chkfile()
            self.fs_energy = scf_obj.energy_tot()
            print("".center(80,'*'))
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
            from pyscf import lo
            if self.fs_unrestricted or scf_obj.mol.spin != 0:
                nelec_a = scf_obj.mol.nelec[0]
                #self.local_mo_coeff[0] = lo.PM(scf_obj.mol, self.mo_coeff[0][:, :nelec_a]).kernel()
                self.local_mo_coeff[0] = lo.ER(scf_obj.mol, self.mo_coeff[0][:, :nelec_a]).kernel()
                #self.local_mo_coeff[0] = lo.Boys(scf_obj.mol, self.mo_coeff[0][:, :nelec_a]).kernel()
                nelec_b = scf_obj.mol.nelec[1]
                #self.local_mo_coeff[1] = lo.PM(scf_obj.mol, self.mo_coeff[1][:, :nelec_b]).kernel()
                self.local_mo_coeff[1] = lo.ER(scf_obj.mol, self.mo_coeff[1][:, :nelec_b]).kernel()
                #self.local_mo_coeff[1] = lo.Boys(scf_obj.mol, self.mo_coeff[1][:, :nelec_b]).kernel()
            else: 
                nelec_a = scf_obj.mol.nelec[0]
                #self.local_mo_coeff[0] = lo.PM(scf_obj.mol, self.mo_coeff[0][:, :nelec_a]).kernel()
                self.local_mo_coeff[0] = lo.ER(scf_obj.mol, self.mo_coeff[0][:, :nelec_a]).kernel()
                #self.local_mo_coeff[0] = lo.Boys(scf_obj.mol, self.mo_coeff[0][:, :nelec_a]).kernel()
                self.local_mo_coeff[1] = self.local_mo_coeff[0]

        return self.fs_energy

    @time_method("Supersystem Nuclear Gradient")
    def get_supersystem_nuc_grad(self, scf_obj=None, readchk=False):

        if self.fs_nuc_grad is None:
            if self.fs_energy is None:
                print ("Calculate Full system energy first")
                self.get_supersystem_energy()
            if scf_obj is None:
                scf_obj = self.fs_scf
            nuc_grad = scf_obj.nuc_grad_method()
            nuc_grad.kernel()
            self.fs_nuc_grad = nuc_grad
        else:
            nuc_grad = self.fs_nuc_grad

        return nuc_grad

    @time_method("Subsystem Energies")
    def get_emb_subsys_elec_energy(self):
        """Calculates subsystem energy
        """ 

        s2s = self.sub2sup
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            subsystem.update_fock()
            FAA = [None, None]
            FAA[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
            FAA[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
            froz_veff = [None, None]
            froz_veff[0] = (FAA[0] - subsystem.env_hcore - subsystem.env_V[0])
            froz_veff[1] = (FAA[1] - subsystem.env_hcore - subsystem.env_V[1])
            subsystem.update_emb_pot(froz_veff)
            subsystem.get_env_energy()
            print (f"Uncorrected Energy:{subsystem.env_energy:>61.8f}")
        #self.correct_env_energy()

    @time_method("Subsystem Nuclear Gradients")
    def get_emb_subsys_nuc_grad(self):
        # Update subsystem embedding gradient. 
        # Assumes closed shell for now.
        s2s = self.sub2sup
        nS = self.mol.nao_nr()
        atms_completed = 0
        full_sys_grad = self.fs_scf.nuc_grad_method()
        hcore_deriv = full_sys_grad.hcore_generator(self.fs_scf.mol)
        s1 = full_sys_grad.get_ovlp(self.fs_scf.mol)
        full_fock = self.fock
        full_fock_cs = (full_fock[0] + full_fock[1])/2.
        prev_p1 = 0

        full_dmat = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        for j in range(len(self.subsystems)):
            sub_temp = self.subsystems[j]
            full_dmat[0][np.ix_(s2s[j], s2s[j])] += sub_temp.dmat[0]
            full_dmat[1][np.ix_(s2s[j], s2s[j])] += sub_temp.dmat[1]

        vhf = full_sys_grad.get_veff(self.fs_scf.mol, full_dmat[0] + full_dmat[1])
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            subsystem.get_env_nuc_grad()
            if isinstance(subsystem, cluster_subsystem.ClusterActiveSubSystem):
                dm = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                dm[0][np.ix_(s2s[i], s2s[i])] += subsystem.dmat[0]
                dm[1][np.ix_(s2s[i], s2s[i])] += subsystem.dmat[1]


                env_dm = np.copy(full_dmat)
                env_dm[0][np.ix_(s2s[i], s2s[i])] -= subsystem.dmat[0]
                env_dm[1][np.ix_(s2s[i], s2s[i])] -= subsystem.dmat[1]

                dm_cs = dm[0] + dm[1]
                env_dm_cs = env_dm[0] + env_dm[1]

                active_dm = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                active_dm[0][np.ix_(s2s[i], s2s[i])] += subsystem.active_dmat[0]
                active_dm[1][np.ix_(s2s[i], s2s[i])] += subsystem.active_dmat[1]

                active_dm_cs = active_dm[0] + active_dm[1]
                    
                atmlist = range(atms_completed, subsystem.mol.natm + atms_completed)
                atms_completed = subsystem.mol.natm

                aoslices = subsystem.mol.aoslice_by_atom() 
                env_emb_de = np.zeros((len(atmlist), 3))
                active_emb_de = np.zeros((len(atmlist), 3))
                env_proj_de = np.zeros((len(atmlist), 3))
                active_proj_de = np.zeros((len(atmlist), 3))
                for k, ia in enumerate(atmlist):
                    p0, p1 = aoslices [k,2:]
                    p0_full = p0 + prev_p1
                    p1_full = p1 + prev_p1
                    h1ao_full = hcore_deriv(ia)
                    h1ao_sub = subsystem.env_hcore_deriv(k)
                    h1ao_env = np.zeros_like(h1ao_full)
                    h1ao_env[0][np.ix_(s2s[i], s2s[i])] += h1ao_sub[0]
                    h1ao_env[1][np.ix_(s2s[i], s2s[i])] += h1ao_sub[1]
                    h1ao_env[2][np.ix_(s2s[i], s2s[i])] += h1ao_sub[2]
                    h1ao_emb = h1ao_full - h1ao_env
                    env_emb_de[k] += np.einsum('xij,ij->x', h1ao_emb,  dm_cs)
                    active_emb_de[k] += np.einsum('xij,ij->x', h1ao_emb,  active_dm_cs)

                    vhf_sub = subsystem.env_vhf_deriv
                    vhf_env = np.zeros_like(vhf)
                    vhf_env[0][np.ix_(s2s[i], s2s[i])] += vhf_sub[0]
                    vhf_env[1][np.ix_(s2s[i], s2s[i])] += vhf_sub[1]
                    vhf_env[2][np.ix_(s2s[i], s2s[i])] += vhf_sub[2]
                    vhf_emb = vhf[:,p0_full:p1_full] - vhf_env[:,p0_full:p1_full]
                    env_emb_de[k] += np.einsum('xij,ij->x', vhf_emb, dm_cs[p0:p1]) * 2
                    active_emb_de[k] += np.einsum('xij,ij->x', vhf_emb, active_dm_cs[p0:p1]) * 2

                    #Get the projection gradient.
                    fock_deriv = h1ao_full + (vhf * 2.)
                    proj_pot = np.zeros_like(fock_deriv)
                    proj_pot[0] += np.dot(np.dot(fock_deriv[0], env_dm_cs), self.smat)
                    proj_pot[1] += np.dot(np.dot(fock_deriv[1], env_dm_cs), self.smat)
                    proj_pot[2] += np.dot(np.dot(fock_deriv[2], env_dm_cs), self.smat)

                    proj_pot[0] += np.dot(np.dot(full_fock_cs, env_dm_cs), s1[0])
                    proj_pot[1] += np.dot(np.dot(full_fock_cs, env_dm_cs), s1[1])
                    proj_pot[2] += np.dot(np.dot(full_fock_cs, env_dm_cs), s1[2])

                    proj_pot[0] += proj_pot[0].transpose()
                    proj_pot[1] += proj_pot[1].transpose()
                    proj_pot[2] += proj_pot[2].transpose()

                    proj_pot = proj_pot * -0.5

                    env_proj_de[k] = np.einsum('xij,ij->x', proj_pot[:,p0_full:p1_full], dm_cs[p0:p1])
                    active_proj_de[k] = np.einsum('xij,ij->x', proj_pot[:,p0_full:p1_full], active_dm_cs[p0:p1])

                prev_p1 = p1_full
                subsystem.env_sub_emb_nuc_grad = env_emb_de
                subsystem.env_sub_proj_nuc_grad = env_proj_de
                subsystem.active_sub_emb_nuc_grad = active_emb_de
                subsystem.active_sub_proj_nuc_grad = active_proj_de


    @time_method("Environment XC Correction")
    # This doesn't work yet.
    def correct_env_energy(self):
        nS = self.mol.nao_nr()
        s2s = self.sub2sup
        #Correct exc energy
        # get interaction energies.
        for i in range(len(self.subsystems)):
            for j in range(i + 1, len(self.subsystems)):
                # if embedding method is dft, must add the correct embedding energy
                if not 'hf' in self.fs_method[:5]:
                    e_corr_1 = 0.0
                    e_corr_2 = 0.0
                    subsystem_1 = self.subsystems[i]
            self.fs_scf = scf_obj
        print(f"KS-DFT  Energy: {self.fs_energy}  ".center(80))
        return self.fs_energy

    @time_method("Subsystem Energies")
    def get_emb_subsys_elec_energy(self):
        """Calculates subsystem energy

        """ 
        s2s = self.sub2sup
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            subsystem.update_fock()
            FAA = [None, None]
            FAA[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
            FAA[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
            froz_veff = [None, None]
            froz_veff[0] = (FAA[0] - subsystem.env_hcore - subsystem.env_V[0])
            froz_veff[1] = (FAA[1] - subsystem.env_hcore - subsystem.env_V[1])
            subsystem.update_emb_pot(froz_veff)
            subsystem.get_env_energy()
            print (f"Uncorrected Energy:{subsystem.env_energy:>61.8f}")
        #self.correct_env_energy()


    @time_method("Environment XC Correction")
    # This doesn't work yet.
    def correct_env_energy(self):
        nS = self.mol.nao_nr()
        s2s = self.sub2sup
        #Correct exc energy
        # get interaction energies.
        for i in range(len(self.subsystems)):
            for j in range(i + 1, len(self.subsystems)):
                # if embedding method is dft, must add the correct embedding energy
                if not 'hf' in self.fs_method[:5]:
                    e_corr_1 = 0.0
                    e_corr_2 = 0.0
                    subsystem_1 = self.subsystems[i]
                    subsystem_2 = self.subsystems[j]
                    dm_subsys = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                    dm_subsys[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += subsystem_1.dmat[0]
                    dm_subsys[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += subsystem_1.dmat[1]
                    dm_subsys[0][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[0]
                    dm_subsys[1][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[1]

                    dm_subsys_1 = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                    dm_subsys_1[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += subsystem_1.dmat[0]
                    dm_subsys_1[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += subsystem_1.dmat[1]

                    dm_subsys_2 = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                    dm_subsys_2[0][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[0]
                    dm_subsys_2[1][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[1]

                    #Subtract the incorrect energy
                    #Assume active system is closed shell
                    veff_combined = self.fs_scf.get_veff(dm=(dm_subsys[0] + dm_subsys[1]))
                    vxc_comb = veff_combined
                    vxc_comb = veff_combined - veff_combined.vj
                    if not veff_combined.vk is None:
                        vxc_comb += veff_combined.vk * 0.5

                    veff_active_sub_1 = self.fs_scf.get_veff(dm=(dm_subsys_1[0] + dm_subsys_1[1]))
                    veff_active_sub_2 = self.fs_scf.get_veff(dm=(dm_subsys_2[0] + dm_subsys_2[1]))
                    vxc_sub_1 = veff_active_sub_1 - veff_active_sub_1.vj
                    vxc_sub_2 = veff_active_sub_2 - veff_active_sub_2.vj
                    if not veff_active_sub_1.vk is None:
                        vxc_sub_1 += veff_active_sub_1.vk * 0.5
                    if not veff_active_sub_2.vk is None:
                        vxc_sub_2 += veff_active_sub_2.vk * 0.5

                    vxc_emb_1 = vxc_comb
                    vxc_emb_2 = vxc_comb
                    vxc_emb_1 = vxc_comb - vxc_sub_1
                    vxc_emb_2 = vxc_comb - vxc_sub_2
                    vxc_emb_sub_1 = vxc_emb_1[np.ix_(s2s[i], s2s[i])]
                    vxc_emb_sub_2 = vxc_emb_2[np.ix_(s2s[j], s2s[j])]

                    #Get Exc Last
                    emb_exc_comb_1 = custom_pyscf_methods.exc_rks(self.fs_scf, dm_subsys[0] + dm_subsys[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 
                    emb_exc_comb_2 = custom_pyscf_methods.exc_rks(self.fs_scf, dm_subsys[0] + dm_subsys[1], dm_subsys_2[0] + dm_subsys_2[1])[1] 
                    emb_exc_1 = custom_pyscf_methods.exc_rks(self.fs_scf, dm_subsys_1[0] + dm_subsys_1[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 
                    emb_exc_2 = custom_pyscf_methods.exc_rks(self.fs_scf, dm_subsys_2[0] + dm_subsys_2[1], dm_subsys_2[0] + dm_subsys_2[1])[1] 

                    #vxc_sub_1 = np.einsum('ij,ji', vxc_emb_sub_1, (subsystem_1.dmat[0] + subsystem_1.dmat[1])) #* .5
                    #vxc_sub_2 = np.einsum('ij,ji', vxc_emb_sub_2, (subsystem_2.dmat[0] + subsystem_2.dmat[1])) #* .5

                    subsystem_1.env_energy -= np.einsum('ij,ji', vxc_emb_sub_1, (subsystem_1.dmat[0] + subsystem_1.dmat[1])) #* 0.5
                    subsystem_2.env_energy -= np.einsum('ij,ji', vxc_emb_sub_2, (subsystem_2.dmat[0] + subsystem_2.dmat[1])) #* 0.5
                    subsystem_1.env_energy += (emb_exc_comb_1 - emb_exc_1) * 2.
                    subsystem_2.env_energy += (emb_exc_comb_2 - emb_exc_2) * 2.

    @time_method("Active Energy")
    def get_hl_energy(self):
        """Determines the hl energy.
        
        """
        #This is crude. Later iterations should be more sophisticated and account for more than 2 subsystems.
        print ("".center(80,'*'))
        print("  HL Subsystems Calculation  ".center(80))
        print ("".center(80,'*'))
        s2s = self.sub2sup
        #ONLY DO FOR THE RO SYSTEM. THIS IS KIND OF A TEST.
        if self.mol.spin != 0 and not self.fs_unrestricted:
            self.update_ro_fock()
        for i in range(len(self.subsystems)):
            sub = self.subsystems[i]
            #Check if subsystem is HLSubSystem but rightnow it is being difficult.
            if i == 0:
                FAA = [None,None]
                FAA[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
                FAA[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
                sub.emb_fock = FAA
                sub.get_hl_in_env_energy()
                act_e = sub.hl_energy
                print(f"Energy:{act_e:>73.8f}")
                print("".center(80,'*'))
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
        print ("".center(80,'*'))
        print("  Env Subsystem Calculation  ".center(80))
        print ("".center(80,'*'))
        for i in range(len(self.subsystems)):
            sub = self.subsystems[i]
            #sub.update_subsys_fock()
            s2s = self.sub2sup
            FAA = np.array([None, None])
            FAA[0] = self.fock[0][np.ix_(s2s[i], s2s[i])]
            FAA[1] = self.fock[1][np.ix_(s2s[i], s2s[i])]
            sub.emb_fock = FAA
            sub.get_env_energy()
            env_e = sub.env_energy
            print(f"Energy Subsystem {i}:{env_e:>61.8f}")
        print("".center(80,'*'))

    @time_method("Correct Exc Energy")
    #Doesn't work yet.
    def correct_active_energy(self):
        #Assumes all are closed shell systems
        # Do for all subsystems
        nS = self.mol.nao_nr()
        s2s = self.sub2sup
        energy_corr = 0.0
        for j in range(1, len(self.subsystems)):
            # if embedding method is dft, must add the correct embedding energy
            if not 'hf' in self.ct_method[:5]:
                subsystem_1 = self.subsystems[0]
                subsystem_2 = self.subsystems[j]
                dm_subsys = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                dm_subsys[0][np.ix_(self.sub2sup[0], self.sub2sup[0])] += subsystem_1.active_dmat[0]
                dm_subsys[1][np.ix_(self.sub2sup[0], self.sub2sup[0])] += subsystem_1.active_dmat[1]
                dm_subsys[0][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[0]
                dm_subsys[1][np.ix_(self.sub2sup[j], self.sub2sup[j])] += subsystem_2.dmat[1]

                dm_subsys_1 = [np.zeros((nS, nS)), np.zeros((nS, nS))]
                dm_subsys_1[0][np.ix_(self.sub2sup[0], self.sub2sup[0])] += subsystem_1.active_dmat[0]
                dm_subsys_1[1][np.ix_(self.sub2sup[0], self.sub2sup[0])] += subsystem_1.active_dmat[1]

                #Assume active system is closed shell
                veff_combined = self.fs_scf.get_veff(dm=(dm_subsys[0] + dm_subsys[1]))
                vxc_comb = veff_combined - veff_combined.vj
                if not veff_combined.vk is None:
                    vxc_comb += veff_combined.vk * 0.5

                veff_active_sub = self.fs_scf.get_veff(dm=(dm_subsys_1[0] + dm_subsys_1[1]))
                vxc_sub = veff_active_sub - veff_active_sub.vj
                if not veff_active_sub.vk is None:
                    vxc_sub += veff_active_sub.vk * 0.5

                vxc_emb = vxc_comb - vxc_sub
                vxc_emb_sub = vxc_emb[np.ix_(s2s[0], s2s[0])]

                emb_exc = custom_pyscf_methods.exc_rks(self.fs_scf, dm_subsys[0] + dm_subsys[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 
                emb_exc_a = custom_pyscf_methods.exc_rks(self.fs_scf, dm_subsys_1[0] + dm_subsys_1[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 

                energy_corr -= np.einsum('ij,ji',vxc_emb_sub, (subsystem_1.active_dmat[0] + subsystem_1.active_dmat[1])) #* 0.5
                energy_corr += (emb_exc - emb_exc_a) * 2.
        
                #vxc_emb_sub = vxc_emb[np.ix_(s2s[0], s2s[0])]
                ##THis is incorrect for more than 2 subsys 
                #emb_exc = custom_pyscf_methods.exc_rks(self.fs_scf, dm_subsys[0] + dm_subsys[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 
                #emb_exc_a = custom_pyscf_methods.exc_rks(self.fs_scf, dm_subsys_1[0] + dm_subsys_1[1], dm_subsys_1[0] + dm_subsys_1[1])[1] 
                #
                ##Subtract the Vxc from the fock and recalculate the energies.
                #if 'hf' in subsystem_1.active_method:
               #    emb_pot = (subsystem_1.emb_pot[0] + subsystem_1.emb_pot[1])/2. - vxc_emb_sub
                #    proj_pot = (subsystem_1.proj_pot[0] + subsystem_1.proj_pot[1])/2.
                #    energy_elec = custom_pyscf_methods.rhf_energy_elec(subsystem_1.active_scf, emb_pot, proj_pot)
                #    energy_corr = energy_elec[0] + (emb_exc - emb_exc_a)
                #elif subsystem_1.active_method == 'ccsd' or subsystem_1.active_method == 'rccsd':
                #    new_eris = subsystem_1.active_cc.ao2mo()
                #    emb_pot = (subsystem_1.emb_pot[0] + subsystem_1.emb_pot[1])/2. - vxc_emb_sub
                #    proj_pot = (subsystem_1.proj_pot[0] + subsystem_1.proj_pot[1])/2.
                #    fock = custom_pyscf_methods.rhf_get_fock(subsystem_1.active_scf, emb_pot, proj_pot)
                #    new_eris.fock = functools.reduce(np.dot, (subsystem_1.active_scf.mo_coeff.conj().T, fock, subsystem_1.active_scf.mo_coeff))
                #    energy_elec = custom_pyscf_methods.rhf_energy_elec(subsystem_1.active_scf, emb_pot, proj_pot)[0]
                #    energy_elec += subsystem_1.active_cc.energy(eris=new_eris)  
                #    energy_corr = energy_elec + (emb_exc - emb_exc_a)

                #else:
                #    energy_elec = 0.0
                #    energy_corr = energy_elec + (emb_exc - emb_exc_a)
        
                #Calculate the Exc using the active density and add it to the energies.

        return energy_corr
                

    @time_method("Env. in Env. Energy")
    def get_env_in_env_energy(self):
        """Calculates the energy of dft-in-dft.

        This is unnecessary for the Total Embedding energy, however it 
        may be a useful diagnostic tool.

        """
        print ("".center(80,'*'))
        print("  Env-in-Env Calculation  ".center(80))
        print ("".center(80,'*'))
        dm_env = self.get_emb_dmat()
        self.env_energy = self.fs_scf.energy_tot(dm=dm_env)

        if self.ft_save_orbs:
            print('Writing DFT-in-DFT Orbitals'.center(80))
            #The MO coefficients must be created from the density.
            #moldenname = os.path.splitext(self.filename)[0] + '_dftindft' + '.molden'
            #dmat = dm_env[0] + dm_env[1]
            #cubegen.density(self.fs_scf.mol, cubename, dmat) 

        print(f"DFT-in-DFT Energy:{self.env_energy:>62.8f}")
        print("".center(80,'*'))
        return self.env_energy

    def update_ro_fock(self):
        """Updates the full system fock  for restricted open shell matrix.

        Parameters
        ----------
        diis : bool
            Whether to use the diis method.
        """
        # Optimization: Rather than recalculate the full V, only calculate the V for densities which changed. 
        # get 2e matrix
        nS = self.mol.nao_nr()
        dm = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        sub_openshell = False
        for i in range(len(self.subsystems)):
            dm[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += (self.subsystems[i].env_dmat[0])
            dm[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += (self.subsystems[i].env_dmat[1])
        temp_fock = self.fs_scf.get_fock(h1e=self.hcore, dm=dm)
        self.fock = [temp_fock, temp_fock]

        s2s = self.sub2sup
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            sub_fock_ro = self.fock[0][np.ix_(s2s[i], s2s[i])]
            sub_fock_0 = self.fock[0].focka[np.ix_(s2s[i], s2s[i])]
            sub_fock_1 = self.fock[0].fockb[np.ix_(s2s[i], s2s[i])]
            subsystem.emb_ro_fock = [sub_fock_ro, sub_fock_0, sub_fock_1]

        return True

    def update_fock(self, diis=True):
        """Updates the full system fock matrix.

        Parameters
        ----------
        diis : bool
            Whether to use the diis method.
        """

        # Optimization: Rather than recalculate the full V, only calculate the V for densities which changed. 
        # get 2e matrix
        nS = self.mol.nao_nr()
        dm = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        sub_openshell = False
        for i in range(len(self.subsystems)):
            if self.subsystems[i].unrestricted:
                sub_openshell = True 
                dm[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].env_dmat[0]
                dm[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].env_dmat[1]
            else:
                dm[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += (self.subsystems[i].env_dmat[0])
                dm[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += (self.subsystems[i].env_dmat[1])
        
        if self.fs_unrestricted or sub_openshell:
            self.fock = self.os_scf.get_fock(h1e=self.hcore, dm=dm)
        elif self.mol.spin != 0:
            temp_fock = self.fs_scf.get_fock(h1e=self.hcore, dm=dm)
            self.fock = [temp_fock, temp_fock]
        else:
            dm = dm[0] + dm[1]
            temp_fock = self.fs_scf.get_fock(h1e=self.hcore, dm=dm)
            self.fock = [temp_fock, temp_fock]

        # to use the scf diis methods, must generate individual fock parts and recombine to make full fock matrix, because the projection operator is necessary to use diis correctly.
        if not self.ft_diis is None and diis:
            if self.fs_unrestricted or sub_openshell or self.mol.spin != 0:
                self.fock[0] = self.ft_diis[0].update(self.fock[0])
                self.fock[1] = self.ft_diis[1].update(self.fock[1])
            else:
                f = self.ft_diis[0].update(self.fock[0])
                self.fock[0] = f
                self.fock[1] = f
        self.veff = [self.fock[0] - self.hcore, self.fock[1] - self.hcore]
        #if not self.ft_diis is None and diis:
        #    #f = self.ft_diis[0].update(self.smat, (dm[0] + dm[1]), self.fock[0], self.fs_scf, self.hcore, V_a)
        #    f = self.ft_diis[0].update(self.fock[0])
        #    self.fock[0] = f
        #    self.fock[1] = f
        #Every fock update also update the subsystem embedding potentials.

        #Add the external potential to each fock.
        self.fock[0] += self.ext_pot[0]
        self.fock[1] += self.ext_pot[1]

        s2s = self.sub2sup
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            sub_fock_0 = self.fock[0][np.ix_(s2s[i], s2s[i])]
            sub_fock_1 = self.fock[1][np.ix_(s2s[i], s2s[i])]
            subsystem.emb_fock = [sub_fock_0, sub_fock_1]

        return True

    def update_proj_pot(self):
        # currently updates both at once. Can easily modify to only update one subsystem, however I don't think it will improve the speed.
        s2s = self.sub2sup
        for i in range(len(self.subsystems)):
            A = i
            nA = self.subsystems[A].mol.nao_nr()
            SAA = self.smat[np.ix_(s2s[A], s2s[A])]
            POp = [np.zeros((nA, nA)), np.zeros((nA, nA))]

            # cycle over all other subsystems
            for B in range(len(self.subsystems)):
                if B==A: continue

                sub_dmat = self.subsystems[B].env_dmat
                SAB = self.smat[np.ix_(s2s[A], s2s[B])]
                SBA = self.smat[np.ix_(s2s[B], s2s[A])]

                # get mu-parameter projection operator
                if isinstance(self.proj_oper, int) or isinstance(self.proj_oper, float):
                    POp[0] += self.proj_oper * np.dot( SAB, np.dot( sub_dmat[0], SBA ))
                    POp[1] += self.proj_oper * np.dot( SAB, np.dot( sub_dmat[1], SBA ))

                elif self.proj_oper in ('huzinaga', 'huz'):
                    FAB = [None, None]
                    FAB[0] = self.fock[0][np.ix_(s2s[A], s2s[B])]
                    FAB[1] = self.fock[1][np.ix_(s2s[A], s2s[B])]
                    FDS = [None, None]
                    FDS[0] = np.dot( FAB[0], np.dot( sub_dmat[0], SBA ))
                    FDS[1] = np.dot( FAB[1], np.dot( sub_dmat[1], SBA ))
                    POp[0] += -1. * ( FDS[0] + FDS[0].transpose() ) 
                    POp[1] += -1. * ( FDS[1] + FDS[1].transpose() )

                elif self.proj_oper in ('huzinagafermi', 'huzfermi'):
                    FAB = [None, None]
                    FAB[0] = self.fock[0][np.ix_(s2s[A], s2s[B])]
                    FAB[1] = self.fock[1][np.ix_(s2s[A], s2s[B])]
                    #The max of the fermi energy
                    efermi = [None, None]
                    if self.ft_setfermi is None:
                        efermi[0] = max([fermi[0] for fermi in self.ft_fermi])
                        efermi[1] = max([fermi[1] for fermi in self.ft_fermi])
                    else:
                        efermi[0] = self.ft_setfermi
                        efermi[1] = self.ft_setfermi #Allow for two set fermi, one for a and one for b

                    FAB[0] -= SAB * efermi[0]
                    FAB[1] -= SAB * efermi[1] #could probably specify fermi for each alpha or beta electron.

                    FDS = [None, None]
                    FDS[0] = np.dot( FAB[0], np.dot( sub_dmat[0], SBA ))
                    FDS[1] = np.dot( FAB[1], np.dot( sub_dmat[1], SBA ))
                    POp[0] += -1. * ( FDS[0] + FDS[0].transpose() ) 
                    POp[1] += -1. * ( FDS[1] + FDS[1].transpose() )
            self.proj_pot[i] = POp.copy()
        return True

    def set_chkfile_index(self, index):
        self.chkfile_index = str(index)
        for i in range(len(self.subsystems)):
            sub = self.subsystems[i]
            sub.chkfile_index = str(index) + "_" + str(i)

    #These should catch exceptions.
    def read_chkfile(self, filename=None):
    # Need to make more robust. Handle errors and such.

        if filename is None:
            if self.filename is None:
                return False
            else:
                filename = os.path.splitext(self.filename)[0] + '.hdf5'
        assert(self.chkfile_index is not None),'Need to set chkfile_index'

        chk_index = self.chkfile_index
        if os.path.isfile(filename):
            try:
                with h5py.File(filename, 'r') as hf:
                    supsys_coeff = hf['supersystem:{chk_index}/mo_coeff']
                    if (self.mol.nao == supsys_coeff.shape[1]):
                        self.mo_coeff = supsys_coeff[:]
                        supsys_occ = hf['supersystem:{chk_index}/mo_occ']
                        self.mo_occ = supsys_occ[:]
                        supsys_energy = hf['supersystem:{chk_index}/mo_energy']
                        self.mo_energy = supsys_energy[:]
                        return True 
                    else:
                        print ("chkfile improperly formatted".center(80))
                        return False
            except TypeError:
                print ("chkfile improperly formatted".center(80))
                return False
        else:
            print ("chkfile NOT found".center(80))
            return False

    def save_chkfile(self, filename=None):
        # current plan is to save mo_coefficients, occupation vector, and energies.
        # becasue of how h5py works we need to check if none and save as the correct filetype (f)
        #Need to make more robust. Handle errors and such.

        # check if file exists. 
        if filename is None:
            if self.filename is None:
                return False
            else:
                filename = os.path.splitext(self.filename)[0] + '.hdf5'
        assert(self.chkfile_index is not None),'Need to set chkfile_index'

        chk_index = self.chkfile_index
        if os.path.isfile(filename):
            try:
                with h5py.File(filename, 'r+') as hf:
                    supsys_coeff = hf['supersystem:{chk_index}/mo_coeff']
                    supsys_coeff[...] = self.mo_coeff
                    supsys_occ = hf['supersystem:{chk_index}/mo_occ']
                    supsys_occ[...] = self.mo_occ
                    supsys_energy = hf['supersystem:{chk_index}/mo_energy']
                    supsys_energy[...] = self.mo_energy
            except TypeError:
                print ("Overwriting existing chkfile".center(80))
                with h5py.File(filename, 'w') as hf:
                    sup_mol = hf.create_group('supersystem:{chk_index}')
                    sup_mol.create_dataset('mo_coeff', data=self.mo_coeff)
                    sup_mol.create_dataset('mo_occ', data=self.mo_occ)
                    sup_mol.create_dataset('mo_energy', data=self.mo_energy)
            except KeyError:
                print ("Updating existing chkfile".center(80))
                with h5py.File(filename, 'a') as hf:
                    sup_mol = hf.create_group('supersystem:{chk_index}')
                    sup_mol.create_dataset('mo_coeff', data=self.mo_coeff)
                    sup_mol.create_dataset('mo_occ', data=self.mo_occ)
                    sup_mol.create_dataset('mo_energy', data=self.mo_energy)

        else:
            with h5py.File(filename, 'w') as hf:
                sup_mol = hf.create_group('supersystem:{chk_index}')
                sup_mol.create_dataset('mo_coeff', data=self.mo_coeff)
                sup_mol.create_dataset('mo_occ', data=self.mo_occ)
                sup_mol.create_dataset('mo_energy', data=self.mo_energy)

    def save_fs_density_file(self, filename=None, density=None):
        from pyscf.tools import cubegen
        if filename is None:
            filename = self.filename
        if density is None:
            density = self.fs_dmat

        print('Writing Full System Density'.center(80))
        if self.mol.spin != 0 or self.fs_unrestricted:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_fs_a.cube'
            cubegen.density(self.mol, cubegen_fn, density[0])
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_fs_b.cube'
            cubegen.density(self.mol, cubegen_fn, density[1])
        else:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_fs.cube'
            cubegen.density(self.mol, cubegen_fn, (density[0] + density[1]))

    def save_fs_spin_density_file(self, filename=None, density=None):
        from pyscf.tools import cubegen
        if filename is None:
            filename = self.filename
        if density is None:
            density = self.fs_dmat

        if self.mol.spin != 0 or self.fs_unrestricted:
            print('Writing Full System Spin Density'.center(80))
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_fs_spinden.cube'
            cubegen.density(self.mol, cubegen_fn, np.subtract(density[0], density[1]))
        else:
            print('Cannot write spin density for a closed shell system.'.center(80))


    def save_ft_density_file(self, filename=None, density=None):
        from pyscf.tools import cubegen
        if filename is None:
            filename = self.filename
        if density is None:
            nS = self.mol.nao_nr()
            dm = [np.zeros((nS, nS)), np.zeros((nS, nS))]
            for i in range(len(self.subsystems)):
                if self.subsystems[i].unrestricted or self.subsystems[i].mol.spin != 0:
                    dm[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].get_dmat()[0]
                    dm[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].get_dmat()[1]
                else:
                    dm[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += (self.subsystems[i].get_dmat()/2.)
                    dm[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += (self.subsystems[i].get_dmat()/2.)
        print('Writing DFT-in-DFT Density'.center(80))
        if self.mol.spin != 0 or self.ft_unrestricted:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index +  '_ft_a.cube'
            cubegen.density(self.mol, cubegen_fn, dm[0])
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index +  '_ft_b.cube'
            cubegen.density(self.mol, cubegen_fn, dm[1])
        else:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index +  '_ft.cube'
            cubegen.density(self.mol, cubegen_fn, (dm[0] + dm[1]))

    def save_ft_spin_density_file(self, filename=None, density=None):
        from pyscf.tools import cubegen
        if filename is None:
            filename = self.filename
        if density is None:
            nS = self.mol.nao_nr()
            dm = [np.zeros((nS, nS)), np.zeros((nS, nS))]
            for i in range(len(self.subsystems)):
                if self.subsystems[i].unrestricted or self.subsystems[i].mol.spin != 0:
                    dm[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].get_dmat()[0]
                    dm[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].get_dmat()[1]
                else:
                    dm[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += (self.subsystems[i].get_dmat()/2.)
                    dm[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += (self.subsystems[i].get_dmat()/2.)
        if self.mol.spin != 0 or self.ft_unrestricted:
            print('Writing DFT-in-DFT Density'.center(80))
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index +  '_ft_spinden.cube'
            cubegen.density(self.mol, cubegen_fn, np.subtract(dm[0], dm[1]))
        else:
            print('Cannot write spin density for a closed shell system.'.center(80))


    def save_fs_orbital_file(self, filename=None, scf_obj=None, mo_occ=None, mo_coeff=None, mo_energy=None):
        from pyscf.tools import molden
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
        molden_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_fs.molden'
        print('Writing Full System Orbitals'.center(80))
        with open(molden_fn, 'w') as fin:
            if not (self.fs_unrestricted or self.mol.spin != 0):
                molden.header(scf_obj.mol, fin)
                molden.orbital_coeff(self.mol, fin, mo_coeff, ene=mo_energy, occ=mo_occ)
            else:
                print ("OPEN SHELL NOT CODED STOP ASKING")

    @time_method("Freeze and Thaw")
    def freeze_and_thaw(self):
        # Optimization: rather than recalculate vA use the existing fock and subtract out the block that is double counted.

        print("".center(80, '*'))
        print("Freeze-and-Thaw".center(80))
        print("".center(80, '*'))
         
        s2s = self.sub2sup
        ft_err = 1.
        self.ft_iter = 0 
        last_cycle = False

        while((ft_err > self.ft_conv) and (self.ft_iter < self.ft_cycles)):
            # cycle over subsystems
            ft_err = 0
            self.ft_iter += 1
            #Correct for DIIS 
            # If fock only updates after cycling, then use python multiprocess todo simultaneously.
            self.update_fock(diis=True)
            self.update_proj_pot()
            for i in range(0, len(self.subsystems)):
              
                sub = self.subsystems[i] 
                sub_old_dm = sub.get_dmat().copy()
                sub.proj_pot = self.proj_pot[i]
                sub.diagonalize()

                new_dm = [None, None]
                if sub.unrestricted or sub.mol.spin != 0:
                    ddm = sp.linalg.norm(sub.get_dmat()[0] - sub_old_dm[0])
                    ddm += sp.linalg.norm(sub.get_dmat()[1] - sub_old_dm[1])
                    proj_e = np.trace(np.dot(sub.get_dmat()[0], self.proj_pot[i][0]))
                    proj_e += np.trace(np.dot(sub.get_dmat()[1], self.proj_pot[i][1]))
                    ft_err += ddm
                    self.ft_fermi[i] = sub.fermi

                    damp = [self.ft_damp, self.ft_damp]
                    if damp[0] < 0:
                        #GeT ODA DAMPING parameters.
                        pass
                    new_dm[0] = ((1 - damp[0]) * sub.get_dmat()[0] + (damp[0] * sub_old_dm[0]))
                    new_dm[1] = ((1 - damp[1]) * sub.get_dmat()[1] + (damp[1] * sub_old_dm[1]))
                    sub.env_dmat = new_dm
                else:
                    damp = self.ft_damp
                    ddm = sp.linalg.norm(sub.get_dmat() - sub_old_dm)
                    proj_e = np.trace(np.dot(sub.get_dmat(), self.proj_pot[i][0]))
                    ft_err += ddm
                    self.ft_fermi[i] = [sub.fermi, sub.fermi]
                    if damp < 0:
                        #GET ODA DAMPING PARAMETER.
                        pass
                    new_dm = ((1. - damp) * sub.get_dmat() + (damp * sub_old_dm))
                    sub.env_dmat = np.array([new_dm/2., new_dm/2.])
                # print output to console.
                print(f"iter:{self.ft_iter:>3d}:{i:<2d}              |ddm|:{ddm:12.6e}               |Tr[DP]|:{proj_e:12.6e}")
                #Check if need to update fock or proj pot.
                if self.ft_updatefock > 0 and ((i + 1) % self.ft_updatefock) == 0:
                    self.update_fock(diis=True)
                if self.ft_updateproj > 0 and ((i + 1) % self.ft_updateproj) == 0:
                    self.update_proj_pot()
            self.save_chkfile()

        print("".center(80))
        self.is_ft_conv = True
        if(ft_err > self.ft_conv):
            print("".center(80))
            print("Freeze-and-Thaw NOT converged".center(80))
        
        self.update_fock(diis=False)
        #Update Proj pot?
        # print subsystem energies 
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            subsystem.get_env_energy()
            print(f"Subsystem {i} Energy:{subsystem.env_energy:>61.8f}")
        print("".center(80))
        print("".center(80, '*'))

        #Assumes closed shell.
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
            print(f'Writing Supersystem FT density'.center(80))
            self.save_ft_density_file()

        if self.ft_save_spin_density:
            print(f'Writing Supersystem FT spin density'.center(80))
            self.save_ft_spin_density_file()
            
    def get_dft_diff_parameters(self, fs_dmat=None, fs_scf=None, dftindft_dmat=None):
        
        if fs_dmat is None:
            fs_dmat = self.fs_dmat
        if fs_scf is None:
            fs_dmat = self.fs_scf
        if dftindft_dmat is None:
            dftindft_dmat = self.dftindft_dmat

        e_diff = self.fs_energy - self.env_energy
        print (f"Energy Difference of KS-DFT to DFT-in-DFT:{e_diff:>38.8f}")

        #trace_diff = (0.5 * np.trace(self.dmat[0] - self.dftindft_dmat[0]) 
        #             + 0.5 * np.trace(self.dmat[1] - self.dftindft_dmat[1]))

        #print (f"Trace Difference of KS-DFT to DFT-in-DFT:{trace_diff:>39.8f}") 

    def get_embedding_nuc_grad(self):
        sup_nuc_grad = self.get_supersystem_nuc_grad().grad()
        self.get_emb_subsys_nuc_grad()
        subsystem_grad = np.zeros_like(sup_nuc_grad)
        s2s = self.sub2sup
        num_atoms_done = 0
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            if isinstance(subsystem, cluster_subsystem.ClusterActiveSubSystem):
                subsystem.get_env_nuc_grad()
                subsystem.get_active_nuc_grad()
                for a in range(subsystem.mol.natm):
                    curr_atom = a + num_atoms_done
                    subsystem_grad[curr_atom] -= subsystem.env_sub_nuc_grad[a] + subsystem.env_sub_emb_nuc_grad[a] + subsystem.env_sub_proj_nuc_grad[a]
                    subsystem_grad[curr_atom] += subsystem.active_sub_nuc_grad[a] + subsystem.active_sub_emb_nuc_grad[a] + subsystem.active_sub_proj_nuc_grad[a]
                num_atoms_done += subsystem.mol.natm

        total_grad = sup_nuc_grad - subsystem_grad
        return total_grad
 
       
