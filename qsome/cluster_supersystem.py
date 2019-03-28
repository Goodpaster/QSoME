# A method to define a cluster supersystem
# Daniel Graham

import os
from qsome import supersystem, custom_pyscf_methods
from pyscf import gto, scf, dft, lib, lo

from pyscf.tools import cubegen, molden

import functools
import time

import numpy as np
import scipy as sp
import h5py
#from multiprocessing import Process, Pipe
#
#def spawn(f):
#    def fun(pipe,x):
#        pipe.send(f(x))
#        pipe.close()
#    return fun
#
#def parmap(f,X):
#    pipe=[Pipe() for x in X]
#    proc=[Process(target=spawn(f),args=(c,x)) for x,(p,c) in zip(X,pipe)]
#    [p.start() for p in proc]
#    [p.join() for p in proc]
#    return [p.recv() for (p,c) in pipe]

def time_method(function_name=None):
    def real_decorator(func):
        @functools.wraps(func)
        def wrapper_time_method(*args, **kwargs):
            ts = time.time()
            result = func(*args, **kwargs)
            te = time.time()
            if function_name is None:
                name = func.__name__.upper()
            else:
                name = function_name
            elapsed_t = (te - ts)
            print( f'{name} {elapsed_t:>50.4f}s')
            return result
        return wrapper_time_method 
    return real_decorator


class ClusterSuperSystem(supersystem.SuperSystem):
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


    def __init__(self, subsystems, fs_method, fs_unrestricted=False, 
                 proj_oper='huz', filename=None, ft_cycles=100, ft_conv=1e-8, 
                 ft_grad=None, ft_diis=1, ft_setfermi=None, ft_damp=0.0,
                 ft_initguess='supmol', ft_updatefock=0, ft_writeorbs=False, 
                 fs_cycles=100, fs_conv=1e-9, fs_grad=None, fs_damp=0, 
                 fs_shift=0, fs_smearsigma=0, fs_initguess=None, grid_level=4, 
                 verbose=3, analysis=False, debug=False, rhocutoff=1e-7, 
                 nproc=None, pmem=None, scr_dir=None, fs_save_orbs=False, 
                 fs_save_density=False, ft_save_orbs=False, 
                 ft_save_density=False, compare_density=False):
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
        self.fs_method = fs_method
        self.fs_unrestricted = fs_unrestricted
        self.proj_oper = proj_oper

        self.nproc = nproc
        self.pmem= pmem
        self.scr_dir = scr_dir
        if filename is None:
            filename = os.getcwd() + '/temp.inp'
        self.filename = filename
        self.chk_filename = os.path.splitext(self.filename)[0] + '.hdf5'

        # freeze and thaw settings
        self.ft_cycles = ft_cycles
        self.ft_conv = ft_conv
        self.ft_grad = ft_grad
        self.ft_damp = ft_damp

        self.ft_setfermi = ft_setfermi
        self.ft_initguess = ft_initguess
        self.ft_updatefock = ft_updatefock
        self.ft_save_orbs = ft_save_orbs
        self.ft_save_density = ft_save_density

        # full system settings
        self.fs_cycles = fs_cycles
        self.fs_conv = fs_conv
        self.fs_grad = fs_grad
        self.rho_cutoff = rhocutoff
        self.fs_damp = fs_damp
        self.fs_shift = fs_shift
        self.fs_smearsigma = fs_smearsigma
        self.fs_initguess = fs_initguess
        self.fs_save_orbs = fs_save_orbs
        self.fs_save_density = fs_save_density

        # general system settings
        self.grid_level = grid_level
        self.verbose = verbose
        self.analysis = analysis #provide a more detail at higher cost
        self.debug = debug
        self.compare_density = compare_density
        

        # Densities are stored separately to allow for alpha and beta.
        self.is_ft_conv = False
        self.mol = self.concat_mols()
        self.sub2sup = self.gen_sub2sup()
        self.fs_scf, self.os_scf = self.init_scf()

        # how to include sigmasmear? Currently not in pyscf.
        self.smat = self.fs_scf.get_ovlp()
        self.mo_coeff = [np.zeros_like(self.smat), np.zeros_like(self.smat)]
        self.local_mo_coeff = [None, None]
        self.mo_occ = [np.zeros_like(self.smat[0]), 
                       np.zeros_like(self.smat[0])]
        self.mo_energy = self.mo_occ.copy()
        self.fock = self.mo_coeff.copy()
        self.hcore = self.fs_scf.get_hcore()
        self.proj_pot = [[0.0, 0.0] for i in range(len(self.subsystems))]
        self.fs_energy = None

        self.dmat = self.init_density()
        self.dftindft_dmat = [None, None]
        self.save_chkfile()

        # There are other diis methods but these don't work with out method due to subsystem projection.
        if ft_diis == 0:
            self.ft_diis = None
        else:
            self.ft_diis = [lib.diis.DIIS(), lib.diis.DIIS()]

        self.update_fock(diis=False)
        self.update_proj_pot()
        self.ft_fermi = [[0., 0.] for i in range(len(subsystems))]

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

        nao = np.array([
            subsystems[i].mol.nao_nr() for i in range(len(subsystems))
            ])
        nssl = [None for i in range(len(subsystems))]

        for i in range(len(subsystems)):
            nssl[i] = np.zeros(subsystems[i].mol.natm, dtype=int)
            for j in range(subsystems[i].mol.natm):
                ib_t = np.where(subsystems[i].mol._bas.transpose()[0] == j)[0]
                ib = ib_t.min()
                ie_t = np.where(subsystems[i].mol._bas.transpose()[0] == j)[0]
                ie = ie_t.max()
                ir = subsystems[i].mol.nao_nr_range(ib, ie + 1)
                ir = ir[1] - ir[0]
                nssl[i][j] = ir

            if nssl[i].sum() != subsystems[i].mol.nao_nr():
                print ('ERROR: naos not equal!') # should throw exception.

        mAB = mol
        nsl = np.zeros(mAB.natm, dtype=int)
        for i in range(mAB.natm):
            ib = np.where(mAB._bas.transpose()[0] == i)[0].min()
            ie = np.where(mAB._bas.transpose()[0] == i)[0].max()
            ir = mAB.nao_nr_range(ib, ie + 1)
            ir = ir[1] - ir[0]
            nsl[i] = ir

        if nsl.sum() != mAB.nao_nr():
            print ('ERROR: naos not equal!') # should throw exception.

        sub2sup = [ None for i in range(len(subsystems)) ]
        for i in range(len(subsystems)):
            sub2sup[i] = np.zeros(nao[i], dtype=int)
            for a in range(subsystems[i].mol.natm):
                match = False
                for b in range(mAB.natm):
                    c1 = subsystems[i].mol.atom_coord(a)
                    c2 = mAB.atom_coord(b)
                    d = np.dot(c1 - c2, c1 - c2)
                    if d < 0.0001:
                        match = True
                        ia = nssl[i][0:a].sum()
                        ja = ia + nssl[i][a]
                        ib = nsl[0:b].sum()
                        jb = ib + nsl[b]
                        sub2sup[i][ia:ja] = range(ib, jb)

                if not match:
                    print ('ERROR: I did not find an atom match!') # should throw exception.
        return sub2sup

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
            verbose = self.verbose
        if damp is None:
            damp = self.fs_damp
        if shift is None:
            shift = self.fs_shift

        if (self.pmem):
            self.mol.max_memory = self.pmem

        if self.fs_unrestricted:
            if fs_method == 'hf':
                scf_obj = scf.UHF(mol) 
                u_scf_obj = scf_obj
            else:
                scf_obj = scf.UKS(mol)
                scf_obj.xc = fs_method
                scf_obj.small_rho_cutoff = self.rho_cutoff
                u_scf_obj = scf_obj

        elif mol.spin != 0:
            if fs_method == 'hf':
                scf_obj = scf.ROHF(mol) 
            else:
                scf_obj = scf.ROKS(mol)
                scf_obj.xc = fs_method
                scf_obj.small_rho_cutoff = self.rho_cutoff
        else:
            if fs_method == 'hf':
               scf_obj = scf.RHF(mol) 
               u_scf_obj = scf.UHF(mol)
            else:
                scf_obj = scf.RKS(mol)
                u_scf_obj = scf.UKS(mol)
                scf_obj.xc = fs_method
                u_scf_obj.xc = fs_method
                scf_obj.small_rho_cutoff = self.rho_cutoff
                u_scf_obj.small_rho_cutoff = self.rho_cutoff

        fs_scf = scf_obj
        fs_scf.max_cycle = self.fs_cycles
        fs_scf.conv_tol = self.fs_conv
        fs_scf.conv_tol_grad = self.fs_grad
        fs_scf.damp = self.fs_damp
        fs_scf.level_shift = self.fs_shift
        fs_scf.verbose = self.verbose

        grids = dft.gen_grid.Grids(mol)
        grids.level = self.grid_level
        grids.build()
        fs_scf.grids = grids
        u_scf_obj.grids = grids
        return fs_scf, u_scf_obj


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

        for i in range(len(subsystems)):
            sub_dmat = [0., 0.]
            subsystem = subsystems[i]
            # Ensure same gridpoints for all systems
            subsystem.env_scf.grids = fs_scf.grids
            sub_guess = subsystem.initguess
            if sub_guess is None:
                sub_guess = self.ft_initguess

            if sub_guess == 'supmol':
                self.get_supersystem_energy(readchk=super_chk)
                sub_dmat[0] = self.dmat[0][np.ix_(s2s[i], s2s[i])] 
                sub_dmat[1] = self.dmat[1][np.ix_(s2s[i], s2s[i])] 
                temp_smat = np.copy(fs_scf.get_ovlp())
                temp_sm = temp_smat[np.ix_(s2s[i], s2s[i])]
                num_e_a = np.trace(np.dot(sub_dmat[0], temp_sm))
                num_e_b = np.trace(np.dot(sub_dmat[1], temp_sm))
                sub_dmat[0] *= subsystem.mol.nelec[0]/num_e_a
                sub_dmat[1] *= subsystem.mol.nelec[1]/num_e_b
                subsystem.update_density(sub_dmat)
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
                occ_order[0] = np.argsort(local_occ[0])[-1*int(subsystem.mol.nelec[0]):]
                occ_order[1] = np.argsort(local_occ[1])[-1*int(subsystem.mol.nelec[1]):]
                local_occ = [np.zeros(len(local_coeff[0])), np.zeros(len(local_coeff[1]))]
                local_occ[0][occ_order[0]] = 1
                local_occ[1][occ_order[1]] = 1
                new_dm = [np.zeros_like(subsystem.dmat[0]), np.zeros_like(subsystem.dmat[1])]

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

                subsystem.update_density(new_dm)
 
            elif sub_guess == 'readchk':
                is_chkfile = self.read_chkfile()
                if is_chkfile:
                    if (np.any(subsystem.env_mo_coeff) 
                      and np.any(subsystem.env_mo_occ)):
                        sub_mo_coeff = subsystem.env_mo_coeff
                        sub_mo_occ = subsystem.env_mo_occ
                        sub_dmat[0] = np.dot((sub_mo_coeff[0] * sub_mo_occ[0]),
                                          sub_mo_coeff[0].T.conjugate())
                        sub_dmat[1] = np.dot((sub_mo_coeff[1] * sub_mo_occ[1]),
                                          sub_mo_coeff[1].T.conjugate())
                        subsystem.update_density(sub_dmat)
                    elif (np.any(self.mo_coeff) and np.any(self.mo_occ)):
                        sup_dmat = [None, None]
                        sup_dmat[0] = np.dot((self.mo_coeff[0] * self.mo_occ[0]), 
                                          self.mo_coeff[0].T.conjugate())
                        sup_dmat[1] = np.dot((self.mo_coeff[1] * self.mo_occ[1]), 
                                          self.mo_coeff[1].T.conjugate())
                        sub_dmat[0] = sup_dmat[0][np.ix_(s2s[i], s2s[i])]
                        sub_dmat[1] = sup_dmat[1][np.ix_(s2s[i], s2s[i])]
                        # Normalize Density
                        temp_smat = np.copy(fs_scf.get_ovlp())
                        temp_sm = temp_smat[np.ix_(s2s[i], s2s[i])]
                        num_e_a = np.trace(np.dot(sub_dmat[0], temp_sm))
                        num_e_b = np.trace(np.dot(sub_dmat[1], temp_sm))
                        sub_dmat[0] *= subsystem.mol.nelec[0]/num_e_a
                        sub_dmat[1] *= subsystem.mol.nelec[1]/num_e_b
                        subsystem.update_density(sub_dmat)
                    else: #default to super
                        self.get_supersystem_energy(readchk=super_chk)
                        sub_dmat[0] = self.dmat[0][np.ix_(s2s[i], s2s[i])]
                        sub_dmat[1] = self.dmat[1][np.ix_(s2s[i], s2s[i])]
                        # Normalize Density
                        temp_smat = np.copy(fs_scf.get_ovlp())
                        temp_sm = temp_smat[np.ix_(s2s[i], s2s[i])]
                        num_e_a = np.trace(np.dot(sub_dmat[0], temp_sm))
                        num_e_b = np.trace(np.dot(sub_dmat[1], temp_sm))
                        sub_dmat[0] *= subsystem.mol.nelec[0]/num_e_a
                        sub_dmat[1] *= subsystem.mol.nelec[1]/num_e_b
                        subsystem.update_density(sub_dmat)
                else: # default to super.
                    self.get_supersystem_energy(readchk=super_chk)
                    sub_dmat[0] = self.dmat[0][np.ix_(s2s[i], s2s[i])]
                    sub_dmat[1] = self.dmat[1][np.ix_(s2s[i], s2s[i])]
                    temp_smat = np.copy(fs_scf.get_ovlp())
                    temp_sm = temp_smat[np.ix_(s2s[i], s2s[i])]
                    num_e_a = np.trace(np.dot(sub_dmat[0], temp_sm))
                    num_e_b = np.trace(np.dot(sub_dmat[1], temp_sm))
                    sub_dmat[0] *= subsystem.mol.nelec[0]/num_e_a
                    sub_dmat[1] *= subsystem.mol.nelec[1]/num_e_b
                    subsystem.update_density(sub_dmat)
            elif sub_guess == 'submol':
                subsystem.env_scf.kernel()
                temp_dmat = subsystem.env_scf.make_rdm1() 
                if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                    t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                    temp_dmat = t_d
                sub_dmat = temp_dmat
                subsystem.update_density(sub_dmat)
            elif sub_guess in ['atom', '1e', 'minao']:
                temp_dmat = subsystem.env_scf.get_init_guess(key=sub_guess)
                if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                    t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                    temp_dmat = t_d
                sub_dmat = temp_dmat
                subsystem.update_density(sub_dmat)

        # Initialize supersystem density.
        if self.fs_initguess == 'supmol':
            self.get_supersystem_energy(readchk=super_chk)
        if self.fs_initguess == 'readchk': 
            is_chkfile = self.read_chkfile()
            if is_chkfile:
                if (np.any(self.mo_coeff) and np.any(self.mo_occ)):
                    dmat[0] = np.dot((self.mo_coeff[0] * self.mo_occ[0]), 
                                      self.mo_coeff[0].T.conjugate())
                    dmat[1] = np.dot((self.mo_coeff[1] * self.mo_occ[1]), 
                                      self.mo_coeff[1].T.conjugate())
                else:
                    temp_dmat = self.fs_scf.get_init_guess()
                    if temp_dmat.ndim == 2:  #Temp dmat is only one dimensional
                        t_d = [temp_dmat.copy()/2., temp_dmat.copy()/2.]
                        temp_dmat = t_d
                    dmat = temp_dmat
        else:
            if self.fs_initguess == "ft":
                pass
            elif self.fs_initguess != None:
                dmat = fs_scf.get_init_guess(key=self.fs_initguess)
            else:
                dmat = fs_scf.get_init_guess()

            if dmat.ndim == 2:  #Temp dmat is only one dimensional
                t_d = [dmat.copy()/2., dmat.copy()/2.]
                dmat = t_d

        print ("".center(80,'*'))
        return dmat

    def concat_mols(self, subsys_list=None):
        """Concatenates Mole objects into one Mole.

        Parameters
        ----------
        subsys_list : list
            List of subsystems to concatenate into single Mole.
        """

        if subsys_list is None:
            subsys_list = self.subsystems
        if len(subsys_list) < 2:
            # Raise not large enough subsystems.
            print ("Cannot concatenate less than 2 mol objects")
            return False
        mol1 = gto.mole.copy(subsys_list[0].mol)
        for n in range(1, len(subsys_list)):
            mol2 = gto.mole.copy(subsys_list[n].mol)
            for j in range(mol2.natm):
                old_name = mol2.atom_symbol(j)
                new_name = mol2.atom_symbol(j) + ':' + str(n)
                mol2._atom[j] = (new_name, mol2._atom[j][1])
                if old_name in mol2._basis.keys():
                    mol2._basis[new_name] = mol2._basis.pop(old_name)
            mol1 = gto.mole.conc_mol(mol1, mol2)

        #Remove overlapping ghost atoms.
        # I still think there is a better way.
        def remove_overlap(atom_list):
            added_already = {}
            no_dup = []
            for i in range(len(atom_list)):
                coord_tuple = tuple(atom_list[i][1])
                atom_name = atom_list[i][0]
                if not 'ghost' in atom_name:
                    no_dup.append(atom_list[i])
                    if coord_tuple in added_already:
                        dup_index = added_already[coord_tuple]
                        if not 'ghost' in no_dup[dup_index][0]:
                            print ("OVERLAPPING ATOMS")
                        else:
                            del no_dup[added_already[coord_tuple]]
                            for key in added_already.keys():
                                if added_already[key] > added_already[coord_tuple]:
                                    added_already[key] -= 1

                    added_already[coord_tuple] = (len(no_dup) - 1)
                else:
                    if coord_tuple in added_already:
                        pass 
                    else:
                        no_dup.append(atom_list[i])
                        added_already[coord_tuple] = (len(no_dup) - 1)
            return no_dup

        mol1._atom = remove_overlap(mol1._atom)    
        mol1.atom = mol1._atom
        mol1.unit = 'bohr'
        mol1.build(basis=mol1._basis)
        return mol1

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
                    ft_dmat[0][np.ix_(s2s[i], s2s[i])] += subsystem.dmat[0]
                    ft_dmat[1][np.ix_(s2s[i], s2s[i])] += subsystem.dmat[1]
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
            self.dmat = scf_obj.make_rdm1()

            #One way of determining electrons.
            if self.analysis:
                temp_dmat = np.copy(self.dmat)
                temp_dmat[:self.subsystems[0].mol.nao_nr(), 
                          :self.subsystems[0].mol.nao_nr()] = 0.0
                temp_dmat[self.subsystems[0].mol.nao_nr():, 
                          self.subsystems[0].mol.nao_nr():] = 0.0
                temp_smat = np.copy(scf_obj.get_ovlp())
                temp_smat[:self.subsystems[0].mol.nao_nr(), 
                          :self.subsystems[0].mol.nao_nr()] = 0.0
                temp_smat[self.subsystems[0].mol.nao_nr():, 
                          self.subsystems[0].mol.nao_nr():] = 0.0
                print ("Interaction Electrion Number")
                print (np.trace(np.dot(temp_dmat, scf_obj.get_ovlp())))
                print ()

                #A localization way
                mull_pop = scf_obj.mulliken_pop(verbose=3)[1]
                print ("Mull 1")
                print (np.sum(mull_pop[:self.subsystems[0].mol.natm]))
                print ("Mull 2")
                print (np.sum(mull_pop[self.subsystems[0].mol.natm:]))

            if self.dmat.ndim == 2: #Always store as alpha and beta, even if closed shell. Makes calculations easier.
                t_d = [self.dmat.copy()/2., self.dmat.copy()/2.]
                self.dmat = t_d
                self.mo_coeff = [self.fs_scf.mo_coeff, self.fs_scf.mo_coeff]
                self.mo_occ = [self.fs_scf.mo_occ/2, self.fs_scf.mo_occ/2]
                self.mo_energy = [self.fs_scf.mo_energy, self.fs_scf.mo_energy]
            
            #self.save_chkfile()
            self.fs_energy = scf_obj.energy_tot()
            print("".center(80,'*'))
            if self.fs_save_density:
                print('Writing Full System Density'.center(80))
                if self.fs_unrestricted or self.mol.spin != 0:
                    cubename_a = os.path.splitext(self.filename)[0] + '_super_a' + '.cube'
                    dmat = self.dmat[0]
                    cubegen.density(mol, cubename_a, dmat) 

                    cubename_b = os.path.splitext(self.filename)[0] + '_super_b' + '.cube'
                    dmat = self.dmat[1]
                    cubegen.density(mol, cubename_b, dmat) 
                else:
                    cubename = os.path.splitext(self.filename)[0] + '_super' + '.cube'
                    dmat = self.dmat[0] + self.dmat[1]
                    cubegen.density(mol, cubename, dmat) 

            if self.fs_save_orbs:
                print('Writing Full System Orbitals'.center(80))
                from pyscf.tools import molden
                if self.fs_unrestricted or self.mol.spin != 0:
                    moldenname_a = os.path.splitext(self.filename)[0] + '_super_a' + '.molden'
                    moldenname_b = os.path.splitext(self.filename)[0] + '_super_b' + '.molden'
                    with open(moldenname_a, 'w') as fin:
                        molden.header(self.mol, fin)
                        molden.orbital_coeff(self.mol, fin, self.mo_coeff[0], ene=self.mo_energy[0], occ=self.mo_occ[0], spin='Alpha')
                    with open(moldenname_b, 'w') as fin:
                        molden.header(self.mol, fin)
                        molden.orbital_coeff(self.mol, fin, self.mo_coeff[1], ene=self.mo_energy[1], occ=self.mo_occ[1], spin='Beta')
                else:
                    moldenname = os.path.splitext(self.filename)[0] + '_super' + '.molden'
                    with open(moldenname, 'w') as fin:
                        molden.header(self.mol, fin)
                        molden.orbital_coeff(self.mol, fin, self.mo_coeff[0], ene=self.mo_energy[0], occ=self.mo_occ[0])
           
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
    def get_active_energy(self):
        """Determines the active energy.
        
        """
        #This is crude. Later iterations should be more sophisticated and account for more than 2 subsystems.
        print ("".center(80,'*'))
        print("  Active Subsystem Calculation  ".center(80))
        print ("".center(80,'*'))
        s2s = self.sub2sup
        FAA = [None, None]
        FAA[0] = self.fock[0][np.ix_(s2s[0], s2s[0])]
        FAA[1] = self.fock[1][np.ix_(s2s[0], s2s[0])]
        self.subsystems[0].update_emb_fock(FAA)
        self.subsystems[0].active_in_env_energy()
        print (f"Uncorrected Energy: {self.subsystems[0].active_energy:>.8f}")
        #CORRECT ACTIVE SETTINGS.
        #act_elec_e = self.correct_active_energy()
        act_elec_e = 0.0
        self.subsystems[0].active_energy += act_elec_e
        act_e = self.subsystems[0].active_energy
        print(f"Energy:{act_e:>73.8f}")
        print("".center(80,'*'))

    @time_method("Env Energy")
    def get_env_energy(self):
        """Determines the subsystem env energy
        
        """
        print ("".center(80,'*'))
        print("  Env Subsystem Calculation  ".center(80))
        print ("".center(80,'*'))
        self.subsystems[0].update_fock()
        s2s = self.sub2sup
        FAA = [None, None]
        FAA[0] = self.fock[0][np.ix_(s2s[0], s2s[0])]
        FAA[1] = self.fock[1][np.ix_(s2s[0], s2s[0])]
        froz_veff = [None, None]
        froz_veff[0] = (FAA[0] - self.subsystems[0].env_hcore - self.subsystems[0].env_V[0])
        froz_veff[1] = (FAA[1] - self.subsystems[0].env_hcore - self.subsystems[0].env_V[1])
        self.subsystems[0].update_emb_pot(froz_veff)
        self.subsystems[0].get_env_energy()
        print (f"Uncorrected Energy:{self.subsystems[0].active_energy:>61.8f}")
        #CORRECT ACTIVE SETTINGS.
        #act_elec_e = self.correct_active_energy()
        env_elec_e = 0.0
        self.subsystems[0].env_energy += env_elec_e
        env_e = self.subsystems[0].env_energy
        print(f"Energy:{env_e:>73.8f}")
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
    def env_in_env_energy(self):
        """Calculates the energy of dft-in-dft.

        This is unnecessary for the Total Embedding energy, however it 
        may be a useful diagnostic tool.

        """
        print ("".center(80,'*'))
        print("  Env-in-Env Calculation  ".center(80))
        print ("".center(80,'*'))
        nS = self.mol.nao_nr()
        dm_env = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        for i in range(len(self.subsystems)):
            dm_env[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].dmat[0]
            dm_env[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].dmat[1]
        if self.fs_unrestricted or self.mol.spin != 0:
            self.env_energy = self.fs_scf.energy_tot(dm=dm_env)
        else:
            self.env_energy = self.fs_scf.energy_tot(dm=(dm_env[0] + dm_env[1]))
            if self.ft_save_density:
                print('Writing DFT-in-DFT Density'.center(80))
                if self.fs_unrestricted or self.mol.spin != 0:
                    cubename_a = os.path.splitext(self.filename)[0] + '_dftindft_a' + '.cube'
                    dmat = dm_env[0]
                    cubegen.density(self.fs_scf.mol, cubename_a, dmat) 
                    cubename_b = os.path.splitext(self.filename)[0] + '_dftindft_b' + '.cube'
                    dmat = dm_env[1]
                    cubegen.density(self.fs_scf.mol, cubename_b, dmat) 
                else:
                    cubename = os.path.splitext(self.filename)[0] + '_dftindft' + '.cube'
                    dmat = dm_env[0] + dm_env[1]
                    cubegen.density(self.fs_scf.mol, cubename, dmat) 
            if self.ft_save_orbs:
                print('Writing DFT-in-DFT Orbitals'.center(80))
                #The MO coefficients must be created from the density.
                #moldenname = os.path.splitext(self.filename)[0] + '_dftindft' + '.molden'
                #dmat = dm_env[0] + dm_env[1]
                #cubegen.density(self.fs_scf.mol, cubename, dmat) 

        self.dftindft_dmat = dm_env
        print(f"DFT-in-DFT Energy:{self.env_energy:>62.8f}")
        print("".center(80,'*'))
        return self.env_energy


    def update_fock(self, diis=True):
        """Updates the full system fock matrix.

        Parameters
        ----------
        diis : bool
            Whether to use the diis method.
        """

        self.fock = [np.copy(self.hcore), np.copy(self.hcore)]

        # Optimization: Rather than recalculate the full V, only calculate the V for densities which changed. 
        # get 2e matrix
        nS = self.mol.nao_nr()
        dm = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        sub_openshell = False
        for i in range(len(self.subsystems)):
            if self.subsystems[i].unrestricted:
                sub_openshell = True 
            dm[0][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].dmat[0]
            dm[1][np.ix_(self.sub2sup[i], self.sub2sup[i])] += self.subsystems[i].dmat[1]

        if self.fs_unrestricted or sub_openshell:
            V = self.os_scf.get_veff(mol=self.mol, dm=dm)
            V_a = V[0]
            V_b = V[1]
        elif self.mol.spin != 0:
            #RO
            pass
        else:
            V_a = self.fs_scf.get_veff(mol=self.mol, dm=(dm[0] + dm[1]))
            V_b = V_a

        self.fock[0] += V_a
        self.fock[1] += V_b

        # to use the scf diis methods, must generate individual fock parts and recombine to make full fock matrix, because the projection operator is necessary to use diis correctly.
        if not self.ft_diis is None and diis:
            if self.fs_method[0] == 'u' or self.fs_method[:2] == 'ro' or sub_openshell:
                self.fock[0] = self.ft_diis[0].update(self.fock[0])
                self.fock[1] = self.ft_diis[0].update(self.fock[1])
            else:
                #f = self.ft_diis[0].update(self.smat, (dm[0] + dm[1]), self.fock[0], self.fs_scf, self.hcore, V_a)
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

        s2s = self.sub2sup
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            sub_fock_0 = self.fock[0][np.ix_(s2s[i], s2s[i])]
            sub_fock_1 = self.fock[1][np.ix_(s2s[i], s2s[i])]
            subsystem.update_emb_fock([sub_fock_0, sub_fock_1])

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

                SAB = self.smat[np.ix_(s2s[A], s2s[B])]
                SBA = self.smat[np.ix_(s2s[B], s2s[A])]

                # get mu-parameter projection operator
                if isinstance(self.proj_oper, int) or isinstance(self.proj_oper, float):
                    POp[0] += self.proj_oper * np.dot( SAB, np.dot( self.subsystems[B].dmat[0], SBA ))
                    POp[1] += self.proj_oper * np.dot( SAB, np.dot( self.subsystems[B].dmat[1], SBA ))

                elif self.proj_oper in ('huzinaga', 'huz'):
                    FAB = [None, None]
                    FAB[0] = self.fock[0][np.ix_(s2s[A], s2s[B])]
                    FAB[1] = self.fock[1][np.ix_(s2s[A], s2s[B])]
                    FDS = [None, None]
                    FDS[0] = np.dot( FAB[0], np.dot( self.subsystems[B].dmat[0], SBA ))
                    FDS[1] = np.dot( FAB[1], np.dot( self.subsystems[B].dmat[1], SBA ))
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
                    FDS[0] = np.dot( FAB[0], np.dot( self.subsystems[B].dmat[0], SBA ))
                    FDS[1] = np.dot( FAB[1], np.dot( self.subsystems[B].dmat[1], SBA ))
                    POp[0] += -1. * ( FDS[0] + FDS[0].transpose() ) 
                    POp[1] += -1. * ( FDS[1] + FDS[1].transpose() )
            self.proj_pot[i] = POp.copy()
        return self.proj_pot

    #These should catch exceptions.
    def read_chkfile(self):
    # Need to make more robust. Handle errors and such.
        if os.path.isfile(self.chk_filename):
            try:
                with h5py.File(self.chk_filename, 'r') as hf:
                    supsys_coeff = hf['supersystem/mo_coeff']
                    self.mo_coeff = supsys_coeff[:]
                    supsys_occ = hf['supersystem/mo_occ']
                    self.mo_occ = supsys_occ[:]
                    supsys_energy = hf['supersystem/mo_energy']
                    self.mo_energy = supsys_energy[:]

                    for i in range(len(self.subsystems)):
                        subsystem = self.subsystems[i]
                        subsys_coeff = hf[f'subsystem:{i}/mo_coeff']
                        subsystem.env_mo_coeff = subsys_coeff[:]
                        subsys_occ = hf[f'subsystem:{i}/mo_occ']
                        subsystem.env_mo_occ = subsys_occ[:]
                        subsys_energy = hf[f'subsystem:{i}/mo_energy']
                        subsystem.env_mo_energy = subsys_energy[:]
                return True 
            except TypeError:
                print ("chkfile improperly formatted".center(80))
                return False
        else:
            print ("chkfile NOT found".center(80))
            return False

    def save_chkfile(self):
        # current plan is to save mo_coefficients, occupation vector, and energies.
        # becasue of how h5py works we need to check if none and save as the correct filetype (f)
        #Need to make more robust. Handle errors and such.
        
        # check if file exists. 
        if os.path.isfile(self.chk_filename):
            try:
                with h5py.File(self.chk_filename, 'r+') as hf:
                    supsys_coeff = hf['supersystem/mo_coeff']
                    supsys_coeff[...] = self.mo_coeff
                    supsys_occ = hf['supersystem/mo_occ']
                    supsys_occ[...] = self.mo_occ
                    supsys_energy = hf['supersystem/mo_energy']
                    supsys_energy[...] = self.mo_energy
                    for i in range(len(self.subsystems)):
                        subsystem = self.subsystems[i]
                        subsys_coeff = hf[f'subsystem:{i}/mo_coeff']
                        subsys_coeff[...] = subsystem.env_mo_coeff
                        subsys_occ = hf[f'subsystem:{i}/mo_occ']
                        subsys_occ[...] = subsystem.env_mo_occ
                        subsys_energy = hf[f'subsystem:{i}/mo_energy']
                        subsys_energy[...] = subsystem.env_mo_energy
            except TypeError:
                print ("Overwriting existing chkfile".center(80))
                with h5py.File(self.chk_filename, 'w') as hf:
                    hf.create_dataset("embedding_chkfile", data=True)
                    sup_mol = hf.create_group('supersystem')
                    sup_mol.create_dataset('mo_coeff', data=self.mo_coeff)
                    sup_mol.create_dataset('mo_occ', data=self.mo_occ)
                    sup_mol.create_dataset('mo_energy', data=self.mo_energy)

                    for i in range(len(self.subsystems)):
                        subsystem = self.subsystems[i]
                        sub_sys_data = hf.create_group(f'subsystem:{i}')
                        sub_sys_data.create_dataset('mo_coeff', data=subsystem.env_mo_coeff)
                        sub_sys_data.create_dataset('mo_occ', data=subsystem.env_mo_occ)
                        sub_sys_data.create_dataset('mo_energy', data=subsystem.env_mo_energy)
         
        else:
            with h5py.File(self.chk_filename, 'w') as hf:
                hf.create_dataset("embedding_chkfile", data=True)
                sup_mol = hf.create_group('supersystem')
                sup_mol.create_dataset('mo_coeff', data=self.mo_coeff)
                sup_mol.create_dataset('mo_occ', data=self.mo_occ)
                sup_mol.create_dataset('mo_energy', data=self.mo_energy)

                for i in range(len(self.subsystems)):
                    subsystem = self.subsystems[i]
                    sub_sys_data = hf.create_group(f'subsystem:{i}')
                    sub_sys_data.create_dataset('mo_coeff', data=subsystem.env_mo_coeff)
                    sub_sys_data.create_dataset('mo_occ', data=subsystem.env_mo_occ)
                    sub_sys_data.create_dataset('mo_energy', data=subsystem.env_mo_energy)

    @time_method("Freeze and Thaw")
    def freeze_and_thaw(self):
        # Optimization: rather than recalculate vA use the existing fock and subtract out the block that is double counted.

        print("".center(80, '*'))
        print("Freeze-and-Thaw".center(80))
        print("".center(80, '*'))
         
        s2s = self.sub2sup
        ft_err = 1.
        ft_iter = 0 
        last_cycle = False

        #def sub_diag(subsystem):
        #    subsystem.diagonalize()
        #    return True

        while((ft_err > self.ft_conv) and (ft_iter < self.ft_cycles)):
            # cycle over subsystems
            ft_err = 0
            ft_iter += 1
            #Correct for DIIS 
            # If fock only updates after cycling, then use python multiprocess todo simultaneously.
            multi_cycle = (len(self.subsystems) - self.ft_updatefock)
            for i in range(0, len(self.subsystems), multi_cycle):
                self.update_fock(diis=True)
                sub_list = [sub for sub in self.subsystems[i:i+multi_cycle] if not sub.freeze]
                #this will slow down calculation. 
                #if self.analysis:
                #    self.get_emb_subsys_elec_energy()
                #    sub_old_e = subsystem.get_env_energy()

                sub_old_dms = [sub.dmat.copy() for sub in sub_list]
             
                for j in range(len(sub_list)):
                    self.update_proj_pot() #could use i as input and only get for that sub.
                    sub_list[j].update_proj_pot(self.proj_pot[j+i])
                    #Remove the following if multiprocessing.
                    sub_list[j].diagonalize()

                #This is where to do multiprocess.  
                #pool = Pool(processes=multi_cycle)
               # result = parmap(sub_diag, sub_list)
                
                for k in range(len(sub_list)):
                    subsystem = sub_list[k]
                    new_dm = [None, None]
                    new_dm[0] = ((1 - self.ft_damp) * subsystem.dmat[0] + (self.ft_damp * sub_old_dms[k][0]))
                    new_dm[1] = ((1 - self.ft_damp) * subsystem.dmat[1] + (self.ft_damp * sub_old_dms[k][1]))
                    subsystem.update_density(new_dm) 
                    ddm = sp.linalg.norm(subsystem.dmat[0] - sub_old_dms[k][0])
                    ddm += sp.linalg.norm(subsystem.dmat[1] - sub_old_dms[k][1])
                    proj_e = np.trace(np.dot(subsystem.dmat[0], self.proj_pot[i+k][0]))
                    proj_e += np.trace(np.dot(subsystem.dmat[1], self.proj_pot[i+k][1]))
                    ft_err += ddm
                    self.ft_fermi[i+k] = subsystem.fermi

                    #This will slow down execution.
                    #if self.analysis:
                        #self.get_emb_subsys_elec_energy()
                        #sub_new_e = subsystem.get_env_energy()
                        #dE = abs(sub_old_e - sub_new_e)

                    # print output to console.
                    if self.analysis:
                        print(f"iter:{ft_iter:>3d}:{i+j:<2d}          |dE|:{dE:12.6e}           |ddm|:{ddm:12.6e}           |Tr[DP]|:{proj_e:12.6e}")
                    else:
                        print(f"iter:{ft_iter:>3d}:{i+j:<2d}              |ddm|:{ddm:12.6e}               |Tr[DP]|:{proj_e:12.6e}")
                self.save_chkfile()

        print("".center(80))
        self.is_ft_conv = True
        if(ft_err > self.ft_conv):
            print("".center(80))
            print("Freeze-and-Thaw NOT converged".center(80))
        
        
        # cycle over subsystems
        self.update_fock(diis=False)
        for i in range(len(self.subsystems)):
            subsystem = self.subsystems[i]
            if not subsystem.freeze:
                #subsystem.update_fock()

                self.update_proj_pot() #could use i as input and only get for that sub.
                SAA = self.smat[np.ix_(s2s[i], s2s[i])]
                #I don't think this changes. Could probably set in the initialize.

                subsystem.update_proj_pot(self.proj_pot[i])
                subsystem.diagonalize(diis=-1)
                # save to file. could be done in larger cycles.
                self.save_chkfile()
                self.ft_fermi[i] = subsystem.fermi

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
                if subsystem.unrestricted or subsystem.mol.spin != 0:
                    cubename_a = os.path.splitext(self.filename)[0] + '_' + str(i+1) + '_a.cube'
                    dmat = subsystem.dmat[0]
                    cubegen.density(subsystem.mol, cubename_a, dmat) 

                    cubename_b = os.path.splitext(self.filename)[0] + '_' + str(i+1) + '_b.cube'
                    dmat = subsystem.dmat[1]
                    cubegen.density(subsystem.mol, cubename_b, dmat) 
                else:
                    cubename = os.path.splitext(self.filename)[0] + '_' + str(i+1) + '.cube'
                    dmat = subsystem.dmat[0] + subsystem.dmat[1]
                    cubegen.density(subsystem.mol, cubename, dmat) 

            if subsystem.save_orbs:
                print(f'Writing Subsystem {i} Env Orbitals'.center(80))
                from pyscf.tools import molden
                if subsystem.unrestricted or subsystem.mol.spin != 0:
                    moldenname_a = os.path.splitext(self.filename)[0] + '_' + str(i+1) + '_a.molden'
                    with open(moldenname_a, 'w') as fin:
                        molden.header(subsystem.mol, fin)
                        molden.orbital_coeff(subsystem.mol, fin, subsystem.env_mo_coeff[0], ene=subsystem.env_mo_energy[0], occ=subsystem.env_mo_occ[0], spin='Alpha')

                    moldenname_b = os.path.splitext(self.filename)[0] + '_' + str(i+1) + '_b.molden'
                    with open(moldenname_b, 'w') as fin:
                        molden.header(subsystem.mol, fin)
                        molden.orbital_coeff(subsystem.mol, fin, subsystem.env_mo_coeff[1], ene=subsystem.env_mo_energy[1], occ=subsystem.env_mo_occ[1], spin='Beta')
                else:
                    moldenname = os.path.splitext(self.filename)[0] + '_' + str(i+1) + '.molden'
                    with open(moldenname, 'w') as fin:
                        molden.header(subsystem.mol, fin)
                        molden.orbital_coeff(subsystem.mol, fin, subsystem.env_mo_coeff[0], ene=subsystem.env_mo_energy[0], occ=subsystem.env_mo_occ[0])
            
    def get_dft_diff_parameters(self, fs_dmat=None, fs_scf=None, dftindft_dmat=None):
        
        if fs_dmat is None:
            fs_dmat = self.dmat
        if fs_scf is None:
            fs_dmat = self.fs_scf
        if dftindft_dmat is None:
            dftindft_dmat = self.dftindft_dmat

        e_diff = self.fs_energy - self.env_energy
        print (f"Energy Difference of KS-DFT to DFT-in-DFT:{e_diff:>38.8f}")

        trace_diff = (0.5 * np.trace(self.dmat[0] - self.dftindft_dmat[0]) 
                     + 0.5 * np.trace(self.dmat[1] - self.dftindft_dmat[1]))

        print (f"Trace Difference of KS-DFT to DFT-in-DFT:{trace_diff:>39.8f}") 
 
       
