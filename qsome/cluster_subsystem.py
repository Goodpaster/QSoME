""" A method to define cluster subsystem objects
Daniel Graham
Dhabih V. Chulhai
"""

import re
import os
from copy import deepcopy as copy
import h5py
import numpy as np
import scipy as sp
from pyscf import gto, scf, mp, cc, mcscf, mrpt, fci, tools
from pyscf import hessian
from pyscf.cc import ccsd_t, uccsd_t
from pyscf.cc import eom_uccsd, eom_rccsd
from pyscf.scf import diis as scf_diis
from pyscf.lib import diis as lib_diis
from qsome import custom_pyscf_methods, custom_diis
from qsome.ext_methods.ext_factory import ExtFactory



class ClusterEnvSubSystem:
    """A base subsystem object for use in projection embedding.

    Attributes
    ----------
    mol : Mole
        The pyscf Mole object specifying the geometry and basis
    env_method : str
        Defines the method to use for environment calculations.
    env_order : int
        An ordering scheme to keep track of subsystems in the big picture.
    env_init_guess : str
        The initial guess for the density matrix.
    env_damp : float
        The damping parameter for F&T calculations.
    env_shift : float
        Orbital shifting parameter.
    env_subcycles : int
        Number of scf subcycles for freeze and thaw cycles.
    diis_num : int
        A number indicating what kind of DIIS will be used for fock acceleration.
    unrestricted : bool
        Whether the subsystem is unrestricted.
    density_fitting : bool
        Whether to use density fitting.
    freeze : bool
        Whether to relax the density matrix
    save_orbs : bool
        Whether to save the env orbitals
    save_density : bool
        Whether to save the env density
    save_spin_density : bool
        Whether to save the spin density.
    filename : str
        A path to the input file.
    chkfile_index : str
        An identifier for the subsystem within the context of the full system.
    nproc : int
        The number of processors accessible to the calculation.
    pmem : float
        The amount of memory per processor (in MB)
    scr_dir : str
        The path to the scratch directory for the calculation.
    fermi : array
        An array of alpha and beta fermi energies.
    env_scf : SCF
        The pyscf SCF object of the subsystem.
    env_hcore : np.float64
        A numpy array of core hamiltonian matrix, compatible with pyscf.
    env_dmat : np.float64
        A numpy array of electron density matrix, compatible with pyscf.
    emb_fock : array
        An array of alpha and beta embedded fock matrices.
    emb_proj_fock : array
        An array of alpha and beta embedded and projected fock matrices.
    subsys_fock : array
        An array of alpha and beta subsystem fock matrices.
    emb_pot : array
        An array of alpha and beta embedding potentials (emb_fock - subsys_fock).
    proj_pot : array
        An array of alpha and beta projection potentials.
    env_mo_coeff : np.float64
        A numpy array of mo coefficients, compatible with pyscf.
    env_mo_occ : np.float
        A numpy array of mo occupations, compatible with psycf
    env_mo_energy : np.float
        A numpy array of mo energies, compatible with psycf
    env_energy : float
        The total energy of this subsystem.
    diis : DIIS object
        The PySCF DIIS object for fock acceleration of the subsystem.

    Methods
    -------
    init_env_scf()
       Initializes the pyscf SCF object.
    init_density()
        Sets the initial subsystem density matrix.
    get_dmat()
        Returns a formatted density matrix.
    update_subsys_fock(dmat, hcore)
        Updates the subsystem fock matrix.
    update_emb_pot(emb_fock)
        Updates the embedding potential.
    get_env_proj_e()
        Returns the energy of the projection potential.
    get_env_emb_e()
        Returns the embedded energy
    get_env_elec_energy()
        Get the electronic energy for the subsystem.
    get_env_energy()
        Get the total energy for the subsystem.
    save_orbital_file()
        Saves the env orbitals to a file.
    save_density_file()
        Save the env electron density to a file.
    save_spin_density_file()
        Save the env electron spin density to a file.
    save_chkfile()
        Saves the electron density to a chkfile for calculation restart purposes.
    read_chkfile()
        Reads an existing chkfile and initializes the electron density to that value.
    diagonalize()
        Diagonalize the env subsystem and return an update density.
    __do_unrestricted_diag()
        Diagonalize an unrestricted subsystem.
    __do_restricted_os_diag()
        Diagonalize a restricted open shell subsystem.
    __do_restricted_diag()
        Diagonalize a restricted closed shell subsystem.
    relax_sub_dmat()
        Relaxes the subsystem based on the fock operator and returns the difference
        between old and new density matrices.
    __set_fermi(e_sorted)
        Sets the fermi parameter of the subsystem based on the list of sorted orbitals
        (esorted).
    __set_occupation()
        Sets the molecular occupation based on the sorted molecular orbital energies.
    """


    def __init__(self, mol, env_method, env_order=1, init_guess=None, damp=0.,
                 shift=0., subcycles=1, diis_num=0, unrestricted=False,
                 density_fitting=False, freeze=False, save_orbs=False, 
                 save_density=False, save_spin_density=False, filename=None,
                 nproc=None, pmem=None, scrdir=None):
        """
        Parameters
        ----------
        mol : Mole
            The pyscf Mole object specifying the geometry and basis
        env_method : str
            Defines the method to use for environment calculations.
        env_order : int, optional
            ID for the subsystem in the full system.
            (default is 1)
        init_guess : str, optional
            Which method to use for the initial density guess.
            (default is None)
        damp : float, optional
            Damping percentage. Mixeas a percent of previous density into
            each new density. (default is 0.)
        shift : float, optional
            How much to level shift orbitals. (default is 0.)
        subcycles : int, optional
            Number of diagonalization cycles. (default is 1)
        diis_num : int, optional
            Specifies DIIS method to use. (default is 0)
        unrestricted : bool, optional
            Whether the subsystem is unrestricted.
            (default is False)
        density_fitting : bool, optional
            Whether to use density fitting for the env method.
            (default is False)
        freeze : bool, optional
            Whether to freeze the electron density.
            (default is False)
        save_orbs : bool, optional
            Whether to save the env orbitals to a file.
            (default is False)
        save_density : bool, optional
            Whether to save the electron density to a file.
            (default is False)
        save_spin_density: bool, optional
            Whether to save the spin density to a file.
            (default is False)
        filename : str, optional
            The path to the input file being read. (default is None)
        nproc : int, optional
            Number of processors provided for calculation. (default is None)
        pmem : int, optional
            Memory per processor available in MB. (default is None)
        scr_dir : str, optional
            Path to the directory used for scratch. (default is None)
        """

        self.mol = mol
        self.env_method = env_method
        self.env_order = env_order

        self.env_init_guess = init_guess
        self.env_damp = damp
        self.env_shift = shift

        self.env_subcycles = subcycles
        self.diis_num = diis_num

        self.unrestricted = unrestricted
        self.density_fitting = density_fitting
        self.freeze = freeze
        self.save_orbs = save_orbs
        self.save_density = save_density
        self.save_spin_density = save_spin_density

        self.filename = filename
        self.chkfile_index = None
        self.nproc = nproc
        if nproc is None:
            self.nproc = 1
        self.pmem = pmem
        if pmem is None:
            self.pmem = 2000
        self.scr_dir = scrdir
        if scrdir is None:
            self.scr_dir = os.getenv('TMPDIR')

        self.fermi = [0., 0.]
        self.env_scf = self.init_env_scf()
        self.env_hcore = self.env_scf.get_hcore()
        self.env_dmat = None
        self.emb_fock = np.array([None, None])
        self.emb_proj_fock = np.array([None, None])
        self.subsys_fock = np.array([None, None])

        self.emb_pot = np.array([np.zeros_like(self.env_hcore),
                                 np.zeros_like(self.env_hcore)])
        self.proj_pot = np.array([np.zeros_like(self.env_hcore),
                                  np.zeros_like(self.env_hcore)])

        self.env_mo_coeff = np.array([np.zeros_like(self.env_hcore),
                                      np.zeros_like(self.env_hcore)])
        self.env_mo_occ = np.array([np.zeros_like(self.env_hcore[0]),
                                    np.zeros_like(self.env_hcore[0])])
        self.env_mo_energy = self.env_mo_occ.copy()
        self.env_energy = 0.0

        if self.diis_num == 1:
            #Use subtractive diis. Most simple
            self.diis = lib_diis.DIIS()
        elif self.diis_num == 2:
            self.diis = scf_diis.CDIIS()
        elif self.diis_num == 3:
            self.diis = scf_diis.EDIIS()
        elif self.diis_num == 4:
            self.diis = scf.diis.ADIIS()
        elif self.diis_num == 5:
            self.diis = custom_diis.EDIIS_DIIS(self.env_scf)
        elif self.diis_num == 6:
            self.diis = custom_diis.ADIIS_DIIS(self.env_scf)
        else:
            self.diis = None

    def init_env_scf(self, mol=None, env_method=None, damp=None, shift=None, 
                     dfit=None):
        """Initializes the environment pyscf scf object.

        Parameters
        ----------
        mol : Mole, optional
            Mole object containing geometry and basis (default is None).
        method : str, optional
            Subsystem method for calculation (default is None).
        rho_cutoff : float, optional
            DFT rho cutoff parameter (default is None).
        damp : float, optional
            Damping parameter (default is None).
        shift : float, optional
            Level shift parameter (default is None).
        """

        if mol is None:
            mol = self.mol
        if env_method is None:
            env_method = self.env_method
        if damp is None:
            damp = self.env_damp
        if shift is None:
            shift = self.env_shift
        if dfit is None:
            dfit = self.density_fitting

        if self.pmem:
            mol.max_memory = self.pmem

        if self.unrestricted:
            if env_method == 'hf':
                scf_obj = scf.UHF(mol)
            else:
                scf_obj = scf.UKS(mol)
                scf_obj.xc = env_method

        elif mol.spin != 0:
            if 'hf' in env_method:
                scf_obj = scf.ROHF(mol)
            else:
                scf_obj = scf.ROKS(mol)
                scf_obj.xc = env_method
        else:
            if env_method == 'hf':
                scf_obj = scf.RHF(mol)
            else:
                scf_obj = scf.RKS(mol)
                scf_obj.xc = env_method

        env_scf = scf_obj
        env_scf.damp = damp
        env_scf.level_shift = shift
        if dfit:
            env_scf = env_scf.density_fit()
        return env_scf

    def init_density(self, in_dmat=None, scf_obj=None, env_method=None,
                     init_guess=None):
        """Initializes the subsystem density..

        Parameters
        ----------
        in_dmat : numpy.float64
            New subsystem density matrix (default is None).
        scf_obj : SCF, optional
            Subsystem SCF object (default is None).
        env_method : str, optional
            Subsystem energy method (default is None).
        init_guess : str, optional
            Subsystem density guess method (default is None).
        """
        if in_dmat is not None:
            in_dmat = np.array(in_dmat)
            self.env_dmat = in_dmat
            return True

        if scf_obj is None:
            scf_obj = self.env_scf
        if env_method is None:
            env_method = self.env_method
        if init_guess is None:
            if self.env_init_guess is None:
                init_guess = 'chk'
            else:
                init_guess = self.env_init_guess

        if init_guess == 'chk':
            try:
                is_chkfile = self.read_chkfile()
            except AssertionError:
                is_chkfile = False
            if is_chkfile:
                if (np.any(self.env_mo_coeff) and np.any(self.env_mo_occ)):
                    #Confirm correct read density dimensions.
                    ndim = scf_obj.mol.nao
                    if (ndim == self.env_mo_coeff.shape[1] and ndim == self.env_mo_coeff.shape[2]):
                        dmat = [0, 0]
                        dmat[0] = np.dot((self.env_mo_coeff[0] * self.env_mo_occ[0]),
                                         self.env_mo_coeff[0].T.conjugate())
                        dmat[1] = np.dot((self.env_mo_coeff[1] * self.env_mo_occ[1]),
                                         self.env_mo_coeff[1].T.conjugate())
                    else:
                        self.env_mo_coeff = [np.zeros_like(self.env_hcore),
                                             np.zeros_like(self.env_hcore)]
                        self.env_mo_occ = [np.zeros_like(self.env_hcore[0]),
                                           np.zeros_like(self.env_hcore[0])]
                        init_guess = 'supmol'
                        dmat = scf_obj.get_init_guess()
                else:
                    init_guess = 'supmol'
                    dmat = scf_obj.get_init_guess()
            else:
                init_guess = 'supmol'
                dmat = scf_obj.get_init_guess()

            #If readchk not found, update the init_guess method
            self.env_init_guess = init_guess

        elif init_guess in ['atom', '1e', 'minao', 'huckel', 'vsap']:
            dmat = scf_obj.get_init_guess(key=init_guess)
        elif init_guess == 'submol':
            scf_obj.kernel()
            dmat = scf_obj.make_rdm1()
        else:
            dmat = scf_obj.get_init_guess()

        #Dmat always stored [alpha, beta]
        if np.array(dmat).ndim == 2:
            dmat = np.array([dmat/2., dmat/2.])
        self.env_dmat = dmat

        #Initialize the subsys fock when density initialized.
        self.update_subsys_fock()
        return True

    def get_dmat(self):
        """Returns the density matrix"""

        dmat = self.env_dmat
        if not (self.unrestricted or self.mol.spin != 0):
            dmat = dmat[0] + dmat[1]
        return dmat

    def update_subsys_fock(self, dmat=None, hcore=None):
        """Update the subsystem fock matrix

        Parameters
        ----------
        dmat : array
        hcore : array

        Returns
        -------
        boolean
        """

        if dmat is None:
            dmat = self.env_dmat
        if hcore is None:
            hcore = self.env_hcore

        if self.unrestricted:
            self.subsys_fock = self.env_scf.get_fock(h1e=hcore, dm=dmat)
        elif self.mol.spin != 0:
            temp_fock = self.env_scf.get_fock(h1e=hcore, dm=dmat)
            self.subsys_fock = [temp_fock, temp_fock]
        else:
            temp_fock = self.env_scf.get_fock(h1e=hcore, dm=(dmat[0] + dmat[1]))
            self.subsys_fock = [temp_fock, temp_fock]
        return True

    def update_emb_pot(self, emb_fock=None):
        """Updates the embededing potential for the system

        Parameters
        ----------
        emb_fock : list
        """

        if emb_fock is None:
            if self.emb_fock[0] is None:
                emb_fock = None
            else:
                emb_fock = self.emb_fock
        self.update_subsys_fock()
        self.emb_pot = [emb_fock[0] - self.subsys_fock[0],
                        emb_fock[1] - self.subsys_fock[1]]

    def get_env_proj_e(self, proj_pot=None, dmat=None):
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

        if proj_pot is None:
            proj_pot = self.proj_pot
        if dmat is None:
            dmat = copy(self.env_dmat)

        e_proj = (np.einsum('ij,ji', proj_pot[0], dmat[0]) +
                  np.einsum('ij,ji', proj_pot[1], dmat[1])).real

        return e_proj

    def get_env_emb_e(self, emb_pot=None, dmat=None):
        """Gets the embedded energy

        Parameters
        ----------
        env_method : str, optional
            Subsystem low level method string (default is None).
        proj_pot : numpy.float64, optional
            Projection potential matrix (default is None).
        dmat : numpy.float64, optional
            Subsystem density matrix (default is None).
        """
        if dmat is None:
            dmat = copy(self.env_dmat)

        if emb_pot is None:
            if self.emb_fock[0] is None:
                emb_pot = [np.zeros_like(dmat[0]), np.zeros_like(dmat[1])]
            else:
                emb_pot = [self.emb_fock[0] - self.subsys_fock[0],
                           self.emb_fock[1] - self.subsys_fock[1]]

        e_emb = (np.einsum('ij,ji', emb_pot[0], dmat[0]) +
                 np.einsum('ij,ji', emb_pot[1], dmat[1])).real

        return e_emb

    def get_env_elec_energy(self, env_method=None, fock=None, dmat=None,
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

        #Need to use embedding fock for freeze and thaw, and not for energies
        if env_method is None:
            env_method = self.env_method
        if dmat is None:
            dmat = copy(self.env_dmat)
        if fock is None:
            self.update_subsys_fock()
            fock = self.subsys_fock
        if env_hcore is None:
            env_hcore = self.env_hcore
        if proj_pot is None:
            proj_pot = self.proj_pot
        if emb_pot is None:
            if self.emb_fock[0] is None:
                emb_pot = [np.zeros_like(dmat[0]), np.zeros_like(dmat[1])]
            else:
                emb_pot = [self.emb_fock[0] - fock[0],
                           self.emb_fock[1] - fock[1]]

        e_emb = self.get_env_emb_e(emb_pot, dmat)
        e_proj = self.get_env_proj_e(proj_pot, dmat)
        if not (self.unrestricted or self.mol.spin != 0):
            dmat = dmat[0] + dmat[1]
        subsys_e = self.env_scf.energy_elec(dm=dmat)[0]
        return subsys_e + e_emb + e_proj

    def get_env_energy(self, mol=None, env_method=None, fock=None, dmat=None,
                       env_hcore=None, proj_pot=None, emb_pot=None):
        """Return the total subsystem energy

        Parameters
        ----------
        mol : Mole, optional
            Subsystem Mole object (default is None).
        """

        if env_method is None:
            env_method = self.env_method
        if dmat is None:
            dmat = copy(self.env_dmat)
        if fock is None:
            self.update_subsys_fock()
            fock = self.subsys_fock
        if env_hcore is None:
            env_hcore = self.env_hcore
        if proj_pot is None:
            proj_pot = self.proj_pot
        if emb_pot is None:
            if self.emb_fock[0] is None:
                emb_pot = [np.zeros_like(dmat[0]), np.zeros_like(dmat[1])]
            else:
                emb_pot = [self.emb_fock[0] - fock[0],
                           self.emb_fock[1] - fock[1]]
        if mol is None:
            mol = self.mol

        
        self.env_energy = self.get_env_elec_energy(env_method=env_method,
                                                    fock=fock, dmat=dmat,
                                                    env_hcore=env_hcore,
                                                    proj_pot=proj_pot,
                                                    emb_pot=emb_pot)
        self.env_energy += mol.energy_nuc()
        return self.env_energy

    def save_orbital_file(self, filename=None, scf_obj=None, mo_occ=None,
                          mo_coeff=None, mo_energy=None):
        """Saves a molden orbital file.

        Parameters
        ----------
        filename : str
        scf_obj : pyscf SCF object
        mo_occ : list
        mo_coeff : list
        mo_energy : list

        Returns
        -------
            bool
        """

        if filename is None:
            if self.filename is None:
                print("Cannot save orbitals because no filename")
                return False
            filename = self.filename
        if scf_obj is None:
            scf_obj = self.env_scf
        if mo_occ is None:
            mo_occ = self.env_mo_occ
        if mo_coeff is None:
            mo_coeff = self.env_mo_coeff
        if mo_energy is None:
            mo_energy = self.env_mo_energy
        print(f'Writing Subsystem {self.chkfile_index} Orbitals'.center(80))
        if not self.unrestricted:
            molden_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv.molden'
            with open(molden_fn, 'w') as fin:
                tools.molden.header(scf_obj.mol, fin)
                tools.molden.orbital_coeff(self.mol, fin, mo_coeff[0],
                                           ene=mo_energy[0],
                                           occ=(mo_occ[0] + mo_occ[1]))
        else:
            molden_fn_a = (os.path.splitext(filename)[0] + '_' +
                           self.chkfile_index + '_subenv_alpha.molden')
            molden_fn_b = (os.path.splitext(filename)[0] + '_' +
                           self.chkfile_index + '_subenv_beta.molden')
            with open(molden_fn_a, 'w') as fin:
                tools.molden.header(scf_obj.mol, fin)
                tools.molden.orbital_coeff(self.mol, fin, mo_coeff[0],
                                           spin='Alpha', ene=mo_energy[0],
                                           occ=mo_occ[0])
            with open(molden_fn_b, 'w') as fin:
                tools.molden.header(scf_obj.mol, fin)
                tools.molden.orbital_coeff(self.mol, fin, mo_coeff[1],
                                           spin='Beta', ene=mo_energy[1],
                                           occ=mo_occ[1])
        return True

    def save_density_file(self, filename=None):
        """Save the electron density as a molden file.

        Parameters
        ----------
        filename : str, optional
            The filename to save the density as.
            (default is None)
        """

        if filename is None:
            if self.filename is None:
                print("Cannot save density because no filename")
                return False
            filename = self.filename
        density = self.get_dmat()
        print(f'Writing Subsystem {self.chkfile_index} Density'.center(80))
        if self.mol.spin != 0 or self.unrestricted:
            cubegen_fn = (os.path.splitext(filename)[0] + '_' +
                          self.chkfile_index + '_subenv_alpha.cube')
            tools.cubegen.density(self.mol, cubegen_fn, density[0])
            cubegen_fn = (os.path.splitext(filename)[0] + '_' +
                          self.chkfile_index + '_subenv_beta.cube')
            tools.cubegen.density(self.mol, cubegen_fn, density[1])
        else:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv.cube'
            tools.cubegen.density(self.mol, cubegen_fn, density)
        return True

    def save_spin_density_file(self, filename=None):
        """Saves a molden file of the spin density

        Parameters
        ----------
        filename : str, optional
            The filename to save the spin density as.
            (default is None)
        """

        if filename is None:
            if self.filename is None:
                print("Cannot save density because no filename")
                return False
            filename = self.filename
        density = self.get_dmat()
        if self.mol.spin != 0 or self.unrestricted:
            print(f'Writing Subsystem {self.chkfile_index} Spin Density'.center(80))
            cubegen_fn = (os.path.splitext(filename)[0] + '_' +
                          self.chkfile_index + '_subenv_spinden.cube')
            tools.cubegen.density(self.mol, cubegen_fn, np.subtract(density[0], density[1]))
        else:
            print('Cannot write spin density for a closed shell system.'.center(80))
            return False
        return True

    def save_chkfile(self, filename=None):
        """Saves a checkpoint file of the electron density.

        Parameters
        ----------
        filename : str
            filename to save the checkpoint file.
            (default is None)
        """

        if filename is None:
            if self.filename is None:
                print("chkfile not saved because no filename set.")
                return False
            filename = os.path.splitext(self.filename)[0] + '.hdf5'
        assert(self.chkfile_index is not None), 'Need to set chkfile_index'

        chk_index = self.chkfile_index
        # check if file exists.
        if os.path.isfile(filename):
            try:
                with h5py.File(filename, 'r+') as fin:
                    subsys_coeff = fin[f'subsystem:{chk_index}/mo_coeff']
                    subsys_coeff[...] = self.env_mo_coeff
                    subsys_occ = fin[f'subsystem:{chk_index}/mo_occ']
                    subsys_occ[...] = self.env_mo_occ
                    subsys_energy = fin[f'subsystem:{chk_index}/mo_energy']
                    subsys_energy[...] = self.env_mo_energy
            except TypeError:
                print("Overwriting existing chkfile".center(80))
                with h5py.File(filename, 'w') as fout:
                    sub_sys_data = fout.create_group(f'subsystem:{chk_index}')
                    sub_sys_data.create_dataset('mo_coeff', data=self.env_mo_coeff)
                    sub_sys_data.create_dataset('mo_occ', data=self.env_mo_occ)
                    sub_sys_data.create_dataset('mo_energy', data=self.env_mo_energy)
            except KeyError:
                print("Missing subsystem data in chkfile".center(80))
                with h5py.File(filename, 'a') as fout:
                    sub_sys_data = fout.create_group(f'subsystem:{chk_index}')
                    sub_sys_data.create_dataset('mo_coeff', data=self.env_mo_coeff)
                    sub_sys_data.create_dataset('mo_occ', data=self.env_mo_occ)
                    sub_sys_data.create_dataset('mo_energy', data=self.env_mo_energy)
        else:
            with h5py.File(filename, 'a') as fout:
                sub_sys_data = fout.create_group(f'subsystem:{chk_index}')
                sub_sys_data.create_dataset('mo_coeff', data=self.env_mo_coeff)
                sub_sys_data.create_dataset('mo_occ', data=self.env_mo_occ)
                sub_sys_data.create_dataset('mo_energy', data=self.env_mo_energy)
        return True

    def read_chkfile(self, filename=None):
        """Reads the embedding checkpoint file and saves the density.

        Parameters
        ----------
        filename : str
            Name of the checkpoint file.
            (default is None)

        Returns
        -------
            bool
        """

        if filename is None:
            if self.filename is None:
                return False
            filename = os.path.splitext(self.filename)[0] + '.hdf5'
        assert(self.chkfile_index is not None), 'Need to set chkfile_index'

        filename = os.path.splitext(filename)[0] + '.hdf5'
        chk_index = self.chkfile_index

        if os.path.isfile(filename):
            try:
                with h5py.File(filename, 'r') as fin:
                    subsys_coeff = fin[f'subsystem:{chk_index}/mo_coeff']
                    self.env_mo_coeff = subsys_coeff[:]
                    subsys_occ = fin[f'subsystem:{chk_index}/mo_occ']
                    self.env_mo_occ = subsys_occ[:]
                    subsys_energy = fin[f'subsystem:{chk_index}/mo_energy']
                    self.env_mo_energy = subsys_energy[:]
                return True
            except TypeError:
                print("chkfile improperly formatted".center(80))
                return False
            except KeyError:
                print("Missing subsystem data in chkfile".center(80))
                return False
        else:
            print("chkfile NOT found".center(80))
            return False


    def diagonalize(self):
        """Diagonalizes the subsystem fock matrix and returns updated density."""

        for i in range(self.env_subcycles):
            if i > 0: #This doesn't work as intended right now.
                self.update_subsys_fock()

            if self.unrestricted:
                self.__do_unrestricted_diag()

            elif self.mol.spin != 0:
                self.__do_restricted_os_diag()

            else:
                self.__do_restricted_diag()

            e_sorted = [np.sort(self.env_mo_energy[0]), np.sort(self.env_mo_energy[1])]
            self.__set_occupation()
            self.__set_fermi()

            self.env_dmat[0] = np.dot((self.env_mo_coeff[0] * self.env_mo_occ[0]),
                                      self.env_mo_coeff[0].transpose().conjugate())
            self.env_dmat[1] = np.dot((self.env_mo_coeff[1] * self.env_mo_occ[1]),
                                      self.env_mo_coeff[1].transpose().conjugate())

            self.save_chkfile()
            return self.env_dmat

    def __do_unrestricted_diag(self):
        """Performs diagonalization on the unrestricted env object."""
        emb_proj_fock = np.array([None, None])
        if self.emb_proj_fock[0] is None:
            fock = self.emb_fock
            if fock[0] is None:
                fock = self.subsys_fock
            emb_proj_fock[0] = fock[0] + self.proj_pot[0]
            emb_proj_fock[1] = fock[1] + self.proj_pot[1]
            if self.diis:
                if self.diis_num == 1:
                    emb_proj_fock = self.diis.update(emb_proj_fock)
                if self.diis_num == 2:
                    dmat = self.get_dmat()
                    ovlp = self.env_scf.get_ovlp()
                    emb_proj_fock = self.diis.update(ovlp, dmat, emb_proj_fock)
        else:
            emb_proj_fock = self.emb_proj_fock
        energy, coeff = self.env_scf.eig(emb_proj_fock, self.env_scf.get_ovlp())
        self.env_mo_energy = [energy[0], energy[1]]
        self.env_mo_coeff = [coeff[0], coeff[1]]

    def __do_restricted_os_diag(self):
        """Performs diagonalization on the restricted open shell env object."""
        emb_proj_fock = np.array([None, None])
        if self.emb_proj_fock[0] is None:
            fock = self.emb_fock
            if fock[0] is None:
                fock = self.subsys_fock

            emb_proj_fock = fock[0] + self.proj_pot[0]
            emb_proj_fock += fock[1] + self.proj_pot[1]
            emb_proj_fock /= 2.
            if self.diis:
                if self.diis_num == 1:
                    emb_proj_fock = self.diis.update(emb_proj_fock)
                if self.diis_num == 2:
                    dmat = self.get_dmat()
                    dmat_tot = dmat[0] + dmat[1]
                    ovlp = self.env_scf.get_ovlp()
                    emb_proj_fock = self.diis.update(ovlp, dmat_tot, emb_proj_fock)
        else:
            emb_proj_fock = (self.emb_proj_fock[0] + self.emb_proj_fock[1]) / 2.

        energy, coeff = self.env_scf.eig(emb_proj_fock, self.env_scf.get_ovlp())

        self.env_mo_energy = [energy, energy]
        self.env_mo_coeff = [coeff, coeff]

    def __do_restricted_diag(self):
        """Performs diagonalization on the restricted env object."""
        emb_proj_fock = np.array([None, None])
        if self.emb_proj_fock[0] is None:
            fock = self.emb_fock
            if fock[0] is None:
                fock = self.subsys_fock

            emb_proj_fock = fock[0] + self.proj_pot[0]
            emb_proj_fock += fock[1] + self.proj_pot[1]
            emb_proj_fock /= 2.

            if self.diis:
                if self.diis_num == 1:
                    emb_proj_fock = self.diis.update(emb_proj_fock)
                if self.diis_num == 2:
                    dmat = self.get_dmat()
                    ovlp = self.env_scf.get_ovlp()
                    emb_proj_fock = self.diis.update(ovlp, dmat, emb_proj_fock)
        else:
            emb_proj_fock = (self.emb_proj_fock[0] + self.emb_proj_fock[1]) / 2.

        energy, coeff = self.env_scf.eig(emb_proj_fock, self.env_scf.get_ovlp())
        self.env_mo_energy = [energy, energy]
        self.env_mo_coeff = [coeff, coeff]

    def relax_sub_dmat(self, damp_param=None):
        """Relaxes the given subsystem density using the updated fock.
        """

        if damp_param is None:
            damp_param = self.env_damp

        sub_old_dm = self.get_dmat().copy()
        self.diagonalize()

        new_dm = [None, None]
        if self.unrestricted or self.mol.spin != 0:
            ddm = sp.linalg.norm(self.get_dmat()[0] - sub_old_dm[0])
            ddm += sp.linalg.norm(self.get_dmat()[1] - sub_old_dm[1])
            damp = [damp_param, damp_param]
            if damp[0] < 0:
                #GeT ODA DAMPING parameters.
                pass
            new_dm[0] = ((1 - damp[0]) * self.get_dmat()[0] + (damp[0] * sub_old_dm[0]))
            new_dm[1] = ((1 - damp[1]) * self.get_dmat()[1] + (damp[1] * sub_old_dm[1]))
            self.env_dmat = new_dm
        else:
            damp = damp_param
            ddm = sp.linalg.norm(self.get_dmat() - sub_old_dm)
            if damp < 0:
                #GET ODA DAMPING PARAMETER.
                pass
            new_dm = ((1. - damp) * self.get_dmat() + (damp * sub_old_dm))
            self.env_dmat = [new_dm/2., new_dm/2.]
        return ddm

    def __set_fermi(self):
        """Sets the fermi level for the subsystem.

        Parameters
        ----------
        e_sorted : list
            A list of the orbital energies sorted lowest to highest.
        """
        self.fermi = [0., 0.]
        nocc_orbs = [self.mol.nelec[0], self.mol.nelec[1]]
        alpha_occ = copy(self.env_mo_occ[0])

        if not np.all(alpha_occ):
            occ_energy_m = np.ma.masked_where(alpha_occ==0, self.env_mo_energy[0])
            alpha_homo = np.max(np.ma.compressed(occ_energy_m))
            unocc_energy_m = np.ma.masked_where(alpha_occ>0, self.env_mo_energy[0])
            alpha_lumo = np.min(np.ma.compressed(unocc_energy_m))
            self.fermi[0] = (alpha_homo + alpha_lumo) / 2.

        beta_occ = copy(self.env_mo_occ[1])
        if not np.all(beta_occ):
            occ_energy_m = np.ma.masked_where(beta_occ==0, self.env_mo_energy[1])
            beta_homo = np.max(np.ma.compressed(occ_energy_m))
            unocc_energy_m = np.ma.masked_where(beta_occ>0, self.env_mo_energy[1])
            beta_lumo = np.min(np.ma.compressed(unocc_energy_m))
            self.fermi[1] = (beta_homo + beta_lumo) / 2.

    def __set_occupation(self):
        """Sets the orbital occupation numbers.
        """
        #Smear sigma may not be right for single elctron
        self.env_mo_occ = [np.zeros_like(self.env_mo_energy[0]),
                           np.zeros_like(self.env_mo_energy[1])]
                           
        #if self.env_smearsigma > 0.:
        #    self.env_mo_occ[0] = ((self.env_mo_energy[0]
        #                           - self.fermi[0]) / self.env_smearsigma)
        #    occ_orb = np.where(self.env_mo_occ[0] < 1000)
        #    vir_orb = np.where(self.env_mo_occ[0] >= 1000)
        #    self.env_mo_occ[0][occ_orb] = 1. / (np.exp(self.env_mo_occ[0][occ_orb]) + 1.)
        #    self.env_mo_occ[0][vir_orb] = 0.

        #    self.env_mo_occ[1] = (self.env_mo_energy[1] - self.fermi[1]) / self.env_smearsigma
        #    occ_orb = np.where(self.env_mo_occ[1] < 1000)
        #    vir_orb = np.where(self.env_mo_occ[1] >= 1000)
        #    self.env_mo_occ[1][occ_orb] = 1. / (np.exp(self.env_mo_occ[1][occ_orb]) + 1.)
        #    self.env_mo_occ[1][vir_orb] = 0.

        if self.unrestricted:
            mo_energy = self.env_mo_energy
            mo_coeff = self.env_mo_coeff
            self.env_mo_occ = self.env_scf.get_occ(mo_energy, mo_coeff)
        elif self.mol.spin != 0:
            mo_energy = self.env_mo_energy[0]
            mo_coeff = self.env_mo_coeff[0]
            mo_occ = self.env_scf.get_occ(mo_energy, mo_coeff)
            alpha_occ = (mo_occ > 0.).astype(int)
            beta_occ = (mo_occ > 1.).astype(int)
            self.env_mo_occ = [alpha_occ, beta_occ]
        else:
            mo_energy = self.env_mo_energy[0]
            mo_coeff = self.env_mo_coeff[0]
            mo_occ = self.env_scf.get_occ(mo_energy, mo_coeff)
            self.env_mo_occ = [mo_occ/2., mo_occ/2.]

class ClusterHLSubSystem(ClusterEnvSubSystem):
    """
    Extends ClusterEnvSubSystem to calculate higher level methods.

    Attributes
    ----------
    hl_method : str
        Which method to use for high level calculation.
    hl_init_guess : str
        Specifies initial dmat guess for hl method.
    hl_sr_method : str
        Specifies which single reference method to use for high level
        calculations.
    hl_spin : int
        The spin of the high level calculation, different from the
        lower level calculation.
    hl_conv : float
        The density convergence criteria of the high level calculation.
    hl_grad : float
        The convergence of the electronic gradient of the
        high level calculation.
    hl_cycles : int
        The number of scf cycles for the high level method.
    hl_damp : float
        The damping parameter for the high level method.
    hl_shift : float
        The orbital shift parameter for the high level method.
    hl_ext : str
        The name of an external code to calculate the high level energy.
    hl_unrestricted : bool
        Whether the high level calculation is unrestricted.
    hl_compress_approx : bool
        Whether to use the compression approximation for the high levem method.
    hl_density_fitting : bool
        Whether to use density fitting for high level calculation.
    hl_save_orbs : bool
        Whether to save the high level orbitals.
    hl_save_density : bool
        Whether to save the high level electron density.
    hl_save_spin_density : bool
        Whether to save high level electron spin density.
    hl_mo_coeff : array
        Array of high level molecular orbital coeffecients
    hl_mo_occ : array
        Array of high level molecular orbital occupation
    hl_mo_energy : array
        Array of high level molecular orbital energies
    hl_dmat : array
        Array of high level molecular electron density
    hl_sr_scf : SCF Object
        PySCF SCF object for the single reference part of high level calculation.
    hl_energy : float
        The energy of the high level method.

    Methods
    -------
    __set_hl_method_settings(hl_dict)
        Set additional object attributes specific to the high level method.
    get_hl_proj_energy(dmat, proj_pot)
        Gets the projection operator energy based on the high level density.
    get_hl_in_env_energy()
        Gets the energy of the high level calculation in the potential of the
        environment.
    __get_ext_energy()
        Uses the specified external code to calculate the high level energies.
    __do_sr_scf()
        Performs the inital single reference calculation for the high level
        method.
    __gen_hf_scf()
        Initializes a hartree-fock single reference calculation as the initial
        guess for high level calculations.
    __gen_dft_scf()
        Initializes a DFT single reference calculation as the initial guess
        for high level calculations.
    __do_cc()
        Performs a coupled cluster calculation as the high level method.
    __do_mp()
        Performs an MP calculation as the high level method.
    __do_casscf()
        Performs a CASSCF calculation as the high level method.
    __do_fci()
        Performs an FCI calculation as the high level method.
    __do_dmrg()
        Performs a DMRG calculation as the high level method.
    __do_shci()
        Performs an SHCI calculation as the high level method.
    __save_fcidump()
        Saves a formatted fcidump file at the specified location.
    __save_hl_density_file()
        Saves the high level electron density to a file.
    __save_hl_orbital_file()
        Saves the high level orbitals to a file.
    """

    def __init__(self, mol, env_method, hl_method, hl_order=1, hl_init_guess=None,
                 hl_sr_method=None, hl_excited=None, hl_spin=None, hl_conv=None, hl_grad=None,
                 hl_cycles=None, hl_damp=0., hl_shift=0., hl_ext=None,
                 hl_unrestricted=False, hl_compress_approx=False,
                 hl_density_fitting=False, hl_save_orbs=False,
                 hl_save_density=False, hl_save_spin_density=False,
                 hl_dict=None, hl_excited_dict=None, **kwargs):
        """
        Parameters
        ----------
        mol : Mole
            The pyscf Mole object specifitying geometry and basis.
        env_method : str
            Defines the method for use in env calculations.
        hl_method : str
            Defines the high level method for the calculations.
        hl_order : int, optional
            Specifies the subsystem within the context of the full system.
            (default is 1)
        hl_init_guess : str, optional
            Specifies initial dmat guess for hl method.
            (default is None)
        hl_sr_method : str, optional
            Specifies which single reference method to use for high level
            calculations.
            (default is None)
        hl_spin : int, optional
            The spin of the high level calculation, different from the
            lower level calculation.
            (default is None)
        hl_conv : float, optional
            The density convergence criteria of the high level calculation.
            (default is None)
        hl_grad : float, optional
            The convergence of the electronic gradient of the
            high level calculation.
            (default is None)
        hl_cycles : int, optional
            The number of scf cycles for the high level method.
            (default is None)
        hl_damp : float, optional
            The damping parameter for the high level method.
            (default is 0.)
        hl_shift : float, optional
            The orbital shift parameter for the high level method.
            (default is 0.)
        hl_ext : str, optional
            The name of an external code to calculate the high level energy.
            (default is None)
        hl_unrestricted : bool, optional
            Whether the high level calculation is unrestricted.
            (default is False)
        hl_compress_approx : bool, optional
            Whether to use the compression approximation for the high levem method.
            (default is False)
        hl_density_fitting : bool, optional
            Whether to use density fitting for high level calculation.
            (default is False)
        hl_save_orbs : bool, optional
            Whether to save the high level orbitals.
            (default is False)
        hl_save_density : bool, optional
            Whether to save the high level electron density.
            (default is False)
        hl_save_spin_density : bool, optional
            Whether to save high level electron spin density.
            (default is False)
        hl_dict : dict, optional
            A dictionary containing method specific keywords.
            (default is None)
        """

        super().__init__(mol, env_method, **kwargs)

        self.hl_method = hl_method
        self.hl_init_guess = hl_init_guess
        self.hl_sr_method = hl_sr_method
        self.hl_excited = hl_excited
        if hl_spin:
            self.hl_spin = hl_spin
        else:
            self.hl_spin = self.mol.spin
        self.hl_conv = hl_conv
        self.hl_grad = hl_grad
        self.hl_cycles = hl_cycles
        self.hl_damp = hl_damp
        self.hl_shift = hl_shift

        self.hl_ext = hl_ext
        self.hl_unrestricted = hl_unrestricted
        self.hl_compress_approx = hl_compress_approx
        self.hl_density_fitting = hl_density_fitting
        self.hl_save_orbs = hl_save_orbs
        self.hl_save_density = hl_save_density
        self.hl_save_spin_density = hl_save_spin_density
        self.__set_hl_method_settings(hl_dict)
        self.__set_hl_excited_settings(hl_excited_dict)

        self.hl_mo_coeff = None
        self.hl_mo_occ = None
        self.hl_mo_energy = None
        self.hl_dmat = None
        self.hl_sr_scf = None
        self.hl_energy = None

    def __set_hl_method_settings(self, hl_dict):
        """Sets the object parameters based on the hl settings

        Parameters
        ----------
        hl_dict : dict
            A dictionary containing the hl specific settings.
        """

        if hl_dict is None:
            hl_dict = {}
        self.hl_dict = hl_dict

        if 'cc' in self.hl_method:
            self.cc_loc_orbs = hl_dict.get("loc_orbs")
            self.cc_init_guess = hl_dict.get("cc_init_guess")
            self.cc_froz_core_orbs = hl_dict.get("froz_core_orbs")
        if 'cas' in self.hl_method:
            self.cas_loc_orbs = hl_dict.get("loc_orbs")
            self.cas_init_guess = hl_dict.get("cas_init_guess")
            self.cas_active_orbs = hl_dict.get("active_orbs")
            self.cas_avas = hl_dict.get("avas")
        if 'dmrg' in self.hl_method:
            self.dmrg_max_m = hl_dict.get("maxM")
            self.dmrg_num_thrds = hl_dict.get("num_thirds")
        if 'shciscf' in self.hl_method:
            self.shci_mpi_prefix = hl_dict.get("mpi_prefix")
            self.shci_sweep_iter = hl_dict.get("sweep_iter")
            self.shci_sweep_epsilon = hl_dict.get("sweep_epsilon")
            self.shci_no_stochastic = hl_dict.get("no_stochastic")
            self.shci_npt_iter = hl_dict.get("NPTiter")
            self.shci_no_rdm = hl_dict.get("NoRDM")

    def __set_hl_excited_settings(self, hl_excited_dict):
        """Sets the object parameters based on the excited settings

        Parameters
        ----------
        hl_excited_dict : dict
            A dictionary containing the hl excited state specific settings.
        """

        if hl_excited_dict is None:
            hl_excited_dict = {}
        self.hl_excited_dict = hl_excited_dict
        self.hl_excited_nroots = hl_excited_dict.get('nroots')
        self.hl_excited_conv = hl_excited_dict.get('conv')
        self.hl_excited_cycles = hl_excited_dict.get('cycles')
        self.hl_excited_type = hl_excited_dict.get('eom_type')
        self.hl_excited_koopmans = hl_excited_dict.get('koopmans')
        self.hl_excited_tda = hl_excited_dict.get('tda')
        self.hl_excited_analyze = hl_excited_dict.get('analyze')
        self.hl_excited_triple = hl_excited_dict.get('Ta_star')

        # set default number of excited states to 3
        if self.hl_excited_nroots is None: self.hl_excited_nroots=3
        if self.hl_excited_type is None: self.hl_excited_type = 'ee'
        if self.hl_excited_type is None: self.hl_excited_type = True


    def get_hl_proj_energy(self, dmat=None, proj_pot=None):
        """Return the projection energy

        Parameters
        ----------
        dmat : numpy.float64, optional
            The hl subsystem density matrix (default is None).
        proj_pot : numpy.float64, optional
            The projection potential (default is None).
        """

        if dmat is None:
            dmat = self.hl_dmat
        if proj_pot is None:
            proj_pot = self.proj_pot
        return np.trace(dmat, proj_pot)


    def get_hl_in_env_energy(self):
        """Returns the embedded high level method energy.

        Returns
        -------
        float
            The energy of the embedded high level calculation.
        """

        if self.emb_fock[0] is None:
            self.emb_pot = [np.zeros_like(self.env_dmat[0]), np.zeros_like(self.env_dmat[1])]
        else:
            self.update_subsys_fock()
            fock = self.subsys_fock
            self.emb_pot = (self.emb_fock[0] - fock[0],
                            self.emb_fock[1] - fock[1])

        #Determine which method to use for the single reference orbitals.
        hf_aliases = ['hf', 'uhf', 'rhf', 'rohf']
        cc_aliases = ['ccsd', 'ccsd(t)', 'uccsd', 'uccsd(t)']
        mp_aliases = ['mp2']
        cas_regex = re.compile(r'cas(pt2)?(\[\d*,\d*\])?')
        dmrg_regex = re.compile(r'dmrg\[.*\].*')
        shci_regex = re.compile(r'shci(scf)?\[.*\].*')
        fci_aliases = ['fci']
        fcidump_aliases = ['fcidump']
        known_methods = hf_aliases + cc_aliases + mp_aliases + fci_aliases + fcidump_aliases
        self.mol.verbose = 4

        if (self.hl_sr_method is None and
                self.hl_method not in known_methods and
                not re.match(cas_regex, self.hl_method)):
            self.hl_sr_method = self.hl_method

        if self.hl_ext is not None:
            self.__get_ext_energy()
            return self.hl_energy

        self.__do_sr_scf()

        if self.hl_method in cc_aliases:
            self.__do_cc()

        elif self.hl_method in mp_aliases:
            self.__do_mp()

        elif re.match(cas_regex, self.hl_method):
            self.__do_casscf()

        elif self.hl_method in fci_aliases:
            self.__do_fci()

        elif re.match(dmrg_regex, self.hl_method):
            self.__do_dmrg()

        elif re.match(shci_regex, self.hl_method):
            self.__do_shci()

        elif self.hl_method in fcidump_aliases:
            self.__save_fcidump()

        return self.hl_energy

    def calc_den_grad(self):
        """Calculates the gradient of the electron density wrt nuc position."""

        self.emb_hess = None
        if self.unrestricted:
            if self.env_method == 'hf':
                self.emb_hess = hessian.uhf.Hessian(self.env_scf)
            else:
                self.emb_hess = hessian.uks.Hessian(self.env_scf)
        elif self.mol.spin == 0:
            if self.env_method == 'hf':
                self.emb_hess = hessian.rhf.Hessian(self.env_scf)
            else:
                self.emb_hess = hessian.rks.Hessian(self.env_scf)
        else:
            print ("NO ROHF Den Grad")

        if not (self.unrestricted or self.mol.spin != 0):
            env_mo_en = self.env_mo_energy[0]
            env_mo_coeff = self.env_mo_coeff[0]
            env_mo_occ = self.env_mo_occ[0] * 2.
        else:
            env_mo_en = self.env_mo_energy
            env_mo_coeff = self.env_mo_coeff
            env_mo_occ = self.env_mo_occ

        #Modify core hamiltonian
        emb_h1ao = np.zeros_like(self.atom_hcore_grad)
        self.emb_dm_grad = np.zeros_like(self.atom_hcore_grad)
        atmlst = range(self.mol.natm)
        for atm in atmlst:
            emb_h1ao[atm] += self.atom_hcore_grad[atm] + self.atom_emb_pot_grad[atm] + self.atom_proj_grad[atm]

        #Get gradient of MOs
        emb_mo1, emb_mo_e1 = self.emb_hess.solve_mo1(env_mo_en, env_mo_coeff, env_mo_occ, emb_h1ao)
        #Calcualate density grad
        env_mocc = env_mo_coeff[:,env_mo_occ>0]
        for atm in atmlst:
            self.emb_dm_grad[atm] = np.einsum('ypi,qi->ypq', emb_mo1[atm], env_mocc)

        return (self.emb_dm_grad)
        


    def calc_nuc_grad(self):
        """Calculates the nuclear gradient of the embedded subsystems."""

        #currently for testing, separated out the weighted density matrix. In the final form, the weighted density matrix can use the mo energies and only do it once to get the full subsystem e.
        
        #ENV
        #Isolated subsystem
        if not (self.unrestricted or self.mol.spin != 0):
            env_mo_en = self.env_mo_energy[0]
            env_mo_coeff = self.env_mo_coeff[0]
            env_mo_occ = self.env_mo_occ[0] * 2.
        else:
            env_mo_en = self.env_mo_energy
            env_mo_coeff = self.env_mo_coeff
            env_mo_occ = self.env_mo_occ

        env_sub_grad_obj = self.env_scf.nuc_grad_method()
        env_sub_de = env_sub_grad_obj.grad_elec(mo_energy=env_mo_en, mo_coeff=env_mo_coeff, mo_occ=env_mo_occ)
        print ('env_sub_de')
        print (env_sub_de)
        #Embedded potential gradient

        self.atom_emb_pot_grad = np.zeros_like(self.atom_full_hcore_grad)
        self.atom_proj_grad = np.zeros_like(self.atom_full_hcore_grad)
        self.atom_hcore_grad = np.zeros_like(self.atom_full_hcore_grad)
        atmlst = range(self.mol.natm)
        aoslices = self.mol.aoslice_by_atom()
        env_dm = self.get_dmat()
        sub_hcore_deriv = env_sub_grad_obj.hcore_generator(self.mol)
        num_rank = self.mol.nao_nr()
        sub_s1_grad = env_sub_grad_obj.get_ovlp(self.mol)
        env_emb_pot_de = np.zeros_like(env_sub_de)
        env_proj_de = np.zeros_like(env_sub_de)

        for atm in atmlst:
            p0, p1 = aoslices[atm,2:]
            atom_sub_hcore_grad = sub_hcore_deriv(atm)
            emb_hcore = self.atom_full_hcore_grad[atm] - atom_sub_hcore_grad
            self.atom_hcore_grad[atm] = atom_sub_hcore_grad

            #emb_hcore = self.atom_full_hcore_grad[atm] - atom_sub_hcore_grad
            #print ("emb_hcore_grad")
            env_emb_pot_de[atm] += np.einsum('xij,ij->x', emb_hcore, env_dm)
            print (env_emb_pot_de[atm])
            env_emb_pot_de[atm] += (np.einsum('xij,ij->x', self.atom_emb_vhf_grad[0], env_dm)) * 4.
            print (env_emb_pot_de[atm])
            #Need to do nuclear-electron attraction I think.

            #print ('emb_vhf_grad')
            #print (np.einsum('xij,ij->x', self.atom_emb_vhf_grad[atm][:,p0:p1], env_dm[p0:p1]))
            #env_emb_pot_de[atm] += np.einsum('xij,ij->x', self.atom_emb_vhf_grad[atm][:,p0:p1], env_dm[p0:p1] * -2.)
            #self.atom_emb_pot_grad[atm] = self.atom_full_hcore_grad[atm] - atom_sub_hcore_grad + (self.atom_emb_vhf_grad[atm] * 2.)
            #print ("VHF EMB GRAD")
            #print (self.atom_emb_vhf_grad[atm])
            #env_emb_pot_de[atm] += np.einsum('xij,ij->x', self.atom_emb_pot_grad[atm], env_dm)
            #env_proj_de[atm] += np.einsum('xij,ij->x', self.atom_proj_grad[atm], env_dm)


        #print ("Calculating subsystem electron density gradient")
        #self.calc_den_grad()
        ##Test the density gradient.
        #sub_hcore = self.env_scf.get_hcore()
        #sub_hcore_grad = []
        #for atm in atmlst:
        #    gradh = np.einsum('xij,ij->x', sub_hcore_deriv(atm), env_dm)
        #    sub_hcore_grad.append(gradh)
        #    graddmh = np.einsum('xij,ij->x', self.emb_dm_grad[atm], sub_hcore)
        #    sub_hcore_grad[atm] += graddmh

        #print ("DMAT")
        #print (np.trace(env_dm))
        #print ("DMAT DERIV")
        #print (np.trace(self.emb_dm_grad[0][2]))
        #print ("HCORE EN")
        #print (np.einsum('ij,ji->', sub_hcore, env_dm))
        #print ("Hcore Grad")
        #print (sub_hcore_grad)

        env_grad = env_sub_de + env_emb_pot_de + env_proj_de
        print ("ENV GRAD")
        print (env_grad)

        #HL
        hf_aliases = ['hf', 'uhf', 'rhf', 'rohf']
        cc_aliases = ['ccsd', 'ccsd(t)', 'uccsd', 'uccsd(t)']
        mp_aliases = ['mp2']
        #Isolated subsystem
        if self.hl_method in hf_aliases:
            hl_sub_grad_obj = self.hl_sr_scf.nuc_grad_method()
            hl_mo_e = self.hl_sr_scf.mo_energy
            hl_mo_coeff = self.hl_sr_scf.mo_coeff
            hl_mo_occ = self.hl_sr_scf.mo_occ
            hl_sub_grad = hl_sub_grad_obj.grad_elec(mo_energy=hl_mo_e, mo_coeff=hl_mo_coeff, mo_occ=hl_mo_occ)
            hl_rdm1e = hl_sub_grad_obj.make_rdm1e(hl_mo_e, hl_mo_coeff, hl_mo_occ)
            hl_dm = self.hl_sr_scf.make_rdm1()

        #print (hl_sub_grad)
        hl_proj_de = np.zeros((len(atmlst),3))
        hl_emb_pot_de = np.zeros((len(atmlst),3))
        hl_sub_vhf_grad = hl_sub_grad_obj.get_veff(self.mol, hl_dm)
        hl_sub_hcore_deriv = hl_sub_grad_obj.hcore_generator(self.mol)
        for atm in atmlst:
            p0, p1 = aoslices[atm,2:]
            hl_proj_de[atm] += np.einsum('xij,ij->x', self.atom_proj_grad[atm], hl_dm)
            hl_emb_pot_de[atm] += np.einsum('xij,ij->x', self.atom_emb_pot_grad[atm], hl_dm)

        print ("HL PROJ")
        print (hl_proj_de)
        print ("HL EMB POT")
        print (hl_emb_pot_de)
        print ("HL_SUB_GRAD")
        print (hl_sub_grad)
        print ("HL EMB GRAD")
        #print (hl_proj_de + hl_emb_pot_de + hl_sub_grad)
        #print (hl_proj_de + hl_sub_grad)
        hl_grad = hl_proj_de + hl_emb_pot_de + hl_sub_grad
        #hl_grad = hl_proj_de + hl_sub_grad
        #print (hl_proj_de + hl_emb_hcore_de + hl_emb_vhf_de + hl_sub_grad)
        #hl_grad = hl_proj_de + hl_emb_hcore_de + hl_emb_vhf_de + hl_sub_grad

        if self.hl_method in cc_aliases:
            pass
       
        print ("TOTAL GRAD")
        print (hl_grad - env_grad)
        return hl_grad - env_grad

    def __get_ext_energy(self):
        """Uses an external method to calculate high level energy.
        """

        print("use external method for hl calculation")
        hcore = self.env_scf.get_hcore()
        emb_proj_pot = [self.emb_pot[0] + self.proj_pot[0], self.emb_pot[1] + self.proj_pot[1]]
        ext_factory = ExtFactory()
        name_no_path = os.path.split(self.filename)[-1]
        name_no_ext = os.path.splitext(name_no_path)[0]
        file_path = os.path.split(self.filename)[0]
        scr_path = self.scr_dir
        ext_mol = gto.copy(self.mol)
        ext_mol.spin = self.hl_spin
        ext_mol.build()
        ext_obj = ext_factory.get_ext_obj(self.hl_ext, ext_mol,
                                          self.hl_method, emb_proj_pot,
                                          core_ham=hcore, filename=name_no_ext,
                                          work_dir=file_path, scr_dir=scr_path,
                                          nproc=self.nproc, pmem=self.pmem,
                                          save_orbs=None, save_density=False,
                                          hl_dict=self.hl_dict)
        energy = ext_obj.get_energy()
        self.hl_energy = energy[0]

    def __do_sr_scf(self):
        """Initializes and runs the single reference hf object
        """

        hf_aliases = ['hf', 'uhf', 'rhf', 'rohf']
        if (self.hl_sr_method is None or self.hl_sr_method in hf_aliases):
            self.__gen_hf_scf()
        else:
            self.__gen_dft_scf()

        if self.hl_init_guess == 'ft':
            dmat = self.get_dmat()
        elif self.hl_init_guess is not None:
            dmat = self.hl_sr_scf.get_init_guess(key=self.hl_init_guess)
        else:
            dmat = self.hl_sr_scf.get_init_guess()

        if self.hl_conv is not None:
            self.hl_sr_scf.conv_tol = self.hl_conv
        if self.hl_grad is not None:
            self.hl_sr_scf.conv_tol_grad = self.hl_grad
        if self.hl_cycles is not None:
            self.hl_sr_scf.max_cycle = self.hl_cycles
        self.hl_sr_scf.level_shift = self.hl_shift
        self.hl_sr_scf.damp = self.hl_damp

        self.hl_energy = self.hl_sr_scf.scf(dm0=dmat)

        #DO TDDFT or TDHF here.
        if self.hl_excited and 'cc' not in self.hl_method:
            from pyscf import tdscf
            if self.hl_excited_tda: 
                hl_sr_tdscf = tdscf.TDA(self.hl_sr_scf)
                print("TDA calculations:") 
            else: 
                try:
                    hl_sr_tdscf = tdscf.TDHF(self.hl_sr_scf)
                    print("TDHF calculations:") 
                except:
                    hl_sr_tdscf = tdscf.TDDFT(self.hl_sr_scf)
                    print("TDDFT calculations:") 
            if self.hl_excited_conv is not None: 
                hl_sr_tdscf.conv_tol=self.hl_excited_conv
            if self.hl_excited_nroots is not None: 
                hl_sr_tdscf.nroots = self.hl_excited_nroots
            if self.hl_excited_cycles is not None: 
                hl_sr_tdscf.max_cycle = self.hl_excited_cycles 
            etd = hl_sr_tdscf.kernel()[0] 
            if self.hl_excited_analyze:
                hl_sr_tdscf.analyze()

    def __gen_hf_scf(self):
        """Initializes the single reference hartree-fock object.
        """

        #Use HF for initial guesses
        if self.hl_unrestricted:
            hl_sr_scf = scf.UHF(self.mol)
            #increase DIIS space
            hl_sr_scf.DIIS = scf.diis.EDIIS
            hl_sr_scf.diis_space = 15
            #Update the fock and electronic energies to use custom methods.
            hl_sr_scf.get_fock = lambda *args, **kwargs: (
                custom_pyscf_methods.uhf_get_fock(hl_sr_scf,
                                                  self.emb_pot, self.proj_pot, *args, **kwargs))
            hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                custom_pyscf_methods.uhf_energy_elec(hl_sr_scf,
                                                     self.emb_pot, self.proj_pot, *args, **kwargs))
        elif self.mol.spin != 0:
            hl_sr_scf = scf.ROHF(self.mol)
            hl_sr_scf.get_fock = lambda *args, **kwargs: (
                custom_pyscf_methods.rohf_get_fock(hl_sr_scf,
                                                   self.emb_pot, self.proj_pot, *args, **kwargs))
            hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                custom_pyscf_methods.rohf_energy_elec(hl_sr_scf,
                                                      self.emb_pot, self.proj_pot, *args, **kwargs))
        else:
            hl_sr_scf = scf.RHF(self.mol)
            emb_pot = (self.emb_pot[0] + self.emb_pot[1])/2.
            proj_pot = (self.proj_pot[0] + self.proj_pot[1])/2.
            hl_sr_scf.get_fock = lambda *args, **kwargs: (
                custom_pyscf_methods.rhf_get_fock(hl_sr_scf,
                                                  emb_pot, proj_pot, *args, **kwargs))
            hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                custom_pyscf_methods.rhf_energy_elec(hl_sr_scf,
                                                     emb_pot, proj_pot, *args, **kwargs))

        self.hl_sr_scf = hl_sr_scf

    def __gen_dft_scf(self):
        """Initializes the single reference dft object.
        """

        #Use DFT for initial guesses
        if self.hl_unrestricted:
            hl_sr_scf = scf.UKS(self.mol)
            #Update the fock and electronic energies to use custom methods.
            hl_sr_scf.get_fock = lambda *args, **kwargs: (
                custom_pyscf_methods.uks_get_fock(hl_sr_scf,
                                                  self.emb_pot, self.proj_pot, *args, **kwargs))
            hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                custom_pyscf_methods.uks_energy_elec(hl_sr_scf,
                                                     self.emb_pot, self.proj_pot, *args, **kwargs))
        elif self.mol.spin != 0:
            hl_sr_scf = scf.ROKS(self.mol)
            #Update the fock and electronic energies to use custom methods.
            hl_sr_scf.get_fock = lambda *args, **kwargs: (
                custom_pyscf_methods.roks_get_fock(hl_sr_scf,
                                                   self.emb_pot, self.proj_pot, *args, **kwargs))
            hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                custom_pyscf_methods.roks_energy_elec(hl_sr_scf,
                                                      self.emb_pot, self.proj_pot, *args, **kwargs))
        else:
            hl_sr_scf = scf.RKS(self.mol)
            hl_sr_scf = scf.RKS(self.mol)
            emb_pot = (self.emb_pot[0] + self.emb_pot[1])/2.
            proj_pot = (self.proj_pot[0] + self.proj_pot[1])/2.
            hl_sr_scf.get_fock = lambda *args, **kwargs: (
                custom_pyscf_methods.rks_get_fock(hl_sr_scf,
                                                  emb_pot, proj_pot, *args, **kwargs))
            hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                custom_pyscf_methods.rks_energy_elec(hl_sr_scf,
                                                     emb_pot, proj_pot, *args, **kwargs))

        #Set grid, rho and xc
        hl_sr_scf.xc = self.hl_sr_method
        self.hl_sr_scf = hl_sr_scf

    def __do_cc(self):
        """Perform the requested coupled cluster calculation."""

        #If dft for sr method, need to convert to hf.
        if self.hl_unrestricted or self.mol.spin != 0:
            hl_cc = cc.UCCSD(self.hl_sr_scf)
        else:
            hl_cc = cc.CCSD(self.hl_sr_scf)

        hl_cc.frozen = self.cc_froz_core_orbs
        hl_cc.diis_space = 15
        if self.hl_conv is not None:
            hl_cc.conv_tol = self.hl_conv
        if self.hl_cycles is not None:
            hl_cc.max_cycle = self.hl_cycles
        if "(t)" in self.hl_method or self.hl_excited:
            eris = hl_cc.ao2mo()
            ecc = hl_cc.kernel(eris=eris)[0]
        else:
            ecc = hl_cc.kernel()[0]
        self.hl_energy += ecc

        if "(t)" in self.hl_method:
            if self.hl_unrestricted or self.mol.spin != 0:
                ecc_t = uccsd_t.kernel(hl_cc, eris=eris)
            else:
                ecc_t = ccsd_t.kernel(hl_cc, eris=eris)
            self.hl_energy += ecc_t

        if self.hl_excited:
            #DO excited state embedding here.
            # in PySCF v1.7, available CC methods are
            # EE/IP/EA/SF-EOM-CCSD, EA/IP-EOM-CCSD_Ta
            # no need to distinguish RCCSD and UCCSD, it is inherited
            if self.hl_excited_conv is not None:
                hl_cc.conv_tol = self.hl_excited_conv
            if self.hl_excited_cycles is not None:
                hl_cc.max_cycle = self.hl_excited_cycles 
            # import constant to convert hartree to eV and cm-1
            from pyscf.data import nist
            from pyscf.cc import eom_rccsd
            eris = hl_cc.ao2mo()
            if 'ee' in self.hl_excited_type:
                print('Only singlet excitations are considered')
                print('Spin-flip excitations are available in PySCF if wanted')
                hl_eom = eom_rccsd.EOMEESinglet(hl_cc)
                hl_eom.kernel(nroots=self.hl_excited_nroots,eris=eris)
                eev = np.around(hl_eom.eee*nist.HARTREE2EV,3)
                ecm = np.around(hl_eom.eee*nist.HARTREE2WAVENUMBER,3)
                print(f"Embedded EE-EOM-CCSD excitation energy:")
                print(f"Results in hartree   :{hl_eom.eee}")
                print(f"Results in eV        :{eev}")
                print(f"Results in wavenumber:{ecm}")
                print(f"Roots converged?     :{hl_eom.converged}")
                print("".center(80, '*'))
            if 'ea' in self.hl_excited_type:
                hl_eom = eom_rccsd.EOMEA(hl_cc)
                hl_eom.kernel(nroots=self.hl_excited_nroots,eris=eris)
                eev = np.around(hl_eom.eea*nist.HARTREE2EV,3)
                ecm = np.around(hl_eom.eea*nist.HARTREE2WAVENUMBER,3)
                print(f"Embedded EA-EOM-CCSD excitation energy:")
                print(f"Results in hartree   :{hl_eom.eea}")
                print(f"Results in eV        :{eev}")
                print(f"Results in wavenumber:{ecm}")
                print(f"Roots converged?     :{hl_eom.converged}")
                print("".center(80, '*'))
                if self.hl_excited_triple:
                    from pyscf.cc import eom_kccsd_rhf
                    #imds = eom_kccsd_rhf._IMDS(mykcc, eris=eris)
                    #imds = imds.make_t3p2_ip_ea(mykcc)
                    myeom = eom_kccsd_rhf.EOMEA_Ta(hl_cc)
                    eea = myeom.eaccsd_star(nroots=self.hl_excited_nroots) 
                    eev = np.around(eea*nist.HARTREE2EV,3)
                    ecm = np.around(eea*nist.HARTREE2WAVENUMBER,3)
                    print(f"Embedded EA-EOM-CCSD(T)(a)* excitation energy:")
                    print(f"Results in hartree   :{eea}")
                    print(f"Results in eV        :{eev}")
                    print(f"Results in wavenumber:{ecm}")
                    print("".center(80, '*'))
            if 'ip' in self.hl_excited_type:
                hl_eom = eom_rccsd.EOMIP(hl_cc)
                hl_eom.kernel(nroots=self.hl_excited_nroots,eris=eris)
                eev = np.around(hl_eom.eip*nist.HARTREE2EV,3)
                ecm = np.around(hl_eom.eip*nist.HARTREE2WAVENUMBER,3)
                print(f"Embedded EA-EOM-CCSD excitation energy:")
                print(f"Results in hartree   :{hl_eom.eip}")
                print(f"Results in eV        :{eev}")
                print(f"Results in wavenumber:{ecm}")
                print(f"Roots converged?     :{hl_eom.converged}")
                print("".center(80, '*'))
                if self.hl_excited_triple:
                    from pyscf.pbc.cc import eom_kccsd_rhf
                    #imds = eom_kccsd_rhf._IMDS(mykcc, eris=eris)
                    #imds = imds.make_t3p2_ip_ea(mykcc)
                    myeom = eom_kccsd_rhf.EOMIP_Ta(hl_cc)
                    eip = myeom.ipccsd_star(nroots=self.hl_excited_nroots) 
                    eev = np.around(eip*nist.HARTREE2EV,3)
                    ecm = np.around(eip*nist.HARTREE2WAVENUMBER,3)
                    print(f"Embedded IP-EOM-CCSD(T)(a)* excitation energy:")
                    print(f"Results in hartree   :{eip}")
                    print(f"Results in eV        :{eev}")
                    print(f"Results in wavenumber:{ecm}")
                    print("".center(80, '*'))

    def __do_mp(self):
        """Perform the requested perturbation calculation"""

        #If dft for sr method, need to convert to hf.
        if self.hl_unrestricted:
            hl_mp = mp.UMP2(self.hl_sr_scf)
        elif self.mol.spin != 0:
            print("ROMP2 Not Implemented.")
        else:
            hl_mp = mp.MP2(self.hl_sr_scf)

        if self.hl_conv is not None:
            hl_mp.conv_tol = self.hl_conv
        if self.hl_cycles is not None:
            hl_mp.max_cycle = self.hl_cycles
        emp = hl_mp.kernel()[0]
        self.hl_energy += emp

    def __do_casscf(self):
        """Perform the requested casscf calculation"""
        #NEED TO MAKE CUSTOM casscf.get_hcore() adding the projection operator.

        str_start = self.hl_method.find("[") + 1
        str_end = self.hl_method.find("]")
        active_space_str_list = self.hl_method[str_start:str_end].split(',')
        active_space = list(map(int, active_space_str_list))
        hl_casscf = mcscf.CASSCF(self.hl_sr_scf, active_space[0], active_space[1])
        if self.hl_conv is not None:
            hl_casscf.conv_tol = self.hl_conv
        if self.hl_cycles is not None:
            hl_casscf.max_cycle = self.hl_cycles

        self.hl_energy = hl_casscf.kernel()[0]

        #Does not have unrestricted nevpt2
        if 'nevpt' in self.hl_method:
            self.hl_energy += mrpt.NEVPT(hl_casscf).kernel()

    def __do_fci(self):
        """Perform the requested fci calculation. This is incomplete."""

        cisolver = fci.FCI(self.mol, self.hl_sr_scf.mo_coeff)
        hl_energy_tot = cisolver.kernel()
        self.hl_energy = hl_energy_tot[0]

    def __do_dmrg(self):
        """Perform the requested dmrg calculation.
        """
        from pyscf import dmrgscf

        mod_hcore = ((self.env_scf.get_hcore()
                      + (self.emb_pot[0] + self.emb_pot[1])/2.
                      + (self.proj_pot[0] + self.proj_pot[1])/2.))
        self.hl_sr_scf.get_hcore = lambda *args, **kwargs: mod_hcore
        str_start = self.hl_method.find("[") + 1
        str_end = self.hl_method.find("]")
        active_space_str_list = self.hl_method[str_start:str_end].split(',')
        active_space = list(map(int, active_space_str_list))
        hl_dmrg = dmrgscf.DMRGSCF(self.hl_sr_scf, active_space[0], active_space[1])

        dmrg_mem = self.pmem
        if dmrg_mem is not None:
            dmrg_mem = float(dmrg_mem) / 1e3 #DMRG Input memory is in GB for some reason.

        hl_dmrg.fcisolver = dmrgscf.DMRGCI(self.mol, maxM=self.dmrg_max_m, memory=dmrg_mem)
        hl_dmrg.fcisolver.num_thrds = self.dmrg_num_thrds
        hl_dmrg.fcisolver.scratchDirectory = self.scr_dir
        edmrg = hl_dmrg.kernel()
        edmrg = 0
        enevpt = 0
        if "nevpt" in self.hl_method:
            if self.hl_compress_approx:
                enevpt = mrpt.NEVPT(hl_dmrg).compress_approx().kernel()
            else:
                enevpt = mrpt.NEVPT(hl_dmrg).kernel()
        self.hl_energy += edmrg + enevpt

    def __do_shci(self):
        """Perform the requested shci calculation.
        """
        from pyscf import shciscf

        mod_hcore = ((self.env_scf.get_hcore()
                      + (self.emb_pot[0] + self.emb_pot[1])/2.
                      + (self.proj_pot[0] + self.proj_pot[1])/2.))
        self.hl_sr_scf.get_hcore = lambda *args, **kwargs: mod_hcore
        str_start = self.hl_method.find("[") + 1
        str_end = self.hl_method.find("]")
        active_space_str_list = self.hl_method[str_start:str_end].split(',')
        active_space = list(map(int, active_space_str_list))
        hl_shci = shciscf.shci.SHCISCF(self.hl_sr_scf, active_space[0], active_space[1])
        hl_shci.fcisolver.mpiprefix = self.shci_mpi_prefix
        hl_shci.fcisolver.stochastic = not self.shci_no_stochastic
        hl_shci.fcisolver.nPTiter = self.shci_npt_iter
        hl_shci.fcisolver.sweep_iter = self.shci_sweep_iter
        hl_shci.fcisolver.DoRDM = not self.shci_no_rdm
        hl_shci.fcisolver.sweep_epsilon = self.shci_sweep_epsilon
        ecc = hl_shci.mc1step()[0]
        ecc = 0
        self.hl_energy += ecc

    def __save_fcidump(self):
        """Saves fcidump file.
        """

        mod_hcore = ((self.env_scf.get_hcore()
                      + (self.emb_pot[0] + self.emb_pot[1])/2.
                      + (self.proj_pot[0] + self.proj_pot[1])/2.))
        self.hl_sr_scf.get_hcore = lambda *args, **kwargs: mod_hcore
        fcidump_filename = (os.path.splitext(self.filename)[0]
                            + '_' + self.chkfile_index + '_.fcidump')
        print(f"FCIDUMP GENERATED AT {fcidump_filename}")
        tools.fcidump.from_scf(self.hl_sr_scf, (
            os.path.splitext(self.filename)[0] + '.fcidump'),
                               tol=1e-200)

    def __save_hl_density_file(self, hl_density):
        """Saves the high level system density to file.

        Parameters
        ----------
        hl_density : array
            An array of the high level density matrix to save.
        """

        if self.filename is None:
            print("Cannot save hl density because no filename")
            return False
        if self.hl_unrestricted:
            cubegen_fn = (os.path.splitext(self.filename)[0] + '_' +
                          self.chkfile_index + '_hl_alpha.cube')
            tools.cubegen.density(self.mol, cubegen_fn, hl_density[0])
            cubegen_fn = (os.path.splitext(self.filename)[0] + '_' +
                          self.chkfile_index +  '_hl_beta.cube')
            tools.cubegen.density(self.mol, cubegen_fn, hl_density[1])
        else:
            cubegen_fn = (os.path.splitext(self.filename)[0] + '_' +
                          self.chkfile_index +  '_hl.cube')
            tools.cubegen.density(self.mol, cubegen_fn, hl_density)
        return True

    def __save_hl_orbital_file(self, hl_mo_coeff, hl_mo_energy, hl_mo_occ):
        '''Save the orbitals generated by the hl method.

        Parameters
        ----------
        hl_mo_coeff : array
            An array of molecular orbital coeffecients from the high level method.
        hl_mo_energy : array
            An array of molecular orbital energies from the high level method.
        hl_mo_occ : array
            An array of molecular occupations from the high level method.
        '''
        if self.filename is None:
            print("Cannot save hl orbitals because no filename")
            return False
        if self.hl_unrestricted:
            molden_fn = os.path.splitext(self.filename)[0] + self.chkfile_index + '_hl_alpha.molden'
            with open(molden_fn, 'w') as fin:
                tools.molden.header(self.mol, fin)
                tools.molden.orbital_coeff(self.mol, fin, hl_mo_coeff[0],
                                           ene=hl_mo_energy[0],
                                           occ=hl_mo_occ[0])
            tools.molden.from_mo(self.mol, molden_fn, hl_mo_coeff[0],
                                 ene=hl_mo_energy[0], occ=hl_mo_occ[0])

            molden_fn = os.path.splitext(self.filename)[0] + self.chkfile_index + '_hl_beta.molden'
            with open(molden_fn, 'w') as fin:
                tools.molden.header(self.mol, fin)
                tools.molden.orbital_coeff(self.mol, fin, hl_mo_coeff[1],
                                           ene=hl_mo_energy[1],
                                           occ=hl_mo_occ[1])
            tools.molden.from_mo(self.mol, molden_fn, hl_mo_coeff[1],
                                 ene=hl_mo_energy[1], occ=hl_mo_occ[1])

        else:
            molden_fn = os.path.splitext(self.filename)[0] + self.chkfile_index + '_hl.molden'
            with open(molden_fn, 'w') as fin:
                tools.molden.header(self.mol, fin)
                tools.molden.orbital_coeff(self.mol, fin, hl_mo_coeff,
                                           ene=hl_mo_energy,
                                           occ=hl_mo_occ)
            tools.molden.from_mo(self.mol, molden_fn, hl_mo_coeff,
                                 ene=hl_mo_energy, occ=hl_mo_occ)
        return True
