# A method to define all cluster supsystem objects
# Daniel Graham
# Dhabih V. Chulhai

import re
import os
from copy import deepcopy as copy
import h5py
import numpy as np
import scipy as sp
from pyscf import gto, scf, mp, cc, mcscf, mrpt, fci, tools
from pyscf.cc import ccsd_t, uccsd_t
from pyscf.scf import diis as scf_diis
from pyscf.lib import diis as lib_diis
from qsome import custom_pyscf_methods, comb_diis
from qsome.ext_methods.ext_factory import ExtFactory



class ClusterEnvSubSystem:
    """A base subsystem object for use in projection embedding.

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


    def __init__(self, mol, env_method, env_order=1, env_smearsigma=0,
                 initguess=None, conv=1e-8, damp=0., shift=0., subcycles=1,
                 setfermi=None, diis=6, unrestricted=False, density_fitting=False,
                 freeze=False, save_orbs=False, save_density=False, save_spin_density=False,
                 verbose=3, filename=None, nproc=None, pmem=None, scrdir=None):
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
        self.env_order = env_order

        self.env_smearsigma = env_smearsigma
        self.env_initguess = initguess
        self.env_conv = conv
        self.env_damp = damp
        self.env_shift = shift

        self.env_subcycles = subcycles
        self.env_setfermi = setfermi
        self.diis_num = diis

        self.unrestricted = unrestricted
        self.density_fitting = density_fitting
        self.freeze = freeze
        self.save_orbs = save_orbs
        self.save_density = save_density
        self.save_spin_density = save_spin_density

        self.verbose = verbose
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

        self.fullsys_cs = True #Is the full system a closed shell calculation? Determines how the rohf calculation is done later.


        self.fermi = [0., 0.]
        self.flip_ros = False
        self.env_scf = self.init_env_scf()
        self.env_hcore = self.env_scf.get_hcore()
        self.env_dmat = None
        self.emb_fock = [None, None]
        self.subsys_fock = [None, None]

        self.emb_pot = [np.zeros_like(self.env_hcore),
                        np.zeros_like(self.env_hcore)]
        self.proj_pot = [np.zeros_like(self.env_hcore),
                         np.zeros_like(self.env_hcore)]

        self.env_mo_coeff = [np.zeros_like(self.env_hcore),
                             np.zeros_like(self.env_hcore)]
        self.env_mo_occ = [np.zeros_like(self.env_hcore[0]),
                           np.zeros_like(self.env_hcore[0])]
        self.env_mo_energy = self.env_mo_occ.copy()
        self.env_energy = 0.0

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

        self.env_sub_nuc_grad = None
        self.env_sub_emb_nuc_grad = None
        self.env_sub_proj_nuc_grad = None
        self.env_hcore_deriv = None
        self.env_vhf_deriv = None

    def init_env_scf(self, mol=None, env_method=None, verbose=None,
                     damp=None, shift=None, dfit=None):
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
        if verbose is None:
            verbose = self.verbose
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
            #Constrained Unrestricted Calculation.
            if mol.spin < 0:
                #This doesn't work properly. As long as only setting spin above 0 should be fine.
                self.flip_ros = True
                mol.spin *= -1
                mol.build()

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
        env_scf.verbose = verbose
        env_scf.damp = damp
        env_scf.level_shift = shift
        if dfit:
            env_scf = env_scf.density_fit()
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
        if in_dmat is not None:
            #Here convert the density matrix into the correct form for the subsystem.
            in_dmat = np.array(in_dmat)
            self.env_dmat = in_dmat
            return True

        if scf_obj is None:
            scf_obj = self.env_scf
        if env_method is None:
            env_method = self.env_method
        if initguess is None:
            if self.env_initguess is None:
                initguess = 'readchk'
            else:
                initguess = self.env_initguess

        if initguess == 'readchk':
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
                        initguess = 'supmol'
                        dmat = scf_obj.get_init_guess()
                else:
                    initguess = 'supmol'
                    dmat = scf_obj.get_init_guess()
            else:
                initguess = 'supmol'
                dmat = scf_obj.get_init_guess()

            #If readchk not found, update teh initguess method
            self.env_initguess = initguess

        elif initguess in ['atom', '1e', 'minao']:
            dmat = scf_obj.get_init_guess(key=initguess)
        elif initguess == 'submol':
            scf_obj.kernel()
            dmat = scf_obj.make_rdm1()
        else:
            dmat = scf_obj.get_init_guess()

        if self.flip_ros:
            temp_dmat = copy(dmat)
            dmat[0] = temp_dmat[1]
            dmat[1] = temp_dmat[0]

        #Dmat always stored [alpha, beta]
        if np.array(dmat).ndim == 2:
            dmat = [dmat/2., dmat/2.]
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
        self.env_energy = self.get_env_elec_energy(env_method=env_method, fock=fock, dmat=dmat, env_hcore=env_hcore, proj_pot=proj_pot, emb_pot=emb_pot) + mol.energy_nuc()
        return self.env_energy

    def get_env_nuc_grad(self, mol=None, scf_obj=None):
        """Return the gradient of just the subsystem in isolation.
           Must occur after a diagonalization."""

        if mol is None:
            mol = self.mol
        if scf_obj is None:
            scf_obj = self.env_scf

        if self.unrestricted or self.mol.spin != 0:
            total_grad = None
        else:
            grad_obj = scf_obj.nuc_grad_method()
            mo_e = self.env_mo_energy[0]
            mo_coeff = self.env_mo_coeff[0]
            mo_occ = self.env_mo_occ[0] * 2.
            hcore_deriv = grad_obj.hcore_generator(mol)
            self.env_hcore_deriv = hcore_deriv
            ovlp_1 = grad_obj.get_ovlp(mol)
            dm0 = self.env_dmat[0] + self.env_dmat[1]

            vhf = grad_obj.get_veff(mol, dm0)
            self.env_vhf_deriv = np.copy(vhf)
            dme0 = grad_obj.make_rdm1e(mo_e, mo_coeff, mo_occ)

            atmlst = range(mol.natm)
            aoslices = mol.aoslice_by_atom()
            deriv = np.zeros((len(atmlst), 3))
            for i, atm in enumerate(atmlst):
                ind_0, ind_1 = aoslices[atm, 2:]
                h1ao = hcore_deriv(atm)
                deriv[i] += np.einsum('xij,ij->x', h1ao, dm0)
                # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
                deriv[i] += np.einsum('xij,ij->x', vhf[:, ind_0:ind_1], dm0[ind_0:ind_1]) * 2
                deriv[i] -= np.einsum('xij,ij->x', ovlp_1[:, ind_0:ind_1], dme0[ind_0:ind_1]) * 2

                #de[k] += grad_obj.extra_force(ia, locals())

            total_grad = deriv

        self.env_sub_nuc_grad = total_grad
        return total_grad

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
                tools.molden.orbital_coeff(self.mol, fin, mo_coeff[0], ene=mo_energy[0], occ=(mo_occ[0] + mo_occ[1]))
        else:
            molden_fn_a = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv_alpha.molden'
            molden_fn_b = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv_beta.molden'
            with open(molden_fn_a, 'w') as fin:
                tools.molden.header(scf_obj.mol, fin)
                tools.molden.orbital_coeff(self.mol, fin, mo_coeff[0], spin='Alpha', ene=mo_energy[0], occ=mo_occ[0])
            with open(molden_fn_b, 'w') as fin:
                tools.molden.header(scf_obj.mol, fin)
                tools.molden.orbital_coeff(self.mol, fin, mo_coeff[1], spin='Beta', ene=mo_energy[1], occ=mo_occ[1])
        return True

    def save_density_file(self, filename=None, density=None):
        """Save the electron density to the file

        Parameters
        ----------
        filename : str
        density : list
        """
        if filename is None:
            if self.filename is None:
                print("Cannot save density because no filename")
                return False
            filename = self.filename
        if density is None:
            density = self.get_dmat()
        print(f'Writing Subsystem {self.chkfile_index} Density'.center(80))
        if self.mol.spin != 0 or self.unrestricted:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv_alpha.cube'
            tools.cubegen.density(self.mol, cubegen_fn, density[0])
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv_beta.cube'
            tools.cubegen.density(self.mol, cubegen_fn, density[1])
        else:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv.cube'
            tools.cubegen.density(self.mol, cubegen_fn, density)
        return True

    def save_spin_density_file(self, filename=None, density=None):
        """Saves a molden file of the spin density

        Parameters
        ----------
        """

        if filename is None:
            if self.filename is None:
                print("Cannot save density because no filename")
                return False
            filename = self.filename
        if density is None:
            density = self.get_dmat()
        if self.mol.spin != 0 or self.unrestricted:
            print(f'Writing Subsystem {self.chkfile_index} Spin Density'.center(80))
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv_spinden.cube'
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

        num_orb_a = self.subsys_fock[0].shape[0]
        num_orb_b = self.subsys_fock[1].shape[0]
        num_orb = [np.zeros((num_orb_a)), np.zeros((num_orb_b))]
        num_orb[0][:self.mol.nelec[0]] = 1.
        num_orb[1][:self.mol.nelec[1]] = 1.

        for i in range(self.env_subcycles):
            #TODO This doesn't work.
            if i > 0:
                self.update_subsys_fock()

            if self.unrestricted:
                self.__do_unrestricted_diag()

            elif self.mol.spin != 0:
                self.__do_restricted_os_diag()
                #return self.env_dmat

            else:
                self.__do_restricted_diag()

            e_sorted = [np.sort(self.env_mo_energy[0]), np.sort(self.env_mo_energy[1])]
            self.__set_fermi(e_sorted)
            self.__set_occupation(e_sorted)

            self.env_dmat[0] = np.dot((self.env_mo_coeff[0] * self.env_mo_occ[0]),
                                      self.env_mo_coeff[0].transpose().conjugate())
            self.env_dmat[1] = np.dot((self.env_mo_coeff[1] * self.env_mo_occ[1]),
                                      self.env_mo_coeff[1].transpose().conjugate())

            self.save_chkfile()
            return self.env_dmat

    def __do_unrestricted_diag(self):
        """Performs diagonalization on the unrestricted env object."""
        emb_proj_fock = [None, None]
        fock = self.emb_fock
        if fock[0] is None:
            fock = self.subsys_fock
        emb_proj_fock[0] = fock[0] + self.proj_pot[0]
        emb_proj_fock[1] = fock[1] + self.proj_pot[1]
        energy, coeff = self.env_scf.eig(emb_proj_fock, self.env_scf.get_ovlp())
        self.env_mo_energy = [energy[0], energy[1]] #May be unnecessary.
        self.env_mo_coeff = [coeff[0], coeff[1]]

    def __do_restricted_os_diag(self):
        """Performs diagonalization on the restricted open shell env object."""
        fock = self.emb_fock
        if fock[0] is None:
            fock = self.subsys_fock

        emb_proj_fock = fock[0] + self.proj_pot[0]
        emb_proj_fock += fock[1] + self.proj_pot[1]
        emb_proj_fock /= 2.

        #emb_proj_fock = scf.rohf.get_roothaan_fock(emb_proj_fock, dmat, scf_obj.get_ovlp())
        energy, coeff = self.env_scf.eig(emb_proj_fock, self.env_scf.get_ovlp())

        self.env_mo_energy = [energy, energy]
        self.env_mo_coeff = [coeff, coeff]
        #occ = self.env_scf.get_occ(energy, coeff)
        #alpha_occ = np.zeros_like(occ)
        #beta_occ = np.zeros_like(occ)
        #for i, occ_val in enumerate(occ):
        #    if occ_val > 0:
        #        alpha_occ[i] = 1.
        #    if occ_val > 1:
        #        beta_occ[i] = 1.
        #env_mo_occ = [alpha_occ, beta_occ]
        #self.env_mo_energy = env_mo_energy
        #self.env_mo_coeff = env_mo_coeff
        #self.env_mo_occ = env_mo_occ
        #self.env_dmat[0] = np.dot((env_mo_coeff[0] * env_mo_occ[0]),
        #                          env_mo_coeff[0].transpose().conjugate())
        #self.env_dmat[1] = np.dot((env_mo_coeff[1] * env_mo_occ[1]),
        #                          env_mo_coeff[1].transpose().conjugate())

    def __do_restricted_diag(self):
        """Performs diagonalization on the restricted env object."""
        fock = self.emb_fock
        if fock[0] is None:
            fock = self.subsys_fock
        emb_proj_fock = fock[0] + self.proj_pot[0]
        emb_proj_fock += fock[1] + self.proj_pot[1]
        emb_proj_fock = emb_proj_fock / 2.
        energy, coeff = self.env_scf.eig(emb_proj_fock, self.env_scf.get_ovlp())
        self.env_mo_energy = [energy, energy]
        self.env_mo_coeff = [coeff, coeff]

    def relax_sub_dmat(self, damp_param=0):
        """Relaxes the given subsystem density using the updated fock.
        """
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
            self.env_dmat = np.array([new_dm/2., new_dm/2.])
        return ddm

    def __set_fermi(self, e_sorted):
        """Sets the fermi level for the subsystem.

        Parameters
        ----------
        e_sorted : list
            A list of the orbital energies sorted lowest to highest.
        nocc_orbs : list
            A list of the number of occupied orbitals for alpha and beta e.
        """
        self.fermi = [0., 0.]
        nocc_orbs = [self.mol.nelec[0], self.mol.nelec[1]]
        if len(e_sorted[0]) > nocc_orbs[0]:
            self.fermi[0] = ((e_sorted[0][nocc_orbs[0]]
                              + e_sorted[0][nocc_orbs[0] -1]) / 2.)
        else:
            self.fermi[0] = 0.    #Minimal basis
        if len(e_sorted[1]) > nocc_orbs[1]:
            self.fermi[1] = ((e_sorted[1][nocc_orbs[1]]
                              + e_sorted[1][nocc_orbs[1] -1]) / 2.)
        else:
            self.fermi[1] = 0.    #Minimal basis

    def __set_occupation(self, e_sorted):
        """Sets the orbital occupation numbers.

        Parameters
        ----------
        e_sorted : list
            A sorted list of molecular orbital energies.
        """
        #Smear sigma may not be right for single elctron
        self.env_mo_occ = [np.zeros_like(self.env_mo_energy[0]),
                           np.zeros_like(self.env_mo_energy[1])]
        nocc_orbs = [self.mol.nelec[0], self.mol.nelec[1]]
        if self.env_smearsigma > 0.:
            self.env_mo_occ[0] = ((self.env_mo_energy[0]
                                   - self.fermi[0]) / self.env_smearsigma)
            occ_orb = np.where(self.env_mo_occ[0] < 1000)
            vir_orb = np.where(self.env_mo_occ[0] >= 1000)
            self.env_mo_occ[0][occ_orb] = 1. / (np.exp(self.env_mo_occ[0][occ_orb]) + 1.)
            self.env_mo_occ[0][vir_orb] = 0.

            self.env_mo_occ[1] = (self.env_mo_energy[1] - self.fermi[1]) / self.env_smearsigma
            occ_orb = np.where(self.env_mo_occ[1] < 1000)
            vir_orb = np.where(self.env_mo_occ[1] >= 1000)
            self.env_mo_occ[1][occ_orb] = 1. / (np.exp(self.env_mo_occ[1][occ_orb]) + 1.)
            self.env_mo_occ[1][vir_orb] = 0.

        else:
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
            #if len(e_sorted[0]) > nocc_orbs[0]:
            #    for i in range(nocc_orbs[0]):
            #        self.env_mo_occ[0][i] = 1
            #else:
            #    self.env_mo_occ[0][:] = 1.

            #if len(e_sorted[1]) > nocc_orbs[1]:
            #    for i in range(nocc_orbs[1]):
            #        self.env_mo_occ[1][i] = 1
            #else:
            #    self.env_mo_occ[1][:] = 1.

#    def get_useable_dmat(self, mat=None):
#        """Return a useable density matrix for use with PySCF.
#        This is because we always store the density matrix as a
#        (2 x nao x nao) matrix."""
#
#        if self.unrestricted:
#            return mat
#        if self.cell.spin != 0:
#            return mat
#        else:
#            return mat[0] + mat[1]


#    def update_stored_dmat(self, mat=None):
#        """Store the density matrix as a (2 x nao x nao) matrix."""
#
#        if mat.ndim == 3:
#            return mat
#        else:
#            out = np.zeros((2, mat.shape[0], mat.shape[1]), dtype=mat.dtype)
#            out[0] = mat / 2.0
#            out[1] = mat / 2.0
#            return out

#    def get_useable_pot(self, mat=None):
#        """Return a useable potential for use with PySCF.
#        This is because we always store the potentials as a
#        (2 x nao x nao) matrix."""
#
#        if mat.ndim == 2:
#            return mat
#        if self.unrestricted:
#            return mat
#        if self.cell.spin != 0:
#            return mat
#        else:
#            return (mat[0] + mat[1])/2.
#
#    def update_stored_pot(self, mat=None):
#        """Store the potential matrix as a (2 x nao x nao) matrix."""
#
#        if mat.ndim == 3:
#            return mat
#        else:
#            out = np.zeros((2, mat.shape[0], mat.shape[1]), dtype=mat.dtype)
#            out[0] = mat
#            out[1] = mat
#            return out
#
#    def print_coordinates(self):
#        """Prints the Mole coordinates."""
#        bohr_2_angstrom = 0.52917720859
#        print(" Atom Coordinates (Angstrom) ".center(80, "-"))
#        if hasattr(self, 'cell'):
#            mol = self.cell
#        else:
#            mol = self.mol
#        for i in range(mol.natm):
#            a = mol.atom_symbol(i)
#            c = mol.atom_coord(i) * bohr_2_angstrom
#            st = "{0:<10} {1:10.4f} {2:10.4f} {3:10.4f}".format(
#                a, c[0], c[1], c[2])
#            print(st)


class ClusterHLSubSystem(ClusterEnvSubSystem):
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

    def __init__(self, mol, env_method, hl_method, hl_order=1, hl_initguess=None,
                 hl_sr_method=None, hl_spin=None, hl_conv=None, hl_grad=None,
                 hl_cycles=None, hl_damp=0., hl_shift=0., hl_freeze_orbs=None,
                 hl_ext=None, hl_unrestricted=False, hl_compress_approx=False,
                 hl_density_fitting=False, hl_save_orbs=False,
                 hl_save_density=False, hl_save_spin_density=False,
                 hl_dict=None, **kwargs):
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

        self.hl_method = hl_method
        self.hl_initguess = hl_initguess
        self.hl_sr_method = hl_sr_method
        self.hl_spin = hl_spin
        self.hl_conv = hl_conv
        self.hl_grad = hl_grad
        self.hl_cycles = hl_cycles
        self.hl_damp = hl_damp
        self.hl_shift = hl_shift

        self.hl_ext = hl_ext
        self.hl_freeze_orbs = hl_freeze_orbs
        self.hl_unrestricted = hl_unrestricted
        self.hl_compress_approx = hl_compress_approx
        self.hl_density_fitting = hl_density_fitting
        self.hl_save_orbs = hl_save_orbs
        self.hl_save_density = hl_save_density
        self.hl_save_spin_density = hl_save_spin_density
        self.__set_hl_method_settings(hl_dict)

        self.hl_mo_coeff = None
        self.hl_mo_occ = None
        self.hl_mo_energy = None
        self.hl_dmat = None
        self.hl_sr_scf = None
        self.hl_energy = None
        self.hl_sub_nuc_grad = None
        self.hl_sub_emb_nuc_grad = None
        self.hl_sub_proj_nuc_grad = None

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
            self.cc_initguess = hl_dict.get("cc_initguess")
            self.cc_froz_orbs = hl_dict.get("froz_orbs")
        if 'cas' in self.hl_method:
            self.cas_loc_orbs = hl_dict.get("loc_orbs")
            self.cas_init_guess = hl_dict.get("cas_initguess")
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
        if (self.hl_sr_method is None and
                self.hl_method not in known_methods and
                not re.match(cas_regex, self.hl_method)):
            self.hl_sr_method = self.hl_method

        if self.hl_ext is not None:
            self.get_ext_energy()
            return self.hl_energy

        self.do_sr_scf()

        if self.hl_method in cc_aliases:
            self.do_cc()

        elif self.hl_method in mp_aliases:
            self.do_mp()

        elif re.match(cas_regex, self.hl_method):
            self.do_casscf()

        elif self.hl_method in fci_aliases:
            self.do_fci()

        elif re.match(dmrg_regex, self.hl_method):
            self.do_dmrg()

        elif re.match(shci_regex, self.hl_method):
            self.do_shci()

        elif self.hl_method in fcidump_aliases:
            self.save_fcidump()

        return self.hl_energy

    def get_ext_energy(self):
        """Uses an external method to calculate high level energy.

        Parameters
        ----------
        emb_pot : np.array
            Array of the embedding potential.
        proj_pot : np.array
            Array of the projection potential.
        """

        print("use external method for hl calculation")
        hcore = self.env_scf.get_hcore()
        #emb_pot_ro = self.emb_ro_fock[0] - self.fock
        emb_proj_pot = [self.emb_pot[0] + self.proj_pot[0], self.emb_pot[1] + self.proj_pot[1]]
        ext_factory = ExtFactory()
        name_no_path = os.path.split(self.filename)[-1]
        name_no_ext = os.path.splitext(name_no_path)[0]
        file_path = os.path.split(self.filename)[0]
        scr_path = self.scr_dir
        ext_obj = ext_factory.get_ext_obj(self.hl_ext, gto.copy(self.mol), self.hl_method, emb_proj_pot, core_ham=hcore, filename=name_no_ext, work_dir=file_path, scr_dir=scr_path, nproc=self.nproc, pmem=self.pmem, save_orbs=None, save_density=False, hl_dict=self.hl_dict)
        energy = ext_obj.get_energy()
        self.hl_energy = energy[0]

    def do_sr_scf(self):
        """Initializes and runs the single reference hf object"""

        hf_aliases = ['hf', 'uhf', 'rhf', 'rohf']
        if (self.hl_sr_method is None or self.hl_sr_method in hf_aliases):
            self.gen_hf_scf()
        else:
            self.gen_dft_scf()

        if self.hl_initguess == 'ft':
            dmat = self.get_dmat()
        elif self.hl_initguess is not None:
            dmat = self.hl_sr_scf.get_init_guess(key=self.hl_initguess)
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

        self.hl_energy = self.hl_sr_scf.kernel(dm0=dmat)

    def gen_hf_scf(self):
        """Initializes the single reference hartree-fock object.

        Parameters
        ----------
        emb_pot : np.array
            Array of the embedding potential.
        proj_pot : np.array
            Array of the projection potential.
        """

        #Use HF for initial guesses
        if self.hl_unrestricted:
            hl_sr_scf = scf.UHF(self.mol)
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

    def gen_dft_scf(self):
        """Initializes the single reference dft object.

        Parameters
        ----------
        xc_fun : str
           The exchange correlation functional.
        emb_pot : np.array
            Array of the embedding potential.
        proj_pot : np.array
            Array of the projection potential.
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

    def do_cc(self):
        """Perform the requested coupled cluster calculation."""

        #If dft for sr method, need to convert to hf.
        if self.hl_unrestricted or self.mol.spin != 0:
            hl_cc = cc.UCCSD(self.hl_sr_scf)
        else:
            hl_cc = cc.CCSD(self.hl_sr_scf)

        hl_cc.frozen = self.cc_froz_orbs
        if self.hl_conv is not None:
            hl_cc.conv_tol = self.hl_conv
        if self.hl_cycles is not None:
            hl_cc.max_cycle = self.hl_cycles
        ecc = hl_cc.kernel()[0]
        self.hl_energy += ecc
        if "(t)" in self.hl_method:
            eris = hl_cc.ao2mo()
            if self.hl_unrestricted or self.mol.spin != 0:
                ecc_t = uccsd_t.kernel(hl_cc, eris)
            else:
                ecc_t = ccsd_t.kernel(hl_cc, eris)
            self.hl_energy += ecc_t

    def do_mp(self):
        """Perform the requested perturbation calculation"""

        #If dft for sr method, need to convert to hf.
        if self.hl_unrestricted:
            hl_mp = mp.UMP2(self.hl_sr_scf)
        elif self.mol.spin != 0:
            print("ro mp2")
        else:
            hl_mp = mp.MP2(self.hl_sr_scf)

        if self.hl_conv is not None:
            hl_mp.conv_tol = self.hl_conv
        if self.hl_cycles is not None:
            hl_mp.max_cycle = self.hl_cycles
        emp = hl_mp.kernel()[0]
        self.hl_energy += emp

    def do_casscf(self):
        """Perform the requested casscf calculation"""

        active_space = [int(i) for i in (self.hl_method[self.hl_method.find("[") + 1:self.hl_method.find("]")]).split(',')]
        hl_casscf = mcscf.CASSCF(self.hl_sr_scf, active_space[0], active_space[1])
        if self.hl_conv is not None:
            hl_casscf.conv_tol = self.hl_conv
        if self.hl_cycles is not None:
            hl_casscf.max_cycle = self.hl_cycles

        self.hl_energy = hl_casscf.kernel()[0]

    def do_fci(self):
        """Perform the requested fci calculation. THIS IS NOT CORRECT."""

        cisolver = fci.FCI(self.mol, self.hl_sr_scf.mo_coeff)
        hl_energy_tot = cisolver.kernel()
        self.hl_energy = hl_energy_tot[0]

    def do_dmrg(self):
        """Perform the requested dmrg calculation.

        Parameters
        ----------
        emb_pot : np.array
            Array of the embedding potential.
        proj_pot : np.array
            Array of the projection potential.
        """
        from pyscf import dmrgscf

        mod_hcore = ((self.env_scf.get_hcore()
                      + (self.emb_pot[0] + self.emb_pot[1])/2.
                      + (self.proj_pot[0] + self.proj_pot[1])/2.))
        self.hl_sr_scf.get_hcore = lambda *args, **kwargs: mod_hcore
        active_space = [int(i) for i in (self.hl_method[self.hl_method.find("[") + 1:self.hl_method.find("]")]).split(',')]
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

    def do_shci(self):
        """Perform the requested shci calculation.

        Parameters
        ----------
        emb_pot : np.array
            Array of the embedding potential.
        proj_pot : np.array
            Array of the projection potential.
        """
        from pyscf import shciscf

        mod_hcore = ((self.env_scf.get_hcore()
                      + (self.emb_pot[0] + self.emb_pot[1])/2.
                      + (self.proj_pot[0] + self.proj_pot[1])/2.))
        self.hl_sr_scf.get_hcore = lambda *args, **kwargs: mod_hcore
        active_space = [int(i) for i in (self.hl_method[self.hl_method.find("[") + 1:self.hl_method.find("]")]).split(',')]
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

    def save_fcidump(self):
        """Saves fcidump file.

        Parameters
        ----------
        emb_pot : np.array
            Array of the embedding potential.
        proj_pot : np.array
            Array of the projection potential.
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

    def save_hl_density_file(self, filename=None, density=None):
        """Saves the high level system density to file.

        Parameters
        ---------
        filename : str
            Name of file to save the cube density file.
        density : np.array
            An array of the density matrix for the subsystem.
        """

        if filename is None:
            if self.filename is None:
                print("Cannot save hl density because no filename")
                return False
            filename = self.filename
        if density is None:
            density = self.get_dmat()
        cubegen_fn = os.path.splitext(filename)[0] + '_hl.cube'
        tools.cubegen.density(self.mol, cubegen_fn, density)
        return True

    def save_hl_orbital_file(self):
        '''Save the orbitals generated by the hl method.

        Parameters
        ----------

        Returns
        -------
        '''
        if self.filename is None:
            print("Cannot save hl orbitals because no filename")
            return False
        molden_fn = os.path.splitext(self.filename)[0] + '_hl.molden'
        with open(molden_fn, 'w') as fin:
            tools.molden.header(self.mol, fin)
            tools.molden.orbital_coeff(self.mol, fin, self.env_mo_coeff,
                                       ene=self.env_mo_energy,
                                       occ=self.env_mo_occ)
        tools.molden.from_mo(self.mol, molden_fn, self.env_mo_coeff,
                             ene=self.env_mo_energy, occ=self.env_mo_occ)
        return True
#class ClusterExcitedSubSystem(ClusterHLSubSystem):
#    """Excited state subsystem object for embedding
#
#    Attributes
#    ----------
#    UNKNOWN
#
#    Methods
#    -------
#    UNKNOWN
#    """
#
#
#    def __init__(self):
#        super().__init__()
#
