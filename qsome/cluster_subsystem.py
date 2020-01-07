# A method to define all cluster supsystem objects
# Daniel Graham
# Dhabih V. Chulhai

import re
from qsome import subsystem, custom_pyscf_methods, molpro_calc, comb_diis
from qsome.ext_methods.ext_factory import ExtFactory
from pyscf import gto, scf, dft, cc, mcscf, tools, mp, fci
from pyscf.cc import ccsd_t, uccsd_t, ccsd_t_rdm_slow, ccsd_t_lambda_slow
from pyscf.scf import diis as scf_diis
from pyscf.lib import diis as lib_diis

import os
import h5py

from functools import reduce

import numpy as np
import scipy as sp
from copy import deepcopy as copy


class ClusterEnvSubSystem:
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


    def __init__(self, mol, env_method, env_order=1, env_smearsigma=0,
                 initguess=None, conv=1e-8, damp=0., shift=0., subcycles=1, 
                 setfermi=None, diis=0, unrestricted=False, density_fitting=False, 
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

    def concat_mol_obj(self, mol):
        """Adds a mol ovject to the existing mol object."""
        self.mol.atom = self.mol._atom + mol._atom
        self.mol.unit = 'bohr'
        self.mol._basis.update(mol._basis)
        self.mol.basis = self.mol._basis
        self.mol.build()
        #self.mol = new_mol
        self.env_scf = self.init_env_scf()
        self.env_hcore = self.env_scf.get_hcore()
        self.emb_pot = [np.zeros_like(self.env_hcore), 
                        np.zeros_like(self.env_hcore)]
        self.proj_pot = [np.zeros_like(self.env_hcore), 
                        np.zeros_like(self.env_hcore)]

        self.env_mo_coeff = [np.zeros_like(self.env_hcore), 
                             np.zeros_like(self.env_hcore)]
        self.env_mo_occ = [np.zeros_like(self.env_hcore[0]), 
                           np.zeros_like(self.env_hcore[0])]
        self.env_mo_energy = self.env_mo_occ.copy()
        #Update diis?



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

        else:
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
                    if (np.any(self.env_mo_coeff) 
                      and np.any(self.env_mo_occ)):
                        #Confirm correct read density dimensions.
                        ndim = scf_obj.mol.nao
                        if (ndim == self.env_mo_coeff.shape[1] and ndim == self.env_mo_coeff.shape[2]):
                            dmat = [0,0]
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
        dmat = self.env_dmat
        if not (self.unrestricted or self.mol.spin != 0):
            dmat = dmat[0] + dmat[1]
        return dmat

    def update_subsys_fock(self, dmat=None, hcore=None):

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
            s1 = grad_obj.get_ovlp(mol)
            dm0 = self.dmat[0] + self.dmat[1]

            vhf = grad_obj.get_veff(mol, dm0)
            self.env_vhf_deriv = np.copy(vhf)
            dme0 = grad_obj.make_rdm1e(mo_e, mo_coeff, mo_occ)

            atmlst = range(mol.natm)
            aoslices = mol.aoslice_by_atom()
            de = np.zeros((len(atmlst),3))
            for k, ia in enumerate(atmlst):
                p0, p1 = aoslices [ia,2:]
                h1ao = hcore_deriv(ia)
                de[k] += np.einsum('xij,ij->x', h1ao, dm0)
                # nabla was applied on bra in vhf, *2 for the contributions of nabla|ket>
                de[k] += np.einsum('xij,ij->x', vhf[:,p0:p1], dm0[p0:p1]) * 2
                de[k] -= np.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2

                #de[k] += grad_obj.extra_force(ia, locals())

            total_grad = de

        self.env_sub_nuc_grad = total_grad
        return total_grad
    
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

    def update_emb_pot(self, emb_fock=None):
        if emb_fock is None:
            if self.emb_fock[0] is None:
                emb_fock = None
            else:
                emb_fock = self.emb_fock
        self.update_subsys_fock()
        self.emb_pot = [emb_fock[0] - self.subsys_fock[0],
                        emb_fock[1] - self.subsys_fock[1]]

    def save_density_file(self, filename=None, density=None):
        from pyscf.tools import cubegen
        if filename is None:
            if self.filename is None:
                print ("Cannot save density because no filename")
                return False
            else:
                filename = self.filename
        if density is None:
            density = self.get_dmat()
        print(f'Writing Subsystem {self.chkfile_index} Density'.center(80))
        if self.mol.spin != 0 or self.unrestricted:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv_a.cube'
            cubegen.density(self.mol, cubegen_fn, density[0])
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv_b.cube'
            cubegen.density(self.mol, cubegen_fn, density[1])
        else:
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv.cube'
            cubegen.density(self.mol, cubegen_fn, density)

    def save_spin_density_file(self, filename=None, density=None):
        from pyscf.tools import cubegen
        if filename is None:
            if self.filename is None:
                print ("Cannot save density because no filename")
                return False
            else:
                filename = self.filename
        if density is None:
            density = self.get_dmat()
        if self.mol.spin != 0 or self.unrestricted:
            print(f'Writing Subsystem {self.chkfile_index} Spin Density'.center(80))
            cubegen_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv_spinden.cube'
            cubegen.density(self.mol, cubegen_fn, np.subtract(density[0],density[1]))
        else:
            print(f'Cannot write spin density for a closed shell system.'.center(80))
    
    def save_orbital_file(self, filename=None, scf_obj=None, mo_occ=None, mo_coeff=None, mo_energy=None):
        from pyscf.tools import molden
        if filename is None:
            if self.filename is None:
                print ("Cannot save orbitals because no filename")
                return False
            else:
                filename = self.filename
        if scf_obj is None:
            scf_obj = self.env_scf
        if mo_occ is None:
            mo_occ = self.env_mo_occ
        if mo_coeff is None:
            mo_coeff = self.env_mo_coeff
        if mo_energy is None:
            mo_energy = self.env_mo_energy
        molden_fn = os.path.splitext(filename)[0] + '_' + self.chkfile_index + '_subenv.molden'
        print(f'Writing Subsystem {self.chkfile_index} Orbitals'.center(80))
        with open(molden_fn, 'w') as fin:
            if not (self.unrestricted or self.mol.spin != 0):
                molden.header(scf_obj.mol, fin)
                molden.orbital_coeff(self.mol, fin, mo_coeff[0], ene=mo_energy[0], occ=(mo_occ[0]*2.))
            else:
                print ("Save open shell orbitals NOT CODED YET")
        #molden.from_mo(scf_obj.mol, molden_fn, mo_coeff, ene=mo_energy, occ=mo_occ)

    #This needs to be checked. I don't know if I can modify the hdf5 file after it is created by another object to add further groups.
    def save_chkfile(self, filename=None):

        if filename is None:
            if self.filename is None:
                print ("chkfile not saved because no filename set.")
                return False
            else:
                filename = os.path.splitext(self.filename)[0] + '.hdf5'
        assert(self.chkfile_index is not None),'Need to set chkfile_index'

        chk_index = self.chkfile_index
        # check if file exists. 
        if os.path.isfile(filename):
            try:
                with h5py.File(filename, 'r+') as hf:
                    subsys_coeff = hf[f'subsystem:{chk_index}/mo_coeff']
                    subsys_coeff[...] = self.env_mo_coeff
                    subsys_occ = hf[f'subsystem:{chk_index}/mo_occ']
                    subsys_occ[...] = self.env_mo_occ
                    subsys_energy = hf[f'subsystem:{chk_index}/mo_energy']
                    subsys_energy[...] = self.env_mo_energy
            except TypeError:
                print ("Overwriting existing chkfile".center(80))
                with h5py.File(filename, 'w') as hf:
                    sub_sys_data = hf.create_group(f'subsystem:{chk_index}')
                    sub_sys_data.create_dataset('mo_coeff', data=self.env_mo_coeff)
                    sub_sys_data.create_dataset('mo_occ', data=self.env_mo_occ)
                    sub_sys_data.create_dataset('mo_energy', data=self.env_mo_energy)
            except KeyError:
                print ("Missing subsystem data in chkfile".center(80))
                with h5py.File(filename, 'a') as hf:
                    sub_sys_data = hf.create_group(f'subsystem:{chk_index}')
                    sub_sys_data.create_dataset('mo_coeff', data=self.env_mo_coeff)
                    sub_sys_data.create_dataset('mo_occ', data=self.env_mo_occ)
                    sub_sys_data.create_dataset('mo_energy', data=self.env_mo_energy)
        else:
            with h5py.File(filename, 'a') as hf:
                sub_sys_data = hf.create_group(f'subsystem:{chk_index}')
                sub_sys_data.create_dataset('mo_coeff', data=self.env_mo_coeff)
                sub_sys_data.create_dataset('mo_occ', data=self.env_mo_occ)
                sub_sys_data.create_dataset('mo_energy', data=self.env_mo_energy)

    def read_chkfile(self, filename=None):

        if filename is None:
            if self.filename is None:
                return False
            else:
                filename = os.path.splitext(self.filename)[0] + '.hdf5'
        assert(self.chkfile_index is not None),'Need to set chkfile_index'

        filename = os.path.splitext(filename)[0] + '.hdf5'
        chk_index = self.chkfile_index

        if os.path.isfile(filename):
            try:
                with h5py.File(filename, 'r') as hf:
                    subsys_coeff = hf[f'subsystem:{chk_index}/mo_coeff']
                    self.env_mo_coeff = subsys_coeff[:]
                    subsys_occ = hf[f'subsystem:{chk_index}/mo_occ']
                    self.env_mo_occ = subsys_occ[:]
                    subsys_energy = hf[f'subsystem:{chk_index}/mo_energy']
                    self.env_mo_energy = subsys_energy[:]
                return True
            except TypeError:
                print ("chkfile improperly formatted".center(80))
                return False
            except KeyError:
                print ("Missing subsystem data in chkfile".center(80))
                return False
        else:
            print ("chkfile NOT found".center(80))
            return False


    def diagonalize(self, scf_obj=None, subcycles=None, env_method=None,
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

        if scf_obj is None:
            scf_obj = self.env_scf
        if subcycles is None:
            subcycles = self.env_subcycles
        if env_method is None:
            env_method = self.env_method
        if dmat is None:
            dmat = copy(self.env_dmat)
        if proj_pot is None:
            proj_pot = self.proj_pot
        if fock is None:
            if self.emb_fock[0] is None:
                self.update_subsys_fock()
                fock = self.subsys_fock
            else:
                fock = self.emb_fock
        if env_hcore is None:
            env_hcore = self.env_hcore
        if diis is None:
            diis = self.diis

        mol = scf_obj.mol

        fock = np.array(fock)
        nA_a = fock[0].shape[0]
        nA_b = fock[1].shape[0]
        N = [np.zeros((nA_a)), np.zeros((nA_b))]
        N[0][:mol.nelec[0]] = 1.
        N[1][:mol.nelec[1]] = 1.

        for i in range(subcycles):
            #TODO This doesn't work.
            if i > 0:
                self.update_subsys_fock()
                fock = self.subsys_fock
                if fock.ndim == 2.:
                   fock = np.array([fock, fock]) 

            if self.unrestricted:
                emb_proj_fock = [None, None]
                emb_proj_fock[0] = fock[0] + proj_pot[0]
                emb_proj_fock[1] = fock[1] + proj_pot[1]

                #This is the costly part. I think.
                E, C = scf_obj.eig(emb_proj_fock, scf_obj.get_ovlp())
                env_mo_energy = [E[0], E[1]]
                env_mo_coeff = [C[0], C[1]]

            elif mol.spin != 0:
                #There are two possible scenarios. 
                #1. The overall WF is spin 0. In this case the potential is just a closed shell potential for all orbitals but the singly occupied ones. These will have a different potential added at the end.
                if self.fullsys_cs:
                    emb_proj_fock = [None, None]
                    emb_proj_fock[0] = fock[0] + proj_pot[0]
                    emb_proj_fock[1] = fock[1] + proj_pot[1]

                    cs_fock = emb_proj_fock[1]
                    os_fock = emb_proj_fock[0]

                    if self.flip_ros:
                        cs_fock = emb_proj_fock[0]
                        os_fock = emb_proj_fock[1]

                    E_cs, C_cs = scf_obj.eig(cs_fock, scf_obj.get_ovlp())
                    E_os, C_os = scf_obj.eig(os_fock, scf_obj.get_ovlp())

                    e_sorted = [np.sort(E_cs), np.sort(E_os)]
                    mo_occ = [np.zeros_like(E_cs), 
                              np.zeros_like(E_os)]

                    num_cs = mol.nelec[1]
                    num_os = mol.nelec[0] - mol.nelec[1]

                    #CS Density
                    if (len(e_sorted[0]) > num_cs):
                        for i in range(num_cs): 
                            mo_occ[1][i] = 1.
                    else:
                        mo_occ[1][:] = 1.

                    cs_dmat = np.dot((C_cs * mo_occ[1]), 
                                      C_cs.transpose().conjugate())

                    #OS Density
                    if (len(e_sorted[0]) > num_cs + num_os):
                        for i in range(num_cs, num_cs + num_os): 
                            mo_occ[0][i] = 1.
                    else:
                        mo_occ[0][num_cs:] = 1.

                    os_dmat = np.dot((C_os * mo_occ[0]), 
                                      C_os.transpose().conjugate())

                    self.env_dmat[0] = cs_dmat + os_dmat
                    self.env_dmat[1] = cs_dmat
                    temp_occ = np.add(mo_occ[0], mo_occ[1])
                    mo_occ[0] = temp_occ
                    if self.flip_ros:
                        self.env_dmat[1] = cs_dmat + os_dmat
                        self.env_dmat[0] = cs_dmat
                        temp_occ = copy(mo_occ)
                        mo_occ[0] = temp_occ[1]
                        mo_occ[1] = temp_occ[0]

                    self.env_mo_occ = mo_occ
                    return self.env_dmat

                #2. The overall WF is not spin 0 and is treated using an RO method. In this case, things get much more complicated.
                else:
                    emb_proj_fock = fock[0] + proj_pot[0] 
                    emb_proj_fock += fock[1] + proj_pot[1]
                    emb_proj_fock /= 2.
                    #emb_proj_fock = [None, None]
                    #emb_proj_fock[0] = fock[0] + proj_pot[0]
                    #emb_proj_fock[1] = fock[1] + proj_pot[1]

                    #if self.flip_ros:
                    #    temp_fock = copy(emb_proj_fock)
                    #    temp_dmat = copy(dmat)
                    #    emb_proj_fock[0] = temp_fock[1]
                    #    emb_proj_fock[1] = temp_fock[0]
                    #    dmat[0] = temp_dmat[1]
                    #    dmat[1] = temp_dmat[0]

                    #emb_proj_fock = scf.rohf.get_roothaan_fock(emb_proj_fock, dmat, scf_obj.get_ovlp())
                    E_ro, C_ro = scf_obj.eig(emb_proj_fock, scf_obj.get_ovlp())

                    E = [E_ro, E_ro]
                    C = [C_ro, C_ro]
                    env_mo_energy = [E[0], E[1]]
                    env_mo_coeff = [C[0], C[1]]
                    occ = scf_obj.get_occ(E_ro, C_ro)
                    alpha_occ = np.zeros_like(occ)
                    beta_occ = np.zeros_like(occ)
                    for k in range(len(occ)):
                        if occ[k] > 0:
                            alpha_occ[k] = 1.
                        if occ[k] > 1:
                            beta_occ[k] = 1.
                    env_mo_occ = [alpha_occ, beta_occ]
                    self.env_mo_energy = env_mo_energy
                    self.env_mo_coeff = env_mo_coeff
                    self.env_mo_occ = env_mo_occ
                    self.env_dmat[0] = np.dot((env_mo_coeff[0] * env_mo_occ[0]), 
                                           env_mo_coeff[0].transpose().conjugate())
                    self.env_dmat[1] = np.dot((env_mo_coeff[1] * env_mo_occ[1]), 
                                           env_mo_coeff[1].transpose().conjugate())
                    if self.flip_ros:
                        temp_mo_e = copy(env_mo_energy)
                        temp_mo_coeff = copy(env_mo_coeff)
                        temp_mo_occ = copy(env_mo_occ)
                        temp_dmat = copy(self.env_dmat)
                        self.env_mo_energy = [temp_mo_e[1], temp_mo_e[0]]
                        self.env_mo_coeff = [temp_mo_coeff[1], temp_mo_coeff[0]]
                        self.env_mo_occ = [temp_mo_occ[1], temp_mo_occ[0]]
                        self.env_dmat = [temp_dmat[1], temp_dmat[0]]

                    return self.env_dmat

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
                        s1e = scf_obj.get_ovlp()
                        dm = dmat[0] + dmat[1]
                        f = emb_fock
                        mf = scf_obj
                        h1e = (env_hcore
                               + (emb_pot[0] + emb_pot[1])/2.
                               + (proj_pot[0] + proj_pot[1])/2.)
                        vhf = scf_obj.get_veff(dm=(self.dmat[0] + self.dmat[1]))
                        emb_fock = diis.update(s1e, dm, f, mf, h1e, vhf)

                E, C = scf_obj.eig(emb_proj_fock, scf_obj.get_ovlp())
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
            if self.env_smearsigma > 0.:
                mo_occ[0] = ((env_mo_energy[0] 
                              - fermi[0]) / self.env_smearsigma)
                ie = np.where( mo_occ[0] < 1000 )
                i0 = np.where( mo_occ[0] >= 1000 )
                mo_occ[0][ie] = 1. / ( np.exp( mo_occ[0][ie] ) + 1. )
                mo_occ[0][i0] = 0.

                mo_occ[1] = (env_mo_energy[1] - fermi[1] ) / self.env_smearsigma
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
            self.env_dmat[0] = np.dot((env_mo_coeff[0] * mo_occ[0]), 
                                   env_mo_coeff[0].transpose().conjugate())
            self.env_dmat[1] = np.dot((env_mo_coeff[1] * mo_occ[1]), 
                                   env_mo_coeff[1].transpose().conjugate())
            self.save_chkfile()
            return self.env_dmat


    def get_useable_dmat(self, mat=None):
        """Return a useable density matrix for use with PySCF.
        This is because we always store the density matrix as a
        (2 x nao x nao) matrix."""

        if self.unrestricted:
            return mat 
        if self.cell.spin != 0:
            return mat 
        else:
            return mat[0] + mat[1]


    def update_stored_dmat(self, mat=None):
        """Store the density matrix as a (2 x nao x nao) matrix."""

        import numpy as np

        if mat.ndim == 3:
            return mat 
        else:
            out = np.zeros((2, mat.shape[0], mat.shape[1]), dtype=mat.dtype)
            out[0] = mat / 2.0 
            out[1] = mat / 2.0 
            return out 


    def get_useable_pot(self, mat=None):
        """Return a useable potential for use with PySCF.
        This is because we always store the potentials as a
        (2 x nao x nao) matrix."""

        if mat.ndim == 2:
            return mat
        if self.unrestricted:
            return mat  
        if self.cell.spin != 0:
            return mat  
        else:
            return (mat[0] + mat[1])/2.


    def update_stored_pot(self, mat=None):
        """Store the potential matrix as a (2 x nao x nao) matrix."""

        import numpy as np

        if mat.ndim == 3:
            return mat  
        else:
            out = np.zeros((2, mat.shape[0], mat.shape[1]), dtype=mat.dtype)
            out[0] = mat
            out[1] = mat
            return out


    def print_coordinates(self):
        """Prints the Mole coordinates."""
        bohr_2_angstrom = 0.52917720859
        print (" Atom Coordinates (Angstrom) ".center(80,"-"))
        if hasattr(self, 'cell'):
            mol = self.cell
        else:
            mol = self.mol
        for i in range(mol.natm):
            a = mol.atom_symbol(i)
            c = mol.atom_coord(i) * bohr_2_angstrom
            st = "{0:<10} {1:10.4f} {2:10.4f} {3:10.4f}".format(
                 a, c[0], c[1], c[2])
            print (st)


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

    def __init__(self, mol, env_method, hl_method, hl_order=1, hl_initguess=None, hl_sr_method=None,
                 hl_spin=None, hl_conv=None, hl_grad=None, hl_cycles=None, 
                 hl_damp=0., hl_shift=0., hl_freeze_orbs=None, hl_ext=None, 
                 hl_unrestricted=False, hl_compress_approx=False, 
                 hl_density_fitting=False, hl_save_orbs=False, hl_save_density=False, hl_save_spin_density=False,
                 cas_loc_orbs=False, cas_init_guess=None, cas_active_orbs=None,
                 cas_avas=None, shci_mpi_prefix=None, shci_sweep_iter=None, 
                 shci_sweep_epsilon=None, shci_nPTiter=None, 
                 shci_no_stochastic=False, shci_NoRDM=False, dmrg_maxM=100, 
                 dmrg_num_thrds=1, **kwargs):
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

        self.cas_loc_orbs = cas_loc_orbs
        self.cas_init_guess = cas_init_guess
        self.cas_active_orbs = cas_active_orbs
        self.cas_avas = cas_avas

        self.shci_mpi_prefix = shci_mpi_prefix
        self.shci_sweep_iter = shci_sweep_iter
        self.shci_sweep_epsilon = shci_sweep_epsilon
        self.shci_no_stochastic = shci_no_stochastic
        self.shci_nPTiter = shci_nPTiter
        self.shci_NoRDM = shci_NoRDM

        self.dmrg_maxM = dmrg_maxM
        self.dmrg_num_thrds = dmrg_num_thrds

        self.hl_mo_coeff = None
        self.hl_mo_occ = None
        self.hl_mo_energy = None

        self.hl_dmat = None
        self.hl_sub_nuc_grad = None
        self.hl_sub_emb_nuc_grad = None
        self.hl_sub_proj_nuc_grad = None
 

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


    def get_hl_in_env_energy(self, mol=None, dmat=None, emb_pot=None, 
                             proj_pot=None, hl_method=None, 
                             hl_conv=None, hl_grad=None, 
                             hl_shift=None, hl_damp=None, 
                             hl_cycles=None, hl_initguess=None,
                             hl_frozen=None, hl_sr_method=None):

        if mol is None:
            mol = self.mol
        if dmat is None:
            dmat = self.env_dmat
        if emb_pot is None:
            if self.emb_fock[0] is None:
                emb_pot = [np.zeros_like(dmat[0]), np.zeros_like(dmat[1])]
            else:
                self.update_subsys_fock()
                fock = self.subsys_fock
                emb_pot = (self.emb_fock[0] - fock[0], 
                           self.emb_fock[1] - fock[1])
        if proj_pot is None:
            proj_pot = self.proj_pot 
        if hl_method is None:
            hl_method = self.hl_method
        if hl_conv is None:
            hl_conv = self.hl_conv
        if hl_grad is None:
            hl_grad = self.hl_grad
        if hl_shift is None:
            hl_shift = self.hl_shift
        if hl_damp is None:
            hl_damp = self.hl_damp
        if hl_cycles is None:
            hl_cycles = self.hl_cycles
        if hl_initguess is None:
            hl_initguess = self.hl_initguess
        #if hl_frozen is None:
        #    hl_frozen = self.hl_frozen
        if hl_sr_method is None:
            hl_sr_method = self.hl_sr_method

        hl_energy = 0.0
        #Determine which method to use for the single reference orbitals.
        #If it is None, or hf use hf. Otherwise assume it is a dft functional.
        #If the method is a dft method and the sr is None, do sr as the dft method.
        #If the method is a dft method and the sr is not none, do sr and use the density to seed the method.

        hf_aliases = ['hf', 'uhf', 'rhf', 'rohf']
        cc_aliases = ['ccsd', 'ccsd(t)', 'uccsd', 'uccsd(t)']
        mp_aliases = ['mp2']
        cas_regex = re.compile('cas(pt2)?(\[\d*,\d*\])?')
        fci_aliases = ['fci']
        fcidump_aliases = ['fcidump']
        known_methods = hf_aliases + cc_aliases + mp_aliases + fci_aliases + fcidump_aliases
        if hl_sr_method is None:
            if hl_method not in known_methods:
                if not re.match(cas_regex, hl_method):
                    hl_sr_method = hl_method

        if self.hl_ext is not None:
            print ("use external method for hl calculation")
            hcore = self.env_scf.get_hcore()
            #emb_pot_ro = self.emb_ro_fock[0] - self.fock
            emb_proj_pot = [emb_pot[0] + proj_pot[0], emb_pot[1] + proj_pot[1]]
            ext_factory = ExtFactory()
            name_no_path = os.path.split(self.filename)[-1]
            name_no_ext = os.path.splitext(name_no_path)[0]
            file_path = os.path.split(self.filename)[0]
            scr_path = self.scr_dir
            ext_obj = ext_factory.get_ext_obj(self.hl_ext, gto.copy(mol), hl_method, emb_proj_pot, core_ham=hcore, filename=name_no_ext, work_dir=file_path, scr_dir=scr_path, nproc=self.nproc, pmem=self.pmem, save_orbs=None, save_density=False, cas_settings={'cas_loc_orbs':self.cas_loc_orbs, 'cas_init_guess':self.cas_init_guess, 'cas_active_orbs':self.cas_active_orbs, 'cas_avas':self.cas_avas})
            energy = ext_obj.get_energy()
            print (energy)
            self.hl_energy = energy[0]
            
            #hl_sr_scf = scf.ROHF(gto.mole.copy(mol))
            #hl_sr_scf.get_fock = lambda *args, **kwargs: (
            #    custom_pyscf_methods.rohf_get_fock(hl_sr_scf, 
            #    emb_pot, proj_pot, *args, **kwargs))
            #hl_sr_scf.energy_elec = lambda *args, **kwargs: (
            #    custom_pyscf_methods.rohf_energy_elec(hl_sr_scf, 
            #    emb_pot, proj_pot, *args, **kwargs))
            #temp = np.copy(self.env_dmat)
            #print (hl_sr_scf.kernel(dm0=copy(self.env_dmat)))
            #y = hl_sr_scf.get_fock()
            return self.hl_energy

        else:
            print ('determine whether to use hf or dft for initial guess')
            if (hl_sr_method is None or hl_sr_method in hf_aliases):
                #Use HF for initial guesses
                if self.hl_unrestricted:
                    hl_sr_scf = scf.UHF(mol)
                    #Update the fock and electronic energies to use custom methods.
                    hl_sr_scf.get_fock = lambda *args, **kwargs: (
                        custom_pyscf_methods.uhf_get_fock(hl_sr_scf, 
                        emb_pot, proj_pot, *args, **kwargs))
                    hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                        custom_pyscf_methods.uhf_energy_elec(hl_sr_scf, 
                        emb_pot, proj_pot, *args, **kwargs))
                elif self.mol.spin != 0:
                    if self.flip_ros:
                        mol.spin *= -1
                        mol.build()
                        temp_emb_pot = copy(emb_pot)
                        temp_proj_pot = copy(proj_pot)
                        emb_pot[0] = temp_emb_pot[1]
                        emb_pot[1] = temp_emb_pot[0]
                        proj_pot[0] = temp_proj_pot[1]
                        proj_pot[1] = temp_proj_pot[0]
                    hl_sr_scf = scf.ROHF(mol)
                    #Update the fock and electronic energies to use custom methods.
                    #print (hl_sr_scf.get_hcore())
                    #print (emb_pot[0] + proj_pot[0])
                    #print (emb_pot[1] + proj_pot[1])
                    #print ("POTS")
                    #print (emb_pot)
                    #print (proj_pot)
                    hl_sr_scf.get_fock = lambda *args, **kwargs: (
                        custom_pyscf_methods.rohf_get_fock(hl_sr_scf, 
                        emb_pot, proj_pot, *args, **kwargs))
                    hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                        custom_pyscf_methods.rohf_energy_elec(hl_sr_scf, 
                        emb_pot, proj_pot, *args, **kwargs))
                else:
                    hl_sr_scf = scf.RHF(mol)
                    #Update the fock and electronic energies to use custom methods.
                    hl_sr_scf.get_fock = lambda *args, **kwargs: (
                        custom_pyscf_methods.rhf_get_fock(hl_sr_scf, 
                        (emb_pot[0] + emb_pot[1])/2., (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
                    hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                        custom_pyscf_methods.rhf_energy_elec(hl_sr_scf, 
                        (emb_pot[0] + emb_pot[1])/2., (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))

            else:
                #Use DFT for initial guesses
                if self.hl_unrestricted:
                    hl_sr_scf = scf.UKS(mol)
                    #Update the fock and electronic energies to use custom methods.
                    hl_sr_scf.get_fock = lambda *args, **kwargs: (
                        custom_pyscf_methods.uks_get_fock(hl_sr_scf, 
                        emb_pot, proj_pot, *args, **kwargs))
                    hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                        custom_pyscf_methods.uks_energy_elec(hl_sr_scf, 
                        emb_pot, proj_pot, *args, **kwargs))
                elif self.mol.spin != 0:
                    if self.flip_ros:
                        mol.spin *= -1
                        mol.build()
                        temp_emb_pot = copy(emb_pot)
                        temp_proj_pot = copy(proj_pot)
                        emb_pot[0] = temp_emb_pot[1]
                        emb_pot[1] = temp_emb_pot[0]
                        proj_pot[0] = temp_proj_pot[1]
                        proj_pot[1] = temp_proj_pot[0]
                    hl_sr_scf = scf.ROKS(mol)
                    #Update the fock and electronic energies to use custom methods.
                    hl_sr_scf.get_fock = lambda *args, **kwargs: (
                        custom_pyscf_methods.roks_get_fock(hl_sr_scf, 
                        emb_pot, proj_pot, *args, **kwargs))
                    hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                        custom_pyscf_methods.roks_energy_elec(hl_sr_scf, 
                        emb_pot, proj_pot, *args, **kwargs))
                else:
                    hl_sr_scf = scf.RKS(mol)
                    #Update the fock and electronic energies to use custom methods.
                    hl_sr_scf.get_fock = lambda *args, **kwargs: (
                        custom_pyscf_methods.rks_get_fock(hl_sr_scf, 
                        (emb_pot[0] + emb_pot[1])/2., (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
                    hl_sr_scf.energy_elec = lambda *args, **kwargs: (
                        custom_pyscf_methods.rks_energy_elec(hl_sr_scf, 
                        (emb_pot[0] + emb_pot[1])/2., (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
                #Set grid, rho and xc
                hl_sr_scf.xc = hl_sr_method #Change
                

            #Set parameters
            if hl_conv is not None:
                hl_sr_scf.conv_tol = hl_conv
            if hl_grad is not None:
                hl_sr_scf.conv_tol_grad = hl_grad
            if hl_cycles is not None:
                hl_sr_scf.max_cycle = hl_cycles

            hl_sr_scf.level_shift = hl_shift
            hl_sr_scf.damp = hl_damp


            if hl_initguess == 'ft':
                dmat = self.get_dmat()
            elif hl_initguess is not None:
                dmat = hl_sr_scf.get_init_guess(key=hl_initguess)
            else:
                dmat = hl_sr_scf.get_init_guess()

            hl_energy = hl_sr_scf.kernel(dm0=dmat)
            self.hl_sr_scf = hl_sr_scf

            if hl_method in hf_aliases:
                self.hl_energy = hl_energy
                return hl_energy
                   
            elif hl_method in cc_aliases:

                #If dft for sr method, need to convert to hf.
                if self.hl_unrestricted:
                    hl_cc = cc.UCCSD(hl_sr_scf)#, frozen=hl_frozen)
                elif self.mol.spin != 0:
                    hl_cc = cc.UCCSD(hl_sr_scf)#, frozen=hl_frozen)
                else:
                    hl_cc = cc.CCSD(hl_sr_scf)#, frozen=hl_frozen)

                if hl_conv is not None:
                    hl_cc.conv_tol = hl_conv
                if hl_cycles is not None:
                    hl_cc.max_cycle = hl_cycles
                ecc, t1, t2 = hl_cc.kernel()
                hl_energy += ecc
                if "(t)" in hl_method:
                    eris = hl_cc.ao2mo()
                    if self.hl_unrestricted:
                        ecc_t = uccsd_t.kernel(hl_cc, eris)
                    else:
                        ecc_t = ccsd_t.kernel(hl_cc, eris)
                    hl_energy += ecc_t

                self.hl_energy = hl_energy
                return hl_energy

            elif hl_method in mp_aliases:
                #If dft for sr method, need to convert to hf.
                if self.hl_unrestricted:
                    hl_mp = mp.UMP2(hl_sr_scf)#, frozen=hl_frozen)
                elif self.mol.spin != 0:
                    print ("ro mp2")
                else:
                    hl_mp = mp.MP2(hl_sr_scf)#, frozen=hl_frozen)

                if hl_conv is not None:
                    hl_mp.conv_tol = hl_conv
                if hl_cycles is not None:
                    hl_mp.max_cycle = hl_cycles
                emp, t1 = hl_mp.kernel()
                hl_energy += emp
                self.hl_energy = hl_energy
                return hl_energy

            elif re.match(cas_regex, hl_method):
                active_space = [int(i) for i in (hl_method[hl_method.find("[") + 1:hl_method.find("]")]).split(',')]
                hl_casscf = mcscf.CASSCF(hl_sr_scf, active_space[0], active_space[1])
                if hl_conv is not None:
                    hl_casscf.conv_tol = hl_conv
                if hl_cycles is not None:
                    hl_casscf.max_cycle = hl_cycles

                hl_energy = hl_casscf.kernel()[0]
                self.hl_energy = hl_energy
                return hl_energy

            elif hl_method in fci_aliases:
                mod_hcore = ((self.env_scf.get_hcore() 
                             + (emb_pot[0] + emb_pot[1])/2.
                             + (proj_pot[0] + proj_pot[1])/2.))
                cisolver = fci.FCI(mol, hl_sr_scf.mo_coeff)
                hl_energy_tot = cisolver.kernel(h1e=mod_hcore)
                self.hl_energy = hl_energy_tot[0]
                return hl_energy

            elif hl_method in fcidump_aliases:
                mod_hcore = ((self.env_scf.get_hcore() 
                             + (emb_pot[0] + emb_pot[1])/2.
                             + (proj_pot[0] + proj_pot[1])/2.))
                hl_sr_scf.get_hcore = lambda *args, **kwargs: mod_hcore
                hl_energy = hl_sr_scf.kernel(dm0=(self.get_dmat()))
                fcidump_filename = (os.path.splitext(self.filename)[0] 
                                    + '_' + self.chkfile_index + '_.fcidump')
                print (f"FCIDUMP GENERATED AT {fcidump_filename}")
                tools.fcidump.from_scf(hl_sr_scf, (
                    os.path.splitext(self.filename)[0] + '.fcidump'), 
                    tol=1e-200)

                self.hl_energy = hl_energy
                return hl_energy

            #dft as high level
            #Assume used for the initial starting orbital guess.
            else:
                self.hl_energy = hl_energy
                return hl_energy
                
           
    #def hl_in_env_energy2(self, mol=None, dmat=None, emb_pot=None, 
    #                         proj_pot=None, hl_method=None, 
    #                         hl_conv=None, hl_grad=None, 
    #                         hl_shift=None, hl_damp=None, 
    #                         hl_cycles=None, hl_initguess=None,
    #                         hl_frozen=None):
    #    """Returns the hl subsystem energy as embedded in the full system.

    #    Parameters
    #    ----------
    #    mol : Mole, optional
    #        pyscf Mole object (default is None).
    #    dmat : numpy.float64, optional
    #        Subsystem density matrix (default is None).
    #    env_hcore : numpy.float64, optional
    #        Subsystem core hamiltonian matrix (default is None).
    #    emb_pot : numpy.float64, optional
    #        Subsystem embedding potential matrix (default is None).
    #    proj_pot : numpy.float64, optional
    #        Subsystem projection potential matrix (default is None).
    #    hl_method : str, optional
    #        Active method string (default is None).
    #    """

    #    if mol is None:
    #        mol = self.mol
    #    if dmat is None:
    #        dmat = self.dmat
    #    if emb_pot is None:
    #        if self.emb_fock is None:
    #            emb_pot = [np.zeros_like(dmat[0]), np.zeros_like(dmat[1])]
    #        else:
    #            if self.unrestricted:
    #                self.fock = (self.env_scf.get_hcore() 
    #                             + self.env_scf.get_veff(dm=dmat))
    #                fock = self.fock
    #                emb_pot = (self.emb_fock[0] - fock[0], 
    #                           self.emb_fock[1] - fock[1])
    #            elif mol.spin != 0:
    #                #RO
    #                pass
    #            else:
    #                self.fock = (self.env_scf.get_hcore() 
    #                    + self.env_scf.get_veff(dm=(dmat[0] + dmat[1])))
    #                fock = self.fock
    #                emb_pot = (self.emb_fock[0] - fock, 
    #                           self.emb_fock[1] - fock)
    #    if proj_pot is None:
    #        proj_pot = self.proj_pot 
    #    if hl_method is None:
    #        hl_method = self.hl_method
    #    if hl_conv is None:
    #        hl_conv = self.hl_conv
    #    if hl_grad is None:
    #        hl_grad = self.hl_grad
    #    if hl_shift is None:
    #        hl_shift = self.hl_shift
    #    if hl_damp is None:
    #        hl_damp = self.hl_damp
    #    if hl_cycles is None:
    #        hl_cycles = self.hl_cycles
    #    if hl_initguess is None:
    #        hl_initguess = self.hl_initguess
    #    #if hl_frozen is None:
    #    #    hl_frozen = self.hl_frozen

    #    hl_energy = 0.0
    #    if self.hl_ext is not None:
    #        mod_hcore = (self.env_scf.get_hcore() 
    #                     + ((emb_pot[0] + emb_pot[1])/2.
    #                     + (proj_pot[0] + proj_pot[1])/2.))
    #        if hl_method[0] == 'r':
    #            hl_method = hl_method[1:]
    #        if hl_method == 'hf':
    #            energy = molpro_calc.molpro_energy(
    #                      mol, mod_hcore, hl_method, self.filename, 
    #                      self.hl_save_orbs, scr_dir=self.scr_dir, 
    #                      nproc=self.nproc, pmem=self.pmem)
    #            hl_energy = energy[0]
    #        elif hl_method == 'ccsd' or hl_method == 'ccsd(t)':
    #            energy = molpro_calc.molpro_energy(
    #                      mol, mod_hcore, hl_method, self.filename, 
    #                      self.hl_save_orbs, scr_dir=self.scr_dir, 
    #                      nproc=self.nproc, pmem=self.pmem)
    #            hl_energy = energy[0]
    #        elif re.match(re.compile('cas(pt2)?\[.*\].*'), hl_method):
    #            energy = molpro_calc.molpro_energy(
    #                      mol, mod_hcore, hl_method, self.filename, 
    #                      self.hl_save_orbs, self.active_orbs, self.avas,
    #                      self.localize_orbitals, scr_dir=self.scr_dir, 
    #                      nproc=self.nproc, pmem=self.pmem)
    #            hl_energy = energy[0]
    #        elif hl_method == 'fcidump':
    #            energy = molpro_calc.molpro_energy(
    #                      mol, mod_hcore, active_method, self.filename, 
    #                      self.active_save_orbs, scr_dir=self.scr_dir, 
    #                      nproc=self.nproc, pmem=self.pmem)
    #            hl_energy = energy[0]
    #        else:
    #            pass

    #    else:
    #        if self.hl_unrestricted: 
    #            if (hl_method == 'hf' or hl_method == 'ccsd' or 
    #                hl_method == 'uccsd' or hl_method == 'ccsd(t)' or 
    #                hl_method == 'uccsd(t)' or 
    #                re.match(re.compile('cas(pt2)?\[.*\].*'), 
    #                                       hl_method) or
    #                re.match(re.compile('shci(scf)?\[.*\].*'), 
    #                                       hl_method) or
    #                re.match(re.compile('dmrg\[.*\].*'), 
    #                                       hl_method)):
    #                hl_scf = scf.UHF(mol)
    #                hl_scf.conv_tol = hl_conv
    #                hl_scf.conv_tol_grad = hl_grad
    #                hl_scf.max_cycle = hl_cycles
    #                hl_scf.level_shift = hl_shift
    #                hl_scf.damp = hl_damp
    #                hl_scf.get_fock = lambda *args, **kwargs: (
    #                    custom_pyscf_methods.uhf_get_fock(hl_scf, 
    #                    emb_pot, proj_pot, *args, **kwargs))
    #                hl_scf.energy_elec = lambda *args, **kwargs: (
    #                    custom_pyscf_methods.uhf_energy_elec(hl_scf, 
    #                    emb_pot, proj_pot, *args, **kwargs))
    #                if hl_initguess != 'ft' and hl_initguess is not None:
    #                    dmat = hl_scf.get_init_guess(key=hl_initguess)
    #                else:
    #                    dmat = hl_scf.get_init_guess()

    #                hl_energy = hl_scf.kernel(dm0=dmat)

    #                self.hl_scf = hl_scf
    #                if 'ccsd' in hl_method:
    #                    hl_cc = cc.UCCSD(hl_scf, frozen=hl_frozen)
    #                    hl_cc.max_cycle = hl_cycles
    #                    hl_cc.conv_tol = hl_conv
    #                    #active_cc.conv_tol_normt = 1e-6
    #                    ecc, t1, t2 = hl_cc.kernel()
    #                    hl_energy += ecc
    #                    if (hl_method == 'ccsd(t)' 
    #                        or hl_method == 'uccsd(t)'):
    #                        eris = hl_cc.ao2mo(hl_scf.mo_coeff)
    #                        ecc_t = uccsd_t.kernel(hl_cc, eris)
    #                        hl_energy += ecc_t

    #                #For the following two not sure how to include the separate potential for alpha and beta.
    #                elif re.match(re.compile('shci(scf)?\[.*\].*'), 
    #                                           hl_method):
    #                    from pyscf.future.shciscf import shci
    #                    mod_hcore = ((self.env_scf.get_hcore() 
    #                             + (emb_pot[0] + emb_pot[1]) /2.
    #                             + (proj_pot[0] + proj_pot[1])/2.))
    #                    hl_scf.get_hcore = lambda *args, **kwargs: mod_hcore
    #                    active_space = [int(i) for i in (method[method.find("[") + 1:method.find("]")]).split(',')]
    #                    hl_shci = shci.SHCISCF(hl_scf, active_space[0], active_space[1])
    #                    hl_shci.fcisolver.mpiprefix = self.shci_mpi_prefix
    #                    hl_shci.fcisolver.stochastic = self.shci_stochastic
    #                    hl_shci.fcisolver.nPTiter = self.shci_nPTiter
    #                    hl_shci.fcisolver.sweep_iter = self.shci_sweep_iter
    #                    hl_shci.fcisolver.DoRDM = self.shci_DoRDM
    #                    hl_shci.fcisolver.sweep_epsilon = self.shci_sweep_epsilon
    #                    ecc = hl_shci.mc1step()[0]

    #                elif re.match(re.compile('dmrg\[.*\].*'), 
    #                                           hl_method):
    #                    from pyscf import dmrgscf
    #                    mod_hcore = ((self.env_scf.get_hcore() 
    #                             + (emb_pot[0] + emb_pot[1])/2.
    #                             + (proj_pot[0] + proj_pot[1])/2.))
    #                    hl_scf.get_hcore = lambda *args, **kwargs: mod_hcore
    #                    active_space = [int(i) for i in (method[method.find("[") + 1:method.find("]")]).split(',')]
    #                    hl_dmrg = dmrgscf.DMRGSCF(hl_scf, active_space[0], active_space[1])
    #                    if self.dmrg_memory is None:
    #                        dmrg_mem = self.pmem 
    #                    else: 
    #                        dmrg_mem = self.dmrg_memory 
    #                    if dmrg_memory is not None:
    #                        dmrg_memory = float(dmrg_memory) / 1e3 #DMRG Input memory is in GB for some reason.
    #                    hl_dmrg.fcisolver = dmrgscf.DMRGCI(self.mol, maxM=self.dmrg_maxM, memory=dmrg_mem)
    #                    hl_dmrg.fcisolver.num_thrds = self.dmrg_numthrds
    #                    hl_dmrg.fcisolver.scratchDirectory = self.scr_dir
    #                    edmrg = hl_dmrg.kernel()
    #                    if "nevpt" in hl_method:
    #                        from pyscf import mrpt
    #                        if self.compress_approx:
    #                            enevpt = mrpt.NEVPT(hl_dmrg).compress_approx().kernel()
    #                        else:
    #                            enevpt = mrpt.NEVPT(hl_dmrg).kernel()
    #                    

    #                elif re.match(re.compile('cas(pt2)?\[.*\].*'), 
    #                                         hl_method):
    #                    hl_space = [int(i) for i in (method[method.find("[") + 1:method.find("]")]).split(',')]
    #            elif hl_method[1:] == 'fci':
    #                pass
    #            else: 
    #                pass
    #        elif mol.spin != 0: 
    #            #RO
    #            pass
    #        elif hl_method == 'fcidump':
    #            hl_scf = scf.RHF(mol)
    #            hl_scf.conv_tol = hl_conv
    #            hl_scf.conv_tol_grad = hl_grad
    #            hl_scf.max_cycle = hl_cycles
    #            hl_scf.level_shift = hl_shift
    #            hl_scf.damp = hl_damp
    #            mod_hcore = ((self.env_scf.get_hcore() 
    #                         + (emb_pot[0] + emb_pot[1])/2.
    #                         + (proj_pot[0] + proj_pot[1])/2.))
    #            hl_scf.get_hcore = lambda *args, **kwargs: mod_hcore
    #            hl_energy = hl_scf.kernel(dm0=(dmat[0] + dmat[1]))
    #            fcidump_filename = (os.path.splitext(self.filename)[0] 
    #                                + '.fcidump')
    #            print (f"FCIDUMP GENERATED AT {fcidump_filename}")
    #            tools.fcidump.from_scf(hl_scf, (
    #                os.path.splitext(self.filename)[0] + '.fcidump'), 
    #                tol=1e-200)
    #        else: 
    #            if hl_method[0] == 'r':
    #                hl_method = hl_method[1:]
    #            if (hl_method == 'hf' or hl_method == 'ccsd' or 
    #                hl_method == 'ccsd(t)' or  
    #                re.match(re.compile('cas(pt2)?\[.*\].*'), 
    #                                       hl_method) or
    #                re.match(re.compile('shci(scf)?\[.*\].*'), 
    #                                       hl_method) or
    #                re.match(re.compile('dmrg\[.*\].*'), 
    #                                       hl_method)):
    #                hl_scf = scf.RHF(mol)
    #                hl_scf.conv_tol = hl_conv
    #                hl_scf.conv_tol_grad = hl_grad
    #                hl_scf.max_cycle = hl_cycles
    #                hl_scf.level_shift = hl_shift
    #                hl_scf.damp = hl_damp
    #                hl_scf.get_fock = lambda *args, **kwargs: (
    #                    custom_pyscf_methods.rhf_get_fock(hl_scf, 
    #                    (emb_pot[0] + emb_pot[1])/2.,
    #                    (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
    #                hl_scf.energy_elec = lambda *args, **kwargs: (
    #                    custom_pyscf_methods.rhf_energy_elec(hl_scf, 
    #                    (emb_pot[0] + emb_pot[1])/2., 
    #                    (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
    #                if hl_initguess == 'ft':
    #                    init_dmat = dmat[0] + dmat[1]
    #                elif hl_initguess is not None:
    #                    init_dmat = hl_scf.get_init_guess(key=hl_initguess)
    #                else:
    #                    init_dmat = hl_scf.get_init_guess()
    #                hl_energy = hl_scf.kernel(dm0=init_dmat)
    #                self.hl_scf = hl_scf
    #                if hl_method == 'hf':
    #                    # this slows down execution.
    #                    self.hl_dmat = self.hl_scf.make_rdm1()
    #                if 'ccsd' in hl_method:
    #                    hl_cc = cc.CCSD(hl_scf, frozen=hl_frozen)
    #                    hl_cc.max_cycle = hl_cycles
    #                    hl_cc.conv_tol = hl_conv
    #                    #hl_cc.conv_tol_normt = 1e-6
    #                    ecc, t1, t2 = hl_cc.kernel()
    #                    eris = hl_cc.ao2mo()
    #                    hl_energy += ecc
    #                    self.hl_scf = hl_cc
    #                    if hl_method == 'ccsd':
    #                        # this slows down execution.
    #                        self.hl_dmat = self.hl_scf.make_rdm1()
    #                    if hl_method == 'ccsd(t)':
    #                        ecc_t = ccsd_t.kernel(hl_cc, eris)
    #                        hl_energy += ecc_t
    #                        #l1, l2 = ccsd_t_lambda_slow.kernel(self.hl_cc, new_eris, t1, t2,)[1:]
    #                        # this slows down execution.
    #                        self.hl_dmat = ccsd_t_rdm_slow.make_rdm1(self.hl_cc, t1, t2, l1, l2, eris=new_eris)
    #                    else:
    #                        pass
    #                        # this slows down execution.
    #                        #self.hl_dmat = self.hl_cc.make_rdm1()

    #                    # Convert to AO form
    #                    #temp_dmat = copy(self.hl_dmat)
    #                    #ao_dmat = reduce (np.dot, (self.hl_cc.mo_coeff, np.dot(temp_dmat, self.hl_cc.mo_coeff.T)))
    #                    #print ("DMATS")
    #                    #print (self.hl_dmat)
    #                    #print (ao_dmat)
    #                    #print (self.dmat[0] + self.dmat[1])
    #                    #self.hl_dmat = ao_dmat

    #                elif re.match(re.compile('cas(pt2)?\[.*\].*'), 
    #                                         hl_method):
    #                    active_space = [int(i) for i in (hl_method[hl_method.find("[") + 1:hl_method.find("]")]).split(',')]
    #                    hl_cc_scf = mcscf.CASSCF(hl_scf, active_space[0], active_space[1])
    #                    hl_cc_scf.kernel()
    #                    if self.hl_save_orbs:
    #                        self.save_orbitals(hl_cc_scf, hl_scf.mo_occ)

    #                elif re.match(re.compile('shci(scf)?\[.*\].*'), 
    #                                                   hl_method):
    #                    from pyscf.future.shciscf import shci
    #                    mod_hcore = (self.env_scf.get_hcore() 
    #                         + ((emb_pot[0] + emb_pot[1])/2.
    #                         + (proj_pot[0] + proj_pot[1])/2.))
    #                    active_space = [int(i) for i in (hl_method[hl_method.find("[") + 1:hl_method.find("]")]).split(',')]
    #                    hl_shci = shci.SHCISCF(hl_scf, active_space[0], active_space[1])
    #                    hl_shci.fcisolver.mpiprefix = self.shci_mpi_prefix
    #                    hl_shci.fcisolver.stochastic = self.shci_stochastic
    #                    hl_shci.fcisolver.nPTiter = self.shci_nPTiter
    #                    hl_shci.fcisolver.sweep_iter = self.shci_sweep_iter
    #                    hl_shci.fcisolver.DoRDM = self.shci_DoRDM
    #                    hl_shci.fcisolver.sweep_epsilon = self.shci_sweep_epsilon
    #                    ecc = hl_shci.mc1step()[0]

    #                elif re.match(re.compile('dmrg\[.*\].*'), 
    #                                                   hl_method):
    #                    from pyscf import dmrgscf
    #                    mod_hcore = (self.env_scf.get_hcore() 
    #                         + ((emb_pot[0] + emb_pot[1])/2.
    #                         + (proj_pot[0] + proj_pot[1])/2.))
    #                    hl_scf.get_hcore = lambda *args, **kwargs: mod_hcore
    #                    active_space = [int(i) for i in (hl_method[hl_method.find("[") + 1:hl_method.find("]")]).split(',')]
    #                    hl_dmrg = dmrgscf.DMRGSCF(hl_scf, active_space[0], active_space[1])
    #                    if self.dmrg_memory is None:
    #                        dmrg_mem = self.pmem
    #                    else: 
    #                        dmrg_mem = self.dmrg_memory
    #                    if dmrg_memory is not None:
    #                        dmrg_memory = float(dmrg_memory) / 1e3 #DMRG Input memory is in GB for some reason.
    #                    hl_dmrg.fcisolver = dmrgscf.DMRGCI(self.mol, maxM=self.dmrg_maxM, memory=dmrg_mem)
    #                    hl_dmrg.fcisolver.num_thrds = self.dmrg_numthrds
    #                    hl_dmrg.fcisolver.scratchDirectory = self.scr_dir
    #                    edmrg = hl_dmrg.kernel()
    #                    if "nevpt" in hl_method:
    #                        from pyscf import mrpt
    #                        if self.compress_approx:
    #                            enevpt = mrpt.NEVPT(hl_dmrg).compress_approx().kernel()
    #                        else:
    #                            enevpt = mrpt.NEVPT(hl_dmrg).kernel()

    #            else: #DFT
    #                hl_scf = scf.RKS(mol)
    #                hl_scf.xc = hl_method
    #                hl_scf.grids = self.env_scf.grids
    #                hl_scf.small_rho_cutoff = self.rho_cutoff
    #                hl_scf.conv_tol = hl_conv
    #                hl_scf.conv_tol_grad = hl_grad
    #                hl_scf.max_cycle = hl_cycles
    #                hl_scf.level_shift = hl_shift
    #                hl_scf.damp = hl_damp
    #                #hl_scf.get_hcore = lambda *args, **kwargs: env_hcore
    #                hl_scf.get_fock = lambda *args, **kwargs: (
    #                    custom_pyscf_methods.rks_get_fock(hl_scf, 
    #                    (emb_pot[0] + emb_pot[1])/2.,
    #                    (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
    #                hl_scf.energy_elec = lambda *args, **kwargs: (
    #                    custom_pyscf_methods.rks_energy_elec(hl_scf, 
    #                    (emb_pot[0] + emb_pot[1])/2., 
    #                    (proj_pot[0] + proj_pot[1])/2., *args, **kwargs))
    #                hl_energy = hl_scf.kernel(dm0=(dmat[0] + dmat[1]))
    #                #Slows down execution
    #                self.hl_scf = hl_scf
    #                self.hl_dmat = self.hl_scf.make_rdm1()

    #            temp_dmat = copy(self.hl_dmat)
    #            self.hl_dmat = [temp_dmat/2., temp_dmat/2.]
    #    self.hl_energy = hl_energy 

    #    if self.hl_save_density:
    #        pass
    #    if self.hl_save_orbs:
    #        pass
    #    return self.hl_energy

    def get_hl_nuc_grad(self, mol=None, scf_obj=None):
        if mol is None:
            mol = self.mol
        if scf_obj is None:
            scf_obj = self.hl_scf

        grad_obj = scf_obj.nuc_grad_method() 
        self.active_sub_nuc_grad = grad_obj.grad_elec(scf_obj.mo_energy, scf_obj.mo_coeff, scf_obj.mo_occ)
        return self.active_sub_nuc_grad

    def save_hl_density_file(self, filename=None, density=None):
        from pyscf.tools import cubegen
        if filename is None:
            if self.filename is None:
                print ("Cannot save hl density because no filename")
                return False
            else:
                filename = self.filename
        if density is None:
            density = self.get_dmat()
        cubegen_fn = os.path.splitext(filename)[0] + '_hl.cube'
        cubegen.density(self.mol, cubegen_fn, density)

    def save_hl_orbital_file(self, filename=None, scf_obj=None, mo_occ=None, mo_coeff=None, mo_energy=None):
        from pyscf.tools import molden
        if filename is None:
            if self.filename is None:
                print ("Cannot save hl orbitals because no filename")
                return False
            else:
                filename = self.filename
        if scf_obj is None:
            scf_obj = self.env_scf
        if mo_occ is None:
            mo_occ = self.env_mo_occ
        if mo_coeff is None:
            mo_coeff = self.env_mo_coeff
        if mo_energy is None:
            mo_energy = self.env_mo_energy
        molden_fn = os.path.splitext(filename)[0] + '_hl.molden'
        with open(molden_fn, 'w') as fin:
            molden.header(scf_obj.mol, fin)
            molden.orbital_coeff(self.mol, fin, mo_coeff, ene=mo_energy, occ=mo_occ)
        molden.from_mo(scf_obj.mol, molden_fn, mo_coeff, ene=mo_energy, occ=mo_occ)
        pass
         

 
class ClusterExcitedSubSystem(ClusterHLSubSystem):

    def __init__(self):
        super().__init__()

