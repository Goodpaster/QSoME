#Module to define periodic subsystems
# Dhabih V. Chulhai

from qsome import cluster_subsystem, subsystem

class PeriodicEnvSubSystem(cluster_subsystem.ClusterEnvSubSystem):
    """
    A base subsystem object for use in periodic projection embedding.
    """

    def __init__(self, cell, env_method, kpoints=None, filename=None, smearsigma=0.,
                 damp=0., shift=0., subcycles=1, diis=0, freeze=False,
                 unrestricted=False, initguess=None, grid_level=3, rhocutoff=1e-7,
                 verbose=3, analysis=False, debug=False,
                 density_fit='df', auxbasis='weigend',
                 save_orbs=False, save_density=False, **kwargs):
        """
        Initializes the periodic subsystem.

        Parameters
        ----------
            cell : pyscf:pbc:gto:Cell object
                The PySCF Cell object with the geomoetry and basis.
            env_method : str
                Defines the method to use for the environment calculations.
            kpoints : ndarray
                kpoints to use in the periodic calculations
            filename : str, optional
                The path to the input file being read. (default is None)
            unrestricted : bool
                Whether to use an unrestricted SCF code. (default is False)
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
                Specifies pyscf Grids size. (default is 3)
            rhocutoff : float, optional
                Numerical integration rho cutoff. (default is 1e-7) 
            verbose : int, optional
                Specifies level of verbose output. (default is 3)
            analysis : bool, optional
                Analysis flag. (default is False)
            debug : bool, optional
                Debug flag. (default is False)
            densit_fit : str, optional
                Density fit method (default is `fftdf` - fast fourier transform).
            auxbasis : str, optional
                Auxillary basis for density fitting (default is `weigend`).
            save_orbs : bool, optional
                Whether to save orbital coefficients. (default is False)
            save_density : bool, optional
                Whether to save density matrix. (default is False)
        """

        import numpy as np
        import os
        from copy import deepcopy as copy

        self.cell = cell
        self.env_method = env_method

        self.initguess = initguess
        self.smearsigma = smearsigma
        self.rho_cutoff = rhocutoff
        self.grid_level = grid_level
        self.damp = damp
        self.shift = shift
        self.density_fit = density_fit
        self.auxbasis = auxbasis
        self.unrestricted = unrestricted

        self.freeze = freeze
        self.subcycles = subcycles

        self.verbose = verbose
        self.analysis = analysis
        self.debug = debug
        self.save_orbs = save_orbs
        self.save_density = save_density

        if filename is None:
            filename = os.getcwd() + '/temp.inp'
        self.filename = filename

        self.kpts = kpoints
        self.nkpts = len(kpoints)
        self.nao = cell.nao_nr()

        self.env_scf = self.init_env_scf()
        self.env_hcore = self.env_scf.get_hcore()
        self.dmat = self.init_density()

        self.smat = self.env_scf.get_ovlp()
        self.emb_pot = np.zeros((2, self.nkpts, self.nao, self.nao), dtype=self.env_hcore.dtype)
        self.proj_pot = np.zeros_like(self.emb_pot)
        self.fock = np.zeros_like(self.emb_pot)
        self.emb_fock = None

        self.env_mo_coeff = None
        self.env_mo_occ = None
        self.env_mo_energy = None
        self.env_energy = 0.0

        # other unsed attributes
        self.env_sub_nuc_grad = None 
        self.env_sub_emb_nuc_grad = None 
        self.env_sub_proj_nuc_grad = None 
        self.env_hcore_deriv = None 
        self.env_vhf_deriv = None

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

        self.env_scf = self.init_env_scf()

        print (" Subsystem ".center(80,"="))
        self.print_coordinates()


    def print_lattice_vectors(self):
        """Prints the Cell lattice parameters."""
        bohr_2_angstrom = 0.52917720859
        print (" Lattice Vectors (Angstrom) ".center(80,"-"))
        l = self.cell.lattice_vectors() * bohr_2_angstrom
        for ai in range(3):
            st = ("a{0:<10} {1:10.4f} {2:10.4f} {3:10.4f}".format(ai+1,
                  l[ai,0], l[ai,1], l[ai,2]))
            print (st)

    def init_env_scf(self, cell=None, env_method=None, kpts=None,
                     rho_cutoff=None, verbose=None, damp=None, shift=None,
                     grid_level=None, density_fit=None, auxbasis=None):
        """
        Initializes the environment pyscf:pbc:scf object.

        Paramters
        ---------
            cell : pyscf:pbc:Cell object, optional
            env_method : str, optional
                Subsystem method for calculation (default is None).
            kpts : ndarray, optional
                Kpoints (in inv. bohr) to use (default is None).
            rho_cutoff : float, optional
                DFT density/rho cutoff parameter (default is None).
            verbose : int, optional
                Verbosity parameter (default is None).
            damp : float, optional
                Damping parameter (default is None).
            shift : float, optional
                level shift parameter for convergence (default is None).
            grid_level : int, optional
                Grid_level for DFT and Coulomb integration (default is None).
            density_fit : str, optional
                The default desnity fit method to use (default is None).
            auxbasis : str, optional
                Auxillary basis for density fitting (default is None).

        Returns
        -------
            cSCF : pyscf.pbc.SCF object
        """

        import numpy as np
        from pyscf.pbc import dft as pbcdft, scf as pbcscf, df as pbcdf

        if cell is None: cell = self.cell
        if env_method is None: env_method = self.env_method.lower()
        if kpts is None: kpts = self.kpts
        if rho_cutoff is None: rho_cutoff = self.rho_cutoff
        if verbose is None: verbose = self.verbose
        if damp is None: damp = self.damp
        if shift is None: shift = self.shift
        if grid_level is None: grid_level = self.grid_level
        if density_fit is None: density_fit = self.density_fit
        density_fit = density_fit.lower()
        if auxbasis is None: auxbasis = self.auxbasis

        if self.unrestricted:
            if env_method in ('hf', 'hartree-fock'):
                cSCF = pbcscf.KUHF(cell, kpts)
            else:
                cSCF = pbcscf.KUKS(cell, kpts)
                cSCF.xc = env_method

        elif cell.spin != 0:
            if env_method in ('hf', 'hartree-fock'):
                cSCF = pbcscf.KROHF(cell, kpts)
            else:
                cSCF = pbcscf.KROKS(cell, kpts)
                cSCF.xc = env_method

        else:
            if env_method in ('hf', 'hartree-fock'):
                cSCF = pbcscf.KRHF(cell, kpts)
            else:
                cSCF = pbcscf.KRKS(cell, kpts)
                cSCF.xc = env_method

        cSCF.kpts = kpts
        cSCF.verbose = verbose

        # grids should probably be for the supersystem
        # Keep this for now, although its probably expensive
        # to calculate the grids for each subsystem
        cSCF.grids = pbcdft.gen_grid.BeckeGrids(cell)
        cSCF.grids.level = grid_level
        cSCF.grids.build()

        # density fitting (can be very time consuming)
        if density_fit == 'fftdf':
            DensityFit = pbcdf.FFTDF
        elif density_fit == 'mdf':
            DensityFit = pbcdf.MDF
        elif density_fit == 'pwdf':
            DensityFit = pbcdf.PWDF
        elif density_fit == 'gdf':
            DensityFit = pbcdf.GDF
        elif density_fit == 'aftdf':
            DensityFit = pbcdf.AFTDF
        else:
            DensityFit = pbcdf.DF

        cSCF.with_df = DensityFit(cell)
        cSCF.with_df.kpts = kpts
        cSCF.with_df.auxbasis = auxbasis

#        # only for local and semi-local functional for now
#        cSCF.with_df.build(j_only=True)

        return cSCF


    def init_density(self, in_dmat=None, scf_obj=None, env_method=None,
        kpts=None):
        """
        Initializes the periodic subsystem density

        Parameters
        ----------
            in_dmat : ndarray, optional
                New subsystem density matrix (default is None).
            scf_obj : pyscf:pbc:scf object, optional
                The PySCF subsystem object (default is None).
            env_method : str, optional
                Subsystem environment energy method (default is None).
            kpts : ndarray, optional
                Kpoints to use in calculation (default is None).

        Returns
        -------
            dmat : ndarray
                The new / guessed density matrix
        """

        import numpy as np
        import pyscf
        from pyscf.scf import RKS, RHF, ROKS, ROHF, UKS, UHF

        if in_dmat is not None:
            return in_dmat

        if scf_obj is None: scf_obj = self.env_scf
        if env_method is None: env_method = self.env_method
        if kpts is None: kpts = self.kpts

        nkpts = len(kpts)
        nao = scf_obj.cell.nao_nr()

        dmat = np.zeros((2,nkpts,nao,nao))

        mol = scf_obj.cell.to_mol()
        mol.verbose = 0

        if self.unrestricted:
            if env_method in ('hf', 'hartree-fock'):
                mf = UHF(mol)
            else:
                mf = UKS(mol)
                mf.xc = env_method
            mf.kernel()
            dtemp = mf.make_rdm1()

        elif mol.spin != 0:
            if env_method in ('hf', 'hartree-fock'):
                mf = ROHF(mol)
            else:
                mf = ROKS(mol)
                mf.xc = env_method
            mf.kernel()
            dtemp = mf.make_rdm1()

        else:
            if env_method in ('hf', 'hartree-fock'):
                mf = RHF(mol)
            else:
                mf = RKS(mol)
                mf.xc = env_method
            mf.kernel()
            dtemp = mf.make_rdm1() / 2.0
            dtemp = [dtemp, dtemp]

        for k in range(nkpts):
            dmat[0,k] = dtemp[0]
            dmat[1,k] = dtemp[0]

        return dmat

    def diagonalize(self, scf_obj=None, subcycles=None, env_method=None,
                    fock=None, env_hcore=None, emb_pot=None, proj_pot=None,
                    dmat=None, diis=None, kpts=None, smearsigma=None):
        """
        Diagonalizes the subsystem fock matrix and returns updated density.

        Parameters
        ----------
            scf_obj : pyscf:pbc:SCF object, optional
            subcycles : int, optional
                Number of diagonalization subcycles (default is None).
            env_method : str, optional
                Which method used to calculate properties
            fock : np.ndarray, optional
                Subsystem fock matrix (default is None).
            env_hcore : np.ndarray, optional
                The environment core hamiltonian (default is None).
            emb_pot : np.ndarray, optional
                The embedding potential (default is None).
            proj_pot : np.ndarray, optional
                The projection operator / potential (default is None).
            dmat : np.ndarray, optional
                Density matrix of subsystem (default is None).
            diis : pyscf:DIIS or int, optional
                Which DIIS to use for diagonalization.
                A negative value turns off DIIS (default is None).
            kpts : ndarray, optional
                The kpoints to use.
            smearsigma : float, optional
                Smears the electrons over orbitals.

        Returns
        -------
            dnew : np.ndarray
                New / updated density matrix.
        """

        import numpy as np
        import scipy as sp

        if scf_obj is None: scf_obj = self.env_scf
        if subcycles is None: subcycles = self.subcycles
        if env_method is None: env_method = self.env_method
        if fock is None:
            if self.emb_fock is None:
                self.fock
            else:
                fock = self.emb_fock
        if env_hcore is None: env_hcore = self.env_hcore
        if emb_pot is None:
            emb_pot = self.emb_pot
        if proj_pot is None: proj_pot = self.proj_pot
        if dmat is None: dmat = self.dmat
        if diis is None: diis = self.diis
        if kpts is None: kpts = self.kpts
        if smearsigma is None: smearsigma = self.smearsigma

        nkpts = len(kpts)
        nao = self.cell.nao_nr()
        nelectron = self.cell.nelectron
        smat = self.smat

        # add projection potential
        fock += proj_pot

        # do diagaonlization for each kpoint
        fock = self.get_useable_pot(fock)
        mo_energy, mo_coeff = scf_obj.eig(fock, smat)
        mo_energy = np.array(mo_energy)
        mo_coeff = np.array(mo_coeff)

        # get fermi energy (average of HOMO and LUMO)
        norbs = ( nelectron * nkpts ) // 2
        e_sorted = np.sort(mo_energy.ravel())
        fermi = ( e_sorted[norbs] + e_sorted[norbs-1] ) / 2.0
        eorder = np.argsort(mo_energy, axis=None)

        # print warning for band gap
        if smearsigma is None:
            bandgap = abs(e_sorted[norbs] - e_sorted[norbs-1])
            if bandgap <= 0.001:
                print (f' WARNING: Small band gap: {bandgap:12.6f} a.u. '.center(80,'!'))
    
        # get molecular occupation
        if smearsigma is None:
            mo_occ = np.zeros_like(mo_energy)
            mo_occ[mo_energy<fermi] = 2.
        else:
            mo_occ = ( mo_energy - fermi ) / smearsigma
            ie = np.where( mo_occ < 1000 )
            i0 = np.where( mo_occ >= 1000 )
            mo_occ[ie] = 2. / ( np.exp( mo_occ[ie] ) + 1. )
            mo_occ[i0] = 0.
    
#            # print occupation
#            print ('MO occupancy:')
#            for i in range(max(0,norbs-4),min(norbs+4,nao*nkpts)):
#                print ('{0:>3d}     {1:12.6f}     {2:12.6f}'.format(
#                       i+1, mo_energy.flatten()[eorder][i],
#                       mo_occ.flatten()[eorder][i]))
    
        # get density matrix
        dmat = np.zeros((nkpts, nao, nao), dtype=smat.dtype)
        for k in range(nkpts):
            dmat[k] = np.dot(mo_coeff[k] * mo_occ[k], mo_coeff[k].transpose().conjugate())
   
        self.dmat = self.update_stored_dmat(dmat)
        self.env_fermi = fermi
        self.env_mo_coeff = mo_coeff
        self.env_mo_occ = mo_occ
        self.env_mo_energy = mo_energy
        return self.dmat


    def get_useable_pot(self, mat=None):
        """Return a useable potential for use with PySCF.
        This is because we always store the potentials as a
        (2 x nkpts x nao x nao) matrix."""

        if mat.ndim == 3:
            return mat
        if self.unrestricted:
            return mat  
        if self.cell.spin != 0:
            return mat  
        else:
            return (mat[0] + mat[1])/2.


    def update_stored_dmat(self, mat=None):
        """Store the mat as a (2 x nkpts x nao x nao) matrix."""

        import numpy as np

        if mat.ndim == 4:
            return mat
        else:
            out = np.zeros((2, mat.shape[0], mat.shape[1], mat.shape[2]),
                             dtype=mat.dtype)
            out[0] = mat / 2.0 
            out[1] = mat / 2.0 
            return out


    def update_stored_pot(self, mat=None):
        """Store the mat as a (2 x nkpts x nao x nao) matrix."""

        import numpy as np

        if mat.ndim == 4:
            return mat 
        else:
            out = np.zeros((2, mat.shape[0], mat.shape[1], mat.shape[2]),
                             dtype=mat.dtype)
            out[0] = mat
            out[1] = mat
            return out


def InitKpoints(cell=None, kpoints=None, kgroup=None):#, **kwargs):
    """
    Initializes the kpoints.

    Paratmers
    ---------
        cell : pyscf:pbc:Cell object
        kpoints : (n,) ndarray
            Number of kpoints in each periodic dimension.
        kgroup : (n,3) ndarray
            Array of kpoints to use.

        (kpoints and kgroup are mutually exclusive)

    Returns
    -------
        kpts : (n,3) ndarray
            The kpoints to use (in inv. bohr)
        nkpts : int
            The number of k-points generated
    """

    import numpy as np

    assert not (kpoints is None and kgroup is None), \
        'Must give one of `kpoints` or `kgroup`!'
    assert not ( kpoints is not None and kgroup is not None ), \
        '`kpoints` and `kgroup` are mutually exclusive!'

    if kpoints is not None:
        kpts = cell.make_kpts(kpoints)
    if kgroup is not None:
        kpts = cell.get_abs_kpts(kgroup)

    return kpts, len(kpts)


class FiniteSubSystem(cluster_subsystem.ClusterActiveSubSystem):
    """
    Extends the PeriodicEnvSubSystem class to calculate higher
    level methods
    """

    def __init__(self, periodic_env_subsystem, active_method, **active_kwargs):
        """
        Creates a new class to hold just the necessary finite properties
        """

        import numpy as np

        self.active_method = active_method

        self.periodic = periodic_env_subsystem
        self.mol = self.periodic.cell.to_mol()
        self.cell = self.mol
        self.kpts = self.periodic.kpts
        self.unrestricted = periodic_env_subsystem.unrestricted

        # set/reset some attributes
        attr_keys = {'env_method'       : 'env_method',
                     'smearsigma'       : 'smearsigma',
                     'rho_cutoff'       : 'rho_cutoff',
                     'grid_level'       : 'grid_level',
                     'damp'             : 'damp',
                     'shift'            : 'shift',
                     'freeze'           : 'freeze',
                     'subcycles'        : 'subcycles',
                     'verbose'          : 'verbose',
                     'debug'            : 'debug',
                     'nproc'            : 'nproc',
                     'pmem'             : 'pmem',
                     'scr_dir'          : 'scr_dir',
                     'save_orbs'        : 'save_orbs',
                     'save_density'     : 'save_density',
                     'filename'         : 'filename',
                     'fermi'            : 'fermi',
                     'unrestricted'     : 'unrestricted',
                    }
        for key in attr_keys:
            if hasattr(self.periodic, attr_keys[key]):
                setattr(self, key, getattr(self.periodic, attr_keys[key]))
            else:
                setattr(self, key, None)

        # transform matrices matrix
        self.env_scf = self.init_env_scf()
        self.smat = self.env_scf.get_ovlp()
        self.cluster_hcore = self.env_scf.get_hcore()

        self.k2origin()

        self.check_transformed_electrons()

        self.diis_num = self.periodic.diis_num
        if self.diis_num == 1:
            #Use subtractive diis. Most simple
            self.diis = lib_diis.DIIS()
        elif self.diis_num == 2:
            self.diis = scf_diis.CDIIS(self.env_scf)
        elif self.diis_num == 3:
            self.diis = scf_diis.EDIIS()
        elif self.diis_num == 4:
            self.diis = scf.diis.ADIIS()
        elif self.diis_num == 5:
            self.diis = comb_diis.EDIIS_DIIS(self.env_scf)
        elif self.diis_num == 6:
            self.diis = comb_diis.ADIIS_DIIS(self.env_scf)
        else:
            self.diis = None
        self.diis = None

        # active keywords
        active_keys = {'active_unrestricted'       : False,
                       'localize_orbitals'         : False,
                       'active_orbs'               : None,
                       'avas'                      : None,
                       'active_conv'               : 1e-9,
                       'active_grad'               : None,
                       'active_cycles'             : 100,
                       'use_molpro'                : False,
                       'active_damp'               : 0,
                       'active_frozen'             : None,
                       'active_shift'              : 0,
                       'active_initguess'          : 'ft',
                       'active_save_orbs'          : False,
                       'active_save_density'       : False,
                       'compress_approx'           : False,
                       'shci_mpi_prefix'           : '',
                       'shci_stochastic'           : True,
                       'shci_nPTiter'              : 0,
                       'shci_sweep_iter'           : None,
                       'shci_DoRDM'                : True,
                       'shci_sweep_epsilon'        : None,
                       'dmrg_maxM'                 : 100,
                       'dmrg_memory'               : None,
                       'dmrg_num_thrds'            : 1,
                      }

        for key in active_keys:
            if key in active_kwargs:
                setattr (self, key, active_kwargs[key])
            else:
                setattr (self, key, active_keys[key])

        self.cluster_in_periodic_env_energy = None


    def k2origin(self):
        """Transforms a k-space matrix to a real-space matrix."""
        from numpy import copy, mean
        def func(mat, axis=1):
            out = mean(mat, axis=axis).real
            return out

        if hasattr (self, 'dmat'):
            self.dmat_kpts = copy(self.dmat)
            self.dmat = func(self.dmat)
        else:
            self.dmat = func(self.periodic.dmat)
        if hasattr (self, 'env_hcore'):
            self.env_hcore = func(self.env_hcore, axis=0)
        else:
            self.env_hcore = func(self.periodic.env_hcore, axis=0)
        if hasattr (self, 'emb_pot'):
            self.emb_pot = func(self.emb_pot)
        else:
            self.emb_pot = func(self.periodic.emb_pot)
        if hasattr (self, 'proj_pot'):
            self.proj_pot = func(self.proj_pot)
        else:
            self.proj_pot = func(self.periodic.proj_pot)
        if hasattr (self, 'fock'):
            self.fock = func(self.fock)
        else:
            self.fock = func(self.periodic.fock)
        if hasattr (self, 'emb_fock'):
            self.emb_fock = func(self.emb_fock)
        else:
            self.emb_fock = func(self.periodic.emb_fock)


    def origin2k(self):
        """Transforms a real-space matrix to k-space"""
        from numpy import copy, zeros, swapaxes
        def func(mat, nkpts):
            out = zeros((nkpts, 2, mat[0].shape[0], mat[0].shape[1]), dtype=self.smat.dtype)
            for k in range(nkpts):
                out[k,0] = mat[0]
                out[k,1] = mat[1]
            out = swapaxes(out, 0, 1)
            return out

        nkpts = len(self.kpts)

        self.dmat_origin = copy(self.dmat)
        self.dmat = func(self.dmat, nkpts)
        self.env_hcore = func([self.env_hcore/2.0, self.env_hcore/2.0], nkpts)
        self.env_hcore = self.env_hcore[0] + self.env_hcore[1]
        self.emb_pot = func(self.emb_pot, nkpts)
        self.proj_pot = func(self.proj_pot, nkpts)
        self.fock = func(self.fock, nkpts)
        self.emb_fock = func(self.emb_fock, nkpts)

    def check_transformed_electrons(self):
        """Checks the integral of the periodic-to-cluster density matrix."""
        from numpy import trace, dot
        n_transformed_ele_a = trace(dot(self.dmat[0], self.smat))
        n_transformed_ele_b = trace(dot(self.dmat[1], self.smat))
        n_transformed_ele = n_transformed_ele_a + n_transformed_ele_b
        if self.unrestricted or self.mol.spin !=0:
            print (f"Number of transformed electrons (alpha): {n_transformed_ele_a:16.8f}")
            print (f"Number of transformed electrons (beta):  {n_transformed_ele_b:16.8f}")
        else:
            print (f"Number of transformed electrons: {n_transformed_ele:16.8f}")
        ele_diff = abs(n_transformed_ele - self.mol.nelectron)
        if ele_diff > 1e-3:
            print (" wARNING: Cluster approximation might not be accurate ".center(80,"!"))

