# A moduel to define all periodic supersystems
# Dhabih V. Chulhai

import pyscf
from pyscf import lib
from qsome import supersystem, cluster_supersystem, periodic_subsystem
import functools
import time
import numpy as np

from qsome.cluster_supersystem import time_method

class PeriodicSuperSystem(cluster_supersystem.ClusterSuperSystem):
    """
    Defines a periodic system ready for embedding.

    Attributes
    ----------


    Methods
    -------
    """

    def __init__(self, subsystems, fs_method, kpoints=None, proj_oper='huz',
                 filename=None, ft_cycles=100, ft_conv=1e-8, ft_grad=None,
                 ft_diis=1, ft_setfermi=None, ft_damp=0.0,
                 ft_initguess='supmol', ft_updatefock=0, ft_writeorbs=False,
                 fs_cycles=100, fs_conv=1e-9, fs_grad=None, fs_damp=0.0,
                 fs_shift=0, fs_smearsigma=0, fs_initguess=None, unrestricted=False,
                 rho_cutoff=1e-7, fs_save_orbs=False, fs_save_density=False,
                 ft_save_orbs=False, ft_save_density=False, compare_density=False,
                 grid_level=3, verbose=3, debug=False, density_fit='df',
                 auxbasis='weigend', exp_to_discard=0.1, **kwargs):
        """
        Initializes the periodic supersystem for embedding.

        Parameters
        ----------
            subsystems : list
                List of PeriodicSubSystem objects
            fs_method : str
                Defines the supersystem/full system method
            kpoints : ndarray
                Kpoints to use in the periodic calculations
            proj_oper : str
                Which projection operator to use (default is `huz`).
            filename : str, optional
                Path to input file.
            ft_cycles : int
                Number of freeze-and-thaw cycles (default is 100).
            ft_conv : float
                Freeze-and-thaw energy convergence (default is 1e-8).
            ft_grad : float
                Freeze-and-that gradient convergence (default is None).
            ft_diis : pyscf:DIIS object
                DIIS method for use for fock matrix during F&T cycles.
            ft_setfermi : ??
            ft_damp : float
                Damping factor to use for F&T dmat (default is 0.0).
            ft_initguess : str
                Method to use for guessing initial F&T density.
                Default is to use the supersystem density.
            ft_updatefock : ???
            ft_writeorbs : bool
                ??
            fs_cycles : int
                Number of cycles to converge full system (default 100).
            fs_conv : float
                Energy convergence for full system (default is 1e-9).
            fs_grad : float
                Energy gradient convergence for full system (default None).
            fs_damp : float
                Damping parameter for full system density matrix (default 0.0).
            fs_shift : float
                Energy shift for full system density matrix (default 0.0).
            fs_smearsigma : float
                Fermi smearing for full system density matrix (default 0.0).
            fs_initguess : str
                Method used for initial density for full system (default None).
            unrestricted : bool
                Whether full system calculations are unrestricted.
            rho_cutoff : float
                Rho cutoff parameter (default 1e-7).
            fs_save_orbs : bool
                Whether to save full system orbitals to file (default False).
            fs_save_density : bool
                Whether to save full system density to file (default False).
            ft_save_orbs : bool
                Wther to save F&T orbitals to file (default False).
            ft_save_density : bool
                Whether to save F&T density to file (default False).
            compare_density : bool
                ???
            grid_level : int
                DFT Becke grid level for integration (default 3).
            verbose : int
                PySCF verbosity level (default 3).
            debug : bool
                Whether to print debug statements (default False).

            Parameters
            ----------
            fs_energy : float
                The full system KS-DFT energy.
        """

        self.subsystems = subsystems
        self.filename   = filename
        self.verbose    = verbose

        self.kpts            = kpoints
        self.nkpts           = len(kpoints)
        self.grid_level      = grid_level
        self.rho_cutoff      = rho_cutoff
        self.density_fit     = density_fit
        self.auxbasis        = auxbasis
        self.exp_to_discard  = exp_to_discard

        self.fs_method       = fs_method
        self.fs_cycles       = fs_cycles
        self.fs_conv         = fs_conv
        self.fs_grad         = fs_grad
        self.fs_damp         = fs_damp
        self.fs_shift        = fs_shift
        self.fs_smearsigma   = fs_smearsigma
        self.unrestricted    = unrestricted
        self.fs_initguess    = fs_initguess
        self.fs_save_orbs    = fs_save_orbs
        self.fs_save_density = fs_save_density

        self.proj_oper       = proj_oper
        self.ft_cycles       = ft_cycles
        self.ft_conv         = ft_conv
        self.ft_grad         = ft_grad
        self.ft_setfermi     = ft_setfermi
        self.ft_damp         = ft_damp
        self.ft_initguess    = ft_initguess
        self.ft_updatefock   = ft_updatefock
        self.ft_writeorbs    = ft_writeorbs
        self.ft_save_orbs    = ft_save_orbs
        self.ft_save_density = ft_save_density

        self.compare_density = compare_density

        self.ft_energy       = None

        if ft_diis == 0:
            self.ft_diis = None
        else:
            self.ft_diis = pyscf.lib.diis.DIIS()

        self.fs_energy = None

        self.nsub = len(subsystems)

        # create supercell and initialze SCF object and density matrix
        self.cell = self.concatenate_cells()
        self.gen_sub2sup()

        self.fs_scf = self.init_scf()
        self.fs_hcore = self.update_stored_pot(self.fs_scf.get_hcore())
        self.fs_dmat = self.init_density(scf_obj=self.fs_scf,
            env_method=self.fs_method, kpts=self.kpts)
        self.fock = np.zeros_like(self.fs_hcore)

        self.print_coordinates()
        self.print_lattice_vectors()

    def print_coordinates(self):
        print (' Supercell '.center(80,'='))
        return periodic_subsystem.PeriodicEnvSubSystem.print_coordinates(self)


    def print_lattice_vectors(self):
        return periodic_subsystem.PeriodicEnvSubSystem.print_lattice_vectors(self)


    def concatenate_cells(self, subsystems=None, verbose=None):
        """
        Concatenates the cells from each subsystem (ignoring
        ghost atoms) in order to create the supersystem cell.

        Parameters
        ----------
            subsystems : list
                List of PeriodicSubSystem objects to concatenate
            verbose : int
                PySCF verbosity parameter

        Returns
        -------
            sup_cell : pyscf:pbc:Cell object
                Supersystem Cell object
        """

        from pyscf.pbc import gto

        if subsystems is None: subsystems = self.subsystems
        assert len(subsystems) > 1, 'Cannot concantenate less than 2 Cell objects'
        if verbose is None: verbose = self.verbose

        sup_cell = gto.Cell()
        sup_cell.unit = 'bohr' # units are stored in bohr by default

        atoms = []; basis = {}; ecp = {}
        for i in range(len(subsystems)):
            temp_cell = subsystems[i].cell
            for j in range(len(temp_cell._atom)):
                old_symbol = temp_cell._atom[j][0]
                if 'ghost' in old_symbol: continue
                new_symbol = old_symbol + '-{0}'.format(str(i))
                atoms.append((new_symbol, temp_cell._atom[j][1]))
                if old_symbol in temp_cell._basis.keys():
                    basis[new_symbol] = temp_cell._basis[old_symbol]
                if old_symbol in temp_cell.ecp.keys():
                    ecp[new_symbol] = temp_cell.ecp[old_symbol]

#        conversion = 1.0
#        if subsystems[0].cell.unit in ('a', 'angstrom'):
#            conversion = 1.8897261328856432

        sup_cell.atom = atoms
        sup_cell.basis = basis
        sup_cell.ecp = ecp

        sup_cell.verbose = verbose
        sup_cell.a = subsystems[0].cell.lattice_vectors()
        sup_cell.dimension = subsystems[0].cell.dimension
        sup_cell.precision = subsystems[0].cell.precision
        sup_cell.ke_cutoff = subsystems[0].cell.ke_cutoff
        sup_cell.rcut = subsystems[0].cell.rcut
        sup_cell.low_dim_ft_type = subsystems[0].cell.low_dim_ft_type
        sup_cell.exp_to_discard = subsystems[0].cell.exp_to_discard

        sup_cell.build(dump_input=False)

        return sup_cell


    def gen_sub2sup(self, sup_cell=None, subsystems=None):
        """
        Generate the transformation indices to relate the subsystems
        to the supersystem.

        Parameters
        ----------
            sup_cell : pyscf:pbc:gto:Cell object, optional
                The generated supersystem Cell object
            subsystems : list, optional
                A list of all periodic_subsystems objects

        Returns
        -------
            sub2sup : list
                transformation indicies
        """

        if sup_cell is None: sup_cell = self.cell
        if subsystems is None: subsystems = self.subsystems
        nsub = len(subsystems)

        # get nao slice for each atom of each subsystem
        nssl = [None for i in range(nsub)]
        for i in range(nsub):
            sub_cell = subsystems[i].cell
            nssl[i] = np.zeros((sub_cell.natm), dtype=int)
            for j in range(sub_cell.natm):
                ib = np.where(sub_cell._bas.transpose()[0]==j)[0][0]
                ie = np.where(sub_cell._bas.transpose()[0]==j)[0][-1]
                ir = sub_cell.nao_nr_range(ib,ie+1)
                ir = ir[1] - ir[0]
                nssl[i][j] = ir
    
        # get nao slice for each atom in supermolecule
        nsl = np.zeros((sup_cell.natm), dtype=int)
        for i in range(sup_cell.natm):
            ib = np.where(sup_cell._bas.transpose()[0]==i)[0][0]
            ie = np.where(sup_cell._bas.transpose()[0]==i)[0][-1]
            ir = sup_cell.nao_nr_range(ib,ie+1)
            ir = ir[1] - ir[0]
            nsl[i] = ir
    
        # see which nucleii matches up
        sub2sup = [None for i in range(nsub)]
        for i in range(nsub):
            sub2sup[i] = np.zeros((subsystems[i].cell.nao_nr()), dtype=int)
            for a in range(subsystems[i].cell.natm):
                match = False
                for b in range(sup_cell.natm):
                    d = sup_cell.atom_coord(b) - subsystems[i].cell.atom_coord(a)
                    d = np.dot(d,d)
                    if d < 1e-3:
                        match = True
                        ia = nssl[i][0:a].sum()
                        ja = ia + nssl[i][a]
                        ib = nsl[0:b].sum()
                        jb = ib + nsl[b]
                        sub2sup[i][ia:ja] = range(ib,jb)
                if not match: raise Exception('Atoms in sub- and supersystems did not match!')

        self.sub2sup = sub2sup


    def init_scf(self, cell=None, method=None, verbose=None, damp=None,
                 shift=None, kpts=None, grid_level=None, rho_cutoff=None,
                 density_fit=None, auxbasis=None):
        """
        Initializes the supersystem PySCF SCF object using given settings.

        Parameters
        ----------
            cell : pyscf.pbc.gto.Cell object
                Concatenated supersystem Cell object
            fs_method : str
                Supersystem SCF method
            verbose : int
                PySCF verbosity parameter for output
            damp : float
                Damping parameter for SCF convergence
            shift : float
                Energy level shift parameter for convergence

        Returns
        -------
            scf_obj : pyscf.pbc.SCF object
                SCF object for the supersystem
        """

        import pyscf
        from pyscf.pbc import scf as pbcscf, df as pbcdf, dft as pbcdft

        if cell is None:
            cell = self.cell
        if method is None:
            method = self.fs_method
        if verbose is None:
            verbose = self.verbose
        if damp is None:
            damp = self.fs_damp
        if shift is None:
            shift = self.fs_shift
        if kpts is None:
            kpts = self.kpts
        if grid_level is None:
            grid_level = self.grid_level
        if rho_cutoff is None:
            rho_cutoff = self.rho_cutoff
        if density_fit is None:
            density_fit = self.density_fit
        if auxbasis is None:
            auxbasis = self.auxbasis

#        if self.pmem:
#            self.cell.max_memory = self.pmem

        nkpts = len(kpts)

        if self.unrestricted:
            if method in ('hf', 'hartree-fock'):
                scf_obj = pbcscf.KUHF(cell, kpts)
            else:
                scf_obj = pbcscf.KUKS(cell, kpts)
                scf_obj.xc = method

        elif cell.spin != 0:
            if method in ('hf', 'hartree-fock'):
                scf_obj = pbcscf.KROHF(cell, kpts)
            else:
                scf_obj = pbcscf.KROKS(cell, kpts)
                scf_obj.xc = method

        else:
            if method in ('hf', 'hartree-fock'):
                scf_obj = pbcscf.KRHF(cell, kpts)
            else:
                scf_obj = pbcscf.KRKS(cell, kpts)
                scf_obj.xc = method

        scf_obj.kpts = kpts
        scf_obj.verbose = verbose

        # build grids
        scf_obj.grids = pbcdft.gen_grid.BeckeGrids(cell)
        scf_obj.grids.level = grid_level
        scf_obj.grids.build()

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

        scf_obj.with_df = DensityFit(cell)
        scf_obj.with_df.kpts = kpts
        scf_obj.with_df.auxbasis = auxbasis

        scf_obj.damp = damp

        self.smat = scf_obj.get_ovlp()

        return scf_obj

    from qsome.periodic_subsystem import PeriodicEnvSubSystem
    init_density = PeriodicEnvSubSystem.init_density

    @time_method("Supersystem Energy")
    def get_supersystem_energy(self, scf_obj=None):
        """
        Calculate/return the full system energy and update the
        density matrix.
        """

        if self.fs_energy is not None:
            return self.fs_energy

        if scf_obj is None: scf_obj = self.fs_scf

        print ("".center(80,"*"))
        print (" Supersystem Calculation ".center(80))
        print ("".center(80,"*"))

        self.fs_energy = scf_obj.kernel(dm0=self.get_useable_dmat(self.fs_dmat))
        self.fs_dmat = self.update_stored_dmat(scf_obj.make_rdm1())

        print ("Periodic Supersystem Environment Energy (E_h) "
              f"{self.fs_energy:16.8f}".center(80))

        return self.fs_energy

    @time_method("Freeze-and-Thaw")
    def freeze_and_thaw(self, subsystems=None):
        """
        Updates subsystem densities in iterative cycles until
        convergence.
        """

        print ("".center(80,"*"))
        print ("Freeze-and-Thaw".center(80))
        print ("".center(80,"*"))

        if subsystems is None: subsystems = self.subsystems
        s2s = self.sub2sup
        ft_err = 1.0
        ft_iter = 0
        nsub = len(self.subsystems)
        last_cycle = False

        # TODO: change/remove this later?
        # Updates subsystem density matrices
        # using the converged supersystem density.
        # (this speeds up freeze-and-thaw.)
        for isub in range(nsub):
            subsystems[isub].dmat = self.fs_dmat[np.ix_(
                range(2), range(len(self.kpts)), s2s[isub], s2s[isub])]

        # update fock matrix
        self.ft_energy = self.update_embedding_fock()

        while ((ft_err > self.ft_conv) and (ft_iter < self.ft_cycles)
               and (ft_err > self.cell.precision)):

            ft_iter += 1

            # cycle over subsystems and do embedding
            for isub in range(nsub):

                if subsystems[isub].freeze: continue

                self.update_proj_pot(isub)
                subsystems[isub].dmat = subsystems[isub].diagonalize()

            # update fock matrix
            energy_new = self.update_embedding_fock()
            ft_err = energy_new - self.ft_energy
            self.ft_energy = energy_new

            print (f"cycle= {ft_iter:>3d}     E= {energy_new:16.8f}"
                   f"     delta_E= {ft_err:16.8f}")

            ft_err = abs(ft_err)

        self.subsystems = subsystems


    def update_embedding_fock(self, fs_scf=None, subsystems=None,
        kpts=None, diis=True):
        """
        Update the fock matrix of the supersystem using the
        density matrices of the subsystems.

        Parameters
        ----------
            fs_scf : pyscf.pbc.scf object, optional
                Full system SCF object
            subsystems : list, optional
                List of subsystem objects
            kpts : ndarray, optional
                Kpoints to use
            diis : bool
                Whether to update using DIIS
        """

        if fs_scf is None: fs_scf = self.fs_scf
        if subsystems is None: subsystems = self.subsystems
        if kpts is None: kpts = self.kpts

        sub_nao = [sub.cell.nao_nr() for sub in subsystems]
        sup_nao = fs_scf.cell.nao_nr()
        nsub = len(subsystems)
        nkpts = len(kpts)
        s2s = self.sub2sup

        # make supersystem density matrix from subsystems
        dm = np.zeros((2, nkpts, sup_nao, sup_nao), dtype=subsystems[1].dmat.dtype)
        for isub in range(nsub):
            dm[np.ix_(range(2), range(nkpts), s2s[isub], s2s[isub])] \
                += subsystems[isub].dmat

        dm = self.get_useable_dmat(dm)

        # get effective 2electron potential
        try:
            Veff = fs_scf.get_veff(dm=dm, kpts=kpts)
        except TypeError:
            try:
                Veff = fs_scf.get_veff(dm_kpts=dm, kpts=kpts)
            except TypeError:
                Veff = fs_scf.get_veff(dm=dm)

        self.veff = self.update_stored_pot(Veff)
        self.fock = self.fs_hcore + self.veff

#        # update with DIIS
#        if self.ft_diis is not None and diis:
#            self.fock = self.ft_diis.update(self.fock)

        energy_tot = fs_scf.energy_tot(dm=dm)
        return energy_tot


    def update_proj_pot(self, isub, subsystems=None, cluster=False):
        """
        Update the projection potential.

        Parameters
        ----------
            isub : int
                The index of the subsystem being embedded
            subsystems : list, optional
                The list of subsystem objects.
            cluster : bool
                Whether the embedded subsystem is a cluster
        """

        if subsystems is None: subsystems = self.subsystems

        s2s = self.sub2sup
        nao = subsystems[isub].cell.nao_nr()
        nsub = len(subsystems)
        kpts = self.kpts
        nkpts = len(kpts)

        if not cluster:
            env_scf = subsystems[isub].env_scf
        else:
            env_scf = subsystems[isub].periodic.env_scf

        dmat = subsystems[isub].get_useable_dmat(subsystems[isub].dmat)

        # get effective 2electron potential
        try:
            Veff = env_scf.get_veff(dm=dmat, kpts=kpts)
        except TypeError:
            try:
                Veff = env_scf.get_veff(dm_kpts=dmat, kpts=kpts)
            except TypeError:
                Veff = env_scf.get_veff(dm=dmat)
        Veff = self.update_stored_pot(Veff)

        # get embedding potential
        env_hcore = self.update_stored_pot(subsystems[isub].env_hcore)
        emb_hcore = self.fs_hcore[np.ix_(range(2), range(nkpts), s2s[isub], s2s[isub])]
        emb_hcore -= env_hcore
        emb_pot = self.fock[np.ix_(range(2), range(nkpts), s2s[isub], s2s[isub])]
        emb_pot -= env_hcore
        emb_pot -= Veff
        proj_pot = np.zeros_like(emb_pot)

        # get mu-parameter projection operator
        if isinstance(self.proj_oper, int) or isinstance(self.proj_oper, float):
            for jsub in range(nsub):
                if isub == jsub: continue

                smat_AB = self.smat[np.ix_(range(nkpts), s2s[isub],
                                    s2s[jsub])]
                smat_BA = self.smat[np.ix_(range(nkpts), s2s[jsub],
                                    s2s[isub])]

                for ispin in range(2):
                    for k in range(nkpts):
                        proj_pot[ispin,k] += self.proj_oper * np.dot(smat_AB[k],
                                   np.dot(subsystems[jsub].dmat[ispin,k], smat_BA[k]))

        # traditional huzinaga operator
        elif self.proj_oper in ('huzinaga', 'huz'):
            for jsub in range(nsub):
                if isub == jsub: continue

                smat_AB = self.smat[np.ix_(range(nkpts), s2s[isub],
                                    s2s[jsub])]
                smat_BA = self.smat[np.ix_(range(nkpts), s2s[jsub],
                                    s2s[isub])]

                fmat_AB = self.fock[np.ix_(range(2), range(nkpts),
                                           s2s[isub], s2s[jsub])]
                fmat_BA = self.fock[np.ix_(range(2), range(nkpts),
                                           s2s[jsub], s2s[isub])]

                for ispin in range(2):
                    for k in range(nkpts):
                        FDS = np.dot(fmat_AB[ispin,k], np.dot(subsystems[jsub].dmat[ispin,k],
                                     smat_BA[k]))
                        SDF = np.dot(smat_AB[k], np.dot(subsystems[jsub].dmat[ispin,k],
                                     fmat_BA[ispin,k]))

                        proj_pot[ispin,k] += -1.0 * (FDS + SDF)

        # fermi-shifted huzinaga operator
        elif self.proj_oper in ('huzinagafermi', 'huzfermi'):

                raise Exception ("Huzinaga-Fermi not yet implemented!")

        else:
            raise NotImplementedError ("Projection operator not recognized!")

        # update subsystem values
        subsystems[isub].proj_pot = proj_pot
        subsystems[isub].emb_pot = emb_pot
        subsystems[isub].emb_hcore = emb_hcore
        subsystems[isub].emb_fock = emb_pot + env_hcore + Veff
        subsystems[isub].fock = self.fock[np.ix_(range(2), range(nkpts), s2s[isub], s2s[isub])]
        self.subsystems = subsystems


    @time_method("Env. in Env. Energy")
    def env_in_env_energy(self, isub=0):
        """
        Determines the subsystem env energy.

        Parameters
        ----------
            isub : int
                The index of the subsystem we're interested in

        Returns
        -------
            env_energy : float
                The subsystem environment energy (the energy of subsystem
                in the environment of the total system at the lower level).
        """

        print ("".center(80,"*"))
        print ("Environment Subsystem Calculation".center(80))
        print ("".center(80,"*"))

        emb_pot = self.get_useable_pot(self.subsystems[isub].emb_pot)
        dmat = self.get_useable_dmat(self.subsystems[isub].dmat)
        env_hcore = self.get_useable_pot(self.subsystems[isub].env_hcore)
        proj_pot = self.get_useable_pot(self.subsystems[isub].proj_pot)
        kpts = self.kpts

        # modify the core hamiltonian and get energy
        hcore_embed = env_hcore + emb_pot + proj_pot
        self.subsystems[isub].env_scf.get_hcore = lambda *args: hcore_embed
        env_energy = self.subsystems[isub].env_scf.kernel(dm0=dmat)

        self.subsystems[isub].env_energy = env_energy
        print (f"Periodic-in-Periodic Environment Energy (E_h) {env_energy:16.8f}".center(80))
        return env_energy


    @time_method("Periodic to Cluster")
    def periodic_to_cluster(self, active_method, isub=0, **active_kwargs):
        """
        Turns the periodic subsystem into an active subsystem.

        Parameters
        ----------
            isub : int
                Index of the subsystem that gets transformed
        """

        print ("".center(80,"*"))
        print ("Periodic to Cluster Approximation".center(80))
        print ("".center(80,"*"))

        self.subsystems[isub] = periodic_subsystem.FiniteSubSystem(
                                self.subsystems[isub], active_method, **active_kwargs)

        self.subsystems[isub].ft_dmat = np.copy(self.subsystems[isub].dmat)

        # do cycles here until convergence
        ft_err = 1.0
        ft_iter = 0

        self.subsystems[isub].origin2k()
        self.ft_energy = self.update_embedding_fock()
        dmat = None

        while ((ft_err > self.ft_conv) and (ft_iter < self.ft_cycles)
               and (ft_err > self.cell.precision)):

            ft_iter += 1
            ft_err = 0.0

            self.update_proj_pot(isub, cluster=True)
            self.subsystems[isub].k2origin()

            # diagonalize
            if dmat is None:
                dmat = np.copy(self.subsystems[isub].dmat)
            else:
                self.subsystems[isub].dmat = np.copy(dmat)
            dmat = self.subsystems[isub].diagonalize()

            self.subsystems[isub].dmat = np.copy(dmat)
            self.subsystems[isub].origin2k()
            energy_new = self.update_embedding_fock()
            ft_err = energy_new - self.ft_energy
            self.ft_energy = energy_new

            print (f"cycle= {ft_iter:>3d}     E= {energy_new:16.8f}"
                   f"     delta_E= {ft_err:16.8f}")

            ft_err = abs(ft_err)

        self.subsystems[isub].k2origin()
        self.subsystems[isub].dmat = np.copy(dmat)
        self.subsystems[isub].active_dmat = self.subsystems[isub].dmat

        # transform full system fock matrix here
        self.fock = np.mean(self.fock, axis=1).real

        # update embedding potential
        emb_fock = self.subsystems[isub].get_useable_pot(self.subsystems[isub].emb_fock)
        dmat = self.subsystems[isub].get_useable_dmat(self.subsystems[isub].dmat)
        Veff = self.subsystems[isub].env_scf.get_veff(dm=dmat)
        self.subsystems[isub].update_emb_pot(self.subsystems[isub].update_stored_pot(
            emb_fock - Veff - self.subsystems[isub].cluster_hcore))

        env_energy = self.subsystems[isub].get_env_energy()
        self.subsystems[isub].cluster_in_periodic_env_energy = env_energy
        self.subsystems[isub].periodic_in_periodic_env_energy = self.subsystems[isub].env_energy
        self.subsystems[isub].env_energy = env_energy
        print ("Cluster-in-Periodic Environment Energy (E_h) "
                f"{env_energy:16.8f}".center(80))


    def get_useable_dmat(self, mat=None):
        return periodic_subsystem.PeriodicEnvSubSystem.get_useable_dmat(self, mat=mat)


    def get_useable_pot(self, mat=None):
        return periodic_subsystem.PeriodicEnvSubSystem.get_useable_pot(self, mat=mat)


    def update_stored_dmat(self, mat=None):
        return periodic_subsystem.PeriodicEnvSubSystem.update_stored_dmat(self, mat=mat)


    def update_stored_pot(self, mat=None):
        return periodic_subsystem.PeriodicEnvSubSystem.update_stored_pot(self, mat=mat) 

