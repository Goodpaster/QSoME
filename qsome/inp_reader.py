#!/usr/bin/env python
# A module to define the input reader object
# Daniel S. Graham
# Dhabih V. Chulhai

from __future__ import print_function, division

import input_reader
import sys
import re
import pwd, os

from pyscf import gto, pbc


class InpReader:
    """
    Reads a specific form of input file for creating objects.

    Takes a formatted text file and generates arg and kwarg 
    dictionaries to generate the specified embedding system objects.

    Attributes
    ----------
    filename : str
        The path to the input file being read.
    inp : InputReader
        The InputReader object of the input file.
    supersystem_kwargs : dict
        Dictionary of settings for creating ClusterSupersystem object.
    env_subsystem_kwargs : dict
        Dictionary of settings for creating ClusterEnvSubSystem object.
    active_settings_kwargs : dict
        Dictionary of settings for creating ClusterActiveSubSystem object.
    subsys_mols : list
        A list of subsystem pyscf Mol objects
    
    Methods
    -------
    read_input(filename)
        Opens the file, reads the settings, 
        and stores as an input_reader object.
    get_supersystem_kwargs(inp=None)
        Creates the kwarg dictionary for the supersystem.
    get_env_subsystem_kwargs(inp=None)
        Creates the kwarg dictionary for subsystems.
    get_active_subsystem_kwargs(inp=None)
        Creates the kwarg dictionary for the active subsystems.
    gen_mols(inp=None)
        Generate the mol objects specified in the inp files.  
    """


    def __init__(self, filename):
        """
        Parameters
        ----------
        filename : str
            Path to input file.
        """

        self.inp = self.read_input(filename)
        self.interaction_mediator_kwargs = self.get_interaction_mediator_kwargs()
        self.env_subsystem_kwargs = self.get_env_subsystem_kwargs()
        self.active_subsystem_kwargs = self.get_active_subsystem_kwargs()
        self.cell_kwargs, self.kpoints_kwargs, self.periodic_kwargs \
            = self.get_periodic_kwargs()
        if self.periodic_kwargs is not None:
            self.periodic = True
        else:
            self.periodic = False
        self.subsys_mols = self.gen_mols()


    def read_input(self, filename):
        """Reads a formatted input file, generates an InputReader object.

        Parameters
        ---------
        filename : str
            Path to input file.
        """

        reader = input_reader.InputReader(comment=['!', '#', '::', '//'],
                 case=False, ignoreunknown=False)
        subsys = reader.add_block_key('subsystem', required=True, repeat=True)

        # Could be shortened using repeating regex pattern.
        # input_reader uses re which does not support repeating patterns.
        subsys.add_regex_line(
         'atoms',
         '\s*([A-Za-z.:\d]+)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)',
         repeat=True)
        subsys.add_line_key('charge', type=int)     
        subsys.add_line_key('spin', type=int)      
        subsys.add_line_key('unit', type=('angstrom','a','bohr','b')) 
        subsys.add_boolean_key('addlinkbasis') # Add link H basis functions
        sub_basis = subsys.add_block_key('basis')              
        sub_basis.add_regex_line('basis_def', '\s*([A-Za-z.:\d]+)\s+.+', 
            repeat=True)

        sub_ecp = subsys.add_block_key('ecp')              
        sub_ecp.add_regex_line('ecp_def', '\s*([A-Za-z.:\d]+)\s+.+', 
            repeat=True)
        subsys.add_line_key('env_method_num', type=int)
        subsys.add_line_key('hl_method_num', type=int)
        # Override default environment settings
        sub_env_settings = subsys.add_block_key('env_method_settings')
        sub_env_settings.add_line_key('smearsigma', type=float)   # fermi smearing sigma
        sub_env_settings.add_line_key('initguess', type=('minao', 'atom', '1e', 
            'readchk', 'supmol', 'submol'))
        sub_env_settings.add_line_key('conv', type=float) # embedding convergence
        sub_env_settings.add_line_key('damp', type=float) # subsys damping parameter
        sub_env_settings.add_line_key('shift', type=float) # SCF level-shift parameter
        sub_env_settings.add_line_key('subcycles', type=int) # num subsys diag. cycles
        sub_env_settings.add_line_key('diis', type=int) # DIIS for subsystem (0 for off)
        sub_env_settings.add_boolean_key('unrestricted')
        sub_env_settings.add_boolean_key('freeze')
        sub_env_settings.add_boolean_key('save_orbs')
        sub_env_settings.add_boolean_key('save_density')

        # Override default high level method settings
        sub_hl_settings = subsys.add_block_key('hl_method_settings')
        sub_hl_settings.add_line_key('initguess', type=('minao', 'atom', '1e', 
            'readchk', 'supmol', 'submol'))
        sub_hl_settings.add_line_key('spin', type=int)       
        sub_hl_settings.add_line_key('conv', type=float)       
        sub_hl_settings.add_line_key('grad', type=float)       
        sub_hl_settings.add_line_key('cycles', type=int)       
        sub_hl_settings.add_line_key('damp', type=float)
        sub_hl_settings.add_line_key('shift', type=float)
        sub_hl_settings.add_line_key('use_ext', type=('molpro', 'bagel',
            'molcas', 'openmolcas'))
        sub_hl_settings.add_boolean_key('unrestricted')
        sub_hl_settings.add_boolean_key('compress_approx')
        sub_hl_settings.add_boolean_key('density_fitting')

        sub_cas_settings = sub_hl_settings.add_block_key('cas_settings')
        sub_cas_settings.add_boolean_key('loc_orbs')
        sub_cas_settings.add_line_key('initguess', type=str) 
        sub_cas_settings.add_line_key('active_orbs', type=str) #Could I use a tuple?
        sub_cas_settings.add_line_key('avas', type=str) #Could I use a tuple?

        sub_shci_settings = sub_hl_settings.add_block_key('shci_settings')
        sub_shci_settings.add_line_key('mpi_prefix', type=str)
        sub_shci_settings.add_line_key('sweep_iter', type=str)
        sub_shci_settings.add_line_key('sweep_epsilon', type=str)
        sub_shci_settings.add_line_key('nPTiter', type=int, default=0)
        sub_shci_settings.add_boolean_key('no_stochastic')
        sub_shci_settings.add_boolean_key('NoRDM')

        sub_dmrg_settings = sub_hl_settings.add_block_key('dmrg_settings')
        sub_dmrg_settings.add_line_key('maxM', type=int)
        sub_dmrg_settings.add_line_key('num_thirds', type=int)

        # Define the environment settings and embedding ops
        env_settings = reader.add_block_key('env_method_settings', required=True, 
                                          repeat=True)
        env_settings.add_line_key('env_num', type=int)
        env_settings.add_line_key('env_method', type=str, required=True)
        env_settings.add_line_key('smearsigma', type=float)
        # Initial guess for the supermolecular calculation
        env_settings.add_line_key('initguess', type=('minao', 'atom', '1e', 
            'readchk', 'supmol', 'submol'))
        env_settings.add_line_key('conv', type=float)
        env_settings.add_line_key('grad', type=float)
        env_settings.add_line_key('damp', type=float)
        env_settings.add_line_key('shift', type=float)
        env_settings.add_line_key('diis', type=int) # DIIS for subsystem (0 for off)
        env_settings.add_line_key('grid', type=int)
        env_settings.add_line_key('rhocutoff', type=float)
        env_settings.add_line_key('verbose', type=int)
        env_settings.add_boolean_key('unrestricted')
        env_settings.add_boolean_key('density_fitting')
        env_settings.add_boolean_key('compare_density')
        env_settings.add_boolean_key('save_orbs')
        env_settings.add_boolean_key('save_density')
        
        # Freeze and thaw settings
        embed = env_settings.add_block_key('embed_settings')
        embed.add_line_key('cycles', type=int) 
        embed.add_line_key('subcycles', type=int) 
        embed.add_line_key('conv', type=float)
        embed.add_line_key('grad', type=float)
        embed.add_line_key('damp', type=float)
        embed.add_line_key('diis', type=int) # Use DIIS for fock. (0 for off)
        # Supersystem fock update frequency. 
        # 0 is after F&T cycle, otherwise after every n subsystem cycles
        embed.add_line_key('updatefock', type=int)
        # Initial guess for the subsystem embedding calculation
        embed.add_line_key('initguess', type=(
            'minao', 'atom', '1e', 'readchk', 'supmol', 'submol', 'localsup'))
        embed.add_boolean_key('unrestricted')
        # Output subsystem orbitals after F&T cycles
        embed.add_boolean_key('save_orbs') 
        embed.add_boolean_key('save_density') 

        # This section needs work. Should be uniform option for setting op.
        operator = embed.add_mutually_exclusive_group(dest='operator', 
                                                      required=False)
        operator.add_line_key('mu', type=float, default=1e6)
        operator.add_boolean_key('manby', action=1e6)
        operator.add_boolean_key('huzinaga', action='huz')
        # Fermi shifted
        operator.add_boolean_key('huzinagafermi', action='huzfermi')
        operator.add_boolean_key('huzfermi', action='huzfermi')
        embed.add_line_key('setfermi', type=float)

        periodic_settings = env_settings.add_block_key('periodic_settings', required=False)
        # lattice vectors
        lattice = periodic_settings.add_block_key('lattice_vectors', required=True)
        lattice.add_regex_line('vector', '\s*(\-?\d+.?\d*)\s+(\-?\d+.?\d*)\s+(\-?\d+.?\d*)',
            repeat=True)
        # k-points by grid or num of points
        kgroup = periodic_settings.add_mutually_exclusive_group(dest='kgroup', required=True)
        kgroup.add_line_key('kpoints', type=[int, int, int])
        kscaled = kgroup.add_block_key('kgrid')
        kscaled.add_regex_line('kpoints', '\s*(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)',
                                repeat=True)
        # grid/mesh points
        mesh = periodic_settings.add_mutually_exclusive_group(dest='mgroup', required=False)
        mesh.add_line_key('gspacing', type=float)
        mesh.add_line_key('gs', type=[int, int, int])
        mesh.add_line_key('mesh', type=[int, int, int])
        # dimensions
        periodic_settings.add_line_key('dimensions', type=(0,1,2,3), required=True)
        # line keys with good defaults (shouldn't need to change these)
        periodic_settings.add_line_key('auxbasis', type=str, case=True)
        periodic_settings.add_line_key('density_fit',
            type=('df', 'mdf', 'pwdf', 'fftdf', 'gdf', 'aftdf'))
        periodic_settings.add_line_key('exxdiv', type=('vcut_sph', 'ewald', 'vcut_ws'))
        periodic_settings.add_line_key('precision', type=float)
        periodic_settings.add_line_key('ke_cutoff', type=float)
        periodic_settings.add_line_key('rcut', type=float)
        periodic_settings.add_line_key('exp_to_discard', type=float)
        periodic_settings.add_line_key('low_dim_ft_type', type=str)
        periodic_settings.add_boolean_key('fractional_coordinates') # fractional input coordinates

        hl_settings = reader.add_block_key('hl_method_settings')
        hl_settings.add_line_key('hl_num', type=int)
        hl_settings.add_line_key('hl_method', type=str)
        hl_settings.add_line_key('initguess', type=('minao', 'atom', '1e', 
            'readchk', 'supmol', 'submol'))
        hl_settings.add_line_key('spin', type=int)
        hl_settings.add_line_key('conv', type=float)       
        hl_settings.add_line_key('grad', type=float)       
        hl_settings.add_line_key('cycles', type=int)       
        hl_settings.add_line_key('damp', type=float)
        hl_settings.add_line_key('shift', type=float)
        hl_settings.add_boolean_key('compress_approx')
        hl_settings.add_boolean_key('unrestricted')
        hl_settings.add_boolean_key('density_fitting')
        hl_settings.add_line_key('use_ext', type=('molpro', 'bagel',
            'molcas', 'openmolcas'))

        cas_settings = hl_settings.add_block_key('cas_settings')
        cas_settings.add_boolean_key('loc_orbs')
        cas_settings.add_line_key('initguess', type=str) 
        cas_settings.add_line_key('active_orbs', type=str) #Could I use a tuple?
        cas_settings.add_line_key('avas', type=str) #Could I use a tuple?

        shci_settings = hl_settings.add_block_key('shci_settings')
        shci_settings.add_line_key('mpi_prefix', type=str)
        shci_settings.add_line_key('sweep_iter', type=str)
        shci_settings.add_line_key('sweep_epsilon', type=str)
        shci_settings.add_line_key('nPTiter', type=int, default=0)
        shci_settings.add_boolean_key('no_stochastic')
        shci_settings.add_boolean_key('NoRDM')

        dmrg_settings = hl_settings.add_block_key('dmrg_settings')
        dmrg_settings.add_line_key('maxM', type=int)

        reader.add_line_key('unit', type=('angstrom','a','bohr','b')) 
        basis = reader.add_block_key('basis')
        basis.add_regex_line('basis_def', '\s*([A-Za-z.:\d]+)\s+.+', repeat=True)
        ecp = reader.add_block_key('ecp')
        ecp.add_regex_line('ecp_def', '\s*([A-Za-z.:\d]+)\s+.+', repeat=True)
        reader.add_line_key('ppmem', type=(int, float)) # MB
        reader.add_line_key('nproc', type=int) # MB
        reader.add_line_key('scrdir', type=str)

        inp = reader.read_input(filename)
        inp.filename = filename 

        #Convert scratch keyword to actual scratch directory path. This is SYSTEM DEPENDENT.
        scr_prefix = {"global":"/scratch.global/", "local":"/scratch.local/", 
                     "ssd":"/scratch.ssd/", "ramdisk":"/dev/shm/"}
        if inp.scrdir:
            if inp.scrdir.lower() in scr_prefix.keys():
                scr_path = scr_prefix[inp.scrdir.lower()]
                username = pwd.getpwuid(os.getuid()).pw_name
                scr_path += username
                file_path = os.path.splitext(inp.filename)[0]
                file_path = file_path.split(username)[1]
                scr_path += file_path
                inp.scrdir = scr_path
                
        print("".center(80, '*'))
        print("Input File".center(80))
        print("".center(80, '*'))
        with open(filename, 'r') as f:
            print(f.read())
        print("".center(80, '*'))
        return inp

    def get_interaction_mediatior_kwargs(self, inp=None):
        """Generates a kwarg dictionary for Interaction Mediator object.

        Parameters
        ----------
        inp : InputReader, optional
            InputReader object to extract supersystem settings from.
            (default is None)
        """

        if inp is None:
            inp = self.inp
        mediator_kwargs = {}
        # Setup supersystem method

        print (inp)
        # There is a way to do this that is way better. This works.
        env_method = inp.embed.env_method
        supersystem_kwargs['fs_method'] = env_method
        supersystem_kwargs['filename'] = inp.filename
        if inp.compare_density:
            supersystem_kwargs['compare_density'] = inp.compare_density
        if inp.scrdir is not None:
            supersystem_kwargs['scr_dir'] = inp.scrdir
        if inp.ppmem:
            supersystem_kwargs['pmem'] = inp.ppmem

        #The following are optional arguments.
        # There is also a better way to do this than conditional statements
        if inp.embed:
            if inp.embed.cycles:
                supersystem_kwargs['ft_cycles'] = inp.embed.cycles
            if inp.embed.conv:
                supersystem_kwargs['ft_conv'] = inp.embed.conv
            if inp.embed.grad:
                supersystem_kwargs['ft_grad'] = inp.embed.grad
            if inp.embed.damp:
                supersystem_kwargs['ft_damp'] = inp.embed.damp
            if not inp.embed.diis is None:
                supersystem_kwargs['ft_diis'] = inp.embed.diis
            if inp.embed.setfermi:
                supersystem_kwargs['ft_setfermi'] = (
                    inp.embed.setfermi)
            if inp.embed.initguess:
                supersystem_kwargs['ft_initguess'] = (
                    inp.embed.initguess)
            if inp.embed.updatefock:
                supersystem_kwargs['ft_updatefock'] = (
                    inp.embed.updatefock)
            if inp.embed.save_orbs:
                supersystem_kwargs['ft_save_orbs'] = (
                    inp.embed.save_orbs)
            if inp.embed.save_density:
                supersystem_kwargs['ft_save_density'] = (
                    inp.embed.save_density)
            if inp.embed.operator:
                supersystem_kwargs['proj_oper'] = inp.embed.operator

        if inp.fullsys_settings:
            if inp.fullsys_settings.unrestricted:
                supersystem_kwargs['fs_unrestricted'] = (
                    inp.fullsys_settings.unrestricted)
            if inp.fullsys_settings.save_orbs:
                supersystem_kwargs['fs_save_orbs'] = (
                    inp.fullsys_settings.save_orbs)
            if inp.fullsys_settings.save_density:
                supersystem_kwargs['fs_save_density'] = (
                    inp.fullsys_settings.save_density)
            if inp.fullsys_settings.cycles:
                supersystem_kwargs['fs_cycles'] = inp.fullsys_settings.cycles
            if inp.fullsys_settings.conv:
                supersystem_kwargs['fs_conv'] = inp.fullsys_settings.conv
            if inp.fullsys_settings.grad:
                supersystem_kwargs['fs_grad'] = inp.fullsys_settings.grad
            if inp.fullsys_settings.damp:
                supersystem_kwargs['fs_damp'] = inp.fullsys_settings.damp
            if inp.fullsys_settings.shift:
                supersystem_kwargs['fs_shift'] = inp.fullsys_settings.shift
            if inp.fullsys_settings.smearsigma:
                supersystem_kwargs['fs_smearsigma'] = (
                    inp.fullsys_settings.smearsigma)
            if inp.fullsys_settings.initguess:
                supersystem_kwargs['fs_initguess'] = (
                    inp.fullsys_settings.initguess)

        if inp.grid:
            supersystem_kwargs['grid_level'] = inp.grid
        if inp.rhocutoff:
            supersystem_kwargs['rhocutoff'] = inp.rhocutoff
        if inp.verbose:
            supersystem_kwargs['verbose'] = inp.verbose
        if inp.analysis:
            supersystem_kwargs['analysis'] = inp.analysis
        if inp.debug:
            supersystem_kwargs['debug'] = inp.debug

        return supersystem_kwargs

        

    def get_env_subsystem_kwargs(self, inp=None):
        """Generates a kwarg dictionary for ClusterEnvSubSystem object.

        Parameters
        ----------
        inp : InputReader, optional
            InputReader object to extract subsystem settings from.
            (default is None)
        """
       
        if inp is None:
            inp = self.inp
        # There is certainly a better way to do this. But this works.
        # subsystem universal settings
        universal_subsys_settings = {}
        universal_subsys_settings['filename'] = inp.filename
        universal_subsys_settings['env_method'] = inp.embed.env_method
        if inp.rhocutoff:
            universal_subsys_settings['rhocutoff'] = inp.rhocutoff
        if inp.grid:
            universal_subsys_settings['grid_level'] = inp.grid
        if inp.verbose:
            universal_subsys_settings['verbose'] = inp.verbose
        if inp.analysis:
            universal_subsys_settings['analysis'] = inp.analysis
        if inp.debug:
            universal_subsys_settings['debug'] = inp.debug

        if inp.ppmem:
            universal_subsys_settings['pmem'] = inp.ppmem
        if inp.scrdir:
            universal_subsys_settings['scr_dir'] = inp.scrdir
        
        env_subsystem_kwargs = []
        for subsystem in inp.subsystem:
            subsys_settings = {}
            if subsystem.unrestricted:
                subsys_settings['unrestricted'] = subsystem.unrestricted
            if subsystem.save_orbs:
                subsys_settings['save_orbs'] = subsystem.save_orbs
            if subsystem.save_density:
                subsys_settings['save_density'] = subsystem.save_density
            if subsystem.smearsigma:
                subsys_settings['smearsigma'] = subsystem.smearsigma
            if subsystem.damp:
                subsys_settings['damp'] = subsystem.damp
            if subsystem.shift:
                subsys_settings['shift'] = subsystem.shift
            if subsystem.subcycles:
                subsys_settings['subcycles'] = subsystem.subcycles
            if not subsystem.diis is None:
                subsys_settings['diis'] = subsystem.diis
            if subsystem.freeze:
                subsys_settings['freeze'] = subsystem.freeze
            if subsystem.initguess:
                subsys_settings['initguess'] = subsystem.initguess
            
            subsys_settings.update(universal_subsys_settings.copy())
            env_subsystem_kwargs.append(subsys_settings.copy())

        return env_subsystem_kwargs
            

    def get_active_subsystem_kwargs(self, inp=None):
        """Generates a kwarg dictionary for ClusterActiveSubSystem object.

        Parameters
        ----------
        inp : InputReader, optional
            InputReader object to extract active subsystem settings from.
            (default is None)
        """

        if inp is None:
            inp = self.inp

        active_subsystem_kwargs = {}
        active_subsystem_kwargs['active_method'] = inp.active_method
        if inp.cas_settings:
            if inp.cas_settings.localize_orbitals:
                active_subsystem_kwargs['localize_orbitals'] = (
                    inp.cas_settings.localize_orbitals)
            if inp.cas_settings.active_orbs:
                active_subsystem_kwargs['active_orbs'] = (
                    inp.cas_settings.active_orbs)

            if inp.cas_settings.avas:
                active_subsystem_kwargs['avas'] = (
                    inp.cas_settings.avas)

        if inp.shci_settings:
            if inp.shci_settings.mpi_prefix:
                active_subsystem_kwargs['shci_mpi_prefix'] = (
                    inp.shci_settings.mpi_prefix)
            if inp.shci_settings.no_stochastic:
                active_subsystem_kwargs['shci_stochastic'] = False
            active_subsystem_kwargs['shci_nPTiter'] = (
                inp.shci_settings.nPTiter)
            if inp.shci_settings.sweep_iter:
                active_subsystem_kwargs['shci_sweep_iter'] = (
                    inp.shci_settings.sweep_iter)
            if inp.shci_settings.NoRDM:
                active_subsystem_kwargs['shci_DoRDM'] = False
            if inp.shci_settings.sweep_epsilon:
                active_subsystem_kwargs['shci_sweep_epsilon'] = (
                    inp.shci_settings.sweep_epsilon)

        if inp.dmrg_settings:
            active_subsystem_kwargs['dmrg_maxM'] = (
                inp.dmrg_settings.maxM)
            if inp.dmrg_settings.memory:
                active_subsystem_kwargs['dmrg_memory'] = (
                    inp.dmrg_settings.memory)
            active_subsystem_kwargs['dmrg_num_thrds'] = (
                inp.dmrg_settings.num_thrds)

        if inp.active_settings:
            if inp.active_settings.unrestricted:
                active_subsystem_kwargs['active_unrestricted'] = (
                    inp.active_settings.unrestricted)
            if inp.active_settings.save_orbs:
                active_subsystem_kwargs['active_save_orbs'] = (
                    inp.active_settings.save_orbs)
            if inp.active_settings.save_density:
                active_subsystem_kwargs['active_save_density'] = (
                    inp.active_settings.save_density)
            if inp.active_settings.initguess:
                active_subsystem_kwargs['active_initguess'] = (
                    inp.active_settings.initguess)
            if inp.active_settings.conv:
                active_subsystem_kwargs['active_conv'] = (
                    inp.active_settings.conv)
            if inp.active_settings.grad:
                active_subsystem_kwargs['active_grad'] = (
                    inp.active_settings.grad)
            if inp.active_settings.cycles:
                active_subsystem_kwargs['active_cycles'] = (
                    inp.active_settings.cycles)
            if inp.active_settings.damp:
                active_subsystem_kwargs['active_damp'] = (
                    inp.active_settings.damp)
            if inp.active_settings.shift:
                active_subsystem_kwargs['active_shift'] = (
                    inp.active_settings.shift)

            active_subsystem_kwargs['use_molpro'] = (
                inp.active_settings.molpro)
            if inp.active_settings.compress_approx:
                active_subsystem_kwargs['compress_approx'] = (
                    inp.active_settings.compress_approx)

        return active_subsystem_kwargs

    def get_periodic_kwargs(self, inp=None):
        """Generates a kwarg dictionary to create PeriodicSupersystem, 
        PeriodicEnvSubsystem, and PeriodicEnvSubsystem objects.

        Parameters
        ----------
        inp : InputReader, optional
            InputReader object to extract supersystem settings from.
            (default is None)

        Returns
        -------
        cell_kwargs : dict
            Keywords necessary for generating a PySCF cell object
        kpoints_kwargs : dict
            Keywords necessary for generating k-points
        periodic_kwrags : dict
            Other periodic keywords
        """

        import numpy as np

        if inp is None:
            inp = self.inp
        if inp.periodic_settings is None:
            return None, None, None
        dimensions = inp.periodic_settings.dimensions

        cell_kwargs = {}
        kpoints_kwargs = {}
        periodic_kwargs = {}

        # cell kwargs
        cell_keys = {
                    'dimensions'        :           'dimension',
                    'auxbasis'          :           'auxbasis',
                    'exxdiv'            :           'exxdiv',
                    'precision'         :           'precision',
                    'ke_cutoff'         :           'ke_cutoff',
                    'rcut'              :           'rcut',
                    'low_dim_ft_type'   :           'low_dim_ft_type',
                    }

        # other periodic kwargs
        periodic_keys = {
                        'density_fit'       :       'density_fit',
                        'exp_to_discard'    :       'exp_to_discard',
                        }


        # get k-points arguments
        kpoints_kwargs = {}
        if inp.periodic_settings.kgroup.__class__ is tuple:
            kpoints_kwargs['kpoints'] = np.array(inp.periodic_settings.kgroup)
        else:
            kabs = np.array([r.group(0).split() for r in inp.periodic_settings.kgroup],
                dtype=float)
            kpoints_kwargs['kgroup'] = kabs

        # lattice vectors
        lattice = []
        for r in inp.periodic_settings.lattice_vectors.vector:
            lattice.append(np.array([r.group(1), r.group(2), r.group(3)], dtype=float))
        cell_kwargs['a'] = np.array(lattice)
        if len(cell_kwargs['a']) < 3:
            raise Exception("LATTICE_VECTORS must be a 3x3 array!")

        # read from all cell_keys to create cell_kwargs
        for key in cell_keys:
            value = getattr(inp.periodic_settings, key)
            if value is not None:
                cell_kwargs[cell_keys[key]] = value

        # read from all other keys to create periodic_kwargs
        for key in periodic_keys:
            value = getattr(inp.periodic_settings, key)
            if value is not None:
                periodic_kwargs[periodic_keys[key]] = value
        if inp.grid:
            periodic_kwargs['grid_level'] = inp.grid

        return cell_kwargs, kpoints_kwargs, periodic_kwargs


    #Add link basis functions when mol objects are generated.
    def gen_mols(self, inp=None):
        """Generates the mol or cell objects specified in inp..

        Parameters
        ----------
        inp : InputReader, optional
            InputReader object to extract mol information from.
            (default is None)
        """

        if inp is None:
            inp = self.inp

        subsys_mols = []
        subsys_ghost = []
        for subsystem in inp.subsystem:
            atom_list = subsystem.atoms
            if self.periodic_kwargs is None:
                mol = gto.Mole()
            else:
                mol = pbc.gto.Cell()
            mol.atom = []
            mol.ghosts = []
            nghost = 0

            for atom in atom_list:
                if ('ghost.' in atom.group(1).lower() 
                    or 'gh.' in atom.group(1).lower()):
                    atom_name = atom.group(1).split('.')[1]
                    nghost += 1
                    mol.ghosts.append(atom_name)
                    ghost_name = f'ghost:{atom_name}'
                    mol.atom.append([ghost_name, (float(atom.group(2)), 
                        float(atom.group(3)), float(atom.group(4)))])
                else:
                    atom_name = atom.group(1)
                    mol.atom.append([atom.group(1), (float(atom.group(2)), 
                        float(atom.group(3)), float(atom.group(4)))])

            if subsystem.charge:
                mol.charge = subsystem.charge
            if subsystem.spin:
                mol.spin = subsystem.spin
            if subsystem.unit:
                mol.unit = subsystem.unit
            if inp.verbose:
                mol.verbose = inp.verbose

            described_basis = {}
            if subsystem.basis is not None:
                for basis in subsystem.basis.basis_def:
                    basis_str = basis.group(0)
                    split_basis = basis_str.split()
                    described_basis[split_basis[0]] = split_basis[1]
                    
            if inp.basis is not None:
                for basis in inp.basis.basis_def:
                    basis_str = basis.group(0)
                    split_basis = basis_str.split()
                    if not split_basis[0] in described_basis.keys():
                        described_basis[split_basis[0]] = split_basis[1]
            if len(described_basis.keys()) == 0:
                print ("YOU HAVE NOT DEFINED A BASIS. USING 3-21g BY DEFAULT") 
                described_basis['default'] = '3-21g'

            described_ecp = {}
            if subsystem.ecp is not None:
                for ecp in subsystem.ecp.ecp_def:
                    ecp_str = ecp.group(0)
                    split_ecp = ecp_str.split()
                    described_ecp[split_ecp[0]] = split_ecp[1]
                    
            if inp.ecp is not None:
                for ecp in inp.ecp.ecp_def:
                    ecp_str = ecp.group(0)
                    split_ecp = ecp_str.split()
                    if not split_ecp[0] in described_ecp.keys():
                        described_ecp[split_ecp[0]] = split_ecp[1]

            mol.basis = described_basis
            mol.ecp = described_ecp

            if self.periodic_kwargs is None:
                mol.build(dump_input=False)

            # other options unique to periodic systems
            else:
                if inp.periodic_settings.exp_to_discard is not None:
                    mol.exp_to_discard = inp.periodic_settings.exp_to_discard
                if inp.periodic_settings.precision is not None:
                    mol.precision = inp.periodic_settings.precision
                mol.build(dump_input=False, **self.cell_kwargs)

            subsys_mols.append(mol) 
            subsys_ghost.append(nghost)

        #Add ghost link atoms Assumes angstroms.
        #max_bond_dist = 1.76
        for i in range(len(inp.subsystem)):
            subsystem = inp.subsystem[i]
        #    ghAtms = []
        #    ghost_link = 'H'
        #    if subsystem.addlinkbasis:
        #        mol1 = subsys_mols[i]
        #        for j in range(i + 1, len(inp.subsystem)):
        #            mol2 = subsys_mols[j]
        #            link_atoms = []
        #            link_basis = {}
        #            for k in range(len(mol1.atom)):
        #                atom1_coord = mol1.atom[k][1]
        #                for m in range(len(mol2.atom)):
        #                    atom2_coord = mol2.atom[m][1]
        #                    atom_dist = bond_dist(atom1_coord, atom2_coord) 
        #                    if atom_dist <= max_bond_dist:
        #                        ghost_num = subsys_ghost[i] + 1
        #                        if subsystem.basis:
        #                            new_atom, new_basis = gen_link_basis(
        #                                                      mol1.atom[k], 
        #                                                      mol2.atom[m], 
        #                                                      subsystem.basis)
        #                        else:
        #                            new_atom, new_basis = gen_link_basis(
        #                                                      mol1.atom[k], 
        #                                                      mol2.atom[m], 
        #                                                      inp.basis)
        #                        subsys_ghost[i] = ghost_num
        #                        ghAtms.append(ghost_link)
        #                        link_atoms.append(new_atom)
        #                        link_basis.update(new_basis)

        #            # Will not work if link atoms are explicitly defined.
        #            subsys_mols[i].atom = (subsys_mols[i].atom 
        #                                   + link_atoms)
        #            subsys_mols[i].ghosts += ghAtms
        #            subsys_mols[i]._basis.update(link_basis)
            subsys_mols[i].build(dump_input=False)

        return subsys_mols


def bond_dist(atom1_coord, atom2_coord):

    total = 0.0
    for i in range(len(atom1_coord)):
        total += (atom2_coord[i] - atom1_coord[i]) ** 2.
    return (total ** 0.5)


def gen_link_basis(atom1, atom2, basis):

    basis_atom = 'H'
    ghost_name = f'ghost:{basis_atom}'
    basis_x = (atom2[1][0] + atom1[1][0]) / 2.
    basis_y = (atom2[1][1] + atom1[1][1]) / 2.
    basis_z = (atom2[1][2] + atom1[1][2]) / 2.

    atm = [ghost_name, (basis_x, basis_y, basis_z)]
    basis = {ghost_name: gto.basis.load(basis, basis_atom)}
    return (atm, basis)
