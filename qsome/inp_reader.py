#!/usr/bin/env python
# A module to define the input reader object
# Daniel Graham

from __future__ import print_function, division

import input_reader
import sys
import re

from pyscf import gto


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
    subsystem_mols : list
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
        self.supersystem_kwargs = self.get_supersystem_kwargs()
        self.env_subsystem_kwargs = self.get_env_subsystem_kwargs()
        self.active_subsystem_kwargs = self.get_active_subsystem_kwargs()
        self.subsystem_mols = self.gen_mols()

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

        # May be shortened by using repeating regex.
        subsys.add_regex_line(
         'atoms',
         '\s*([A-Za-z.]+)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)',
         repeat=True)
        subsys.add_line_key('charge', type=int)     
        subsys.add_line_key('spin', type=int)      
        subsys.add_line_key('basis')              
        subsys.add_line_key('smearsigma', type=float)   # fermi smearing sigma
        subsys.add_line_key('unit', type=('angstrom','a','bohr','b')) 
        subsys.add_boolean_key('freeze')
        subsys.add_line_key('initguess', type=('minao', 'atom', '1e', 
            'readchk', 'supmol', 'submol'))
        subsys.add_line_key('damp', type=float) # subsys damping parameter
        subsys.add_line_key('shift', type=float) # SCF level-shift parameter
        subsys.add_line_key('subcycles', type=int) # num subsys diag. cycles
        subsys.add_line_key('diis', type=int) # DIIS for subsystem (0 for off)
        subsys.add_boolean_key('addlinkbasis') # Add link H basis functions

        # Freeze and thaw settings
        embed = reader.add_block_key('embed', required=True)
        embed.add_line_key('cycles', type=int) 
        embed.add_line_key('conv', type=float)
        embed.add_line_key('grad', type=float)
        embed.add_line_key('env_method', type=str) #low-level method
        embed.add_line_key('diis', type=int) # Use DIIS for fock. (0 for off)
        # Supersystem fock update frequency. 
        # 0 is after F&T cycle, 1 is after every subsystem iteration
        embed.add_line_key('updatefock', type=int)
        embed.add_line_key('initguess', type=(
            'minao', 'atom', '1e', 'readchk', 'supmol', 'submol', 'localsup'))
        # Output subsystem orbitals after F&T cycles
        embed.add_boolean_key('writeorbs') 

        # This section needs work. Should be uniform option for setting op.
        operator = embed.add_mutually_exclusive_group(dest='operator', 
                                                      required=True)
        operator.add_line_key('mu', type=float, default=1e6)
        operator.add_boolean_key('manby', action=1e6)
        operator.add_boolean_key('huzinaga', action='huz')
        operator.add_boolean_key('hm', action='hm')
        # Fermi shifted
        operator.add_boolean_key('huzinagafermi', action='huzfermi')
        operator.add_boolean_key('huzfermi', action='huzfermi')
        embed.add_line_key('setfermi', type=float)

        reader.add_line_key('memory', type=(int, float)) # MB
        reader.add_line_key('unit', type=('angstrom','a','bohr','b')) 
        reader.add_line_key('basis')
        # Supersystem method restricted, restricted os, unrestricted
        reader.add_line_key('ct_method', type=('r', 'ro', 'u'), default='r')

        # Supersystem calculation settings
        ct_settings = reader.add_block_key('ct_settings')
        ct_settings.add_line_key('conv', type=float)
        ct_settings.add_line_key('grad', type=float)
        ct_settings.add_line_key('cycles', type=int)
        ct_settings.add_line_key('damp', type=float)
        ct_settings.add_line_key('shift', type=float)
        ct_settings.add_line_key('smearsigma', type=float)
        ct_settings.add_line_key('initguess', type=('minao', 'atom', '1e', 
                                                    'readchk', 'supmol'))

        # High level subsystem settings.
        reader.add_line_key('active_method', type=str, required=True)
        active_settings = reader.add_block_key('active_settings')
        active_settings.add_line_key('conv', type=float)       
        active_settings.add_line_key('grad', type=float)       
        active_settings.add_line_key('cycles', type=int)       
        active_settings.add_line_key('damp', type=float)
        active_settings.add_line_key('shift', type=float)
        active_settings.add_boolean_key('molpro')
        active_settings.add_boolean_key('writeorbs')

        cas_settings = reader.add_block_key('cas_settings')
        # Localize HF orbitals prior to CAS
        cas_settings.add_boolean_key('localize_orbitals')
        #A list to specity active orbitals by number. Ex. [5,6]
        cas_settings.add_line_key('active_orbs', type=str, default='') 

        reader.add_line_key('grid', type=int)
        reader.add_line_key('rhocutoff', type=float)
        reader.add_line_key('verbose', type=int)
        reader.add_line_key('gencube', type=str) # Generate final cubefile
        reader.add_line_key('compden', type=str) # Compare DFT-in-DFT to KS-DFT
        reader.add_boolean_key('analysis')
        reader.add_boolean_key('debug')

        inp  = reader.read_input(filename)
        inp.filename = filename 
        inp.active_method = inp.active_method.lower()
        inp.embed.env_method = inp.embed.env_method.lower()

        # Extract CAS space
        if re.match(re.compile('cas(pt2)?\[.*\].*'), inp.active_method):
            cas_str = inp.active_method
            inp.cas_space = [int(i.strip()) 
                             for i in (cas_str[cas_str.find("[") + 1:
                                               cas_str.find("]")]).split(',')]

        # This could be done better.
        if inp.cas_settings and inp.cas_settings.active_orbs:
            inp.cas_settings.active_orbs = eval(inp.cas_settings.active_orbs)

        print("".center(80, '*'))
        print("Input File".center(80))
        print("".center(80, '*'))
        with open(filename, 'r') as f:
            print(f.read())
        print("".center(80, '*'))
        return inp

    def get_supersystem_kwargs(self, inp=None):
        """Generates a kwarg dictionary for ClusterSupersystem object.

        Parameters
        ----------
        inp : InputReader, optional
            InputReader object to extract supersystem settings from.
            (default is None)
        """

        if inp is None:
            inp = self.inp
        supersystem_kwargs = {}
        if not inp.ct_method:
            spin = 0
            for system in inp.subsystem:
                if system.spin:
                    spin += system.spin
            if spin != 0:
                inp.ct_method = 'ro'
            else:
                inp.ct_method = 'r'

        # Setup supersystem method
        # There is a way to do this that is way better. This works.
        env_method = inp.embed.env_method
        if env_method[:2] == 'ro':
            supersystem_kwargs['ct_method'] = (inp.ct_method 
                                               + env_method[2:])
        elif env_method[:1] == 'u' or env_method[:1] == 'r':
            supersystem_kwargs['ct_method'] = (inp.ct_method
                                               + env_method[1:])
        else:
            supersystem_kwargs['ct_method'] = (inp.ct_method
                                                   + env_method)

        supersystem_kwargs['proj_oper'] = inp.embed.operator   
        supersystem_kwargs['filename'] = inp.filename

        #The following are optional arguments.
        # There is also a better way to do this than conditional statements
        if inp.embed:
            if inp.embed.cycles:
                supersystem_kwargs['ft_cycles'] = inp.embed.cycles
            if inp.embed.conv:
                supersystem_kwargs['ft_conv'] = inp.embed.conv
            if inp.embed.grad:
                supersystem_kwargs['ft_grad'] = inp.embed.grad
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

            supersystem_kwargs['ft_writeorbs'] = inp.embed.writeorbs

        if inp.ct_settings:
            if inp.ct_settings.cycles:
                supersystem_kwargs['cycles'] = inp.ct_settings.cycles
            if inp.ct_settings.conv:
                supersystem_kwargs['conv'] = inp.ct_settings.conv
            if inp.ct_settings.grad:
                supersystem_kwargs['grad'] = inp.ct_settings.grad
            if inp.ct_settings.damp:
                supersystem_kwargs['damp'] = inp.ct_settings.damp
            if inp.ct_settings.shift:
                supersystem_kwargs['shift'] = inp.ct_settings.shift
            if inp.ct_settings.smearsigma:
                supersystem_kwargs['smearsigma'] = (
                    inp.ct_settings.smearsigma)
            if inp.ct_settings.initguess:
                supersystem_kwargs['initguess'] = (
                    inp.ct_settings.initguess)

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
        
        env_subsystem_kwargs = []
        for subsystem in inp.subsystem:
            subsys_settings = {}
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

        if inp.active_settings:
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
            active_subsystem_kwargs['writeorbs'] = (
                inp.active_settings.writeorbs)

        return active_subsystem_kwargs

    #Add link basis functions when mol objects are generated.
    def gen_mols(self, inp=None):
        """Generates the mol objects specified in inp..

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
            mol = gto.Mole()
            mol.atom = []
            mol.ghosts = []
            mol.basis = {}
            nghost = 0

            if subsystem.basis:
                basis = subsystem.basis
            else:
                basis = inp.basis

            for atom in atom_list:
                if ('ghost.' in atom.group(1).lower() 
                    or 'gh.' in atom.group(1).lower()):
                    atom_name = atom.group(1).split('.')[1]
                    nghost += 1
                    mol.ghosts.append(atom_name)
                    ghost_name = f'ghost:{nghost}'
                    mol.atom.append([ghost_name, (float(atom.group(2)), 
                        float(atom.group(3)), float(atom.group(4)))])
                    mol.basis.update({ghost_name: gto.basis.load(basis, 
                                                                 atom_name)})
                else:
                    atom_name = atom.group(1)
                    mol.atom.append([atom.group(1), (float(atom.group(2)), 
                        float(atom.group(3)), float(atom.group(4)))])
                    mol.basis.update({atom_name: gto.basis.load(basis, 
                                                                atom_name)})

            if subsystem.charge:
                mol.charge = subsystem.charge
            if subsystem.spin:
                mol.spin = subsystem.spin
            if subsystem.unit:
                mol.unit = subsystem.unit
            if inp.verbose:
                mol.verbose = inp.verbose
            mol.build(dump_input=False)
            subsys_mols.append(mol) 
            subsys_ghost.append(nghost)

        #Add ghost link atoms Assumes angstroms.
        max_bond_dist = 1.76
        for i in range(len(inp.subsystem)):
            subsystem = inp.subsystem[i]
            ghAtms = []
            ghost_link = 'H'
            if subsystem.addlinkbasis:
                mol1 = subsys_mols[i]
                for j in range(i + 1, len(inp.subsystem)):
                    mol2 = subsys_mols[j]
                    link_atoms = []
                    link_basis = {}
                    for k in range(len(mol1.atom)):
                        atom1_coord = mol1.atom[k][1]
                        for m in range(len(mol2.atom)):
                            atom2_coord = mol2.atom[m][1]
                            atom_dist = bond_dist(atom1_coord, atom2_coord) 
                            if atom_dist <= max_bond_dist:
                                ghost_num = subsys_ghost[i] + 1
                                if subsystem.basis:
                                    new_atom, new_basis = gen_link_basis(
                                                              mol1.atom[k], 
                                                              mol2.atom[m], 
                                                              subsystem.basis, 
                                                              ghost_num)
                                else:
                                    new_atom, new_basis = gen_link_basis(
                                                              mol1.atom[k], 
                                                              mol2.atom[m], 
                                                              inp.basis, 
                                                              ghost_num)
                                subsys_ghost[i] = ghost_num
                                ghAtms.append(ghost_link)
                                link_atoms.append(new_atom)
                                link_basis.update(new_basis)

                    # Will not work if link atoms are explicitly defined.
                    subsys_mols[i].atom = (subsys_mols[i].atom 
                                           + link_atoms)
                    subsys_mols[i].ghosts += ghAtms
                    subsys_mols[i].basis.update(link_basis)
            subsys_mols[i].build(dump_input=False)

        return subsys_mols


def bond_dist(atom1_coord, atom2_coord):


    total = 0.0
    for i in range(len(atom1_coord)):
        total += (atom2_coord[i] - atom1_coord[i]) ** 2.
    return (total ** 0.5)


def gen_link_basis(atom1, atom2, basis, ghost_num):


    basis_atom = 'H'
    ghost_name = f'ghost:{ghost_num}'
    basis_x = (atom2[1][0] + atom1[1][0]) / 2.
    basis_y = (atom2[1][1] + atom1[1][1]) / 2.
    basis_z = (atom2[1][2] + atom1[1][2]) / 2.

    atm = [ghost_name, (basis_x, basis_y, basis_z)]
    basis = {ghost_name: gto.basis.load(basis, basis_atom)}
    return (atm, basis)
