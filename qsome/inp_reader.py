#!/usr/bin/env python
# A module to define the input reader object
# Daniel Graham

from __future__ import print_function, division

import input_reader
import sys
import re

class InpReader:

    def __init__(self, filename):
        self.read_input(filename)

        self.get_supersystem_kwargs()
        self.env_subsystem_kwargs = None
        self.active_subsystem_kwargs = None
        self.subsys_mols = []

    def read_input(self, filename):
        '''Reads a formatted pySCF input file, and generates
        an inp object with the relevant attributes.

        Input: <filename> the filename of the pySCF input file
        Output: a formatted inp object
        '''
        # initialize reader for a pySCF input
        reader = input_reader.InputReader(comment=['!', '#', '::', '//'],
                 case=False, ignoreunknown=False)

        # add subsystems block
        subsys = reader.add_block_key('subsystem', required=True, repeat=True)
        #subsys.add_regex_line('atoms', '\s*([A-Za-z.]+)(\s+(\-?\d+\.?\d*)){3}',
                              #repeat=True)

        subsys.add_regex_line('atoms',
            '\s*([A-Za-z.]+)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)\s+(\-?\d+\.?\d*)', #This can be shortened. No need to repeat the same regex pattern search.
            repeat=True)
        subsys.add_line_key('charge', type=int)     
        subsys.add_line_key('spin', type=int)      
        subsys.add_line_key('basis')              
        subsys.add_line_key('smearsigma', type=float)   # fermi smearing sigma  
        subsys.add_line_key('unit', type=('angstrom','a','bohr','b')) 
        subsys.add_boolean_key('freeze')                        
        subsys.add_line_key('initguess', type=('minao', 'atom', '1e', 'readchk', 'supmol'))            # Set initial guess supmol does supermolecular first and localizes.

        # add embedding block
        embed = reader.add_block_key('embed', required=True)
        embed.add_line_key('cycles', type=int)         # max freeze-and-thaw cycles
        embed.add_line_key('conv', type=float)         # f&t conv tolerance
        embed.add_line_key('grad', type=float)         # f&t conv tolerance
        embed.add_line_key('env_method', type=str)      
        embed.add_line_key('diis', type=int)           # start DIIS (0 to turn off)
        embed.add_line_key('subcycles', type=int)      # number of subsys diagonalizations
        embed.add_line_key('update_fock', type=int)    # frequency of updating fock matrix. 0 is after an embedding cycle, 1 is after every subsystem.
        embed.add_line_key('damp', type=float)         # SCF damping parameter
        embed.add_line_key('shift', type=float)        # SCF level-shift parameter
        embed.add_line_key('initguess', type=('minao', 'atom', '1e', 'readchk', 'supmol'))            # Set initial guess supmol does supermolecular first and localizes.

        # This section needs work.
        operator = embed.add_mutually_exclusive_group(dest='operator', required=True)
        operator.add_line_key('mu', type=float, default=1e6)        # manby operator by mu
        operator.add_boolean_key('manby', action=1e6)               # manby operator
        operator.add_boolean_key('huzinaga', action='huz')     # huzinaga operator
        operator.add_boolean_key('hm', action='hm')                 # modified huzinaga
        operator.add_boolean_key('huzinagafermi', action='huzfermi')# huzinaga-fermi shifted operator
        operator.add_boolean_key('huzfermi', action='huzfermi')     # huzinaga-fermi shifted operator
        embed.add_line_key('setfermi', type=float)

        # add simple line keys
        reader.add_line_key('memory', type=(int, float))             # max memory in MB
        reader.add_line_key('unit', type=('angstrom','a','bohr','b')) 
        reader.add_line_key('basis')                                 # global basis
        reader.add_line_key('ct_method', type=('r', 'ro', 'u'), default='r') # supersystem env method.

        ct_settings = reader.add_block_key('ct_settings')
        ct_settings.add_line_key('conv', type=float)
        ct_settings.add_line_key('grad', type=float)
        ct_settings.add_line_key('cycles', type=int)
        ct_settings.add_line_key('damp', type=float)         # SCF damping parameter
        ct_settings.add_line_key('shift', type=float)        # SCF level-shift parameter
        ct_settings.add_line_key('smearsig', type=float)   # supermolecular dft smearing
        ct_settings.add_line_key('initguess', type=('minao', 'atom', '1e', 'readchk', 'supmol'))            # Set initial guess supmol does supermolecular first and localizes.

        reader.add_line_key('active_method', type=str)                  # Active method
        active_settings = reader.add_block_key('active_settings')
        active_settings.add_line_key('conv', type=float)       
        active_settings.add_line_key('grad', type=float)       
        active_settings.add_line_key('cycles', type=int)       
        active_settings.add_line_key('damp', type=float)         # SCF damping parameter
        active_settings.add_line_key('shift', type=float)        # SCF level-shift parameter
        active_settings.add_line_key('smearsig', type=float)        # SCF level-shift parameter
        active_settings.add_line_key('initguess', type=('minao', 'atom', '1e', 'readchk', 'supmol'))            # Set initial guess supmol does supermolecular first and localizes.

        cas_settings = reader.add_block_key('cas_settings')         # CAS method settings
        cas_settings.add_boolean_key('localize_orbitals')           # Localize orbitals prior to CAS
        cas_settings.add_line_key('active_orbs', type=str, default='')         # Which orbitals to include in the active space

        reader.add_line_key('grid', type=int)            # becke integration grid
        reader.add_line_key('verbose', type=int)         # pySCF verbose level
        reader.add_line_key('gencube', type=str)              # generate Cube file
        reader.add_line_key('compden', type=str)              # compare densities

        # add simple boolean keys
        reader.add_boolean_key('analysis')                          # whether to do pySCF analysis
        reader.add_boolean_key('debug')                             # debug flag

        # read the input filename
        inp  = reader.read_input(filename)
        inp.filename = filename # save filename for later

        # some defaults
        inp.active_method = inp.active_method.lower()
        inp.embed.env_method = inp.embed.env_method.lower()

        # extract CAS space
        if re.match(re.compile('cas(pt2)?\[.*\].*'), inp.active_method):
            cas_str = inp.active_method
            inp.cas_space = [int(i) for i in (cas_str[cas_str.find("[") + 1:cas_str.find("]")]).split(',')]

        if inp.cas_settings and inp.cas_settings.active_orbs:
            inp.cas_settings.active_orbs = eval(inp.cas_settings.active_orbs)

        # print input file to output.
        print("".center(80, '*'))
        print("Input File".center(80))
        print("".center(80, '*'))
        with open(filename, 'r') as f:
            print(f.read())

        print("".center(80, '*'))
        
        self.inp = inp

    def get_supersystem_kwargs(self):
        self.supersystem_kwargs = {}
        if not self.inp.ct_method:
            spin = 0
            for system in self.inp.subsystem:
                if system.spin:
                    spin += system.spin
            if spin != 0:
                self.inp.ct_method = 'ro'
            else:
                self.inp.ct_method = 'r'

        # Convert method type to actual charge transfer method.
        # There is a way to do this that is way better. This works.
        env_method = self.inp.embed.env_method
        if env_method[:2] == 'ro':
            self.supersystem_kwargs['ct_method'] = self.inp.ct_method + env_method[2:]
        elif env_method[:1] == 'u' or env_method[:1] == 'r':
            self.supersystem_kwargs['ct_method'] = self.inp.ct_method + env_method[1:]
        else:
            self.supersystem_kwargs['ct_method'] = self.inp.ct_method + env_method

        
        self.supersystem_kwargs['proj_oper'] = self.inp.embed.operator   
        self.supersystem_kwargs['filename'] = self.inp.filename

        #The following are optional arguments.

        # There is also a better way to do this. rather than conditional statements
        if self.inp.embed:
            if self.inp.embed.cycles:
                self.supersystem_kwargs['ft_cycles'] = self.inp.embed.cycles
            if self.inp.embed.conv:
                self.supersystem_kwargs['ft_conv'] = self.inp.embed.conv
            if self.inp.embed.grad:
                self.supersystem_kwargs['ft_grad'] = self.inp.embed.grad
            if self.inp.embed.diis:
                self.supersystem_kwargs['ft_diis'] = self.inp.embed.diis
            if self.inp.embed.subcycles:
                self.supersystem_kwargs['ft_subcycles'] = self.inp.embed.subcycles
            if self.inp.embed.damp:
                self.supersystem_kwargs['ft_damp'] = self.inp.embed.damp
            if self.inp.embed.shift:
                self.supersystem_kwargs['ft_shift'] = self.inp.embed.shift
            if self.inp.embed.setfermi:
                self.supersystem_kwargs['ft_setfermi'] = self.inp.embed.setfermi
            if self.inp.embed.initguess:
                self.supersystem_kwargs['ft_initguess'] = self.inp.embed.initguess
            if self.inp.embed.update_fock:
                self.supersystem_kwargs['ft_update_fock'] = self.inp.embed.update_fock

        if self.inp.ct_settings:
            if self.inp.ct_settings.cycles:
                self.supersystem_kwargs['cycles'] = self.inp.ct_settings.cycles
            if self.inp.ct_settings.conv:
                self.supersystem_kwargs['conv'] = self.inp.ct_settings.conv
            if self.inp.ct_settings.grad:
                self.supersystem_kwargs['grad'] = self.inp.ct_settings.grad
            if self.inp.ct_settings.damp:
                self.supersystem_kwargs['damp'] = self.inp.ct_settings.damp
            if self.inp.ct_settings.shift:
                self.supersystem_kwargs['shift'] = self.inp.ct_settings.shift
            if self.inp.ct_settings.smearsig:
                self.supersystem_kwargs['smearsig'] = self.inp.ct_settings.smearsig
            if self.inp.ct_settings.initguess:
                self.supersystem_kwargs['initguess'] = self.inp.ct_settings.initguess

        if self.inp.grid:
            self.supersystem_kwargs['grid'] = self.inp.grid
        if self.inp.verbose:
            self.supersystem_kwargs['verbose'] = self.inp.verbose
        if self.inp.analysis:
            self.supersystem_kwargs['analysis'] = self.inp.analysis
        if self.inp.debug:
            self.supersystem_kwargs['debug'] = self.inp.debug

        

    def get_env_subsystem_args(self):
        pass
    def get_active_subsystem_args(self):
        pass
    def gen_mols():
        self.subsys_mols = []
        for subsystem in self.inp.subsystems:
            atom_list = subsystem.atom_list
            mol = gto.Mole()
            mol.atom = []
            mol.ghosts = []
            mol.basis = {}
            ghbasis = []
            for atom in atom_list:
                if 'ghost.' in atom.group(1).lower() or 'gh.' in atom.group(1).lower():
                    atom_name = atom.group(1).split('.')[1]
                    ghbasis.append(atom_name)
                    mol.ghosts.append(atom_name)
                    ghost_name = 'ghost:{0}'.format(len(ghbasis))
                    mol.atom.append([ghost_name, (float(atom.group(2)), 
                        float(atom.group(3)), float(atom.group(4)))])
                    mol.basis.update({ghost_name: gto.basis.load(basis, atom_name)})
                else:
                    atom_name = atom.group(1)
                    mol.atom.append([atom.group(1), (float(atom.group(2)), 
                        float(atom.group(3)), float(atom.group(4)))])
                    mol.basis.update({atom_name: gto.basis.load(basis, atom_name)})

            mol.charge = subsystem.charge
            mol.spin = spin
            mol.units = units
            mol.verbose = verbose
            mol.build(dump_input=False)
        pass
