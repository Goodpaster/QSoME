#!/usr/bin/env python
# A module to define the input reader object
# Daniel Graham

from __future__ import print_function, division

import input_reader
import sys
import re

from pyscf import gto

class InpReader:

    def __init__(self, filename):
        self.read_input(filename)

        self.get_supersystem_kwargs()
        self.get_env_subsystem_kwargs()
        self.get_active_subsystem_kwargs()
        self.gen_mols()

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
        subsys.add_line_key('damp', type=float)         # SCF damping parameter
        subsys.add_line_key('shift', type=float)        # SCF level-shift parameter
        subsys.add_line_key('subcycles', type=int)      # number of subsys diagonalizations

        # add embedding block
        embed = reader.add_block_key('embed', required=True)
        embed.add_line_key('cycles', type=int)         # max freeze-and-thaw cycles
        embed.add_line_key('conv', type=float)         # f&t conv tolerance
        embed.add_line_key('grad', type=float)         # f&t conv tolerance
        embed.add_line_key('env_method', type=str)      
        embed.add_line_key('diis', type=int)           # start DIIS (0 to turn off)
        embed.add_line_key('updatefock', type=int)    # frequency of updating fock matrix. 0 is after an embedding cycle, 1 is after every subsystem.
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
        ct_settings.add_line_key('smearsigma', type=float)   # supermolecular dft smearing
        ct_settings.add_line_key('initguess', type=('minao', 'atom', '1e', 'readchk', 'supmol'))            # Set initial guess supmol does supermolecular first and localizes.
        ct_settings.add_boolean_key('includeghost') 

        reader.add_line_key('active_method', type=str, required=True)                  # Active method
        active_settings = reader.add_block_key('active_settings')
        active_settings.add_line_key('conv', type=float)       
        active_settings.add_line_key('grad', type=float)       
        active_settings.add_line_key('cycles', type=int)       
        active_settings.add_line_key('damp', type=float)         # SCF damping parameter
        active_settings.add_line_key('shift', type=float)        # SCF level-shift parameter

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
            if self.inp.embed.setfermi:
                self.supersystem_kwargs['ft_setfermi'] = self.inp.embed.setfermi
            if self.inp.embed.initguess:
                self.supersystem_kwargs['ft_initguess'] = self.inp.embed.initguess
            if self.inp.embed.updatefock:
                self.supersystem_kwargs['ft_updatefock'] = self.inp.embed.updatefock

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
            if self.inp.ct_settings.smearsigma:
                self.supersystem_kwargs['smearsigma'] = self.inp.ct_settings.smearsigma
            if self.inp.ct_settings.initguess:
                self.supersystem_kwargs['initguess'] = self.inp.ct_settings.initguess
            if self.inp.ct_settings.includeghost:
                self.supersystem_kwargs['includeghost'] = self.inp.ct_settings.includeghost

        if self.inp.grid:
            self.supersystem_kwargs['grid_level'] = self.inp.grid
        if self.inp.verbose:
            self.supersystem_kwargs['verbose'] = self.inp.verbose
        if self.inp.analysis:
            self.supersystem_kwargs['analysis'] = self.inp.analysis
        if self.inp.debug:
            self.supersystem_kwargs['debug'] = self.inp.debug

        

    def get_env_subsystem_kwargs(self):
        
        # There is certainly a better way to do this. But this works.
        # subsystem universal settings
        universal_subsys_settings = {}
        universal_subsys_settings['filename'] = self.inp.filename
        universal_subsys_settings['env_method'] = self.inp.embed.env_method
        if self.inp.grid:
            universal_subsys_settings['grid_level'] = self.inp.grid
        if self.inp.verbose:
            universal_subsys_settings['verbose'] = self.inp.verbose
        if self.inp.analysis:
            universal_subsys_settings['analysis'] = self.inp.analysis
        if self.inp.debug:
            universal_subsys_settings['debug'] = self.inp.debug
        
        self.env_subsystem_kwargs = []
        for subsystem in self.inp.subsystem:
            subsys_settings = {}
            if subsystem.smearsigma:
                subsys_settings['smearsigma'] = subsystem.smearsigma
            if subsystem.damp:
                subsys_settings['damp'] = subsystem.damp
            if subsystem.shift:
                subsys_settings['shift'] = subsystem.shift
            if subsystem.subcycles:
                subsys_settings['subcycles'] = subsystem.subcycles
            if subsystem.freeze:
                subsys_settings['freeze'] = subsystem.freeze
            if subsystem.initguess:
                subsys_settings['initguess'] = subsystem.initguess
            
            subsys_settings.update(universal_subsys_settings.copy())
            self.env_subsystem_kwargs.append(subsys_settings.copy())
            



    def get_active_subsystem_kwargs(self):
        self.active_subsystem_kwargs = {}
        self.active_subsystem_kwargs['active_method'] = self.inp.active_method
        if self.inp.cas_settings:
            if self.inp.cas_settings.localize_orbitals:
                self.active_subsystem_kwargs['localize_orbitals'] = self.inp.cas_settings.localize_orbitals
            if self.inp.cas_settings.active_orbs:
                self.active_subsystem_kwargs['active_orbs'] = self.inp.cas_settings.active_orbs

        if self.inp.active_settings:
            if self.inp.active_settings.conv:
                self.active_subsystem_kwargs['active_conv'] = self.inp.active_settings.conv 
            if self.inp.active_settings.grad:
                self.active_subsystem_kwargs['active_grad'] = self.inp.active_settings.grad 
            if self.inp.active_settings.cycles:
                self.active_subsystem_kwargs['active_cycles'] = self.inp.active_settings.cycles
            if self.inp.active_settings.damp:
                self.active_subsystem_kwargs['active_damp'] = self.inp.active_settings.damp
            if self.inp.active_settings.shift:
                self.active_subsystem_kwargs['active_shift'] = self.inp.active_settings.shift
             
    
        
    def gen_mols(self):

        self.subsys_mols = []
        for subsystem in self.inp.subsystem:
            atom_list = subsystem.atoms
            mol = gto.Mole()
            mol.atom = []
            mol.ghosts = []
            mol.basis = {}
            nghost = 0

            if subsystem.basis:
                basis = subsystem.basis
            else:
                basis = self.inp.basis

            for atom in atom_list:
                if 'ghost.' in atom.group(1).lower() or 'gh.' in atom.group(1).lower():
                    atom_name = atom.group(1).split('.')[1]
                    nghost += 1
                    mol.ghosts.append(atom_name)
                    ghost_name = f'ghost:{nghost}'
                    mol.atom.append([ghost_name, (float(atom.group(2)), 
                        float(atom.group(3)), float(atom.group(4)))])
                    mol.basis.update({ghost_name: gto.basis.load(basis, atom_name)})
                else:
                    atom_name = atom.group(1)
                    mol.atom.append([atom.group(1), (float(atom.group(2)), 
                        float(atom.group(3)), float(atom.group(4)))])
                    mol.basis.update({atom_name: gto.basis.load(basis, atom_name)})

            if subsystem.charge:
                mol.charge = subsystem.charge
            if subsystem.spin:
                mol.spin = subsystem.spin
            if subsystem.unit:
                mol.unit = subsystem.unit
            if self.inp.verbose:
                mol.verbose = self.inp.verbose
            mol.build(dump_input=False)
            self.subsys_mols.append(mol) 
