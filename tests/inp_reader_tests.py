# Tests for the input reader object
# Daniel Graham


import unittest
import os
import shutil
import re

from qsome import inp_reader
from pyscf import gto


def_filename = "default.inp"
default_str = """
subsystem
He    0.0000    0.0000    0.0000
end

subsystem
C    2.0000    0.0000    0.0000
end

embed
 env_method pbe
 huzinaga
end

basis 3-21g
active_method hf
"""

exp_set_filename = 'exp_set.inp'
exp_set_str = """
subsystem
C    0.0000    0.0000    0.0000
charge -1
spin 1
basis aug-cc-pVDZ
smearsigma 0.1
unit angstrom
initguess minao
damp 0.2
shift 0.1
subcycles 4
end

subsystem
Si    2.0000    0.0000    0.0000
charge +2
spin -2
smearsigma 0.01
unit bohr
freeze
initguess supmol
damp 0.1
shift 0.2
end

embed
 env_method pbe
 huzfermi
 cycles 10
 conv 1e-4
 grad 1e-4
 diis 2
 updatefock 2
 initguess atom
 setfermi -4
end

ct_method u
basis 3-21g
ct_settings
 conv 1e-3
 grad 1e-2
 cycles 300
 damp 0.1
 shift 0.3
 smearsigma 0.2
 initguess 1e
 includeghost
end

active_method caspt2[2,2]
cas_settings
 localize_orbitals
 active_orbs [4,5]
end
 
active_settings
 conv 1e-10
 grad 1e-11
 cycles 500
 damp 0.1
 shift 0.2
end

rhocutoff 1e-4
grid 5
verbose 1
gencube test1
compden test2
analysis
debug
"""

hcore_guess_filename = 'hcore_guess.inp'
hcore_guess_str = default_str.replace("""
hl_method hf
""","""
hl_method hf
initguess 1e
""", 1)

chk_guess_filename = 'chk_guess.inp'
chk_guess_str = default_str.replace("""
hl_method hf
""","""
hl_method hf
initguess readchk
""", 1)

ccsd_filename = "ccsd_test.inp"
ccsd_str = default_str.replace("hl_method hf", """
h1_method ccsd""", 1)

cas_filename = "cas_test.inp"
cas_str = default_str.replace("hl_method hf", """
hl_method caspt2[2,2]
cas_settings
 localize_orbitals
 active_orbs [6,7]
end 
""", 1)

super_guess_filename = 'super_guess.inp'
super_guess_str = default_str.replace("""
basis 3-21g
""",
"""
basis 3-21g
initguess supmol
""", 1)

dft_hl_filename = 'dft_hl.inp'
dft_hl_str = default_str.replace("""
hl_method hf
""","""
hl_method pbe
""", 1)
dft_hl_xc_fun_filename = 'diff_dft_hl.inp'
dft_hl_xc_fun_str = default_str.replace("""
hl_method hf
""","""
hl_method m06
""", 1)
partial_ghost_filename = 'partial_ghost.inp'
partial_ghost_str = default_str.replace("""
He    0.0000    0.0000    0.0000
""","""
He    0.0000    0.0000    0.0000
gh.C    2.0000    0.0000    0.0000
""", 1)
ghost_filename = 'ghost.inp'
ghost_str = partial_ghost_str.replace("""
subsystem
C    2.0000    0.0000    0.0000
""","""
subsystem
C    2.0000    0.0000    0.0000
gh.He    0.0000    0.0000    0.0000
""", 1)

mixed_basis_filename = 'mixed_basis.inp'
mixed_basis_str = default_str.replace("""
He    0.0000    0.0000    0.0000
""","""
He    0.0000    0.0000    0.0000
basis 6-311g
""", 1)

subsys_charge_filename = 'subsys_charge.inp'
subsys_charge_str = """"""
subsys_spin_filename = 'subsys_spin.inp'
subsys_spin_str = """"""

multi_subsystem_str = """"""
ghost_multi_subsystem_str = """"""
subsystem_str = """"""
fullsystem_str = """"""
periodic_str = """"""
temp_inp_dir = "/temp_input/"

class TestInputReader(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+exp_set_filename, "w") as f:
            f.write(exp_set_str)

    def test_default_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        inp = in_obj.inp

        #Check atoms
        self.assertEqual(inp.subsystem[0].atoms[0].group(1), 'He')
        self.assertEqual(float(inp.subsystem[0].atoms[0].group(2)), 0.0)
        self.assertEqual(float(inp.subsystem[0].atoms[0].group(3)), 0.0)
        self.assertEqual(float(inp.subsystem[0].atoms[0].group(4)), 0.0)

        self.assertEqual(inp.subsystem[1].atoms[0].group(1), 'C')
        self.assertEqual(float(inp.subsystem[1].atoms[0].group(2)), 2.0)
        self.assertEqual(float(inp.subsystem[1].atoms[0].group(3)), 0.0)
        self.assertEqual(float(inp.subsystem[1].atoms[0].group(4)), 0.0)

        #Check operator and methods
        self.assertEqual(inp.embed.operator, 'huz')
        self.assertEqual(inp.embed.env_method, 'pbe')
        self.assertEqual(inp.active_method, 'hf')
        self.assertEqual(inp.basis, '3-21g')
        #self.assertEqual(inp.subsystem[0].atom_list,  
         
    def test_explicit_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + exp_set_filename)
        inp = in_obj.inp

        #Check Subsystems
        self.assertEqual(inp.subsystem[0].charge, -1) 
        self.assertEqual(inp.subsystem[0].spin, 1) 
        self.assertEqual(inp.subsystem[0].basis, 'aug-cc-pvdz') 
        self.assertEqual(inp.subsystem[0].smearsigma, 0.1) 
        self.assertEqual(inp.subsystem[0].unit, 'angstrom')
        self.assertEqual(inp.subsystem[0].subcycles, 4)
        self.assertEqual(inp.subsystem[0].damp, 0.2)
        self.assertEqual(inp.subsystem[0].shift, 0.1)
        self.assertEqual(inp.subsystem[0].initguess, 'minao')

        self.assertEqual(inp.subsystem[1].charge, 2)
        self.assertEqual(inp.subsystem[1].spin, -2) 
        self.assertEqual(inp.subsystem[1].smearsigma, 0.01) 
        self.assertEqual(inp.subsystem[1].unit, 'bohr')
        self.assertEqual(inp.subsystem[1].damp, 0.1)
        self.assertEqual(inp.subsystem[1].shift, 0.2)
        self.assertTrue(inp.subsystem[1].freeze)
        self.assertEqual(inp.subsystem[1].initguess, 'supmol')

        #Check embedding settings
        self.assertEqual(inp.embed.env_method, 'pbe')
        self.assertEqual(inp.embed.operator, 'huzfermi')
        self.assertEqual(inp.embed.cycles, 10)
        self.assertEqual(inp.embed.conv, 1e-4)
        self.assertEqual(inp.embed.grad, 1e-4)
        self.assertEqual(inp.embed.diis, 2)
        self.assertEqual(inp.embed.updatefock, 2)
        self.assertEqual(inp.embed.initguess, 'atom')
        self.assertEqual(inp.embed.setfermi, -4.0)

        #Check CT settings
        self.assertEqual(inp.basis, '3-21g')
        self.assertEqual(inp.ct_method, 'u')
        self.assertEqual(inp.ct_settings.conv, 1e-3)
        self.assertEqual(inp.ct_settings.grad, 1e-2)
        self.assertEqual(inp.ct_settings.cycles, 300)
        self.assertEqual(inp.ct_settings.damp, 0.1)
        self.assertEqual(inp.ct_settings.shift, 0.3)
        self.assertEqual(inp.ct_settings.smearsigma, 0.2)
        self.assertEqual(inp.ct_settings.initguess, '1e')

        #Check active settings
        self.assertEqual(inp.active_method, 'caspt2[2,2]')
        self.assertEqual(inp.active_settings.conv, 1e-10)
        self.assertEqual(inp.active_settings.grad, 1e-11)
        self.assertEqual(inp.active_settings.cycles, 500)
        self.assertEqual(inp.active_settings.damp, 0.1)
        self.assertEqual(inp.active_settings.shift, 0.2)

        #Check system settings
        self.assertEqual(inp.grid, 5)
        self.assertEqual(inp.rhocutoff, 1e-4)
        self.assertEqual(inp.verbose, 1)
        self.assertEqual(inp.gencube, 'test1')
        self.assertEqual(inp.compden, 'test2')
        self.assertTrue(inp.analysis)
        self.assertTrue(inp.debug)
    

    def test_cas_settings(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + exp_set_filename)
        inp = in_obj.inp

        self.assertTrue(inp.cas_settings.localize_orbitals)
        self.assertEqual(inp.cas_settings.active_orbs, [4,5])

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestSuperSystemKwargs(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+exp_set_filename, "w") as f:
            f.write(exp_set_str)

    def test_default_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        sup_kwargs = in_obj.supersystem_kwargs

        self.assertEqual(sup_kwargs['ct_method'], 'rpbe')
        self.assertEqual(sup_kwargs['proj_oper'], 'huz')
        self.assertEqual(sup_kwargs['filename'], path + def_filename)


    def test_exp_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + exp_set_filename)
        sup_kwargs = in_obj.supersystem_kwargs

        # Embedding kwargs
        self.assertEqual(sup_kwargs['proj_oper'], 'huzfermi')
        self.assertEqual(sup_kwargs['ft_cycles'], 10)
        self.assertEqual(sup_kwargs['ft_conv'], 1e-4)
        self.assertEqual(sup_kwargs['ft_grad'], 1e-4)
        self.assertEqual(sup_kwargs['ft_diis'], 2)
        self.assertEqual(sup_kwargs['ft_updatefock'], 2)
        self.assertEqual(sup_kwargs['ft_initguess'], 'atom')
        self.assertEqual(sup_kwargs['ft_setfermi'], -4.)

        # Supersystem environment calculation kwargs
        self.assertEqual(sup_kwargs['ct_method'], 'upbe')
        self.assertEqual(sup_kwargs['cycles'], 300)
        self.assertEqual(sup_kwargs['conv'], 1e-3)
        self.assertEqual(sup_kwargs['grad'], 1e-2)
        self.assertEqual(sup_kwargs['damp'], 0.1)
        self.assertEqual(sup_kwargs['shift'], 0.3)
        self.assertEqual(sup_kwargs['initguess'], '1e')
        self.assertEqual(sup_kwargs['includeghost'], True)

        #System settings
        self.assertEqual(sup_kwargs['filename'], path + exp_set_filename)
        self.assertEqual(sup_kwargs['grid_level'], 5)
        self.assertEqual(sup_kwargs['rhocutoff'], 1e-4)
        self.assertEqual(sup_kwargs['verbose'], 1)
        self.assertEqual(sup_kwargs['analysis'], True)
        self.assertEqual(sup_kwargs['debug'], True)

       

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    


class TestEnvSubSystemKwargs(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+exp_set_filename, "w") as f:
            f.write(exp_set_str)

    def test_default_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        sub_kwargs = in_obj.env_subsystem_kwargs[0]

        self.assertEqual(sub_kwargs['env_method'], 'pbe')
        self.assertEqual(sub_kwargs['filename'], path + def_filename)

        sub_kwargs = in_obj.env_subsystem_kwargs[1]

        self.assertEqual(sub_kwargs['env_method'], 'pbe')
        self.assertEqual(sub_kwargs['filename'], path + def_filename)

    def test_exp_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + exp_set_filename)
        sub_kwargs = in_obj.env_subsystem_kwargs[0]

        self.assertEqual(sub_kwargs['env_method'], 'pbe')
        self.assertEqual(sub_kwargs['filename'], path + exp_set_filename)

        # subsystem specific options
        self.assertEqual(sub_kwargs['smearsigma'], 0.1)
        self.assertEqual(sub_kwargs['initguess'], 'minao')
        self.assertEqual(sub_kwargs['damp'], 0.2)
        self.assertEqual(sub_kwargs['shift'], 0.1)
        self.assertEqual(sub_kwargs['subcycles'], 4)

        # other options
        self.assertEqual(sub_kwargs['grid_level'], 5)
        self.assertEqual(sub_kwargs['rhocutoff'], 1e-4)
        self.assertEqual(sub_kwargs['verbose'], 1)
        self.assertEqual(sub_kwargs['analysis'], True)
        self.assertEqual(sub_kwargs['debug'], True)

        sub_kwargs = in_obj.env_subsystem_kwargs[1]
        self.assertEqual(sub_kwargs['env_method'], 'pbe')
        self.assertEqual(sub_kwargs['filename'], path + exp_set_filename)

        # subsystem specific options
        self.assertEqual(sub_kwargs['smearsigma'], 0.01)
        self.assertEqual(sub_kwargs['initguess'], 'supmol')
        self.assertEqual(sub_kwargs['freeze'], True)

        # other options
        self.assertEqual(sub_kwargs['damp'], 0.1)
        self.assertEqual(sub_kwargs['shift'], 0.2)
        self.assertEqual(sub_kwargs['grid_level'], 5)
        self.assertEqual(sub_kwargs['rhocutoff'], 1e-4)
        self.assertEqual(sub_kwargs['verbose'], 1)
        self.assertEqual(sub_kwargs['analysis'], True)
        self.assertEqual(sub_kwargs['debug'], True)

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestActiveSubSystemKwargs(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+exp_set_filename, "w") as f:
            f.write(exp_set_str)

    def test_default_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        sub_kwargs = in_obj.active_subsystem_kwargs

        self.assertEqual(sub_kwargs['active_method'], 'hf')

    def test_exp_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + exp_set_filename)
        sub_kwargs = in_obj.active_subsystem_kwargs
        self.assertEqual(sub_kwargs['active_method'], 'caspt2[2,2]')
        self.assertEqual(sub_kwargs['localize_orbitals'], True)
        self.assertEqual(sub_kwargs['active_orbs'], [4,5])

        # other options
        self.assertEqual(sub_kwargs['active_conv'], 1e-10)
        self.assertEqual(sub_kwargs['active_grad'], 1e-11)
        self.assertEqual(sub_kwargs['active_cycles'], 500)
        self.assertEqual(sub_kwargs['active_damp'], 0.1)
        self.assertEqual(sub_kwargs['active_shift'], 0.2)

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestGenMols(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+exp_set_filename, "w") as f:
            f.write(exp_set_str)

        with open(path+partial_ghost_filename, "w") as f:
            f.write(partial_ghost_str)

        with open(path+ghost_filename, "w") as f:
            f.write(ghost_str)

        with open(path+mixed_basis_filename, "w") as f:
            f.write(mixed_basis_str)

    def test_def_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        mols = in_obj.subsys_mols

        curr_atom = in_obj.inp.subsystem[0].atoms[0]
        self.assertEqual(mols[0].atom[0][0], curr_atom.group(1))
        self.assertEqual(mols[0].atom[0][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[0].atom[0][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[0].atom[0][1][2], float(curr_atom.group(4)))

        he_pyscf_basis = {'He': gto.basis.load(in_obj.inp.basis, 'He')} 
        self.assertEqual(mols[0].basis, he_pyscf_basis)
        self.assertEqual(mols[0].charge, 0)
        self.assertEqual(mols[0].spin, 0)
        self.assertEqual(mols[0].unit, 'angstrom')

        curr_atom = in_obj.inp.subsystem[1].atoms[0]
        self.assertEqual(mols[1].atom[0][0], curr_atom.group(1))
        self.assertEqual(mols[1].atom[0][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[1].atom[0][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[1].atom[0][1][2], float(curr_atom.group(4)))

        c_pyscf_basis = {'C': gto.basis.load(in_obj.inp.basis, 'C')} 
        self.assertEqual(mols[1].basis, c_pyscf_basis)
        self.assertEqual(mols[1].charge, 0)
        self.assertEqual(mols[1].spin, 0)
        self.assertEqual(mols[1].unit, 'angstrom')

    def test_exp_set_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + exp_set_filename)
        mols = in_obj.subsys_mols

        curr_atom = in_obj.inp.subsystem[0].atoms[0]
        self.assertEqual(mols[0].atom[0][0], curr_atom.group(1))
        self.assertEqual(mols[0].atom[0][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[0].atom[0][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[0].atom[0][1][2], float(curr_atom.group(4)))

        c_pyscf_basis = {'C': gto.basis.load(in_obj.inp.subsystem[0].basis, 'C')} 
        self.assertEqual(mols[0].basis, c_pyscf_basis)
        self.assertEqual(mols[0].charge, -1)
        self.assertEqual(mols[0].spin, 1)
        self.assertEqual(mols[0].unit, 'angstrom')

        curr_atom = in_obj.inp.subsystem[1].atoms[0]
        self.assertEqual(mols[1].atom[0][0], curr_atom.group(1))
        self.assertEqual(mols[1].atom[0][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[1].atom[0][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[1].atom[0][1][2], float(curr_atom.group(4)))

        si_pyscf_basis = {'Si': gto.basis.load(in_obj.inp.basis, 'Si')} 
        self.assertEqual(mols[1].basis, si_pyscf_basis)
        self.assertEqual(mols[1].charge, 2)
        self.assertEqual(mols[1].spin, -2)
        self.assertEqual(mols[1].unit, 'bohr')
        
    def test_partial_ghost_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + partial_ghost_filename)
        mols = in_obj.subsys_mols

        curr_atom = in_obj.inp.subsystem[0].atoms[0]
        self.assertEqual(mols[0].atom[0][0], curr_atom.group(1))
        self.assertEqual(mols[0].atom[0][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[0].atom[0][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[0].atom[0][1][2], float(curr_atom.group(4)))

        curr_atom = in_obj.inp.subsystem[1].atoms[0]
        self.assertEqual(mols[0].atom[1][0], 'ghost:1')
        self.assertEqual(mols[0].atom[1][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[0].atom[1][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[0].atom[1][1][2], float(curr_atom.group(4)))

        he_pyscf_basis = {'He': gto.basis.load(in_obj.inp.basis, 'He')} 
        c_pyscf_basis = {'C': gto.basis.load(in_obj.inp.basis, 'C')} 
        self.assertEqual(mols[0].basis['He'], he_pyscf_basis['He'])
        self.assertEqual(mols[0].basis['ghost:1'], c_pyscf_basis['C'])
        self.assertEqual(mols[0].charge, 0)
        self.assertEqual(mols[0].spin, 0)
        self.assertEqual(mols[0].unit, 'angstrom')

    def test_ghost_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + ghost_filename)
        mols = in_obj.subsys_mols

        curr_atom = in_obj.inp.subsystem[0].atoms[0]
        self.assertEqual(mols[0].atom[0][0], curr_atom.group(1))
        self.assertEqual(mols[0].atom[0][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[0].atom[0][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[0].atom[0][1][2], float(curr_atom.group(4)))

        curr_atom = in_obj.inp.subsystem[1].atoms[0]
        self.assertEqual(mols[0].atom[1][0], 'ghost:1')
        self.assertEqual(mols[0].atom[1][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[0].atom[1][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[0].atom[1][1][2], float(curr_atom.group(4)))

        he_pyscf_basis = {'He': gto.basis.load(in_obj.inp.basis, 'He')} 
        c_pyscf_basis = {'C': gto.basis.load(in_obj.inp.basis, 'C')} 
        self.assertEqual(mols[0].basis['He'], he_pyscf_basis['He'])
        self.assertEqual(mols[0].basis['ghost:1'], c_pyscf_basis['C'])

        curr_atom = in_obj.inp.subsystem[1].atoms[0]
        self.assertEqual(mols[1].atom[0][0], curr_atom.group(1))
        self.assertEqual(mols[1].atom[0][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[1].atom[0][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[1].atom[0][1][2], float(curr_atom.group(4)))

        curr_atom = in_obj.inp.subsystem[0].atoms[0]
        self.assertEqual(mols[1].atom[1][0], 'ghost:1')
        self.assertEqual(mols[1].atom[1][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[1].atom[1][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[1].atom[1][1][2], float(curr_atom.group(4)))

        he_pyscf_basis = {'He': gto.basis.load(in_obj.inp.basis, 'He')} 
        self.assertEqual(mols[1].basis['C'], c_pyscf_basis['C'])
        self.assertEqual(mols[1].basis['ghost:1'], he_pyscf_basis['He'])

        self.assertEqual(mols[0].charge, 0)
        self.assertEqual(mols[0].spin, 0)
        self.assertEqual(mols[0].unit, 'angstrom')

    def test_mixed_basis_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + mixed_basis_filename)
        mols = in_obj.subsys_mols

        curr_atom = in_obj.inp.subsystem[0].atoms[0]
        self.assertEqual(mols[0].atom[0][0], curr_atom.group(1))
        self.assertEqual(mols[0].atom[0][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[0].atom[0][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[0].atom[0][1][2], float(curr_atom.group(4)))

        he_pyscf_basis = {'He': gto.basis.load(in_obj.inp.subsystem[0].basis, 'He')} 
        self.assertEqual(mols[0].basis, he_pyscf_basis)
        self.assertEqual(mols[0].charge, 0)
        self.assertEqual(mols[0].spin, 0)
        self.assertEqual(mols[0].unit, 'angstrom')

        curr_atom = in_obj.inp.subsystem[1].atoms[0]
        self.assertEqual(mols[1].atom[0][0], curr_atom.group(1))
        self.assertEqual(mols[1].atom[0][1][0], float(curr_atom.group(2)))
        self.assertEqual(mols[1].atom[0][1][1], float(curr_atom.group(3)))
        self.assertEqual(mols[1].atom[0][1][2], float(curr_atom.group(4)))

        he_pyscf_basis = {'He': gto.basis.load(in_obj.inp.basis, 'He')} 
        c_pyscf_basis = {'C': gto.basis.load(in_obj.inp.basis, 'C')} 
        self.assertEqual(mols[1].basis, c_pyscf_basis)
        self.assertEqual(mols[1].charge, 0)
        self.assertEqual(mols[1].spin, 0)
        self.assertEqual(mols[1].unit, 'angstrom')

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

if __name__ == "__main__":
    unittest.main()
