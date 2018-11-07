#A module to test system inits.
# Daniel Graham


import unittest
import os
import shutil
import re

from qsome import inp_reader, cluster_subsystem, cluster_supersystem
from pyscf import gto, lib

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
Be    0.0000    0.0000    0.0000
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
Mg    2.0000    0.0000    0.0000
charge +1
spin -1
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

grid 5
rhocutoff 1e-4
verbose 1
gencube test1
compden test2
analysis
debug
"""

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

temp_inp_dir = "/temp_input/"
class TestSuperSystem(unittest.TestCase):
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

    def test_concat_mols_def(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)

        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)

        # Default
        concat_mol = supersystem.mol

        self.assertEqual(concat_mol.charge, subsystems[0].mol.charge + subsystems[1].mol.charge)
        self.assertEqual(concat_mol.spin, subsystems[0].mol.spin + subsystems[1].mol.spin)

        self.assertEqual(concat_mol.atom[0][0].split(':')[0], subsystems[0].mol.atom[0][0])
        #assesrtAlmostEqual
        self.assertAlmostEqual(concat_mol.atom[0][1][0], (1.88972612457 * float(subsystems[0].mol.atom[0][1][0])), delta=1e-8) # convert to bohr
        self.assertAlmostEqual(concat_mol.atom[0][1][1], (1.88972612457 * float(subsystems[0].mol.atom[0][1][1])), delta=1e-8) # convert to bohr
        self.assertAlmostEqual(concat_mol.atom[0][1][2], (1.88972612457 * float(subsystems[0].mol.atom[0][1][2])), delta=1e-8) # convert to bohr

        self.assertEqual(concat_mol.atom[1][0].split(':')[0], subsystems[1].mol.atom[0][0])
        self.assertAlmostEqual(concat_mol.atom[1][1][0], (1.88972612457 * float(subsystems[1].mol.atom[0][1][0])), delta=1e-8) # convert to bohr
        self.assertAlmostEqual(concat_mol.atom[1][1][1], (1.88972612457 * float(subsystems[1].mol.atom[0][1][1])), delta=1e-8) # convert to bohr
        self.assertAlmostEqual(concat_mol.atom[1][1][2], (1.88972612457 * float(subsystems[1].mol.atom[0][1][2])), delta=1e-8) # convert to bohr
        self.assertEqual(len(concat_mol.basis.keys()), 2)

    def test_concat_mols_ghost(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + ghost_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)

        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)

        # Full Ghost atoms
        concat_mol = supersystem.mol
        self.assertEqual(concat_mol.basis['He'], subsystems[0].mol.basis['He'])
        self.assertEqual(concat_mol.basis['C:1'], subsystems[1].mol.basis['C'])
        self.assertEqual(len(concat_mol.basis.keys()), 2)

    def test_concat_mols_partial_ghost(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + partial_ghost_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)

        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)

        # Partial Ghost atoms
        concat_mol = supersystem.mol
        self.assertEqual(concat_mol.basis['He'], subsystems[0].mol.basis['He'])
        self.assertEqual(concat_mol.basis['C:1'], subsystems[1].mol.basis['C'])
        self.assertEqual(len(concat_mol.basis.keys()), 2)

    def test_concat_mols_mix_basis(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + mixed_basis_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)

        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)

        # Mixed basis
        concat_mol = supersystem.mol
        self.assertEqual(concat_mol.basis['He'], subsystems[0].mol.basis['He'])
        self.assertEqual(concat_mol.basis['C:1'], subsystems[1].mol.basis['C'])
        self.assertEqual(len(concat_mol.basis.keys()), 2)

    def test_from_inp_def(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)

        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)

        self.assertEqual(supersystem.ct_method, 'rpbe')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, path+def_filename)
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertIsInstance(supersystem.ft_diis[0], lib.diis.DIIS)
        self.assertIsInstance(supersystem.ft_diis[1], lib.diis.DIIS)
        self.assertEqual(supersystem.ft_setfermi, None)
        self.assertEqual(supersystem.ft_initguess, None)
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.cycles, 100)
        self.assertEqual(supersystem.conv, 1e-9)
        self.assertEqual(supersystem.grad, None)
        self.assertEqual(supersystem.damp, 0)
        self.assertEqual(supersystem.shift, 0)
        self.assertEqual(supersystem.smearsigma, 0)
        self.assertEqual(supersystem.initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-20)
        self.assertEqual(supersystem.verbose, 3)
        self.assertEqual(supersystem.analysis, False)
        self.assertEqual(supersystem.debug, False)

    #@unittest.skip('skipping supermol calc')
    def test_from_inp_exp_set(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + exp_set_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)

        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)

        self.assertEqual(supersystem.ct_method, 'upbe')
        self.assertEqual(supersystem.proj_oper, 'huzfermi')
        self.assertEqual(supersystem.filename, path+exp_set_filename)
        self.assertEqual(supersystem.ft_cycles, 10)
        self.assertEqual(supersystem.ft_conv, 1e-4)
        self.assertEqual(supersystem.ft_grad, 1e-4)
        self.assertIsInstance(supersystem.ft_diis[0], lib.diis.DIIS)
        self.assertIsInstance(supersystem.ft_diis[1], lib.diis.DIIS)
        self.assertEqual(supersystem.ft_setfermi, -4.)
        self.assertEqual(supersystem.ft_initguess, 'atom')
        self.assertEqual(supersystem.ft_updatefock, 2)

        self.assertEqual(supersystem.cycles, 300)
        self.assertEqual(supersystem.conv, 1e-3)
        self.assertEqual(supersystem.grad, 1e-2)
        self.assertEqual(supersystem.damp, 0.1)
        self.assertEqual(supersystem.shift, 0.3)
        self.assertEqual(supersystem.smearsigma, 0.2)
        self.assertEqual(supersystem.initguess, '1e')
        self.assertEqual(supersystem.grid_level, 5)
        self.assertEqual(supersystem.rho_cutoff, 1e-4)
        self.assertEqual(supersystem.verbose, 1)
        self.assertEqual(supersystem.analysis, True)
        self.assertEqual(supersystem.debug, True)

    def test_custom_obj_def(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        '''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 1.0 0.0 0.0
        O 3.0 0.0 0.0'''
        mol2.basis = 'aug-cc-pVDZ'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'rb3lyp')
        self.assertEqual(supersystem.ct_method, 'rb3lyp')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertIsInstance(supersystem.ft_diis[0], lib.diis.DIIS)
        self.assertIsInstance(supersystem.ft_diis[1], lib.diis.DIIS)
        self.assertEqual(supersystem.ft_setfermi, None)
        self.assertEqual(supersystem.ft_initguess, None)
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.cycles, 100)
        self.assertEqual(supersystem.conv, 1e-9)
        self.assertEqual(supersystem.grad, None)
        self.assertEqual(supersystem.damp, 0)
        self.assertEqual(supersystem.shift, 0)
        self.assertEqual(supersystem.smearsigma, 0)
        self.assertEqual(supersystem.initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-20)
        self.assertEqual(supersystem.verbose, 3)
        self.assertEqual(supersystem.analysis, False)
        self.assertEqual(supersystem.debug, False)

    def test_custom_obj_set(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 1.0 0.0 0.0
        O 3.0 0.0 0.0'''
        mol2.basis = 'aug-cc-pVDZ'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2],
                          'rb3lyp', proj_oper='huzfermi', ft_cycles=2, 
                          ft_conv=1e-1, ft_grad=1e-4, ft_diis=3, 
                          ft_setfermi=-0.1, ft_initguess='1e', ft_updatefock=1, 
                          cycles=3, conv=2, grad=4, damp=1, shift=2.1, 
                          smearsigma=0.1, initguess='atom', 
                          grid_level=2, rhocutoff=1e-2, verbose=1, 
                          analysis=True, debug=True)

        self.assertEqual(supersystem.ct_method, 'rb3lyp')
        self.assertEqual(supersystem.proj_oper, 'huzfermi')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 2)
        self.assertEqual(supersystem.ft_conv, 1e-1)
        self.assertEqual(supersystem.ft_grad, 1e-4)
        self.assertIsInstance(supersystem.ft_diis[0], lib.diis.DIIS)
        self.assertIsInstance(supersystem.ft_diis[1], lib.diis.DIIS)
        self.assertEqual(supersystem.ft_setfermi, -0.1)
        self.assertEqual(supersystem.ft_initguess, '1e')
        self.assertEqual(supersystem.ft_updatefock, 1)

        self.assertEqual(supersystem.cycles, 3)
        self.assertEqual(supersystem.conv, 2)
        self.assertEqual(supersystem.grad, 4)
        self.assertEqual(supersystem.damp, 1)
        self.assertEqual(supersystem.shift, 2.1)
        self.assertEqual(supersystem.smearsigma, 0.1)
        self.assertEqual(supersystem.initguess, 'atom')
        self.assertEqual(supersystem.grid_level, 2)
        self.assertEqual(supersystem.rho_cutoff, 1e-2)
        self.assertEqual(supersystem.verbose, 1)
        self.assertEqual(supersystem.analysis, True)
        self.assertEqual(supersystem.debug, True)

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestEnvSubSystem(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+exp_set_filename, "w") as f:
            f.write(exp_set_str)

    def test_from_inp_def(self):

        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)

        self.assertEqual(subsystems[0].mol, in_obj.subsys_mols[0])
        self.assertEqual(subsystems[0].env_method, 'pbe')
        self.assertEqual(subsystems[0].filename, in_obj.inp.filename)

        # testing defaults
        self.assertEqual(subsystems[0].smearsigma, 0)
        self.assertEqual(subsystems[0].damp, 0)
        self.assertEqual(subsystems[0].shift, 0)
        self.assertEqual(subsystems[0].subcycles, 1)
        self.assertEqual(subsystems[0].freeze, False)
        self.assertEqual(subsystems[0].initguess, None)
        self.assertEqual(subsystems[0].grid_level, 4)
        self.assertEqual(subsystems[0].rho_cutoff, 1e-20)
        self.assertEqual(subsystems[0].verbose, 3)
        self.assertEqual(subsystems[0].analysis, False)
        self.assertEqual(subsystems[0].debug, False)

        self.assertEqual(subsystems[1].mol, in_obj.subsys_mols[1])
        self.assertEqual(subsystems[1].env_method, 'pbe')
        self.assertEqual(subsystems[1].filename, in_obj.inp.filename)

        # testing defaults
        self.assertEqual(subsystems[1].smearsigma, 0)
        self.assertEqual(subsystems[1].damp, 0)
        self.assertEqual(subsystems[1].shift, 0)
        self.assertEqual(subsystems[1].subcycles, 1)
        self.assertEqual(subsystems[1].freeze, False)
        self.assertEqual(subsystems[1].initguess, None)
        self.assertEqual(subsystems[1].grid_level, 4)
        self.assertEqual(subsystems[1].rho_cutoff, 1e-20)
        self.assertEqual(subsystems[1].verbose, 3)
        self.assertEqual(subsystems[1].analysis, False)
        self.assertEqual(subsystems[1].debug, False)

    def test_from_inp_exp_set(self):

        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + exp_set_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)

        self.assertEqual(subsystems[0].mol, in_obj.subsys_mols[0])
        self.assertEqual(subsystems[0].env_method, 'pbe')
        self.assertEqual(subsystems[0].filename, in_obj.inp.filename)

        self.assertEqual(subsystems[0].smearsigma, 0.1)
        self.assertEqual(subsystems[0].damp, 0.2)
        self.assertEqual(subsystems[0].shift, 0.1)
        self.assertEqual(subsystems[0].subcycles, 4)
        self.assertEqual(subsystems[0].freeze, False)
        self.assertEqual(subsystems[0].initguess, 'minao')
        self.assertEqual(subsystems[0].grid_level, 5)
        self.assertEqual(subsystems[0].rho_cutoff, 1e-4)
        self.assertEqual(subsystems[0].verbose, 1)
        self.assertEqual(subsystems[0].analysis, True)
        self.assertEqual(subsystems[0].debug, True)

        self.assertEqual(subsystems[1].mol, in_obj.subsys_mols[1])
        self.assertEqual(subsystems[1].env_method, 'pbe')
        self.assertEqual(subsystems[1].filename, in_obj.inp.filename)

        self.assertEqual(subsystems[1].smearsigma, 0.01)
        self.assertEqual(subsystems[1].damp, 0.1)
        self.assertEqual(subsystems[1].shift, 0.2)
        self.assertEqual(subsystems[1].subcycles, 1)
        self.assertEqual(subsystems[1].freeze, True)
        self.assertEqual(subsystems[1].initguess, 'supmol')
        self.assertEqual(subsystems[1].grid_level, 5)
        self.assertEqual(subsystems[1].rho_cutoff, 1e-4)
        self.assertEqual(subsystems[1].verbose, 1)
        self.assertEqual(subsystems[1].analysis, True)
        self.assertEqual(subsystems[1].debug, True)

    def test_custom_obj_def(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'

        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method)

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'm06')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.smearsigma, 0)
        self.assertEqual(subsys.damp, 0)
        self.assertEqual(subsys.shift, 0)
        self.assertEqual(subsys.subcycles, 1)
        self.assertEqual(subsys.freeze, False)
        self.assertEqual(subsys.initguess, None)
        self.assertEqual(subsys.grid_level, 4)
        self.assertEqual(subsys.rho_cutoff, 1e-20)
        self.assertEqual(subsys.verbose, 3)
        self.assertEqual(subsys.analysis, False)
        self.assertEqual(subsys.debug, False)
        
    def test_custom_obj_set(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'

        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, 
            smearsigma=0.5, damp=1, shift=1, subcycles=10, freeze=True, 
            initguess='supmol', grid_level=1, rhocutoff=1e-1, verbose=2, 
            analysis=True, debug=True)

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'm06')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.smearsigma, 0.5)
        self.assertEqual(subsys.damp, 1)
        self.assertEqual(subsys.shift, 1)
        self.assertEqual(subsys.subcycles, 10)
        self.assertEqual(subsys.freeze, True)
        self.assertEqual(subsys.initguess, 'supmol')
        self.assertEqual(subsys.grid_level, 1)
        self.assertEqual(subsys.rho_cutoff, 1e-1)
        self.assertEqual(subsys.verbose, 2)
        self.assertEqual(subsys.analysis, True)
        self.assertEqual(subsys.debug, True)

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestActiveSubSystem(unittest.TestCase):
    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+exp_set_filename, "w") as f:
            f.write(exp_set_str)

    def test_from_inp_def(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        mol = in_obj.subsys_mols[0]
        env_method = in_obj.env_subsystem_kwargs[0].pop('env_method')
        env_kwargs = in_obj.env_subsystem_kwargs[0]
        active_method = in_obj.active_subsystem_kwargs.pop('active_method')
        active_kwargs = in_obj.active_subsystem_kwargs
        active_kwargs.update(env_kwargs)
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method, **active_kwargs)
        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'pbe')
        self.assertEqual(subsys.filename, in_obj.inp.filename)

        self.assertEqual(subsys.smearsigma, 0)
        self.assertEqual(subsys.damp, 0)
        self.assertEqual(subsys.shift, 0)
        self.assertEqual(subsys.subcycles, 1)
        self.assertEqual(subsys.freeze, False)
        self.assertEqual(subsys.initguess, None)
        self.assertEqual(subsys.grid_level, 4)
        self.assertEqual(subsys.rho_cutoff, 1e-20)
        self.assertEqual(subsys.verbose, 3)
        self.assertEqual(subsys.analysis, False)
        self.assertEqual(subsys.debug, False)

        # active methods
        self.assertEqual(subsys.active_method, 'hf')
        self.assertEqual(subsys.localize_orbitals, False)
        self.assertEqual(subsys.active_orbs, None)
        self.assertEqual(subsys.active_conv, 1e-9)
        self.assertEqual(subsys.active_grad, None)
        self.assertEqual(subsys.active_cycles, 100)
        self.assertEqual(subsys.active_damp, 0)
        self.assertEqual(subsys.active_shift, 0)


    def test_from_inp_exp_set(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + exp_set_filename)
        mol = in_obj.subsys_mols[0]
        env_method = in_obj.env_subsystem_kwargs[0].pop('env_method')
        env_kwargs = in_obj.env_subsystem_kwargs[0]
        active_method = in_obj.active_subsystem_kwargs.pop('active_method')
        active_kwargs = in_obj.active_subsystem_kwargs
        active_kwargs.update(env_kwargs)
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method, **active_kwargs)
        self.assertEqual(subsys.mol, in_obj.subsys_mols[0])
        self.assertEqual(subsys.env_method, 'pbe')
        self.assertEqual(subsys.filename, in_obj.inp.filename)

        self.assertEqual(subsys.smearsigma, 0.1)
        self.assertEqual(subsys.damp, 0.2)
        self.assertEqual(subsys.shift, 0.1)
        self.assertEqual(subsys.subcycles, 4)
        self.assertEqual(subsys.freeze, False)
        self.assertEqual(subsys.initguess, 'minao')
        self.assertEqual(subsys.grid_level, 5)
        self.assertEqual(subsys.rho_cutoff, 1e-4)
        self.assertEqual(subsys.verbose, 1)
        self.assertEqual(subsys.analysis, True)
        self.assertEqual(subsys.debug, True)

        # active settings

        self.assertEqual(subsys.active_method, 'caspt2[2,2]')
        self.assertEqual(subsys.localize_orbitals, True)
        self.assertEqual(subsys.active_orbs, [4,5])
        self.assertEqual(subsys.active_conv, 1e-10)
        self.assertEqual(subsys.active_grad, 1e-11)
        self.assertEqual(subsys.active_cycles, 500)
        self.assertEqual(subsys.active_damp, 0.1)
        self.assertEqual(subsys.active_shift, 0.2)

    def test_custom_obj_def(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'

        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'm06')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.smearsigma, 0)
        self.assertEqual(subsys.damp, 0)
        self.assertEqual(subsys.shift, 0)
        self.assertEqual(subsys.subcycles, 1)
        self.assertEqual(subsys.freeze, False)
        self.assertEqual(subsys.initguess, None)
        self.assertEqual(subsys.grid_level, 4)
        self.assertEqual(subsys.rho_cutoff, 1e-20)
        self.assertEqual(subsys.verbose, 3)
        self.assertEqual(subsys.analysis, False)
        self.assertEqual(subsys.debug, False)

        # active methods
        self.assertEqual(subsys.active_method, 'ccsd')
        self.assertEqual(subsys.localize_orbitals, False)
        self.assertEqual(subsys.active_orbs, None)
        self.assertEqual(subsys.active_conv, 1e-9)
        self.assertEqual(subsys.active_grad, None)
        self.assertEqual(subsys.active_cycles, 100)
        self.assertEqual(subsys.active_damp, 0)
        self.assertEqual(subsys.active_shift, 0)
        
    def test_custom_obj_set(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'
        active_method  = 'mp2'

        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, 
            active_method, localize_orbitals=True, active_orbs=[2,3,4,5],
            active_conv=1e-9, active_grad=1e-8, active_cycles=2, 
            active_damp=0.1, active_shift=0.001, smearsigma=0.5, damp=1, 
            shift=1, subcycles=10, freeze=True, initguess='supmol', grid_level=1, 
            rhocutoff=1e-3, verbose=2, analysis=True, debug=True)

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'm06')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.smearsigma, 0.5)
        self.assertEqual(subsys.damp, 1)
        self.assertEqual(subsys.shift, 1)
        self.assertEqual(subsys.subcycles, 10)
        self.assertEqual(subsys.freeze, True)
        self.assertEqual(subsys.initguess, 'supmol')
        self.assertEqual(subsys.grid_level, 1)
        self.assertEqual(subsys.rho_cutoff, 1e-3)
        self.assertEqual(subsys.verbose, 2)
        self.assertEqual(subsys.analysis, True)

        # active methods
        self.assertEqual(subsys.active_method, 'mp2')
        self.assertEqual(subsys.localize_orbitals, True)
        self.assertEqual(subsys.active_orbs, [2,3,4,5])
        self.assertEqual(subsys.active_conv, 1e-9)
        self.assertEqual(subsys.active_grad, 1e-8)
        self.assertEqual(subsys.active_cycles, 2)
        self.assertEqual(subsys.active_damp, 0.1)
        self.assertEqual(subsys.active_shift, 0.001)

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestExcitedSubSystem(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
