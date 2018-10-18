# A module to test Freeze and thaw method
# Daniel Graham

import unittest
import os
import shutil
import re

from qsome import inp_reader, cluster_subsystem, cluster_supersystem
from pyscf import gto, dft

import numpy as np

def_filename = "default.inp"
default_str = """
subsystem
H    -0.8000    0.0000    0.0000
H    0.0000    0.0000    0.0000
H    0.8000    0.0000    0.0000
H    1.6000    0.0000    0.0000
end

subsystem
H    0.0000    2.0000    0.0000
H    0.8000    2.0000    0.0000
end

embed
 env_method pbe
 huzinaga
 cycles 150
end

basis 3-21g
active_method hf
"""

partial_ghost_filename = 'partial_ghost.inp'
partial_ghost_str = default_str.replace("""
H    -0.8000    0.0000    0.0000
H    0.0000    0.0000    0.0000
H    0.8000    0.0000    0.0000
H    1.6000    0.0000    0.0000
""","""
H    -0.8000    0.0000    0.0000
H    0.0000    0.0000    0.0000
H    0.8000    0.0000    0.0000
H    1.6000    0.0000    0.0000
gh.H    0.0000    2.0000    0.0000
gh.H    0.8000    2.0000    0.0000
""", 1)

ghost_filename = 'ghost.inp'
ghost_str = partial_ghost_str.replace("""
subsystem
H    0.0000    2.0000    0.0000
H    0.8000    2.0000    0.0000
""","""
subsystem
H    0.0000    2.0000    0.0000
H    0.8000    2.0000    0.0000
gh.H    -0.8000    0.0000    0.0000
gh.H    0.0000    0.0000    0.0000
gh.H    0.8000    0.0000    0.0000
gh.H    1.6000    0.0000    0.0000
""", 1)

widesep_filename = 'wide_sep.inp'
widesep_str = default_str.replace("""
H    0.0000    2.0000    0.0000
H    0.8000    2.0000    0.0000
""", """
H    0.0000    200.0000    0.0000
H    0.8000    200.0000    0.0000
""", 1)

hyb_func_filename = 'hyb_fun.inp'
hyb_func_str = default_str.replace("""
 env_method pbe
""","""
 env_method m06
""",1)

lda_func_filename = 'lda_fun.inp'
lda_func_str = default_str.replace("""
 env_method pbe
""","""
 env_method lda,vwn
""",1)

# Need to also test huzfermi and mu operator

temp_inp_dir = "/temp_input/"

class TestFockConstruction(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, 'w') as f:
            f.write(default_str)

        with open(path+ghost_filename, 'w') as f:
            f.write(ghost_str)

        with open(path+widesep_filename, 'w') as f:
            f.write(widesep_str)

    def test_default(self):
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

        supersystem.update_fock()

        # py_scf = dft.RKS(supersystem.mol)
        # py_scf.xc = 'pbe'
        # dm_0 = py_scf.get_init_guess()
        # py_fock = py_scf.get_fock(dm=dm_0)
        # self.assertTrue(np.array_equal(supersystem.fock[0], py_fock))
        # self.assertTrue(np.array_equal(supersystem.fock[1], py_fock))
        # # Need tests

    def test_ghost(self):
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

        supersystem.update_fock()
        # Need tests

    def test_widesep(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + widesep_filename)
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

        supersystem.update_fock()

        pyscf_fock = [None, None]
        pyscf_fock[0] = supersystem.ct_scf.get_fock(dm=(supersystem.ct_scf.get_init_guess())) 
        pyscf_fock[1] = supersystem.ct_scf.get_fock(dm=(supersystem.ct_scf.get_init_guess())) 
        self.assertTrue(np.array_equal(supersystem.fock, pyscf_fock))

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestProjectionConstruction(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, 'w') as f:
            f.write(default_str)

        with open(path+ghost_filename, 'w') as f:
            f.write(ghost_str)

        with open(path+widesep_filename, 'w') as f:
            f.write(widesep_str)

    #@unittest.skip("skip")
    def test_default(self):
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

        supersystem.update_proj_pot()

        # need tests.

    #@unittest.skip("skip")
    def test_ghost(self):
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

        supersystem.ft_cycles = 2
        supersystem.freeze_and_thaw()

        nS = supersystem.mol.nao_nr()
        for i in range(len(subsystems)):
            #dm_subsys = [np.zeros((nS, nS)), np.zeros((nS, nS))]
            subsystem = subsystems[i]
            #dm_subsys[0][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += subsystem.dmat[0]
            #dm_subsys[1][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += subsystem.dmat[1]
            self.assertAlmostEqual(np.trace(np.dot(supersystem.proj_pot[i][0], subsystem.dmat[0])), 0.0, delta=1e-15)
            self.assertAlmostEqual(np.trace(np.dot(supersystem.proj_pot[i][1], subsystem.dmat[1])), 0.0, delta=1e-15)

    #@unittest.skip("skip")
    def test_widesep(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + widesep_filename)
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

        supersystem.update_proj_pot()

        self.assertEqual(np.sum(supersystem.proj_pot[0][0]), 0.0)
        self.assertEqual(np.sum(supersystem.proj_pot[0][1]), 0.0)
        self.assertEqual(np.sum(supersystem.proj_pot[1][0]), 0.0)
        self.assertEqual(np.sum(supersystem.proj_pot[1][1]), 0.0)

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestDiagonalize(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, 'w') as f:
            f.write(default_str)

        with open(path+ghost_filename, 'w') as f:
            f.write(ghost_str)

        with open(path+widesep_filename, 'w') as f:
            f.write(widesep_str)

    def test_default(self):
        pass
    def test_ghost(self):
        pass
    def test_widesep(self):
        pass

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestFreezeAndThaw(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, 'w') as f:
            f.write(default_str)

        with open(path+ghost_filename, 'w') as f:
            f.write(ghost_str)

        with open(path+widesep_filename, 'w') as f:
            f.write(widesep_str)

        with open(path+hyb_func_filename, 'w') as f:
            f.write(hyb_func_str)

        with open(path+lda_func_filename, 'w') as f:
            f.write(lda_func_str)

    def test_default(self):
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
        supersystem.freeze_and_thaw()
        supersystem.env_in_env_energy()
        sup_env_e = supersystem.env_energy
        sub1_env_e = supersystem.subsystems[0].env_energy
        sub2_env_e = supersystem.subsystems[1].env_energy
        sub1_env_proj_e = supersystem.subsystems[0].get_env_proj_e()
        sub2_env_proj_e = supersystem.subsystems[1].get_env_proj_e()
        self.assertAlmostEqual(sup_env_e, sub1_env_e - sub1_env_proj_e + sub2_env_e -sub2_env_proj_e + supersystem.mol.energy_nuc() - supersystem.subsystems[0].mol.energy_nuc() - supersystem.subsystems[1].mol.energy_nuc(), delta=1e-8)


        #unsure how to test...
    def test_ghost(self):
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

        supersystem.freeze_and_thaw()
        # compare dft-in-dft to full system energy
        sup_mo_e = supersystem.get_supersystem_energy()
        sup_env_in_env_e = supersystem.env_in_env_energy() 
        self.assertAlmostEqual(sup_mo_e, sup_env_in_env_e, delta=1e-8)

        sup_env_e = supersystem.env_energy
        sub1_env_e = supersystem.subsystems[0].env_energy
        sub2_env_e = supersystem.subsystems[1].env_energy
        self.assertAlmostEqual(sup_env_e, sub1_env_e + sub2_env_e + supersystem.mol.energy_nuc() - supersystem.subsystems[0].mol.energy_nuc() - supersystem.subsystems[1].mol.energy_nuc(), delta=1e-8)

    def test_widesep(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + widesep_filename)
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
        sup_env_in_env_e = supersystem.env_in_env_energy() 
        supersystem.freeze_and_thaw()

        # compare dft-in-dft to full system energy
        sup_mo_e = supersystem.get_supersystem_energy()
        sup_env_in_env_e = supersystem.env_in_env_energy() 
        self.assertAlmostEqual(sup_mo_e, sup_env_in_env_e, delta=1e-8)

        # supersystem energy should be equal to individual He atoms
        he_pyscf = dft.RKS(subsystems[0].mol)
        he_pyscf.xc = env_method
        he_pyscf.grids = supersystem.grids
        he_pyscf.small_rho_cutoff = 1e-20
        he_e = he_pyscf.kernel()

        he2_pyscf = dft.RKS(subsystems[1].mol)
        he2_pyscf.xc = env_method
        he_pyscf.grids = supersystem.grids
        he_pyscf.small_rho_cutoff = 1e-20
        he2_e = he2_pyscf.kernel()
        #self.assertAlmostEqual(supersystem.subsystems[0].get_env_energy(), he_e, delta=1e-10)

        self.assertAlmostEqual(sup_mo_e, he_e + he2_e, delta=1e-8)

        sup_env_e = supersystem.env_energy
        sub1_env_e = supersystem.subsystems[0].env_energy
        sub2_env_e = supersystem.subsystems[1].env_energy

        self.assertAlmostEqual(sup_env_e, sub1_env_e + sub2_env_e + supersystem.mol.energy_nuc() - supersystem.subsystems[0].mol.energy_nuc() - supersystem.subsystems[1].mol.energy_nuc(), delta=1e-8)
    #@unittest.skip("test")
    def test_hyb_func(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + hyb_func_filename)
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
        sup_env_in_env_e = supersystem.env_in_env_energy() 
        supersystem.freeze_and_thaw()

        sup_env_in_env_e = supersystem.env_in_env_energy() 

        sup_env_e = supersystem.env_energy
        sub1_env_e = supersystem.subsystems[0].env_energy
        sub2_env_e = supersystem.subsystems[1].env_energy
        sub1_env_proj_e = supersystem.subsystems[0].get_env_proj_e()
        sub2_env_proj_e = supersystem.subsystems[1].get_env_proj_e()
        self.assertAlmostEqual(sup_env_e, sub1_env_e - sub1_env_proj_e + sub2_env_e -sub2_env_proj_e + supersystem.mol.energy_nuc() - supersystem.subsystems[0].mol.energy_nuc() - supersystem.subsystems[1].mol.energy_nuc(), delta=1e-8)

    #@unittest.skip("test")
    def test_lda_func(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + lda_func_filename)
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
        sup_env_in_env_e = supersystem.env_in_env_energy() 
        supersystem.freeze_and_thaw()

        # compare dft-in-dft to full system energy
        sup_mo_e = supersystem.get_supersystem_energy()
        sup_env_in_env_e = supersystem.env_in_env_energy() 

        sup_env_e = supersystem.env_energy
        sub1_env_e = supersystem.subsystems[0].env_energy
        sub2_env_e = supersystem.subsystems[1].env_energy
        sub1_env_proj_e = supersystem.subsystems[0].get_env_proj_e()
        sub2_env_proj_e = supersystem.subsystems[1].get_env_proj_e()
        self.assertAlmostEqual(sup_env_e, sub1_env_e - sub1_env_proj_e + sub2_env_e -sub2_env_proj_e + supersystem.mol.energy_nuc() - supersystem.subsystems[0].mol.energy_nuc() - supersystem.subsystems[1].mol.energy_nuc(), delta=1e-8)

    #@unittest.skip("skip")
    def test_readchk(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            env_kwargs['initguess'] = 'readchk'
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)
        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem_kwargs['initguess'] = 'readchk'
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)
        supersystem.initguess = 'readchk'

        supersystem.freeze_and_thaw()

        sup_env_in_env_e = supersystem.env_in_env_energy() 
        sub1_e = subsystems[0].env_energy
        sub2_e = subsystems[1].env_energy

        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            env_kwargs['initguess'] = 'readchk'
            subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, **env_kwargs)
            subsystems.append(subsys)
        ct_method = in_obj.supersystem_kwargs.pop('ct_method')
        supersystem_kwargs = in_obj.supersystem_kwargs
        supersystem_kwargs['initguess'] = 'readchk'
        supersystem = cluster_supersystem.ClusterSuperSystem(subsystems, 
            ct_method, **supersystem_kwargs)
        supersystem.ft_cycles = 1

        supersystem.freeze_and_thaw()

        self.assertAlmostEqual(sub1_e, subsystems[0].env_energy, delta=1e-9)
        self.assertAlmostEqual(sub2_e, subsystems[1].env_energy, delta=1e-9)
        self.assertAlmostEqual(sup_env_in_env_e, supersystem.env_in_env_energy(), delta=1e-9)
        

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

if __name__ == "__main__":
    unittest.main()
