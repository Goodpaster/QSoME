# A module to test final energies.
# Daniel Graham

import unittest
import os
import shutil
import re

from qsome import inp_reader, cluster_subsystem, cluster_supersystem
from pyscf import gto, dft, scf, cc
from pyscf.cc import ccsd_t

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


temp_inp_dir = "/temp_input/"
class TestHFEnergy(unittest.TestCase):

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

    @unittest.skip
    def test_default(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        for i in range(len(in_obj.subsys_mols)):
            if i == 0:
                mol = in_obj.subsys_mols[i]
                env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
                env_kwargs = in_obj.env_subsystem_kwargs[i]
                active_method = in_obj.active_subsystem_kwargs.pop('active_method')
                active_kwargs = in_obj.active_subsystem_kwargs
                active_kwargs.update(env_kwargs)
                subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method,  **active_kwargs)
                subsystems.append(subsys)
            else:
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
        supersystem.get_active_energy()

        h_pyscf = scf.RHF(subsystems[0].mol)
        h_e = h_pyscf.kernel()

        #This is missing nuclear repulsion in the h_e term.
        embedding_nuc_e = 0.0
        #nuc_charges_a = supersystem.subsystems[0].mol.atom_charges()
        #nuc_coords_a = supersystem.subsystems[0].mol.atom_coords()
        #nuc_charges_b = supersystem.subsystems[1].mol.atom_charges()
        #nuc_coords_b = supersystem.subsystems[1].mol.atom_coords()
        #for i in range(len(supersystem.subsystems[0].mol._atm)):
        #    q1 = nuc_charges_a[i]
        #    r1 = nuc_coords_a[i]
        #    for j in range(len(supersystem.subsystems[1].mol._atm)):
        #        q2 = nuc_charges_b[j]
        #        r2 = nuc_coords_b[j]
        #        r = np.linalg.norm(r1-r2)
        #        embedding_nuc_e += q1 * q2 / r

        #print (embedding_nuc_e)
        self.assertNotAlmostEqual(subsystems[0].active_energy+embedding_nuc_e, h_e, delta=1e-7)
      
        #not sure how to tests.

    @unittest.skip
    def test_ghost(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + ghost_filename)
        for i in range(len(in_obj.subsys_mols)):
            if i == 0:
                mol = in_obj.subsys_mols[i]
                env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
                env_kwargs = in_obj.env_subsystem_kwargs[i]
                active_method = in_obj.active_subsystem_kwargs.pop('active_method')
                active_kwargs = in_obj.active_subsystem_kwargs
                active_kwargs.update(env_kwargs)
                subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method,  **active_kwargs)
                subsystems.append(subsys)
            else:
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
        supersystem.get_active_energy()

    def test_widesep(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + widesep_filename)
        for i in range(len(in_obj.subsys_mols)):
            if i == 0:
                mol = in_obj.subsys_mols[i]
                env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
                env_kwargs = in_obj.env_subsystem_kwargs[i]
                active_method = in_obj.active_subsystem_kwargs.pop('active_method')
                active_kwargs = in_obj.active_subsystem_kwargs
                active_kwargs.update(env_kwargs)
                subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method,  **active_kwargs)
                subsystems.append(subsys)
            else:
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
        supersystem.get_active_energy()

        # subsystem energies should be equal to individual He atoms
        h_pyscf = scf.RHF(subsystems[0].mol)
        h_e = h_pyscf.kernel()

        embedding_nuc_e = 0.0
        nuc_charges_a = supersystem.subsystems[0].mol.atom_charges()
        nuc_coords_a = supersystem.subsystems[0].mol.atom_coords()
        nuc_charges_b = supersystem.subsystems[1].mol.atom_charges()
        nuc_coords_b = supersystem.subsystems[1].mol.atom_coords()
        for i in range(len(supersystem.subsystems[0].mol._atm)):
            q1 = nuc_charges_a[i]
            r1 = nuc_coords_a[i]
            for j in range(len(supersystem.subsystems[1].mol._atm)):
                q2 = nuc_charges_b[j]
                r2 = nuc_coords_b[j]
                r = np.linalg.norm(r1-r2)
                embedding_nuc_e += q1 * q2 / r

        self.assertAlmostEqual(supersystem.subsystems[0].active_energy + embedding_nuc_e/2., h_e, delta=1e-7)
        
    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestCCSDEnergy(unittest.TestCase):

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

    def test_widesep(self):
        subsystems = []
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + widesep_filename)
        for i in range(len(in_obj.subsys_mols)):
            if i == 0:
                mol = in_obj.subsys_mols[i]
                env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
                env_kwargs = in_obj.env_subsystem_kwargs[i]
                active_method = in_obj.active_subsystem_kwargs.pop('active_method')
                active_method = 'ccsd(t)'
                active_kwargs = in_obj.active_subsystem_kwargs
                active_kwargs.update(env_kwargs)
                subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method,  **active_kwargs)
                subsystems.append(subsys)
            else:
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
        supersystem.get_active_energy()

        # subsystem energies should be equal to individual He atoms
        he_pyscf = scf.RHF(subsystems[0].mol)
        he_e = he_pyscf.kernel()
        mCCSD = cc.CCSD(he_pyscf)
        ecc, t1, t2 = mCCSD.kernel()
        ecc += ccsd_t.kernel(mCCSD, mCCSD.ao2mo())

        embedding_nuc_e = 0.0
        nuc_charges_a = supersystem.subsystems[0].mol.atom_charges()
        nuc_coords_a = supersystem.subsystems[0].mol.atom_coords()
        nuc_charges_b = supersystem.subsystems[1].mol.atom_charges()
        nuc_coords_b = supersystem.subsystems[1].mol.atom_coords()
        for i in range(len(supersystem.subsystems[0].mol._atm)):
            q1 = nuc_charges_a[i]
            r1 = nuc_coords_a[i]
            for j in range(len(supersystem.subsystems[1].mol._atm)):
                q2 = nuc_charges_b[j]
                r2 = nuc_coords_b[j]
                r = np.linalg.norm(r1-r2)
                embedding_nuc_e += q1 * q2 / r

        self.assertAlmostEqual(supersystem.subsystems[0].active_energy + embedding_nuc_e/2., he_e + ecc, delta=1e-7)
        

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    
class TestMREnergy(unittest.TestCase):

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

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestFCIEnergy(unittest.TestCase):

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
    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestMLEnergy(unittest.TestCase):

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

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

if __name__ == "__main__":
    unittest.main()
