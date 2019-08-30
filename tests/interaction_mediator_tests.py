# Tests for the input reader object
# Daniel Graham


import unittest
import os
import shutil
import re

from qsome.cluster_subsystem import ClusterEnvSubSystem, ClusterHLSubSystem
from qsome.cluster_supersystem import ClusterSuperSystem
from qsome.interaction_mediator import InteractionMediator
from pyscf import gto

import numpy as np


class TestSetup(unittest.TestCase):


    def test_simple_subsystems(self):

        mol1 = gto.Mole()
        mol1.verbose = 3
        mol1.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol1.basis = 'sto-3g'
        mol1.build()

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 1.0 0.0 0.0
        H 1. -2.757 2.857
        H 1. 2.757 2.857'''
        mol2.basis = '3-21g'
        mol2.build()

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom = '''
        O 2.0 0.0 0.0
        H 2. -2.757 2.857
        H 2. 2.757 2.857'''
        mol3.basis = '6-31g'
        mol3.build()

        mol4 = gto.Mole()
        mol4.verbose = 3
        mol4.atom = '''
        O 3.0 0.0 0.0
        H 3. -2.757 2.857
        H 3. 2.757 2.857'''
        mol4.basis = 'aug-cc-pVTZ'
        mol4.build()

        mol5 = gto.Mole()
        mol5.verbose = 3
        mol5.atom = '''
        O 4.0 0.0 0.0
        H 4. -2.757 2.857
        H 4. 2.757 2.857'''
        mol5.basis = '3-21g'
        mol5.build()

        sub1 = ClusterEnvSubSystem(mol1, 'lda', env_order=1)
        sub2 = ClusterEnvSubSystem(mol2, 'lda', env_order=1)
        sub3 = ClusterEnvSubSystem(mol3, 'pbe', env_order=2)
        sub4 = ClusterEnvSubSystem(mol4, 'm06', env_order=3)
        sub5 = ClusterHLSubSystem(mol5, 'm06', 'rhf', env_order=3)
        subsystems = [sub1, sub2, sub3, sub4, sub5]

        sup1_alt_sub_mol = gto.Mole()
        sup1_alt_sub_mol.verbose = 3
        sup1_alt_sub_mol.atom = '''
        O-0 2.0 0.0 0.0
        H-0 2. -2.757 2.857
        H-0 2. 2.757 2.857
        O-1 3.0 0.0 0.0
        H-1 3. -2.757 2.857
        H-1 3. 2.757 2.857
        O-2 4.0 0.0 0.0
        H-2 4. -2.757 2.857
        H-2 4. 2.757 2.857'''
        sup1_alt_sub_mol.basis = {'O-0': '6-31g', 'H-0': '6-31g', 'O-1': 'aug-cc-pVTZ', 'H-1': 'aug-cc-pVTZ', 'O-2': '3-21g', 'H-2': '3-21g'}
        sup1_alt_sub = ClusterEnvSubSystem(sup1_alt_sub_mol, 'pbe', env_order=2)
        supersystem_1 = ClusterSuperSystem([sub1, sub2, sup1_alt_sub], 'lda') 

        sup2_alt_sub_mol = gto.Mole()
        sup2_alt_sub_mol.verbose = 3
        sup2_alt_sub_mol.atom = '''
        O-0 3.0 0.0 0.0
        H-0 3. -2.757 2.857
        H-0 3. 2.757 2.857
        O-1 4.0 0.0 0.0
        H-1 4. -2.757 2.857
        H-1 4. 2.757 2.857'''
        sup2_alt_sub_mol.basis = {'O-0': 'aug-cc-pVTZ', 'H-0': 'aug-cc-pVTZ', 'O-1': '3-21g', 'H-1': '3-21g'}

        sup2_alt_sub = ClusterEnvSubSystem(sup2_alt_sub_mol, 'm06', env_order=3)
        supersystem_2 = ClusterSuperSystem([sub3, sup2_alt_sub], 'pbe', env_order=2)
        supersystem_3 = ClusterSuperSystem([sub4, sub5], 'm06', env_order=3)
        supersystem_list = [supersystem_1, supersystem_2, supersystem_3]
        mediator = InteractionMediator(subsystems)
        #Ensure the densities are the same and the methods are correct should be enough.
        self.assertEqual(len(mediator.supersystems), 3)
        for i in range(len(mediator.supersystems)):
            test = mediator.supersystems[i]
            self.assertTrue(np.allclose(test.get_dmat(), supersystem_list[i].get_dmat()))
            self.assertEqual(test.fs_method, supersystem_list[i].fs_method)

    def test_explicit_subsystems(self):
        #Use the fs settings for the smaller system for the subsystem settings.
        mol1 = gto.Mole()
        mol1.verbose = 3
        mol1.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol1.basis = 'sto-3g'
        mol1.build()

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 1.0 0.0 0.0
        H 1. -2.757 2.857
        H 1. 2.757 2.857'''
        mol2.basis = '3-21g'
        mol2.build()

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom = '''
        O 2.0 0.0 0.0
        H 2. -2.757 2.857
        H 2. 2.757 2.857'''
        mol3.basis = '6-31g'
        mol3.build()

        mol4 = gto.Mole()
        mol4.verbose = 3
        mol4.atom = '''
        O 3.0 0.0 0.0
        H 3. -2.757 2.857
        H 3. 2.757 2.857'''
        mol4.basis = 'aug-cc-pVTZ'
        mol4.build()

        mol5 = gto.Mole()
        mol5.verbose = 3
        mol5.atom = '''
        O 4.0 0.0 0.0
        H 4. -2.757 2.857
        H 4. 2.757 2.857'''
        mol5.basis = '3-21g'
        mol5.build()

        sub1 = ClusterEnvSubSystem(mol1, 'lda', env_order=1)
        sub2 = ClusterEnvSubSystem(mol2, 'lda', env_order=1)
        sub3 = ClusterEnvSubSystem(mol3, 'pbe', env_order=2)
        sub4 = ClusterEnvSubSystem(mol4, 'm06', env_order=3)
        sub5 = ClusterHLSubSystem(mol5, 'm06', 'rhf', env_order=3)
        subsystems = [sub1, sub2, sub3, sub4, sub5]

        sup1_alt_sub_mol = gto.Mole()
        sup1_alt_sub_mol.verbose = 3
        sup1_alt_sub_mol.atom = '''
        O-0 2.0 0.0 0.0
        H-0 2. -2.757 2.857
        H-0 2. 2.757 2.857
        O-1 3.0 0.0 0.0
        H-1 3. -2.757 2.857
        H-1 3. 2.757 2.857
        O-2 4.0 0.0 0.0
        H-2 4. -2.757 2.857
        H-2 4. 2.757 2.857'''
        sup1_alt_sub_mol.basis = {'O-0': '6-31g', 'H-0': '6-31g', 'O-1': 'aug-cc-pVTZ', 'H-1': 'aug-cc-pVTZ', 'O-2': '3-21g', 'H-2': '3-21g'}
        sup1_alt_sub = ClusterEnvSubSystem(sup1_alt_sub_mol, 'pbe', env_order=2)
        supersystem_1 = ClusterSuperSystem([sub1, sub2, sup1_alt_sub], 'lda', fs_conv=1e-9) 

        sup2_alt_sub_mol = gto.Mole()
        sup2_alt_sub_mol.verbose = 3
        sup2_alt_sub_mol.atom = '''
        O-0 3.0 0.0 0.0
        H-0 3. -2.757 2.857
        H-0 3. 2.757 2.857
        O-1 4.0 0.0 0.0
        H-1 4. -2.757 2.857
        H-1 4. 2.757 2.857'''
        sup2_alt_sub_mol.basis = {'O-0': 'aug-cc-pVTZ', 'H-0': 'aug-cc-pVTZ', 'O-1': '3-21g', 'H-1': '3-21g'}

        sup2_alt_sub = ClusterEnvSubSystem(sup2_alt_sub_mol, 'm06', env_order=3)
        supersystem_2 = ClusterSuperSystem([sub3, sup2_alt_sub], 'pbe', env_order=2, fs_unrestricted=True)
        supersystem_3 = ClusterSuperSystem([sub4, sub5], 'm06', env_order=3, fs_cycles=10)
        supersystem_list = [supersystem_1, supersystem_2, supersystem_3]


        supersystem_dicts = [{'fs_conv': 1e-9, 'env_order': 1}, {'fs_unrestricted': True, 'env_order': 2}, {'fs_cycles': 10, 'env_order': 3}]
        mediator = InteractionMediator(subsystems, supersystem_dicts)
        #Ensure the densities are the same and the methods are correct should be enough.
        self.assertEqual(len(mediator.supersystems), 3)
        for i in range(len(mediator.supersystems)):
            test = mediator.supersystems[i]
            self.assertTrue(np.allclose(test.get_dmat(), supersystem_list[i].get_dmat()))
            self.assertEqual(test.fs_method, supersystem_list[i].fs_method)
            self.assertEqual(test.fs_conv, supersystem_list[i].fs_conv)
            self.assertEqual(test.fs_unrestricted, supersystem_list[i].fs_unrestricted)
            self.assertEqual(test.fs_cycles, supersystem_list[i].fs_cycles)

        self.assertTrue(mediator.supersystems[0].subsystems[-1].unrestricted)

class TestFreezeAndThaw(unittest.TestCase):
    def test_simple_subsystems(self):
        pass
    def test_explicit_subsystems(self):
        pass

class TestEmbeddingEnergies(unittest.TestCase):

    def test_simple_subsystems(self):
        mol1 = gto.Mole()
        mol1.verbose = 3
        mol1.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol1.basis = 'sto-3g'
        mol1.build()

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 1.0 0.0 0.0
        H 1. -2.757 2.857
        H 1. 2.757 2.857'''
        mol2.basis = '3-21g'
        mol2.build()

        sub1 = ClusterHLSubSystem(mol1, 'lda', 'rhf')
        sub2 = ClusterEnvSubSystem(mol2, 'lda')
        subsystems = [sub1, sub2]
        mediator = InteractionMediator(subsystems)
        mediator.get_emb_energy()

        self.assertTrue(False)

    def test_explicit_subsystems(self):
        pass
    
if __name__ == "__main__":
    unittest.main()
