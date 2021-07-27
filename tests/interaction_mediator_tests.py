# Tests for the input reader object
# Daniel Graham


import unittest
import os
import shutil
import re
import tempfile

from qsome.cluster_subsystem import ClusterEnvSubSystem, ClusterHLSubSystem
from qsome.cluster_supersystem import ClusterSuperSystem
from qsome.interaction_mediator import InteractionMediator
from qsome import helpers
from pyscf import gto

import numpy as np

from copy import deepcopy as copy


class TestSetup(unittest.TestCase):


    #@unittest.skip
    def test_density_multiple_subsystems(self):

        mol1 = gto.Mole()
        mol1.verbose = 3
        mol1.atom = '''
        H 0.758602  0.000000  0.504284
        H 0.758602  0.000000  -0.504284 
        O 0.0 0.0 0.0'''
        mol1.basis = '3-21g'
        mol1.build()

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        H 0.758602  2.000000  0.504284
        H 0.758602  2.000000  -0.504284 
        O 0.0 2.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom = '''
        H 0.758602  4.000000  0.504284
        H 0.758602  4.000000  -0.504284 
        O 0.0 4.0 0.0'''
        mol3.basis = '6-31g'
        mol3.build()

        mol4 = gto.Mole()
        mol4.verbose = 3
        mol4.atom = '''
        H 0.758602  6.000000  0.504284
        H 0.758602  6.000000  -0.504284 
        O 0.0 6.0 0.0'''
        mol4.basis = 'cc-pVDZ'
        mol4.build()

        mol5 = gto.Mole()
        mol5.verbose = 3
        mol5.atom = '''
        H 0.758602  8.000000  0.504284
        H 0.758602  8.000000  -0.504284 
        O 0.0 8.0 0.0'''
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
        H-0 0.758602  4.000000  0.504284
        H-0 0.758602  4.000000  -0.504284 
        O-0 0.0 4.0 0.0
        H-1 0.758602  6.000000  0.504284
        H-1 0.758602  6.000000  -0.504284 
        O-1 0.0 6.0 0.0
        H-2 0.758602  8.000000  0.504284
        H-2 0.758602  8.000000  -0.504284 
        O-2 0.0 8.0 0.0'''
        sup1_alt_sub_mol.basis = {'O-0': '6-31g', 'H-0': '6-31g', 'O-1': 'cc-pVDZ', 'H-1': 'cc-pVDZ', 'O-2': '3-21g', 'H-2': '3-21g'}
        sup1_alt_sub_mol.build()
        sup1_alt_sub = ClusterEnvSubSystem(sup1_alt_sub_mol, 'pbe', env_order=2)
        mol123 = helpers.concat_mols([sub1.mol, sub2.mol, sup1_alt_sub.mol])
        fs_scf_obj = helpers.gen_scf_obj(mol123, 'lda')
        supersystem_1 = ClusterSuperSystem([sub1, sub2, sup1_alt_sub], 'lda', fs_scf_obj)
        supersystem_1.init_density()

        sup2_alt_sub_mol = gto.Mole()
        sup2_alt_sub_mol.verbose = 3
        sup2_alt_sub_mol.atom = '''
        H-0 0.758602  6.000000  0.504284
        H-0 0.758602  6.000000  -0.504284 
        O-0 0.0 6.0 0.0
        H-1 0.758602  8.000000  0.504284
        H-1 0.758602  8.000000  -0.504284 
        O-1 0.0 8.0 0.0'''
        sup2_alt_sub_mol.basis = {'O-0': 'cc-pVDZ', 'H-0': 'cc-pVDZ', 'O-1': '3-21g', 'H-1': '3-21g'}
        sup2_alt_sub_mol.build()

        sup2_alt_sub = ClusterEnvSubSystem(sup2_alt_sub_mol, 'm06', env_order=3)
        mol123 = helpers.concat_mols([sub3.mol, sup2_alt_sub.mol])
        fs_scf_obj = helpers.gen_scf_obj(mol123, 'pbe')
        supersystem_2 = ClusterSuperSystem([sub3, sup2_alt_sub], 'pbe', fs_scf_obj, env_order=2)
        supersystem_2.init_density()
        mol123 = helpers.concat_mols([sub4.mol, sub5.mol])
        fs_scf_obj = helpers.gen_scf_obj(mol123, 'm06')
        supersystem_3 = ClusterSuperSystem([sub4, sub5], 'm06', fs_scf_obj, env_order=3)
        supersystem_3.init_density()
        supersystem_list = [supersystem_1, supersystem_2, supersystem_3]
        mediator = InteractionMediator(subsystems)
        #Ensure the densities are the same and the methods are correct should be enough.
        self.assertEqual(len(mediator.supersystems), 3)
        for i in range(len(mediator.supersystems)):
            test = mediator.supersystems[i]
            self.assertEqual(test.env_method, supersystem_list[i].env_method)

    #@unittest.skip
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
        sup1_alt_sub_mol.build()
        sup1_alt_sub = ClusterEnvSubSystem(sup1_alt_sub_mol, 'pbe', env_order=2)
        mol123 = helpers.concat_mols([sub1.mol, sub2.mol, sup1_alt_sub.mol])
        fs_scf_obj = helpers.gen_scf_obj(mol123, 'lda', conv_tol=1e-9)
        supersystem_1 = ClusterSuperSystem([sub1, sub2, sup1_alt_sub], 'lda', fs_scf_obj) 

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

        sup2_alt_sub_mol.build()
        sup2_alt_sub = ClusterEnvSubSystem(sup2_alt_sub_mol, 'm06', env_order=3)
        mol123 = helpers.concat_mols([sub3.mol, sup2_alt_sub.mol])
        fs_scf_obj = helpers.gen_scf_obj(mol123, 'pbe', unrestricted=True)
        supersystem_2 = ClusterSuperSystem([sub3, sup2_alt_sub], 'pbe', fs_scf_obj, env_order=2, unrestricted=True)
        mol123 = helpers.concat_mols([sub4.mol, sub5.mol])
        fs_scf_obj = helpers.gen_scf_obj(mol123, 'm06', max_cycle=10)
        supersystem_3 = ClusterSuperSystem([sub4, sub5], 'm06', fs_scf_obj, env_order=3)
        supersystem_list = [supersystem_1, supersystem_2, supersystem_3]


        supersystem_dicts = [{'env_order': 1, 'fs_env_settings':{'conv_tol': 1e-9, 'env_method':'lda'}}, {'env_order': 2, 'fs_env_settings':{'unrestricted':True, 'env_method':'pbe'}}, {'env_order': 3, 'fs_env_settings': {'env_method':'m06', 'max_cycle':10}}]
        mediator = InteractionMediator(subsystems, supersystem_dicts)
        #Ensure the densities are the same and the methods are correct should be enough.
        self.assertEqual(len(mediator.supersystems), 3)
        for i in range(len(mediator.supersystems)):
            test = mediator.supersystems[i]
            self.assertEqual(test.env_method, supersystem_list[i].env_method)
            self.assertEqual(test.fs_scf_obj.conv_tol, supersystem_list[i].fs_scf_obj.conv_tol)
            if hasattr(test.fs_scf_obj, 'unrestricted'):
                self.assertEqual(test.fs_scf_obj.unrestricted, supersystem_list[i].fs_scf_obj.unrestricted)
            self.assertEqual(test.fs_scf_obj.max_cycle, supersystem_list[i].fs_scf_obj.max_cycle)


    @unittest.skip
    def test_read_chkfile(self):
        pass

class TestFreezeAndThaw(unittest.TestCase):

    def setUp(self):
        mol1 = gto.Mole()
        mol1.verbose = 3
        mol1.atom = '''
        O 0.0 0.0 0.0
        H 0.758602  0.000000  0.504284
        H 0.758602  0.000000  -0.504284'''
        mol1.basis = 'sto-3g'
        mol1.build()
        self.mol1 = mol1

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 0.0 2.0 0.0
        H 0.758602  2.000000  0.504284
        H 0.758602  2.000000  -0.504284'''
        mol2.basis = '3-21g'
        mol2.build()
        self.mol2 = mol2


    #@unittest.skip
    def test_simple_subsystems(self):
        sub1 = ClusterHLSubSystem(self.mol1, 'lda', 'rhf')
        sub2 = ClusterEnvSubSystem(self.mol2, 'lda')
        subsystems = [sub1, sub2]
        mediator = InteractionMediator(subsystems)
        mediator.do_embedding()

        #True Values
        sub1 = ClusterHLSubSystem(self.mol1, 'lda', 'rhf')
        sub2 = ClusterEnvSubSystem(self.mol2, 'lda')
        subsystems = [sub1, sub2]
        mol12 = helpers.concat_mols([sub1.mol, sub2.mol])
        fs_scf_obj = helpers.gen_scf_obj(mol12, 'lda')
        supersystem = ClusterSuperSystem(subsystems, 'lda', fs_scf_obj)
        supersystem.init_density()
        sup_e = supersystem.get_supersystem_energy()
        supersystem.freeze_and_thaw()
        self.assertTrue(np.allclose(supersystem.get_emb_dmat(), mediator.supersystems[0].get_emb_dmat()))
        self.assertTrue(np.allclose(supersystem.fs_dmat, mediator.supersystems[0].fs_dmat))

    @unittest.skip
    def test_explicit_subsystems(self):
        pass

    #@unittest.skip
    def test_fs_save_density(self):
        t_file = tempfile.NamedTemporaryFile()
        sub1 = ClusterHLSubSystem(self.mol1, 'lda', 'rhf')
        sub2 = ClusterEnvSubSystem(self.mol2, 'lda')
        subsystems = [sub1, sub2]
        sup_kwargs = [{'env_order': 1, 'env_method': 'lda', 'fs_env_settings':{'save_density':True}}]
        mediator = InteractionMediator(subsystems, sup_kwargs, filename=t_file.name)
        mediator.do_embedding()
        mediator.get_emb_energy()
        #Assert that the density file exists.
        for sup in mediator.supersystems:
            self.assertTrue(os.path.exists(t_file.name + '_' + str(sup.chkfile_index) + '_fs.cube'))
            

    #@unittest.skip
    def test_sub_save_density(self):
        t_file = tempfile.NamedTemporaryFile()
        sub1 = ClusterHLSubSystem(self.mol1, 'lda', 'rhf', init_guess='supmol', save_density=True)
        sub2 = ClusterEnvSubSystem(self.mol2, 'lda', init_guess='supmol', save_density=True)
        subsystems = [sub1, sub2]
        sup_kwargs = [{'env_order': 1, 'env_method': 'lda', 'embed_settings': {'save_density':True}}]
        mediator = InteractionMediator(subsystems, sup_kwargs, filename=t_file.name)
        mediator.do_embedding()
        #Assert that the density file exists.

    #@unittest.skip
    def test_fs_save_orbitals(self):
        t_file = tempfile.NamedTemporaryFile()
        sub1 = ClusterHLSubSystem(self.mol1, 'lda', 'rhf', init_guess='supmol', save_orbs=True)
        sub2 = ClusterEnvSubSystem(self.mol2, 'lda', init_guess='supmol', save_orbs=True)
        subsystems = [sub1, sub2]
        sup_kwargs = [{'env_order': 1, 'env_method': 'lda', 'fs_env_settings': {'save_orbs':True}}]
        mediator = InteractionMediator(subsystems, sup_kwargs, filename=t_file.name)
        mediator.do_embedding()
        for sup in mediator.supersystems:
            self.assertTrue(os.path.exists(t_file.name + '_' + str(sup.chkfile_index) + '_fs.molden'))

    def test_read_chkfile(self):
        t_file = tempfile.NamedTemporaryFile()
        sub1 = ClusterHLSubSystem(self.mol1, 'lda', 'rhf')
        sub2 = ClusterEnvSubSystem(self.mol2, 'lda')
        subsystems = [sub1, sub2]
        sub3 = ClusterHLSubSystem(self.mol1.copy(), 'lda', 'rhf')
        sub4 = ClusterEnvSubSystem(self.mol2.copy(), 'lda')
        subsystems2 = [sub3, sub4]
        mediator = InteractionMediator(subsystems, filename=t_file.name)
        mediator.do_embedding()

        emb_dict = {'init_guess': 'chk'}
        sup_kwargs_test = [{'env_order':1, 'embed_settings': emb_dict}]
        mediator2 = InteractionMediator(subsystems2, supersystem_kwargs=sup_kwargs_test, filename=t_file.name)
        for i in range(len(mediator.supersystems)):
            test = mediator.supersystems[i]
            test2 = mediator2.supersystems[i]
            self.assertTrue(np.allclose(test.get_emb_dmat(), test2.get_emb_dmat()))

class TestEmbeddingEnergies(unittest.TestCase):

    def setUp(self):
        mol1 = gto.Mole()
        mol1.verbose = 3
        mol1.atom = '''
        O 0.0 0.0 0.0
        H 0.758602  0.000000  0.504284
        H 0.758602  0.000000  -0.504284'''
        mol1.basis = 'sto-3g'
        mol1.build()
        self.mol1 = mol1

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 0.0 1.0 0.0
        H 0.758602  1.000000  0.504284
        H 0.758602  1.000000  -0.504284'''
        mol2.basis = '3-21g'
        mol2.build()
        self.mol2 = mol2

    #@unittest.skip
    def test_simple_subsystems(self):
        sub1 = ClusterHLSubSystem(self.mol1, 'lda', 'rhf')
        sub2 = ClusterEnvSubSystem(self.mol2, 'lda')
        subsystems = [sub1, sub2]
        mediator = InteractionMediator(subsystems)
        mediator.do_embedding()
        test_emb_energy = mediator.get_emb_energy()

        #True Values
        sub1 = ClusterHLSubSystem(self.mol1, 'lda', 'rhf')
        sub2 = ClusterEnvSubSystem(self.mol2, 'lda')
        subsystems = [sub1, sub2]
        mol12 = helpers.concat_mols([sub1.mol, sub2.mol])
        fs_scf_obj = helpers.gen_scf_obj(mol12, 'lda')
        supersystem = ClusterSuperSystem(subsystems, 'lda', fs_scf_obj)
        supersystem.init_density()
        sup_e = supersystem.get_supersystem_energy()
        supersystem.freeze_and_thaw()
        supersystem.get_env_energy()
        supersystem.get_hl_energy()
        env_e = supersystem.subsystems[0].env_energy
        hl_e = supersystem.subsystems[0].hl_energy
        true_emb_energy = sup_e - env_e + hl_e
        self.assertAlmostEqual(true_emb_energy, test_emb_energy, delta=1e-9)

    @unittest.skip
    def test_explicit_subsystems(self):
        pass

    @unittest.skip
    def test_save_density(self):
        sub1 = ClusterHLSubSystem(self.mol1, 'lda', 'rhf', init_guess='supmol', hl_save_density=True)
        sub2 = ClusterEnvSubSystem(self.mol2, 'lda', init_guess='supmol')
        subsystems = [sub1, sub2]
        t_file = tempfile.NamedTemporaryFile()
        mediator = InteractionMediator(subsystems, filename=t_file.name)
        mediator.do_embedding()
        test_emb_energy = mediator.get_emb_energy()

    @unittest.skip
    def test_save_orbitals(self):
        sub1 = ClusterHLSubSystem(self.mol1, 'lda', 'rhf', init_guess='supmol', hl_save_orbs=True)
        sub2 = ClusterEnvSubSystem(self.mol2, 'lda', init_guess='supmol')
        subsystems = [sub1, sub2]
        t_file = tempfile.NamedTemporaryFile()
        mediator = InteractionMediator(subsystems, filename=t_file.name)
        mediator.do_embedding()
        test_emb_energy = mediator.get_emb_energy()
    
if __name__ == "__main__":
    unittest.main()
