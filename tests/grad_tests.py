# Tests for the analytical nuclear gradients
# Daniel Graham

import unittest
import os
import time
import shutil
import re
import copy
import numpy as np

from functools import reduce
from qsome import cluster_supersystem, cluster_subsystem, helpers
from pyscf import gto

class TestEnvSubSystemGradients(unittest.TestCase):

    def setUp(self):

        cs_mol = gto.M()
        cs_mol.atom = '''
        N    1.7030   29.2921   -0.3884
        '''
        cs_mol.basis = 'cc-pVDZ'
        cs_mol.charge = -3
        cs_mol.build()
        self.cs_small_mol1 = cs_mol

        cs_mol = gto.M()
        cs_mol.atom = '''
        F    0.1341   29.9216    0.1245
        F    1.1937   28.0518    0.1245
        F    2.2823   29.9034    0.1243
        '''
        cs_mol.basis = 'cc-pVDZ'
        cs_mol.charge = 3
        cs_mol.build()
        self.cs_small_mol2 = cs_mol

        cs_mol = gto.M()
        cs_mol.atom = '''
        F    0.1341   29.9216    0.1245
        F    1.1937   28.0518    0.1245
        F    2.2823   29.9034    0.1243
        '''
        cs_mol.basis = 'cc-pVDZ'
        cs_mol.charge = 3
        cs_mol.build()
        self.cs_twoatm_mol1 = cs_mol

        cs_mol = gto.M()
        cs_mol.atom = '''
        N    1.7030   29.2921   -0.3884
        '''
        cs_mol.basis = 'cc-pVDZ'
        cs_mol.charge = -3
        cs_mol.build()
        self.cs_twoatm_mol2 = cs_mol

        os_mol = gto.M()
        os_mol.atom = '''
        C    1.2026    0.7046    0.0000
        '''
        os_mol.basis = 'cc-pVDZ'
        os_mol.spin = 2
        os_mol.charge = -2
        os_mol.build()
        self.os_sub0_small_mol1 = os_mol

        os_mol = gto.M()
        os_mol.atom = '''
        F    0.8919    1.9804    0.0000 
        F    0.0403    0.0932    0.0000
        '''
        os_mol.basis = 'cc-pVDZ'
        os_mol.charge = 2
        os_mol.build()
        self.os_sub1_small_mol1 = os_mol

        os_mol = gto.M()
        os_mol.atom = '''
        H    1.8306    0.3589    0.8863
        C    1.2048    0.6943    0.0548
        '''
        os_mol.basis = 'cc-pVDZ'
        os_mol.spin = 1
        os_mol.charge = -2
        os_mol.build()
        self.os_sub0_twoatm_mol1 = os_mol

        os_mol = gto.M()
        os_mol.atom = '''
        F    0.0261    0.0744   -0.0159
        F    1.0682    2.0190   -0.0159
        '''
        os_mol.basis = 'cc-pVDZ'
        os_mol.charge = 2
        os_mol.build()
        self.os_sub1_twoatm_mol1 = os_mol

    @unittest.skip('takes forever')
    def test_rhf_den_grad(self):

        env_method = 'hf'
        hl_method = 'ccsd'
        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        ao_dmat = supersystem.get_sub_den_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_small_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_small_mol2)

                    coord_2 = self.cs_small_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_small_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_small_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_small_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_small_mol1)

                    coord_2 = self.cs_small_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_small_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_small_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num density grad terms
                num_sub1_den_grad = (subsys2.get_dmat() - subsys0.get_dmat())/(dim_diff*2.)
                num_sub2_den_grad = (env_subsys2.get_dmat() - env_subsys0.get_dmat())/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                self.assertTrue(np.allclose(ao_dmat[0][atm][dim], num_sub1_den_grad, atol=1e-5))
                self.assertTrue(np.allclose(ao_dmat[1][atm][dim], num_sub2_den_grad, atol=1e-5))


        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_twoatm_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_twoatm_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_twoatm_mol1, self.cs_twoatm_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        ao_dmat = supersystem.get_sub_den_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_twoatm_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_twoatm_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_twoatm_mol2)

                    coord_2 = self.cs_twoatm_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_twoatm_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_twoatm_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_twoatm_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_twoatm_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_twoatm_mol1)

                    coord_2 = self.cs_twoatm_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_twoatm_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_twoatm_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num density grad terms
                num_sub1_den_grad = (subsys2.get_dmat() - subsys0.get_dmat())/(dim_diff*2.)
                num_sub2_den_grad = (env_subsys2.get_dmat() - env_subsys0.get_dmat())/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                self.assertTrue(np.allclose(ao_dmat[0][atm][dim], num_sub1_den_grad, atol=1e-5))
                self.assertTrue(np.allclose(ao_dmat[1][atm][dim], num_sub2_den_grad, atol=1e-5))

    @unittest.skip('takes forever')
    def test_rks_den_grad(self):

        env_method = 'lda'
        hl_method = 'ccsd'
        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method, grid_level=5)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()

        supersystem.get_supersystem_nuc_grad()
        ao_dmat = supersystem.get_sub_den_grad()


        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_small_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_small_mol2)

                    coord_2 = self.cs_small_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_small_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_small_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_small_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_small_mol1)

                    coord_2 = self.cs_small_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_small_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_small_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method, grid_level=5)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method, grid_level=5)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num density grad terms
                num_sub1_den_grad = (subsys2.get_dmat() - subsys0.get_dmat())/(dim_diff*2.)
                num_sub2_den_grad = (env_subsys2.get_dmat() - env_subsys0.get_dmat())/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                self.assertTrue(np.allclose(ao_dmat[0][atm][dim], num_sub1_den_grad, atol=1e-5))
                self.assertTrue(np.allclose(ao_dmat[1][atm][dim], num_sub2_den_grad, atol=1e-5))


        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_twoatm_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_twoatm_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_twoatm_mol1, self.cs_twoatm_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        ao_dmat = supersystem.get_sub_den_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_twoatm_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_twoatm_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_twoatm_mol2)

                    coord_2 = self.cs_twoatm_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_twoatm_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_twoatm_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_twoatm_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_twoatm_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_twoatm_mol1)

                    coord_2 = self.cs_twoatm_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_twoatm_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_twoatm_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num density grad terms
                num_sub1_den_grad = (subsys2.get_dmat() - subsys0.get_dmat())/(dim_diff*2.)
                num_sub2_den_grad = (env_subsys2.get_dmat() - env_subsys0.get_dmat())/(dim_diff*2.)
                
                self.assertTrue(np.allclose(ao_dmat[0][atm][dim], num_sub1_den_grad, atol=1e-4))
                self.assertTrue(np.allclose(ao_dmat[1][atm][dim], num_sub2_den_grad, atol=1e-4))

        env_method = 'pbe'
        hl_method = 'ccsd'
        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        ao_dmat = supersystem.get_sub_den_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_small_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_small_mol2)

                    coord_2 = self.cs_small_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_small_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_small_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_small_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_small_mol1)

                    coord_2 = self.cs_small_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_small_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_small_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num density grad terms
                num_sub1_den_grad = (subsys2.get_dmat() - subsys0.get_dmat())/(dim_diff*2.)
                num_sub2_den_grad = (env_subsys2.get_dmat() - env_subsys0.get_dmat())/(dim_diff*2.)
                
                self.assertTrue(np.allclose(ao_dmat[0][atm][dim], num_sub1_den_grad, atol=1e-4))
                self.assertTrue(np.allclose(ao_dmat[1][atm][dim], num_sub2_den_grad, atol=1e-4))


        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_twoatm_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_twoatm_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_twoatm_mol1, self.cs_twoatm_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        ao_dmat = supersystem.get_sub_den_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_twoatm_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_twoatm_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_twoatm_mol2)

                    coord_2 = self.cs_twoatm_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_twoatm_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_twoatm_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_twoatm_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_twoatm_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_twoatm_mol1)

                    coord_2 = self.cs_twoatm_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_twoatm_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_twoatm_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num density grad terms
                num_sub1_den_grad = (subsys2.get_dmat() - subsys0.get_dmat())/(dim_diff*2.)
                num_sub2_den_grad = (env_subsys2.get_dmat() - env_subsys0.get_dmat())/(dim_diff*2.)
                
                self.assertTrue(np.allclose(ao_dmat[0][atm][dim], num_sub1_den_grad, atol=1e-4))
                self.assertTrue(np.allclose(ao_dmat[1][atm][dim], num_sub2_den_grad, atol=1e-4))

        env_method = 'b3lyp'
        hl_method = 'ccsd'
        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        ao_dmat = supersystem.get_sub_den_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_small_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_small_mol2)

                    coord_2 = self.cs_small_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_small_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_small_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_small_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_small_mol1)

                    coord_2 = self.cs_small_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_small_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_small_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num density grad terms
                num_sub1_den_grad = (subsys2.get_dmat() - subsys0.get_dmat())/(dim_diff*2.)
                num_sub2_den_grad = (env_subsys2.get_dmat() - env_subsys0.get_dmat())/(dim_diff*2.)
                
                self.assertTrue(np.allclose(ao_dmat[0][atm][dim], num_sub1_den_grad, atol=1e-4))
                self.assertTrue(np.allclose(ao_dmat[1][atm][dim], num_sub2_den_grad, atol=1e-4))


        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_twoatm_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_twoatm_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_twoatm_mol1, self.cs_twoatm_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        ao_dmat = supersystem.get_sub_den_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_twoatm_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_twoatm_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_twoatm_mol2)

                    coord_2 = self.cs_twoatm_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_twoatm_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_twoatm_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_twoatm_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_twoatm_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_twoatm_mol1)

                    coord_2 = self.cs_twoatm_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_twoatm_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_twoatm_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num density grad terms
                num_sub1_den_grad = (subsys2.get_dmat() - subsys0.get_dmat())/(dim_diff*2.)
                num_sub2_den_grad = (env_subsys2.get_dmat() - env_subsys0.get_dmat())/(dim_diff*2.)
                
                self.assertTrue(np.allclose(ao_dmat[0][atm][dim], num_sub1_den_grad, atol=1e-4))
                self.assertTrue(np.allclose(ao_dmat[1][atm][dim], num_sub2_den_grad, atol=1e-4))

    @unittest.skip('takes forever')
    def test_rhf_grad(self):
        env_method = 'hf'
        hl_method = 'ccsd'
        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        atm_grad = supersystem.get_env_nuc_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_small_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_small_mol2)

                    coord_2 = self.cs_small_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_small_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_small_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_small_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_small_mol1)

                    coord_2 = self.cs_small_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_small_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_small_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num env energy grad terms
                num_sub1_env_grad = (subsys2.get_env_elec_energy() - subsys0.get_env_elec_energy())/(dim_diff*2.)
                num_sub2_env_grad = (env_subsys2.get_env_elec_energy() - env_subsys0.get_env_elec_energy())/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                print (np.max(np.abs(atm_grad[0][atm][dim] - num_sub1_env_grad)))
                print (np.max(np.abs(atm_grad[1][atm][dim] - num_sub2_env_grad)))
                self.assertTrue(np.allclose(atm_grad[0][atm][dim], num_sub1_env_grad, atol=1e-5))
                self.assertTrue(np.allclose(atm_grad[1][atm][dim], num_sub2_env_grad, atol=1e-5))


        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_twoatm_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_twoatm_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_twoatm_mol1, self.cs_twoatm_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        atm_grad = supersystem.get_env_nuc_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_twoatm_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_twoatm_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_twoatm_mol2)

                    coord_2 = self.cs_twoatm_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_twoatm_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_twoatm_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_twoatm_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_twoatm_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_twoatm_mol1)

                    coord_2 = self.cs_twoatm_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_twoatm_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_twoatm_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num env energy grad terms
                num_sub1_env_grad = (subsys2.get_env_elec_energy() - subsys0.get_env_elec_energy())/(dim_diff*2.)
                num_sub2_env_grad = (env_subsys2.get_env_elec_energy() - env_subsys0.get_env_elec_energy())/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                print (np.max(np.abs(atm_grad[0][atm][dim] - num_sub1_env_grad)))
                print (np.max(np.abs(atm_grad[1][atm][dim] - num_sub2_env_grad)))
                self.assertTrue(np.allclose(atm_grad[0][atm][dim], num_sub1_env_grad, atol=1e-5))
                self.assertTrue(np.allclose(atm_grad[1][atm][dim], num_sub2_env_grad, atol=1e-5))

    @unittest.skip('takes forever')
    def test_rks_grad(self):
        env_method = 'pbe'
        hl_method = 'ccsd'
        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method, grid_level=5)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        atm_grad = supersystem.get_env_nuc_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_small_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_small_mol2)

                    coord_2 = self.cs_small_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_small_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_small_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_small_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_small_mol1)

                    coord_2 = self.cs_small_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_small_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_small_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method, grid_level=5)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method, grid_level=5)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num env energy grad terms
                num_sub1_env_grad = (subsys2.get_env_elec_energy() - subsys0.get_env_elec_energy())/(dim_diff*2.)
                num_sub2_env_grad = (env_subsys2.get_env_elec_energy() - env_subsys0.get_env_elec_energy())/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                print (np.max(np.abs(atm_grad[0][atm][dim] - num_sub1_env_grad)))
                print (np.max(np.abs(atm_grad[1][atm][dim] - num_sub2_env_grad)))
                self.assertTrue(np.allclose(atm_grad[0][atm][dim], num_sub1_env_grad, atol=1e-5))
                self.assertTrue(np.allclose(atm_grad[1][atm][dim], num_sub2_env_grad, atol=1e-5))


        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_twoatm_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_twoatm_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_twoatm_mol1, self.cs_twoatm_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method, grid_level=5)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_supersystem_nuc_grad()
        atm_grad = supersystem.get_env_nuc_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_twoatm_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_twoatm_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_twoatm_mol2)

                    coord_2 = self.cs_twoatm_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_twoatm_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_twoatm_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_twoatm_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_twoatm_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_twoatm_mol1)

                    coord_2 = self.cs_twoatm_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_twoatm_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_twoatm_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method, grid_level=5)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method, grid_level=5)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()

                #Num env energy grad terms
                num_sub1_env_grad = (subsys2.get_env_elec_energy() - subsys0.get_env_elec_energy())/(dim_diff*2.)
                num_sub2_env_grad = (env_subsys2.get_env_elec_energy() - env_subsys0.get_env_elec_energy())/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                print (np.max(np.abs(atm_grad[0][atm][dim] - num_sub1_env_grad)))
                print (np.max(np.abs(atm_grad[1][atm][dim] - num_sub2_env_grad)))
                self.assertTrue(np.allclose(atm_grad[0][atm][dim], num_sub1_env_grad, atol=1e-5))
                self.assertTrue(np.allclose(atm_grad[1][atm][dim], num_sub2_env_grad, atol=1e-5))

    def test_uhf_grad(self):
        env_method = 'hf'
        hl_method = 'ccsd'
    def test_rohf_grad(self):
        env_method = 'hf'
        hl_method = 'ccsd'
    def test_uks_grad(self):
        pass
    def test_roks_grad(self):
        pass

    @unittest.skip('takes forever')
    def test_rhf_ccsd_grad(self):
        env_method = 'hf'
        hl_method = 'ccsd'
        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_hl_energy()
        supersystem.get_supersystem_nuc_grad()
        env_atm_grad = supersystem.get_env_nuc_grad()
        hl_atm_grad = supersystem.get_hl_nuc_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_small_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_small_mol2)

                    coord_2 = self.cs_small_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_small_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_small_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_small_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_small_mol1)

                    coord_2 = self.cs_small_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_small_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_small_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()
                supersystem0.get_hl_energy()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()
                supersystem2.get_hl_energy()

                #Num env energy grad terms
                num_sub1_hl_grad = (subsys2.hl_energy - subsys0.hl_energy)/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                print (np.max(np.abs(hl_atm_grad[atm][dim] - num_sub1_hl_grad)))
                self.assertTrue(np.allclose(hl_atm_grad[atm][dim], num_sub1_hl_grad, atol=1e-5))

        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_twoatm_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_twoatm_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_twoatm_mol1, self.cs_twoatm_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_hl_energy()
        supersystem.get_supersystem_nuc_grad()
        atm_grad = supersystem.get_env_nuc_grad()
        env_atm_grad = supersystem.get_env_nuc_grad()
        hl_atm_grad = supersystem.get_hl_nuc_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_twoatm_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_twoatm_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_twoatm_mol2)

                    coord_2 = self.cs_twoatm_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_twoatm_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_twoatm_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_twoatm_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_twoatm_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_twoatm_mol1)

                    coord_2 = self.cs_twoatm_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_twoatm_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_twoatm_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()
                supersystem0.get_hl_energy()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()
                supersystem2.get_hl_energy()

                #Num env energy grad terms
                num_sub1_hl_grad = (subsys2.hl_energy - subsys0.hl_energy)/(dim_diff*2.)
                num_sub1_hl_grad -= (subsys2.mol.energy_nuc() - subsys0.mol.energy_nuc())/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                print (np.max(np.abs(hl_atm_grad[atm][dim] - num_sub1_hl_grad)))
                self.assertTrue(np.allclose(hl_atm_grad[atm][dim], num_sub1_hl_grad, atol=1e-5))

    @unittest.skip('takes forever')
    def test_rks_ccsd_grad(self):
        env_method = 'lda'
        hl_method = 'ccsd'
        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_hl_energy()
        supersystem.get_supersystem_nuc_grad()
        env_atm_grad = supersystem.get_env_nuc_grad()
        hl_atm_grad = supersystem.get_hl_nuc_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_small_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_small_mol2)

                    coord_2 = self.cs_small_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_small_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_small_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_small_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_small_mol1)

                    coord_2 = self.cs_small_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_small_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_small_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()
                supersystem0.get_hl_energy()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()
                supersystem2.get_hl_energy()

                #Num env energy grad terms
                num_sub1_hl_grad = (subsys2.hl_energy - subsys0.hl_energy)/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                print (np.max(np.abs(hl_atm_grad[atm][dim] - num_sub1_hl_grad)))
                self.assertTrue(np.allclose(hl_atm_grad[atm][dim], num_sub1_hl_grad, atol=1e-4))

        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_twoatm_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_twoatm_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_twoatm_mol1, self.cs_twoatm_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_hl_energy()
        supersystem.get_supersystem_nuc_grad()
        atm_grad = supersystem.get_env_nuc_grad()
        env_atm_grad = supersystem.get_env_nuc_grad()
        hl_atm_grad = supersystem.get_hl_nuc_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_twoatm_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_twoatm_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_twoatm_mol2)

                    coord_2 = self.cs_twoatm_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_twoatm_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_twoatm_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_twoatm_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_twoatm_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_twoatm_mol1)

                    coord_2 = self.cs_twoatm_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_twoatm_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_twoatm_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()
                supersystem0.get_hl_energy()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()
                supersystem2.get_hl_energy()

                #Num env energy grad terms
                num_sub1_hl_grad = (subsys2.hl_energy - subsys0.hl_energy)/(dim_diff*2.)
                num_sub1_hl_grad -= (subsys2.mol.energy_nuc() - subsys0.mol.energy_nuc())/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                print (np.max(np.abs(hl_atm_grad[atm][dim] - num_sub1_hl_grad)))
                self.assertTrue(np.allclose(hl_atm_grad[atm][dim], num_sub1_hl_grad, atol=1e-4))

    
    def test_emb_full_grad(self):
        env_method = 'hf'
        hl_method = 'ccsd'
        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_small_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_small_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_small_mol1, self.cs_small_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_hl_energy()
        supersystem.get_supersystem_nuc_grad()
        emb_atm_grad = supersystem.get_emb_nuc_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_small_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_small_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_small_mol2)

                    coord_2 = self.cs_small_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_small_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_small_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_small_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_small_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_small_mol1)

                    coord_2 = self.cs_small_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_small_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_small_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()
                supersystem0.get_hl_energy()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()
                supersystem2.get_hl_energy()

                #Num env energy grad terms
                num_sub1_emb_grad = ((supersystem2.get_supersystem_energy() - subsys2.env_energy + subsys2.hl_energy) - (supersystem0.get_supersystem_energy() - subsys0.env_energy + subsys0.hl_energy))/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                print (num_sub1_emb_grad)
                print (emb_atm_grad[atm][dim])
                print (np.max(np.abs(emb_atm_grad[atm][dim] - num_sub1_emb_grad)))
                self.assertTrue(np.allclose(emb_atm_grad[atm][dim], num_sub1_emb_grad, atol=1e-5))

        subsys1 = cluster_subsystem.ClusterHLSubSystem(self.cs_twoatm_mol1, env_method, hl_method)
        env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(self.cs_twoatm_mol2, env_method)
        sup_mol11 = helpers.concat_mols([self.cs_twoatm_mol1, self.cs_twoatm_mol2])
        fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_hl_energy()
        supersystem.get_supersystem_nuc_grad()
        atm_grad = supersystem.get_env_nuc_grad()
        emb_atm_grad = supersystem.get_emb_nuc_grad()

        atmlist = range(supersystem.mol.natm)
        for atm in atmlist:
            for dim in range(3):
                dim_diff = 0.0001
                if atm in supersystem.atm_sub2sup[0]:
                    coord_0 = self.cs_twoatm_mol1.atom_coords()
                    coord_0[atm][dim] = coord_0[atm][dim] - dim_diff
                    mol0 = self.cs_twoatm_mol1.set_geom_(coord_0, 'B', inplace=False)
                    env_mol0 = copy.copy(self.cs_twoatm_mol2)

                    coord_2 = self.cs_twoatm_mol1.atom_coords()
                    coord_2[atm][dim] = coord_2[atm][dim] + dim_diff
                    mol2 = self.cs_twoatm_mol1.set_geom_(coord_2, 'B', inplace=False)
                    env_mol2 = copy.copy(self.cs_twoatm_mol2)
                else:
                    sub1_atms = supersystem.atm_sub2sup[0][-1] + 1
                    coord_0 = self.cs_twoatm_mol2.atom_coords()
                    coord_0[atm-sub1_atms][dim] = coord_0[atm-sub1_atms][dim] - dim_diff
                    env_mol0 = self.cs_twoatm_mol2.set_geom_(coord_0, 'B', inplace=False)
                    mol0 = copy.copy(self.cs_twoatm_mol1)

                    coord_2 = self.cs_twoatm_mol2.atom_coords()
                    coord_2[atm-sub1_atms][dim] = coord_2[atm-sub1_atms][dim] + dim_diff
                    env_mol2 = self.cs_twoatm_mol2.set_geom_(coord_2, 'B', inplace=False)
                    mol2 = copy.copy(self.cs_twoatm_mol1)

                subsys0 = cluster_subsystem.ClusterHLSubSystem(mol0, env_method, hl_method)
                env_subsys0 = cluster_subsystem.ClusterEnvSubSystem(env_mol0, env_method)
                sup_mol00 = helpers.concat_mols([mol0, env_mol0])
                fs_scf_obj0 = helpers.gen_scf_obj(sup_mol00, env_method)
                supersystem0 = cluster_supersystem.ClusterSuperSystem([subsys0, env_subsys0], env_method, fs_scf_obj0)
                supersystem0.init_density()
                supersystem0.freeze_and_thaw()
                supersystem0.get_hl_energy()

                subsys2 = cluster_subsystem.ClusterHLSubSystem(mol2, env_method, hl_method)
                env_subsys2 = cluster_subsystem.ClusterEnvSubSystem(env_mol2, env_method)
                sup_mol22 = helpers.concat_mols([mol2, env_mol2])
                fs_scf_obj2 = helpers.gen_scf_obj(sup_mol22, env_method)
                supersystem2 = cluster_supersystem.ClusterSuperSystem([subsys2, env_subsys2], env_method, fs_scf_obj2)
                supersystem2.init_density()
                supersystem2.freeze_and_thaw()
                supersystem2.get_hl_energy()

                #Num env energy grad terms
                num_sub1_emb_grad = (supersystem2.get_supersystem_energy() - subsys2.env_energy + subsys2.hl_energy - (supersystem0.get_supersystem_energy() - subsys0.env_energy + subsys0.hl_energy))/(dim_diff*2.)
                
                s2s = supersystem.sub2sup
                print (np.max(np.abs(emb_atm_grad[atm][dim] - num_sub1_emb_grad)))
                self.assertTrue(np.allclose(emb_atm_grad[atm][dim], num_sub1_emb_grad, atol=1e-5))

    def test_rks_ccsd_grad(self):
        env_method = 'lda'

if __name__ == "__main__":
    unittest.main()
