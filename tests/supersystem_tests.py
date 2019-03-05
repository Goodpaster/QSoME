#A module to test the methods of the supersystem aside from the F&t Method.

#get_supersystem_energy
#get_emb_subsys_elec_energy
#correct_env_energy
#get_active_energy
#correct_active_energy
#env_in_env_energy
#update_fock
#update_proj_pot
#read_chkfile
#save_chkfile
#freeze_and_thaw

# A module to tests the methods of the SuperSystem

import unittest
import os
import shutil
import re

from copy import copy

from qsome import cluster_subsystem, cluster_supersystem
from pyscf import gto, lib, scf, dft

import numpy as np

class TestClusterSuperSystemMethods(unittest.TestCase):

    #@unittest.skip
    def test_get_supersystem_energy(self):

        #Closed Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')
        supsystem_e = supersystem.get_supersystem_energy()

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0
        '''
        mol3.basis = '3-21g'
        mol3.build()
        test_scf = dft.RKS(mol3)
        test_scf.xc = 'b3lyp'
        test_scf.conv_tol = supersystem.conv
        grids = dft.gen_grid.Grids(mol3)
        grids.level = supersystem.grid_level
        grids.build()
        test_scf.grids = grids
        test_e = test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertAlmostEqual(test_e, supsystem_e)
        self.assertTrue(np.allclose(test_dmat, (supersystem.dmat[0] + supersystem.dmat[1])))

        # Unrestricted Open Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        mol.basis = 'aug-cc-pVDZ'
        mol.spin = 1
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method, active_unrestricted=True, unrestricted=True)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        He 3.0 0.0 0.0'''
        mol2.basis = 'aug-cc-pVDZ'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', fs_unrestricted=True)
        supsystem_e = supersystem.get_supersystem_energy()

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom = '''
        Li 0.0 0.0 0.0
        He 3.0 0.0 0.0
        '''
        mol3.basis = 'aug-cc-pVDZ'
        mol3.spin = 1
        mol3.build()
        test_scf = dft.UKS(mol3)
        test_scf.xc = 'b3lyp'
        test_scf.conv_tol = supersystem.conv
        grids = dft.gen_grid.Grids(mol3)
        grids.level = supersystem.grid_level
        grids.build()
        test_scf.grids = grids
        test_e = test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertAlmostEqual(test_e, supsystem_e)
        self.assertTrue(np.allclose(test_dmat[0], supersystem.dmat[0]))
        self.assertTrue(np.allclose(test_dmat[1], supersystem.dmat[1]))

    def test_get_emb_subsys_elec_energy(self):
        pass 
    #@unittest.skip
    def test_get_emb_subsys_elec_energy(self):

        # Closed Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom ='''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol3.basis = '3-21g'
        mol3.build()
        mf = dft.RKS(mol3)
        mf.xc = 'b3lyp'
        grids = dft.gen_grid.Grids(mol3)
        grids.level = supersystem.grid_level
        grids.build()
        mf.grids = grids
        mf.small_rho_cutoff = supersystem.rho_cutoff
        mf_t_dmat = mf.get_init_guess(key='minao')
        mf_init_dmat = np.zeros_like(mf_t_dmat)

        #get energies of two embedded systems. 
        s2s = supersystem.sub2sup
        mf_init_dmat[np.ix_(s2s[0], s2s[0])] += subsys.dmat[0] + subsys.dmat[1]
        mf_init_dmat[np.ix_(s2s[1], s2s[1])] += subsys2.dmat[0] + subsys2.dmat[1]
        mf_hcore = mf.get_hcore()
        mf_init_veff = mf.get_veff(dm=mf_init_dmat)

        dm_1 = mf_init_dmat[np.ix_(s2s[0], s2s[0])]
        hcore_1_emb = mf_hcore[np.ix_(s2s[0], s2s[0])]
        veff_1_emb = mf_init_veff[np.ix_(s2s[0], s2s[0])]
        mf_1 = dft.RKS(mol)
        mf_1.xc = 'm06'
        mf_1.grids = supersystem.fs_scf.grids
        hcore_1_emb = hcore_1_emb - mf_1.get_hcore()
        veff_1 = mf_1.get_veff(dm=dm_1)
        veff_1_emb = veff_1_emb - veff_1
        test_sub1_e = mf_1.energy_elec(dm=dm_1)[0] + np.einsum('ij,ji', hcore_1_emb, dm_1) + np.einsum('ij,ji', veff_1_emb, dm_1)

        dm_2 = mf_init_dmat[np.ix_(s2s[1], s2s[1])]
        hcore_2_emb = mf_hcore[np.ix_(s2s[1], s2s[1])]
        veff_2_emb = mf_init_veff[np.ix_(s2s[1], s2s[1])]
        mf_2 = dft.RKS(mol2)
        mf_2.xc = 'm06'
        mf_2.grids = supersystem.fs_scf.grids
        hcore_2_emb = hcore_2_emb - mf_2.get_hcore()
        veff_2 = mf_2.get_veff(dm=dm_2)
        veff_2_emb = veff_2_emb - veff_2
        test_sub2_e = mf_2.energy_elec(dm=dm_2)[0] + np.einsum('ij,ji', hcore_2_emb, dm_2) + np.einsum('ij,ji', veff_2_emb, dm_2)

        sub1_e = supersystem.subsystems[0].get_env_elec_energy()
        sub2_e = supersystem.subsystems[1].get_env_elec_energy()
        self.assertAlmostEqual(test_sub1_e, sub1_e, delta=1e-8)
        self.assertAlmostEqual(test_sub2_e, sub2_e, delta=1e-8)

        # Unrestricted Open Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        mol.basis = '3-21g'
        mol.spin = 1
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method, unrestricted=True, active_unrestricted=True)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        H 1.595 0.0 0.0'''
        mol2.basis = '3-21g'
        mol2.spin = -1
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao')

        supsystem_e = supersystem.get_supersystem_energy()
        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom = '''
        Li 0.0 0.0 0.0
        H 1.595 0.0 0.0
        '''
        mol3.basis = '3-21g'
        mol3.build()
        mf = dft.RKS(mol3)
        mf.xc = 'm06'
        grids = dft.gen_grid.Grids(mol3)
        grids.level = supersystem.grid_level
        grids.build()
        mf.grids = grids
        mf.small_rho_cutoff = supersystem.rho_cutoff
        mf_t_dmat = mf.get_init_guess(key='minao')
        mf_init_dmat = np.zeros_like(mf_t_dmat)

        #get energies of two embedded systems. 
        s2s = supersystem.sub2sup
        mf_init_dmat[np.ix_(s2s[0], s2s[0])] += subsys.dmat[0] + subsys.dmat[1]
        mf_init_dmat[np.ix_(s2s[1], s2s[1])] += subsys2.dmat[0] + subsys2.dmat[1]
        mf_hcore = mf.get_hcore()
        mf_init_veff = mf.get_veff(dm=mf_init_dmat)

        dm_1 = [mf_init_dmat[np.ix_(s2s[0], s2s[0])]/2., mf_init_dmat[np.ix_(s2s[0], s2s[0])]/2.]
        hcore_1_emb = mf_hcore[np.ix_(s2s[0], s2s[0])]
        veff_1_emb = [mf_init_veff[np.ix_(s2s[0], s2s[0])], mf_init_veff[np.ix_(s2s[0], s2s[0])]]
        mf_1 = dft.UKS(mol)
        mf_1.xc = 'm06'
        mf_1.grids = supersystem.fs_scf.grids
        hcore_1_emb = hcore_1_emb - mf_1.get_hcore()
        veff_1 = mf_1.get_veff(dm=dm_1)
        veff_1_emb = [veff_1_emb[0] - veff_1[0], veff_1_emb[1] - veff_1[1]]
        test_sub1_e = mf_1.energy_elec(dm=dm_1)[0] + np.einsum('ij,ji', hcore_1_emb, dm_1[0]) + np.einsum('ij,ji', hcore_1_emb, dm_1[1]) +  np.einsum('ij,ji', veff_1_emb[0], dm_1[0]) + np.einsum('ij,ji', veff_1_emb[1], dm_1[1])

        dm_2 = [mf_init_dmat[np.ix_(s2s[1], s2s[1])]/2., mf_init_dmat[np.ix_(s2s[1], s2s[1])]/2.]
        hcore_2_emb = mf_hcore[np.ix_(s2s[1], s2s[1])]
        veff_2_emb = [mf_init_veff[np.ix_(s2s[1], s2s[1])], mf_init_veff[np.ix_(s2s[1], s2s[1])]]
        mf_2 = dft.UKS(mol2)
        mf_2.xc = 'm06'
        mf_2.grids = supersystem.fs_scf.grids
        hcore_2_emb = hcore_2_emb - mf_2.get_hcore()
        veff_2 = mf_2.get_veff(dm=dm_2)
        veff_2_emb = [veff_2_emb[0] - veff_2[0], veff_2_emb[1] - veff_2[1]]
        test_sub2_e = mf_2.energy_elec(dm=dm_2)[0] + np.einsum('ij,ji', hcore_2_emb, dm_2[0]) + np.einsum('ij,ji', hcore_2_emb, dm_2[1]) +  np.einsum('ij,ji', veff_2_emb[0], dm_2[0]) + np.einsum('ij,ji', veff_2_emb[1], dm_2[1])


        sub1_e = supersystem.subsystems[0].get_env_elec_energy()
        sub2_e = supersystem.subsystems[1].get_env_elec_energy()
        self.assertAlmostEqual(test_sub1_e, sub1_e, delta=1e-8)
        self.assertAlmostEqual(test_sub2_e, sub2_e, delta=1e-8)


    @unittest.skip
    def test_correct_env_energy(self):
        pass

    @unittest.skip
    def test_get_active_energy(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao') 
        pass

    @unittest.skip
    def test_correct_active_energy(self):
        pass

    #@unittest.skip
    def test_update_fock(self):

        # Closed Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')
        supersystem.update_fock()

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom ='''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol3.basis = '3-21g'
        mol3.build()
        mf = dft.RKS(mol3)
        mf.xc = 'b3lyp'
        grids = dft.gen_grid.Grids(mol3)
        grids.level = supersystem.grid_level
        grids.build()
        mf.grids = grids
        mf.small_rho_cutoff = supersystem.rho_cutoff
        mf_t_dmat = mf.get_init_guess(key='minao')
        mf_init_dmat = np.zeros_like(mf_t_dmat)
        #get energies of two embedded systems. 
        s2s = supersystem.sub2sup
        mf_init_dmat[np.ix_(s2s[0], s2s[0])] += subsys.dmat[0] + subsys.dmat[1]
        mf_init_dmat[np.ix_(s2s[1], s2s[1])] += subsys2.dmat[0] + subsys2.dmat[1]
        mf_hcore = mf.get_hcore()
        mf_init_veff = mf.get_veff(dm=mf_init_dmat)
        full_fock = mf_hcore + mf_init_veff
        self.assertTrue(np.allclose(full_fock, supersystem.fock[0]))

        test_fock1 = full_fock[np.ix_(s2s[0], s2s[0])]
        test_fock2 = full_fock[np.ix_(s2s[1], s2s[1])]
        self.assertTrue(np.allclose(test_fock1, supersystem.subsystems[0].emb_fock[0]))
        self.assertTrue(np.allclose(test_fock2, supersystem.subsystems[1].emb_fock[0]))

        # Unrestricted Open Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        mol.basis = '3-21g'
        mol.spin = 1
        mol.build()
        env_method = 'um06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        H 1.595 0.0 0.0'''
        mol2.basis = '3-21g'
        mol2.spin = -1
        mol2.build()
        env_method = 'um06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao')
        supersystem.update_fock()

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom = '''
        Li 0.0 0.0 0.0
        H 1.595 0.0 0.0
        '''
        mol3.basis = '3-21g'
        mol3.build()
        mf = dft.UKS(mol3)
        mf.xc = 'm06'
        grids = dft.gen_grid.Grids(mol3)
        grids.level = supersystem.grid_level
        grids.build()
        mf.grids = grids
        mf.small_rho_cutoff = supersystem.rho_cutoff
        mf_t_dmat = mf.get_init_guess(key='minao')
        mf_init_dmat = np.zeros_like(mf_t_dmat)

        #get energies of two embedded systems. 
        s2s = supersystem.sub2sup
        mf_init_dmat[0][np.ix_(s2s[0], s2s[0])] += subsys.dmat[0]
        mf_init_dmat[0][np.ix_(s2s[1], s2s[1])] += subsys2.dmat[0]
        mf_init_dmat[1][np.ix_(s2s[0], s2s[0])] += subsys.dmat[1]
        mf_init_dmat[1][np.ix_(s2s[1], s2s[1])] += subsys2.dmat[1]
        mf_hcore = mf.get_hcore()
        mf_init_veff = mf.get_veff(dm=mf_init_dmat)
        full_fock = [mf_hcore + mf_init_veff[0], mf_hcore + mf_init_veff[1]]
        self.assertTrue(np.allclose(full_fock[0], supersystem.fock[0]))
        self.assertTrue(np.allclose(full_fock[1], supersystem.fock[1]))

        test_fock1 = [full_fock[0][np.ix_(s2s[0], s2s[0])], full_fock[1][np.ix_(s2s[0], s2s[0])]]
        test_fock2 = [full_fock[0][np.ix_(s2s[1], s2s[1])], full_fock[1][np.ix_(s2s[1], s2s[1])]]
        self.assertTrue(np.allclose(test_fock1[0], supersystem.subsystems[0].emb_fock[0]))
        self.assertTrue(np.allclose(test_fock1[1], supersystem.subsystems[0].emb_fock[1]))
        self.assertTrue(np.allclose(test_fock2[0], supersystem.subsystems[1].emb_fock[0]))
        self.assertTrue(np.allclose(test_fock2[1], supersystem.subsystems[1].emb_fock[1]))

    #@unittest.skip
    def test_update_proj_pot(self):
        """This test is crude, but the only other way to do it would 
            be to actually calculate the projection operator."""

        # Closed Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        He 1.0 2.0 0.0
        He 3.0 2.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')

        # Calculate the huzinaga projector potential.
        s2s = supersystem.sub2sup
        sub_pop_list = [0,0]
        for i in range(len(supersystem.subsystems)):
            A = i
            nA = supersystem.subsystems[A].mol.nao_nr()
            SAA = supersystem.smat[np.ix_(s2s[A], s2s[A])]
            POp = [np.zeros((nA, nA)), np.zeros((nA, nA))]

            # cycle over all other subsystems
            for B in range(len(supersystem.subsystems)):
                if B==A: continue

                SAB = supersystem.smat[np.ix_(s2s[A], s2s[B])]
                SBA = supersystem.smat[np.ix_(s2s[B], s2s[A])]
                FAB = [None, None]
                FAB[0] = supersystem.fock[0][np.ix_(s2s[A], s2s[B])]
                FAB[1] = supersystem.fock[1][np.ix_(s2s[A], s2s[B])]
                FDS = [None, None]
                FDS[0] = np.dot( FAB[0], np.dot( supersystem.subsystems[B].dmat[0], SBA ))
                FDS[1] = np.dot( FAB[1], np.dot( supersystem.subsystems[B].dmat[1], SBA ))
                POp[0] += -1. * ( FDS[0] + FDS[0].transpose() ) 
                POp[1] += -1. * ( FDS[0] + FDS[0].transpose() )

            sub_pop_list[i] = POp

        self.assertTrue(np.allclose(sub_pop_list[0][0], supersystem.proj_pot[0][0]))
        self.assertTrue(np.allclose(sub_pop_list[0][1], supersystem.proj_pot[0][1]))
        self.assertTrue(np.allclose(sub_pop_list[1][0], supersystem.proj_pot[1][0]))
        self.assertTrue(np.allclose(sub_pop_list[1][1], supersystem.proj_pot[1][1]))

        # Unrestricted Open Shell

    #@unittest.skip
    def test_read_chkfile(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([copy(subsys), copy(subsys2)], 'b3lyp', ft_initguess='1e')
        old_dmat = supersystem.dmat
        old_sub1_dmat = supersystem.subsystems[0].dmat
        old_sub2_dmat = supersystem.subsystems[1].dmat
        supersystem2 = cluster_supersystem.ClusterSuperSystem([copy(subsys), copy(subsys2)], 'b3lyp', ft_initguess='readchk')
        new_dmat = supersystem2.dmat
        new_sub1_dmat = supersystem2.subsystems[0].dmat
        new_sub2_dmat = supersystem2.subsystems[1].dmat
        self.assertTrue(np.equal(old_dmat, new_dmat).all)
        self.assertTrue(np.equal(old_sub1_dmat, new_sub1_dmat).all)
        self.assertTrue(np.equal(old_sub2_dmat, new_sub2_dmat).all)

        # Unrestricted Open Shell 

    #@unittest.skip
    def test_save_chkfile(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')

        self.assertTrue(os.path.isfile('temp.hdf5'))

    #@unittest.skip
    def test_freeze_and_thaw(self):

        #Supermolecular test.
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0.74 0. 0.
        H 0. 0. 0.
        ghost:H 1.48 0.0 0.0
        ghost:H 2.22 0.0 0.0
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method, save_orbs=True, save_density=True)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        H 1.48 0.0 0.0
        H 2.22 0.0 0.0
        ghost:H 0.74 0. 0.
        ghost:H 0. 0. 0.
        '''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, save_orbs=True, save_density=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao', fs_save_orbs=True, fs_save_density=True, ft_save_orbs=True, ft_save_density=True)
        supersystem.freeze_and_thaw()
        supersystem.env_in_env_energy()
        supersystem.get_supersystem_energy()
        self.assertTrue(os.path.isfile('temp_1.molden'))
        self.assertTrue(os.path.isfile('temp_2.molden'))
        self.assertTrue(os.path.isfile('temp_1.cube'))
        self.assertTrue(os.path.isfile('temp_2.cube'))
        self.assertTrue(os.path.isfile('temp_dftindft.cube'))
        self.assertTrue(os.path.isfile('temp_super.molden'))
        self.assertTrue(os.path.isfile('temp_super.cube'))

        projector_energy = np.trace(np.dot(subsys.dmat[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.dmat[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.dmat[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.dmat[1], supersystem.proj_pot[1][1]))
        self.assertAlmostEqual(0.0, projector_energy, delta=1e-10)

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom ='''
        H 0.74 0. 0.
        H 0. 0. 0.
        H 1.48 0.0 0.0
        H 2.22 0.0 0.0
        '''
        mol3.basis = '3-21g'
        mol3.build()
        mf = dft.RKS(mol3)
        mf.xc = 'm06'
        grids = dft.gen_grid.Grids(mol3)
        grids.level = supersystem.grid_level
        grids.build()
        mf.grids = grids
        mf.kernel()
        test_dmat = mf.make_rdm1()
        test_e = mf.energy_tot()
        sup_e = supersystem.env_in_env_energy()
        #self.assertTrue(np.allclose(test_dmat, (supersystem.dmat[0] + supersystem.dmat[1]), atol=1e-6))
        self.assertAlmostEqual(test_e, sup_e, delta=1e-10)

        #Long distance test.
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0.74 0. 0.
        H 0. 0. 0.
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        H 11.48 0.0 0.0
        H 12.22 0.0 0.0
        '''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao')
        supersystem.freeze_and_thaw()

        projector_energy = np.trace(np.dot(subsys.dmat[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.dmat[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.dmat[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.dmat[1], supersystem.proj_pot[1][1]))
        self.assertAlmostEqual(0.0, projector_energy, delta=1e-15)

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom ='''
        H 0.74 0. 0.
        H 0. 0. 0.
        H 11.48 0.0 0.0
        H 12.22 0.0 0.0
        '''
        mol3.basis = '3-21g'
        mol3.build()
        mf = dft.RKS(mol3)
        mf.xc = 'm06'
        grids = dft.gen_grid.Grids(mol3)
        grids.level = supersystem.grid_level
        grids.build()
        mf.grids = grids
        mf.kernel()
        test_dmat = mf.make_rdm1()
        test_e = mf.energy_tot()
        sup_e = supersystem.env_in_env_energy()
        #self.assertTrue(np.allclose(test_dmat, (supersystem.dmat[0] + supersystem.dmat[1]), atol=1e-6))
        self.assertAlmostEqual(test_e, sup_e, delta=1e-10)

        #Projection energy
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0.74 0. 0.
        H 0. 0. 0.
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        H 1.48 0.0 0.0
        H 2.22 0.0 0.0
        '''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='supmol')

        initial_projector_energy = 0.0
        initial_projector_energy = np.trace(np.dot(subsys.dmat[0], supersystem.proj_pot[0][0]))
        initial_projector_energy += np.trace(np.dot(subsys.dmat[1], supersystem.proj_pot[0][1]))
        initial_projector_energy += np.trace(np.dot(subsys2.dmat[0], supersystem.proj_pot[1][0]))
        initial_projector_energy += np.trace(np.dot(subsys2.dmat[1], supersystem.proj_pot[1][1]))

        supersystem.freeze_and_thaw()

        projector_energy = np.trace(np.dot(subsys.dmat[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.dmat[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.dmat[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.dmat[1], supersystem.proj_pot[1][1]))
        self.assertGreaterEqual(0.0, projector_energy-initial_projector_energy)

        # Unrestricted Open Shell
        ##Supermolecular test.
        #mol = gto.Mole()
        #mol.verbose = 3
        #mol.atom = '''
        #Li 0. 0. 0.
        #ghost:H 1.595 0.0 0.0
        #'''
        #mol.basis = 'cc-pVDZ'
        #mol.spin = 1
        #mol.build()
        #env_method = 'm06'
        #active_method = 'ccsd'
        #subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method, unrestricted=True, active_unrestricted=True)

        #mol2 = gto.Mole()
        #mol2.verbose = 3
        #mol2.atom = '''
        #H 1.595 0.0 0.0
        #ghost:Li 0. 0. 0.
        #'''
        #mol2.basis = 'cc-pVDZ'
        #mol2.spin = -1
        #mol2.build()
        #env_method = 'm06'
        #subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, unrestricted=True)
        #supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06')
        #supersystem.freeze_and_thaw()

        #projector_energy = np.trace(np.dot(subsys.dmat[0], supersystem.proj_pot[0][0]))
        #projector_energy += np.trace(np.dot(subsys.dmat[1], supersystem.proj_pot[0][1]))
        #projector_energy += np.trace(np.dot(subsys2.dmat[0], supersystem.proj_pot[1][0]))
        #projector_energy += np.trace(np.dot(subsys2.dmat[1], supersystem.proj_pot[1][1]))
        #self.assertAlmostEqual(0.0, projector_energy, delta=1e-15)

        #mol3 = gto.Mole()
        #mol3.verbose = 3
        #mol3.atom ='''
        #Li 0. 0. 0.
        #H 1.595 0.0 0.0
        #'''
        #mol3.basis = 'cc-pVDZ'
        #mol3.build()
        #mf = dft.RKS(mol3)
        #mf.xc = 'm06'
        #grids = dft.gen_grid.Grids(mol3)
        #grids.level = supersystem.grid_level
        #grids.build()
        #mf.grids = grids
        #mf.kernel()
        #test_dmat = mf.make_rdm1()
        #test_e = mf.energy_tot()
        #sup_e = supersystem.env_in_env_energy()
        #self.assertTrue(np.allclose(test_dmat, (supersystem.dmat[0] + supersystem.dmat[1]), atol=1e-6))
        #self.assertAlmostEqual(test_e, sup_e, delta=1e-10)

        ##Long distance test.
        #mol = gto.Mole()
        #mol.verbose = 3
        #mol.atom = '''
        #Li 0. 0. 0.
        #'''
        #mol.basis = '3-21g'
        #mol.spin = 1
        #mol.build()
        #nv_method = 'm06'
        #active_method = 'ccsd'
        #subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method, unrestricted=True, active_unrestricted=True)

        #mol2 = gto.Mole()
        #mol2.verbose = 3
        #mol2.atom = '''
        #H 12.22 0.0 0.0
        #'''
        #mol2.basis = '3-21g'
        #mol2.spin = -1
        #mol2.build()
        #env_method = 'm06'
        #subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, unrestricted=True)
        #supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao')
        #supersystem.freeze_and_thaw()

        #projector_energy = np.trace(np.dot(subsys.dmat[0], supersystem.proj_pot[0][0]))
        #projector_energy += np.trace(np.dot(subsys.dmat[1], supersystem.proj_pot[0][1]))
        #projector_energy += np.trace(np.dot(subsys2.dmat[0], supersystem.proj_pot[1][0]))
        #projector_energy += np.trace(np.dot(subsys2.dmat[1], supersystem.proj_pot[1][1]))
        #self.assertAlmostEqual(0.0, projector_energy, delta=1e-14)

        #mol3 = gto.Mole()
        #mol3.verbose = 3
        #mol3.atom ='''
        #Li 0. 0. 0.
        #H 12.22 0.0 0.0
        #'''
        #mol3.basis = '3-21g'
        #mol3.build()
        #mf = dft.UKS(mol3)
        #mf.xc = 'm06'
        #grids = dft.gen_grid.Grids(mol3)
        #grids.level = supersystem.grid_level
        #grids.build()
        #mf.grids = grids
        #mf.kernel()
        #test_dmat = mf.make_rdm1()
        #test_e = mf.energy_tot()
        #sup_e = supersystem.env_in_env_energy()
        ##self.assertTrue(np.allclose(test_dmat, (supersystem.dmat[0] + supersystem.dmat[1]), atol=1e-6))
        ##self.assertAlmostEqual(test_e, sup_e, delta=1e-10)

        ##Projection energy
        #mol = gto.Mole()
        #mol.verbose = 3
        #mol.atom = '''
        #Li 0. 0. 0.
        #'''
        #mol.basis = '3-21g'
        #mol.spin = 1
        #mol.build()
        #env_method = 'm06'
        #active_method = 'ccsd'
        #subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method, unrestricted=True, active_unrestricted=True)

        #mol2 = gto.Mole()
        #mol2.verbose = 3
        #mol2.atom = '''
        #H 1.595 0.0 0.0
        #'''
        #mol2.basis = '3-21g'
        #mol2.spin = -1
        #mol2.build()
        #env_method = 'm06'
        #subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, unrestricted=True)
        #supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='supmol')

        #initial_projector_energy = 0.0
        #initial_projector_energy = np.trace(np.dot(subsys.dmat[0], supersystem.proj_pot[0][0]))
        #initial_projector_energy += np.trace(np.dot(subsys.dmat[1], supersystem.proj_pot[0][1]))
        #initial_projector_energy += np.trace(np.dot(subsys2.dmat[0], supersystem.proj_pot[1][0]))
        #initial_projector_energy += np.trace(np.dot(subsys2.dmat[1], supersystem.proj_pot[1][1]))

        #supersystem.freeze_and_thaw()

        #projector_energy = np.trace(np.dot(subsys.dmat[0], supersystem.proj_pot[0][0]))
        #projector_energy += np.trace(np.dot(subsys.dmat[1], supersystem.proj_pot[0][1]))
        #projector_energy += np.trace(np.dot(subsys2.dmat[0], supersystem.proj_pot[1][0]))
        #projector_energy += np.trace(np.dot(subsys2.dmat[1], supersystem.proj_pot[1][1]))
        #self.assertGreaterEqual(0.0, projector_energy-initial_projector_energy)
