# A module to tests the methods of the Subsystem

import unittest
import os
import shutil
import re

from copy import copy

from qsome import cluster_subsystem, cluster_supersystem
from pyscf import gto, lib, scf, dft

import numpy as np

class TestEnvSubsystemMethods(unittest.TestCase):

    def test_get_env_elec_energy(self):

        # Closed Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'cc-pVDZ'
        mol.build()
        env_method = 'm06'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method)
        # Default test
        def_elec_e = subsys.get_env_elec_energy()
        sub_dmat = subsys.dmat[0] + subsys.dmat[1]
        test_scf = dft.RKS(mol)
        test_scf.xc = env_method
        test_elec_e = test_scf.energy_elec(dm=sub_dmat)
        
        self.assertAlmostEqual(test_elec_e[0], def_elec_e, delta=1e-10)

        # With just embedding potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_embed_e = np.einsum('ij,ji', (emb_potent[0] + emb_potent[1])/2.,
                                 (sub_dmat)).real
        
        def_elec_e_embed = subsys.get_env_elec_energy(emb_pot=emb_potent)
        def_emb_e = def_elec_e_embed - def_elec_e
        self.assertAlmostEqual(test_embed_e, def_emb_e, delta=1e-10)
        
        # With just projeciton potential
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', (proj_potent[0] + proj_potent[1])/2.,
                                 (sub_dmat)).real
        def_elec_e_proj = subsys.get_env_elec_energy(proj_pot=proj_potent)
        def_proj_e = def_elec_e_proj - def_elec_e
        self.assertAlmostEqual(test_proj_e, def_proj_e, delta=1e-10)
       
        # With both. 
        emb_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', (proj_potent[0] + proj_potent[1])/2.,
                                 (sub_dmat)).real
        test_embed_e = np.einsum('ij,ji', (emb_potent[0] + emb_potent[1])/2.,
                                 (sub_dmat)).real
        def_elec_e_tot = subsys.get_env_elec_energy(emb_pot=emb_potent, proj_pot=proj_potent)
        def_proj_e = def_elec_e_tot - def_elec_e
        self.assertAlmostEqual(test_proj_e + test_embed_e, def_proj_e, delta=1e-10)


        # Unrestricted Open Shell 
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        mol.basis = 'cc-pVDZ'
        mol.spin = 1
        mol.build()
        env_method = 'm06'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, unrestricted=True)
        # Default test
        def_elec_e = subsys.get_env_elec_energy()
        sub_dmat = [subsys.dmat[0], subsys.dmat[1]]
        test_scf = dft.UKS(mol)
        test_scf.xc = env_method
        test_elec_e = test_scf.energy_elec(dm=sub_dmat)
        self.assertAlmostEqual(test_elec_e[0], def_elec_e, delta=1e-10)

        # With just embedding potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_embed_e = np.einsum('ij,ji', emb_potent[0],
                                 sub_dmat[0]).real
        test_embed_e += np.einsum('ij,ji', emb_potent[1],
                                 sub_dmat[1]).real
        
        def_elec_e_embed = subsys.get_env_elec_energy(emb_pot=emb_potent)
        def_emb_e = def_elec_e_embed - def_elec_e
        self.assertAlmostEqual(test_embed_e, def_emb_e, delta=1e-10)
        
        # With just projeciton potential
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', proj_potent[0],
                                 sub_dmat[0]).real
        test_proj_e += np.einsum('ij,ji', proj_potent[1],
                                 sub_dmat[1]).real
        def_elec_e_proj = subsys.get_env_elec_energy(proj_pot=proj_potent)
        def_proj_e = def_elec_e_proj - def_elec_e
        self.assertAlmostEqual(test_proj_e, def_proj_e, delta=1e-10)
       
        # With both. 
        emb_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', proj_potent[0],
                                 sub_dmat[0]).real
        test_proj_e += np.einsum('ij,ji', proj_potent[1],
                                 sub_dmat[1]).real
        test_embed_e = np.einsum('ij,ji', emb_potent[0],
                                 sub_dmat[0]).real
        test_embed_e += np.einsum('ij,ji', emb_potent[1],
                                 sub_dmat[1]).real

        def_elec_e_tot = subsys.get_env_elec_energy(emb_pot=emb_potent, proj_pot=proj_potent)
        def_proj_e = def_elec_e_tot - def_elec_e
        self.assertAlmostEqual(test_proj_e + test_embed_e, def_proj_e, delta=1e-10)

    @unittest.skip
    def test_get_env_energy(self):
        # This is pretty simple, probably don't need to test.
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
        pass
        

    def test_update_proj_pot(self):
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
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        subsys.update_proj_pot(proj_potent)
        self.assertTrue(np.array_equal(proj_potent, subsys.proj_pot))

    def test_get_env_proj_e(self):

        # Closed Shell
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
        sub_dmat = subsys.dmat[0] + subsys.dmat[1]
        # With 0 potential.
        no_proj_e = subsys.get_env_proj_e()
        self.assertEqual(no_proj_e, 0.0)
        # With potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', (proj_potent[0] + proj_potent[1])/2.,
                                 (sub_dmat)).real
        subsys.update_proj_pot(proj_potent)
        proj_e = subsys.get_env_proj_e()
        self.assertEqual(test_proj_e, proj_e)

        # Unrestricted Open Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        mol.basis = 'aug-cc-pVDZ'
        mol.spin = 1
        mol.build()
        env_method = 'pbe'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, unrestricted=True)
        sub_dmat = [subsys.dmat[0], subsys.dmat[1]]
        # With 0 potential.
        no_proj_e = subsys.get_env_proj_e()
        self.assertEqual(no_proj_e, 0.0)
        # With potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', proj_potent[0],
                                 sub_dmat[0]).real
        test_proj_e += np.einsum('ij,ji', proj_potent[1],
                                 sub_dmat[1]).real
        subsys.update_proj_pot(proj_potent)
        proj_e = subsys.get_env_proj_e()
        self.assertEqual(test_proj_e, proj_e)

    def test_update_emb_pot(self):
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
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        subsys.update_emb_pot(emb_potent)
        self.assertTrue(np.array_equal(emb_potent, subsys.emb_pot))

    def test_update_fock(self):

        # Closed Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'lda'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method)
        sub_dmat = subsys.dmat[0] + subsys.dmat[1]
        test_scf = dft.RKS(mol)
        test_scf.xc = env_method
        #grids = dft.gen_grid.Grids(mol)
        #grids.level = subsys.grid_level
        #grids.build()
        #test_scf.grids = grids
        test_fock = test_scf.get_fock(dm=sub_dmat)
        test_hcore = test_scf.get_hcore()
        test_veff = test_scf.get_veff(dm=sub_dmat)
        sub_fock = subsys.env_scf.get_fock(dm=sub_dmat)
        sub_hcore = subsys.env_scf.get_hcore()
        sub_veff = subsys.env_scf.get_veff(dm=sub_dmat)
        self.assertTrue(np.allclose(test_hcore, sub_hcore))
        self.assertTrue(np.allclose(test_veff, sub_veff))
        self.assertTrue(np.allclose(test_fock, sub_fock))

        # Unrestricted Open Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        mol.basis = 'aug-cc-pVDZ'
        mol.spin = 1
        mol.build()
        env_method = 'lda'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, unrestricted=True)
        sub_dmat = [subsys.dmat[0],subsys.dmat[1]]
        test_scf = dft.UKS(mol)
        test_scf.xc = env_method
        #grids = dft.gen_grid.Grids(mol)
        #grids.level = subsys.grid_level
        #grids.build()
        #test_scf.grids = grids
        test_fock = test_scf.get_fock(dm=sub_dmat)
        test_hcore = test_scf.get_hcore()
        test_veff = test_scf.get_veff(dm=sub_dmat)
        sub_fock = subsys.env_scf.get_fock(dm=sub_dmat)
        sub_hcore = subsys.env_scf.get_hcore()
        sub_veff = subsys.env_scf.get_veff(dm=sub_dmat)
        self.assertTrue(np.allclose(test_hcore, sub_hcore))
        self.assertTrue(np.allclose(test_veff, sub_veff))
        self.assertTrue(np.allclose(test_fock, sub_fock))
        
    def test_update_density(self):
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
        dim0 = subsys.dmat[0].shape[0]
        dim1 = subsys.dmat[1].shape[1]
        new_dmat = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        subsys.update_density(new_dmat)
        self.assertTrue(np.array_equal(subsys.dmat, new_dmat))

    def test_save_orbitals(self):
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

    def test_diagonalize(self):
        # Closed Shell
        # Unsure how to test this with embedding potential or projection pot.
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
        subsys_fock = subsys.env_scf.get_fock(dm=(subsys.dmat[0] + subsys.dmat[1]))
        subsys.diagonalize()
        test_scf = dft.RKS(mol)
        test_scf.max_cycle = 1
        test_scf.xc = env_method
        test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(test_dmat, (subsys.dmat[0] + subsys.dmat[1])))

        # Unrestricted Open Shell
        # Unsure how to test this with embedding potential or projection pot.
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        mol.basis = 'aug-cc-pVDZ'
        mol.spin = 1
        mol.build()
        env_method = 'lda'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, unrestricted=True)
        subsys_fock = subsys.env_scf.get_fock(dm=(subsys.dmat))
        subsys.diagonalize()
        test_scf = dft.UKS(mol)
        test_scf.max_cycle = 1
        test_scf.xc = env_method
        test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(test_dmat[0], subsys.dmat[0]))
        self.assertTrue(np.allclose(test_dmat[1], subsys.dmat[1]))
         
class TestActiveSubsystemMethods(unittest.TestCase):

    def test_active_proj_energy(self):
        pass
        #mol = gto.Mole()
        #mol.verbose = 3
        #mol.atom = '''
        #O 0.0 0.0 0.0
        #H 0. -2.757 2.857
        #H 0. 2.757 2.857'''
        #mol.basis = 'aug-cc-pVDZ'
        #mol.build()
        #env_method = 'm06'
        #active_method  = 'mp2'
        #subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)
        #sub_dmat = subsys.dmat[0] + subsys.dmat[1]
        #dim0 = subsys.emb_pot[0].shape[0]
        #dim1 = subsys.emb_pot[1].shape[1]
        #proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        #test_proj_e = np.einsum('ij,ji', (proj_potent[0] + proj_potent[1])/2.,
        #                         (sub_dmat)).real
        #subsys.update_proj_pot(proj_potent)
        #proj_e = subsys.active_proj_energy()
        #self.assertEqual(test_proj_e, proj_e)

    def test_active_in_env_energy(self):
        # Closed Shell
        # Yet to test including embedding or projection potentials.
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'cc-pVDZ'
        mol.build()
        env_method = 'm06'
        active_method  = 'ccsd'
        conv_param = 1e-10
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method, active_conv=conv_param)
        subsys_fock = subsys.env_scf.get_fock(dm=(subsys.dmat[0] + subsys.dmat[1]))
        subsys_active_e = subsys.active_in_env_energy()
        test_scf = scf.RHF(mol)
        hf = test_scf.kernel()
        from pyscf import cc
        cc_calc = cc.CCSD(test_scf)
        cc_calc.conv_tol = 1e-10
        cc_calc.conv_tol_normt = 1e-6
        test_active_e = cc_calc.kernel()[0]
        self.assertAlmostEqual(subsys_active_e, hf + test_active_e, delta=1e-5)

        #Unrestricted Open Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        mol.basis = 'cc-pVDZ'
        mol.spin = 1
        mol.build()
        env_method = 'm06'
        active_method  = 'uccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method, active_unrestricted=True, unrestricted=True, active_conv=conv_param)
        subsys_fock = subsys.env_scf.get_fock(dm=(subsys.dmat[0] + subsys.dmat[1]))
        subsys_active_e = subsys.active_in_env_energy()
        test_scf = scf.UHF(mol)
        hf = test_scf.kernel()
        from pyscf import cc
        cc_calc = cc.UCCSD(test_scf)
        cc_calc.conv_tol = 1e-10
        test_active_e = cc_calc.kernel()[0]
        self.assertAlmostEqual(subsys_active_e, hf + test_active_e, delta=1e-5)

