# A module to tests the methods of the Subsystem

import unittest
import os
import shutil
import re

from copy import copy

from qsome import cluster_subsystem, cluster_supersystem
from pyscf import gto, lib, scf, dft, cc, mp, mcscf, tools

from pyscf.cc import ccsd_t, uccsd_t

import numpy as np

import tempfile

class TestEnvSubsystemMethods(unittest.TestCase):

    def setUp(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0.758602 0.00 0.504284
        H 0.758602 0.00 -0.504284'''
        mol.basis = '3-21g'
        mol.build()
        self.cs_mol = mol

        os_mol = gto.Mole()
        os_mol.verbose = 3
        os_mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        os_mol.basis = '3-21g'
        os_mol.spin = 1
        os_mol.build()
        self.os_mol = os_mol

        self.env_method = 'lda'

    #@unittest.skip
    def test_init_density(self):

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.init_density()
        init_dmat = scf.get_init_guess(self.cs_mol)
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method, initguess='atom')
        subsys.init_density()
        init_dmat = scf.get_init_guess(self.cs_mol, key='atom')
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method, initguess='1e')
        subsys.init_density()
        init_dmat = scf.get_init_guess(self.cs_mol, key='1e')
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method, initguess='minao')
        subsys.init_density()
        init_dmat = scf.get_init_guess(self.cs_mol, key='minao')
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method, initguess='supmol')
        subsys.init_density()
        init_dmat = scf.get_init_guess(self.cs_mol)
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method, initguess='submol')
        subsys.init_density()
        scf_obj = subsys.env_scf
        scf_obj.kernel()
        init_dmat = scf_obj.make_rdm1()
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        #Test Unrestricted Open Shell.
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
        subsys.init_density()
        init_dmat = scf.uhf.get_init_guess(self.os_mol)
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, initguess='atom', unrestricted=True)
        subsys.init_density()
        init_dmat = scf.uhf.get_init_guess(self.os_mol, key='atom')
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, initguess='1e', unrestricted=True)
        subsys.init_density()
        init_dmat = scf.uhf.get_init_guess(self.os_mol, key='1e')
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, initguess='minao', unrestricted=True)
        subsys.init_density()
        init_dmat = scf.uhf.get_init_guess(self.os_mol, key='minao')
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, initguess='supmol', unrestricted=True)
        subsys.init_density()
        init_dmat = scf.uhf.get_init_guess(self.os_mol)
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, initguess='submol', unrestricted=True)
        subsys.init_density()
        scf_obj = subsys.env_scf
        scf_obj.kernel()
        init_dmat = scf_obj.make_rdm1()
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        #Test Restricted Open Shell.
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method)
        subsys.init_density()
        init_dmat = scf.rhf.get_init_guess(self.os_mol)
        init_dmat = [init_dmat/2., init_dmat/2.]
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, initguess='atom')
        subsys.init_density()
        init_dmat = scf.rhf.get_init_guess(self.os_mol, key='atom')
        init_dmat = [init_dmat/2., init_dmat/2.]
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, initguess='1e')
        subsys.init_density()
        true_hf = scf.rohf.ROHF(self.os_mol)
        init_dmat = true_hf.get_init_guess(key='1e')
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, initguess='minao')
        subsys.init_density()
        init_dmat = scf.rhf.get_init_guess(self.os_mol, key='minao')
        init_dmat = [init_dmat/2., init_dmat/2.]
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, initguess='supmol')
        subsys.init_density()
        true_hf = scf.rohf.ROHF(self.os_mol)
        init_dmat = true_hf.get_init_guess(self.os_mol)
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, initguess='submol')
        subsys.init_density()
        scf_obj = subsys.env_scf
        scf_obj.kernel()
        init_dmat = scf_obj.make_rdm1()
        self.assertTrue(np.allclose(init_dmat, subsys.get_dmat()))

    #@unittest.skip
    def test_update_density(self):
        #Closed Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.init_density()
        dim0 = subsys.get_dmat().shape[0]
        dim1 = subsys.get_dmat().shape[1]
        new_dmat = np.array([np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)])
        subsys.env_dmat = new_dmat
        new_dmat = new_dmat[0] + new_dmat[1]
        self.assertTrue(np.array_equal(subsys.get_dmat(), new_dmat))

        #Unrestricted
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
        subsys.init_density()
        dim0 = subsys.get_dmat().shape[0]
        dim1 = subsys.get_dmat().shape[1]
        new_dmat = np.array([np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)])
        subsys.env_dmat = new_dmat
        self.assertTrue(np.array_equal(subsys.get_dmat(), new_dmat))

        #Restricted open shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method)
        subsys.init_density()
        dim0 = subsys.get_dmat().shape[0]
        dim1 = subsys.get_dmat().shape[1]
        new_dmat = np.array([np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)])
        subsys.env_dmat = new_dmat
        self.assertTrue(np.array_equal(subsys.get_dmat(), new_dmat))

    #@unittest.skip
    def test_update_subsys_fock(self):
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.init_density()
        subsys.update_subsys_fock()
        subsys_dmat = subsys.get_dmat()
        sub_scf = dft.RKS(self.cs_mol)
        sub_scf.xc = self.env_method
        true_fock = sub_scf.get_fock(dm=subsys_dmat)
        true_fock = [true_fock, true_fock]
        self.assertTrue(np.allclose(subsys.subsys_fock, true_fock))

        #Unrestricted
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
        subsys.init_density()
        subsys.update_subsys_fock()
        subsys_dmat = subsys.get_dmat()
        sub_scf = dft.UKS(self.os_mol)
        sub_scf.xc = self.env_method
        true_fock = sub_scf.get_fock(dm=subsys_dmat)
        true_fock = [true_fock, true_fock]
        self.assertTrue(np.allclose(subsys.subsys_fock, true_fock))

        #Restricted Open Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method)
        subsys.init_density()
        subsys.update_subsys_fock()
        subsys_dmat = subsys.get_dmat()
        sub_scf = dft.ROKS(self.os_mol)
        sub_scf.xc = self.env_method
        true_fock = sub_scf.get_fock(dm=subsys_dmat)
        true_fock = [true_fock, true_fock]
        self.assertTrue(np.allclose(subsys.subsys_fock, true_fock))

    #@unittest.skip
    def test_update_emb_pot(self):
        #Closed Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.init_density()
        subsys.update_subsys_fock()
        subsys.emb_fock[0] = subsys.subsys_fock[0]
        subsys.emb_fock[1] = subsys.subsys_fock[1]
        dim0 = subsys.emb_fock[0].shape[0]
        dim1 = subsys.emb_fock[1].shape[1]
        emb_fock = np.array([np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)])
        subsys.emb_fock = emb_fock
        subsys.update_emb_pot()
        true_emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                        emb_fock[1] - subsys.subsys_fock[1]]
        self.assertTrue(np.array_equal(true_emb_pot, subsys.emb_pot))

        #Unrestricted
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
        subsys.init_density()
        subsys.update_subsys_fock()
        subsys.emb_fock[0] = subsys.subsys_fock[0]
        subsys.emb_fock[1] = subsys.subsys_fock[1]
        dim0 = subsys.emb_fock[0].shape[0]
        dim1 = subsys.emb_fock[1].shape[1]
        emb_fock = np.array([np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)])
        subsys.emb_fock = emb_fock
        subsys.update_emb_pot()
        true_emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                        emb_fock[1] - subsys.subsys_fock[1]]
        self.assertTrue(np.array_equal(true_emb_pot, subsys.emb_pot))

        #Restricted Open Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method)
        subsys.init_density()
        subsys.update_subsys_fock()
        subsys.emb_fock[0] = subsys.subsys_fock[0]
        subsys.emb_fock[1] = subsys.subsys_fock[1]
        dim0 = subsys.emb_fock[0].shape[0]
        dim1 = subsys.emb_fock[1].shape[1]
        emb_fock = np.array([np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)])
        subsys.emb_fock = emb_fock
        subsys.update_emb_pot()
        true_emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                        emb_fock[1] - subsys.subsys_fock[1]]
        self.assertTrue(np.array_equal(true_emb_pot, subsys.emb_pot))


    #@unittest.skip
    def test_env_proj_e(self):

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.init_density()
        sub_dmat = subsys.get_dmat()
        # With 0 potential.
        no_proj_e = subsys.get_env_proj_e()
        self.assertEqual(no_proj_e, 0.0)
        # With potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', proj_potent[0],
                                 (sub_dmat/2.)).real
        test_proj_e += np.einsum('ij,ji', proj_potent[1],
                                 (sub_dmat/2.)).real
        subsys.proj_pot = proj_potent
        proj_e = subsys.get_env_proj_e()
        self.assertEqual(test_proj_e, proj_e)

        # Unrestricted Open Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
        subsys.init_density()
        sub_dmat = subsys.get_dmat()
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
        subsys.proj_pot = proj_potent
        proj_e = subsys.get_env_proj_e()
        self.assertEqual(test_proj_e, proj_e)

        # Restricted Open Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method)
        subsys.init_density()
        sub_dmat = subsys.get_dmat()
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
        subsys.proj_pot = proj_potent
        proj_e = subsys.get_env_proj_e()
        self.assertEqual(test_proj_e, proj_e)

    #@unittest.skip
    def test_env_embed_e(self):

        # Closed Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.init_density()
        sub_dmat = subsys.get_dmat()
        # With 0 potential.
        no_embed_e = subsys.get_env_emb_e()
        self.assertEqual(no_embed_e, 0.0)
        # With potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                   emb_fock[1] - subsys.subsys_fock[1]]
        true_emb_e = np.einsum('ij,ji', emb_pot[0],
                                 (sub_dmat/2.)).real
        true_emb_e += np.einsum('ij,ji', emb_pot[1],
                                 (sub_dmat/2.)).real
        subsys.emb_fock = emb_fock
        emb_e = subsys.get_env_emb_e()
        self.assertEqual(true_emb_e, emb_e)

        # Unrestricted Open Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
        subsys.init_density()
        sub_dmat = subsys.get_dmat()
        # With 0 potential.
        no_embed_e = subsys.get_env_emb_e()
        self.assertEqual(no_embed_e, 0.0)
        # With potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = emb_fock - subsys.subsys_fock
        true_emb_e = np.einsum('ij,ji', emb_pot[0],
                                 sub_dmat[0]).real
        true_emb_e += np.einsum('ij,ji', emb_pot[1],
                                 sub_dmat[1]).real
        subsys.emb_fock = emb_fock
        embed_e = subsys.get_env_emb_e()
        self.assertEqual(true_emb_e, embed_e)

        # Restricted Open Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method)
        subsys.init_density()
        sub_dmat = subsys.get_dmat()
        # With 0 potential.
        no_embed_e = subsys.get_env_emb_e()
        self.assertEqual(no_embed_e, 0.0)
        # With potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                   emb_fock[1] - subsys.subsys_fock[1]]
        true_emb_e = np.einsum('ij,ji', emb_pot[0],
                                 sub_dmat[0]).real
        true_emb_e += np.einsum('ij,ji', emb_pot[1],
                                 sub_dmat[1]).real
        subsys.emb_fock = emb_fock
        embed_e = subsys.get_env_emb_e()
        self.assertEqual(true_emb_e, embed_e)

    #@unittest.skip
    def test_get_env_elec_energy(self):

        # Closed Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.init_density()
        # Default test
        def_elec_e = subsys.get_env_elec_energy()
        sub_dmat = subsys.get_dmat()
        test_scf = dft.RKS(self.cs_mol)
        test_scf.xc = self.env_method
        test_elec_e = test_scf.energy_elec(dm=sub_dmat)
        
        self.assertAlmostEqual(test_elec_e[0], def_elec_e, delta=1e-10)

        # With just embedding potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                   emb_fock[1] - subsys.subsys_fock[1]]
        test_embed_e = np.einsum('ij,ji', (emb_pot[0] + emb_pot[1])/2.,
                                 (sub_dmat)).real
        
        def_elec_e_embed = subsys.get_env_elec_energy(emb_pot=emb_pot)
        def_emb_e = def_elec_e_embed - def_elec_e
        self.assertAlmostEqual(test_embed_e, def_emb_e, delta=1e-10)
        
        # With just projection potential
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', (proj_potent[0] + proj_potent[1])/2.,
                                 (sub_dmat)).real
        def_elec_e_proj = subsys.get_env_elec_energy(proj_pot=proj_potent)
        def_proj_e = def_elec_e_proj - def_elec_e
        self.assertAlmostEqual(test_proj_e, def_proj_e, delta=1e-10)
       
        # With both. 
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                   emb_fock[1] - subsys.subsys_fock[1]]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', (proj_potent[0] + proj_potent[1])/2.,
                                 (sub_dmat)).real
        test_embed_e = np.einsum('ij,ji', (emb_pot[0] + emb_pot[1])/2.,
                                 (sub_dmat)).real
        def_elec_e_tot = subsys.get_env_elec_energy(emb_pot=emb_pot, proj_pot=proj_potent)
        def_proj_e = def_elec_e_tot - def_elec_e
        self.assertAlmostEqual(test_proj_e + test_embed_e, def_proj_e, delta=1e-10)


        # Unrestricted Open Shell 
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
        subsys.init_density()
        # Default test
        def_elec_e = subsys.get_env_elec_energy()
        sub_dmat = subsys.env_dmat
        test_scf = dft.UKS(self.os_mol)
        test_scf.xc = self.env_method
        test_elec_e = test_scf.energy_elec(dm=sub_dmat)
        self.assertAlmostEqual(test_elec_e[0], def_elec_e, delta=1e-10)

        # With just embedding potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = emb_fock - subsys.subsys_fock
        test_embed_e = np.einsum('ij,ji', emb_pot[0],
                                 sub_dmat[0]).real
        test_embed_e += np.einsum('ij,ji', emb_pot[1],
                                 sub_dmat[1]).real
        
        def_elec_e_embed = subsys.get_env_elec_energy(emb_pot=emb_pot)
        def_emb_e = def_elec_e_embed - def_elec_e
        self.assertAlmostEqual(test_embed_e, def_emb_e, delta=1e-10)
        
        # With just projection potential
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', proj_potent[0],
                                 sub_dmat[0]).real
        test_proj_e += np.einsum('ij,ji', proj_potent[1],
                                 sub_dmat[1]).real
        def_elec_e_proj = subsys.get_env_elec_energy(proj_pot=proj_potent)
        def_proj_e = def_elec_e_proj - def_elec_e
        self.assertAlmostEqual(test_proj_e, def_proj_e, delta=1e-10)
       
        # With both. 
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = emb_fock - subsys.subsys_fock
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', proj_potent[0],
                                 sub_dmat[0]).real
        test_proj_e += np.einsum('ij,ji', proj_potent[1],
                                 sub_dmat[1]).real
        test_embed_e = np.einsum('ij,ji', emb_pot[0],
                                 sub_dmat[0]).real
        test_embed_e += np.einsum('ij,ji', emb_pot[1],
                                 sub_dmat[1]).real

        def_elec_e_tot = subsys.get_env_elec_energy(emb_pot=emb_pot, proj_pot=proj_potent)
        def_proj_emb_e = def_elec_e_tot - def_elec_e
        self.assertAlmostEqual(test_proj_e + test_embed_e, def_proj_emb_e, delta=1e-10)

        # Restricted Open Shell 
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method)
        subsys.init_density()
        # Default test
        def_elec_e = subsys.get_env_elec_energy()
        sub_dmat = subsys.env_dmat
        test_scf = dft.ROKS(self.os_mol)
        test_scf.xc = self.env_method
        test_elec_e = test_scf.energy_elec(dm=sub_dmat)
        self.assertAlmostEqual(test_elec_e[0], def_elec_e, delta=1e-10)

        # With just embedding potential
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                   emb_fock[1] - subsys.subsys_fock[1]]
        test_embed_e = np.einsum('ij,ji', emb_pot[0],
                                 sub_dmat[0]).real
        test_embed_e += np.einsum('ij,ji', emb_pot[1],
                                 sub_dmat[1]).real
        
        def_elec_e_embed = subsys.get_env_elec_energy(emb_pot=emb_pot)
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
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                   emb_fock[1] - subsys.subsys_fock[1]]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        test_proj_e = np.einsum('ij,ji', proj_potent[0],
                                 sub_dmat[0]).real
        test_proj_e += np.einsum('ij,ji', proj_potent[1],
                                 sub_dmat[1]).real
        test_embed_e = np.einsum('ij,ji', emb_pot[0],
                                 sub_dmat[0]).real
        test_embed_e += np.einsum('ij,ji', emb_pot[1],
                                 sub_dmat[1]).real

        def_elec_e_tot = subsys.get_env_elec_energy(emb_pot=emb_pot, proj_pot=proj_potent)
        def_proj_emb_e = def_elec_e_tot - def_elec_e
        self.assertAlmostEqual(test_proj_e + test_embed_e, def_proj_emb_e, delta=1e-10)

    #@unittest.skip
    def test_get_env_energy(self):

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.init_density()
        sub_dmat = subsys.get_dmat()
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                   emb_fock[1] - subsys.subsys_fock[1]]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        true_proj_e = np.einsum('ij,ji', (proj_potent[0] + proj_potent[1])/2.,
                                 (sub_dmat)).real
        true_embed_e = np.einsum('ij,ji', (emb_pot[0] + emb_pot[1])/2.,
                                 (sub_dmat)).real
        true_scf = dft.RKS(self.cs_mol)
        true_scf.xc = self.env_method
        true_subsys_e = true_scf.energy_tot(dm=sub_dmat)
        subsys_e_tot = subsys.get_env_energy(emb_pot=emb_pot, proj_pot=proj_potent)
        true_e_tot = true_subsys_e + true_proj_e + true_embed_e
        self.assertAlmostEqual(true_e_tot, subsys_e_tot, delta=1e-10)

        #Unrestricted
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
        subsys.init_density()
        sub_dmat = subsys.get_dmat()
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                   emb_fock[1] - subsys.subsys_fock[1]]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        true_proj_e = np.einsum('ij,ji', proj_potent[0],
                                 sub_dmat[0].real)
        true_proj_e += np.einsum('ij,ji', proj_potent[1],
                                 sub_dmat[1].real)
        true_embed_e = np.einsum('ij,ji', emb_pot[0],
                                 (sub_dmat[0]).real)
        true_embed_e += np.einsum('ij,ji', emb_pot[1],
                                 (sub_dmat[1]).real)
        true_scf = dft.UKS(self.os_mol)
        true_scf.xc = self.env_method
        true_subsys_e = true_scf.energy_tot(dm=sub_dmat)
        subsys_e_tot = subsys.get_env_energy(emb_pot=emb_pot, proj_pot=proj_potent)
        true_e_tot = true_subsys_e + true_proj_e + true_embed_e
        self.assertAlmostEqual(true_e_tot, subsys_e_tot, delta=1e-10)

        #Restricted Open Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method)
        subsys.init_density()
        sub_dmat = subsys.get_dmat()
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                   emb_fock[1] - subsys.subsys_fock[1]]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        true_proj_e = np.einsum('ij,ji', proj_potent[0],
                                 sub_dmat[0].real)
        true_proj_e += np.einsum('ij,ji', proj_potent[1],
                                 sub_dmat[1].real)
        true_embed_e = np.einsum('ij,ji', emb_pot[0],
                                 (sub_dmat[0]).real)
        true_embed_e += np.einsum('ij,ji', emb_pot[1],
                                 (sub_dmat[1]).real)
        true_scf = dft.ROKS(self.os_mol)
        true_scf.xc = self.env_method
        true_subsys_e = true_scf.energy_tot(dm=sub_dmat)
        subsys_e_tot = subsys.get_env_energy(emb_pot=emb_pot, proj_pot=proj_potent)
        true_e_tot = true_subsys_e + true_proj_e + true_embed_e
        self.assertAlmostEqual(true_e_tot, subsys_e_tot, delta=1e-10)


    #@unittest.skip
    def test_save_orbs(self):
        import tempfile
        from pyscf.tools import molden
        t_file = tempfile.NamedTemporaryFile()
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method, filename=t_file.name)
        subsys.init_density()
        subsys.chkfile_index = '0'
        subsys.diagonalize()
        sub_mo_coeff = subsys.env_mo_coeff
        sub_mo_energy = subsys.env_mo_energy
        sub_mo_occ = subsys.env_mo_occ
        chkfile_index = subsys.chkfile_index
        subsys.save_orbital_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        molden.from_mo(self.cs_mol, true_ftmp.name, sub_mo_coeff[0], ene=sub_mo_energy[0], occ=(sub_mo_occ[0] * 2.))

        with open(t_file.name + '_' + chkfile_index + '_subenv.molden', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data, true_den_data)

        #Unrestricted open shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True, filename=t_file.name)
        subsys.init_density()
        subsys.chkfile_index = '0'
        subsys.diagonalize()
        sub_mo_coeff = subsys.env_mo_coeff
        sub_mo_energy = subsys.env_mo_energy
        sub_mo_occ = subsys.env_mo_occ
        chkfile_index = subsys.chkfile_index
        subsys.save_orbital_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        molden.from_mo(self.os_mol, true_ftmp.name, sub_mo_coeff[0], spin='Alpha', ene=sub_mo_energy[0], occ=sub_mo_occ[0])

        with open(t_file.name + '_' + chkfile_index + '_subenv_alpha.molden', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()
        self.assertEqual(test_den_data, true_den_data)

        true_ftmp = tempfile.NamedTemporaryFile()
        molden.from_mo(self.os_mol, true_ftmp.name, sub_mo_coeff[1], spin='Beta', ene=sub_mo_energy[1], occ=sub_mo_occ[1])

        with open(t_file.name + '_' + chkfile_index + '_subenv_beta.molden', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()
        self.assertEqual(test_den_data, true_den_data)

        #Restricted Open Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, filename=t_file.name)
        subsys.init_density()
        subsys.chkfile_index = '0'
        subsys.diagonalize()
        sub_mo_coeff = subsys.env_mo_coeff
        sub_mo_energy = subsys.env_mo_energy
        sub_mo_occ = subsys.env_mo_occ
        chkfile_index = subsys.chkfile_index
        subsys.save_orbital_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        molden.from_mo(self.os_mol, true_ftmp.name, sub_mo_coeff[0], ene=sub_mo_energy[0], occ=(sub_mo_occ[0] + sub_mo_occ[1]))

        with open(t_file.name + '_' + chkfile_index + '_subenv.molden', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data, true_den_data)

    #@unittest.skip
    def test_save_density(self):
        import tempfile
        from pyscf.tools import cubegen
        t_file = tempfile.NamedTemporaryFile()
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method, filename=t_file.name)
        subsys.init_density()
        subsys.chkfile_index = '0'
        subsys.diagonalize()
        subsys.save_density_file()
        sub_dmat = subsys.get_dmat()
        true_ftmp = tempfile.NamedTemporaryFile()
        cubegen.density(self.cs_mol, true_ftmp.name, sub_dmat)

        with open(t_file.name + '_' + subsys.chkfile_index + '_subenv.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        #Unrestricted open shell
        t_file = tempfile.NamedTemporaryFile()
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, filename=t_file.name, unrestricted=True)
        subsys.init_density()
        subsys.chkfile_index = '0'
        subsys.diagonalize()
        subsys.save_density_file()
        sub_dmat = subsys.get_dmat()
        true_ftmp = tempfile.NamedTemporaryFile()
        cubegen.density(self.os_mol, true_ftmp.name, sub_dmat[0])

        with open(t_file.name + '_' + subsys.chkfile_index + '_subenv_alpha.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        true_ftmp = tempfile.NamedTemporaryFile()
        cubegen.density(self.os_mol, true_ftmp.name, sub_dmat[1])

        with open(t_file.name + '_' + subsys.chkfile_index + '_subenv_beta.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        #Restricted open shell
        t_file = tempfile.NamedTemporaryFile()
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, filename=t_file.name)
        subsys.init_density()
        subsys.chkfile_index = '0'
        subsys.diagonalize()
        subsys.save_density_file()
        sub_dmat = subsys.get_dmat()
        true_ftmp = tempfile.NamedTemporaryFile()
        cubegen.density(self.os_mol, true_ftmp.name, sub_dmat[0])

        with open(t_file.name + '_' + subsys.chkfile_index + '_subenv_alpha.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        true_ftmp = tempfile.NamedTemporaryFile()
        cubegen.density(self.os_mol, true_ftmp.name, sub_dmat[1])

        with open(t_file.name + '_' + subsys.chkfile_index + '_subenv_beta.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()


    #@unittest.skip
    def test_save_spin_density(self):
        import tempfile
        from pyscf.tools import cubegen

        #Unrestricted Open Shell
        t_file = tempfile.NamedTemporaryFile()
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, filename=t_file.name, unrestricted=True)
        subsys.init_density()
        subsys.chkfile_index = '0'
        subsys.diagonalize()
        subsys.save_spin_density_file()
        sub_dmat = subsys.get_dmat()
        true_ftmp = tempfile.NamedTemporaryFile()
        cubegen.density(self.os_mol, true_ftmp.name, np.subtract(sub_dmat[0],sub_dmat[1]))

        with open(t_file.name + '_' + subsys.chkfile_index + '_subenv_spinden.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        #Restricted Open Shell
        t_file = tempfile.NamedTemporaryFile()
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, filename=t_file.name)
        subsys.init_density()
        subsys.chkfile_index = '0'
        subsys.diagonalize()
        subsys.save_spin_density_file()
        sub_dmat = subsys.get_dmat()
        true_ftmp = tempfile.NamedTemporaryFile()
        cubegen.density(self.os_mol, true_ftmp.name, np.subtract(sub_dmat[0],sub_dmat[1]))

        with open(t_file.name + '_' + subsys.chkfile_index + '_subenv_spinden.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])


    #@unittest.skip
    def test_save_read_chkfile(self):
        import h5py
        t_file = tempfile.NamedTemporaryFile()
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method, filename=t_file.name)
        subsys.chkfile_index = '0'
        subsys.init_density()
        subsys.diagonalize()
        subsys.save_chkfile()

        with h5py.File(t_file.name + '.hdf5', 'r') as hf:
            subsys_coeff = hf[f'subsystem:0/mo_coeff']
            sub_env_mo_coeff = subsys_coeff[:]
            subsys_occ = hf[f'subsystem:0/mo_occ']
            sub_env_mo_occ = subsys_occ[:]
            subsys_energy = hf[f'subsystem:0/mo_energy']
            sub_env_mo_energy = subsys_energy[:]

        self.assertTrue(np.array_equal(subsys.env_mo_coeff, sub_env_mo_coeff))
        self.assertTrue(np.array_equal(subsys.env_mo_occ, sub_env_mo_occ))
        self.assertTrue(np.array_equal(subsys.env_mo_energy, sub_env_mo_energy))

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method, filename=t_file.name, initguess='chk')
        subsys2.chkfile_index = '0'
        subsys2.init_density()
        self.assertTrue(np.array_equal(subsys.env_mo_coeff, subsys2.env_mo_coeff))
        self.assertTrue(np.array_equal(subsys.env_mo_occ, subsys2.env_mo_occ))
        self.assertTrue(np.array_equal(subsys.env_mo_energy, subsys2.env_mo_energy))

    #@unittest.skip
    def test_diagonalize(self):
        # Closed Shell
        # Unsure how to test this with embedding potential or projection pot.
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.chkfile_index = '0'
        subsys.init_density()
        subsys.diagonalize()
        test_scf = dft.RKS(self.cs_mol)
        test_scf.max_cycle = 1
        test_scf.xc = self.env_method
        test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(test_dmat, subsys.get_dmat()))

        # Unrestricted Open Shell
        # Unsure how to test this with embedding potential or projection pot.
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
        subsys.chkfile_index = '0'
        subsys.init_density()
        subsys.diagonalize()
        test_scf = dft.UKS(self.os_mol)
        test_scf.max_cycle = 1
        test_scf.xc = self.env_method
        test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(test_dmat[0], subsys.env_dmat[0]))
        self.assertTrue(np.allclose(test_dmat[1], subsys.env_dmat[1]))

        # Restricted Open Shell
        # Unsure how to test this with embedding potential or projection pot.
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method)
        subsys.chkfile_index = '0'
        subsys.init_density()
        subsys.diagonalize()
        test_scf = dft.ROKS(self.os_mol)
        test_scf.max_cycle = 1
        test_scf.xc = self.env_method
        test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(test_dmat[0], subsys.env_dmat[0]))
        self.assertTrue(np.allclose(test_dmat[1], subsys.env_dmat[1]))
    
class TestHLSubsystemMethods(unittest.TestCase):

    def setUp(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  0.758602  0.000000  -0.504284'''
        mol.basis = '3-21g'
        mol.build()
        self.cs_mol = mol
        self.env_method = 'lda'

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        Li 0.0 0.0 0.0
        '''
        mol2.basis = '3-21g'
        mol2.spin = 1
        mol2.build()
        self.os_mol = mol2

    #@unittest.skip
    def test_hl_init_guess(self):
        hl_method  = 'hf'
        conv_param = 1e-10
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="1e")
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.RHF(self.cs_mol)
        correct_dmat = test_scf.get_init_guess(key="1e")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="minao")
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.RHF(self.cs_mol)
        correct_dmat = test_scf.get_init_guess(key="minao")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="atom")
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.RHF(self.cs_mol)
        correct_dmat = test_scf.get_init_guess(key="atom")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        #Use the embedded density as the hl guess.
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="ft")
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.RHF(self.cs_mol)
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=subsys.get_dmat())
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        #Unrestricted Open Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="1e", hl_unrestricted=True)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.UHF(self.os_mol)
        correct_dmat = test_scf.get_init_guess(key="1e")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="minao", hl_unrestricted=True)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.UHF(self.os_mol)
        correct_dmat = test_scf.get_init_guess(key="minao")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="atom", hl_unrestricted=True)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.UHF(self.os_mol)
        correct_dmat = test_scf.get_init_guess(key="atom")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        #Use the embedded density as the hl guess.
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="ft", hl_unrestricted=True)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.UHF(self.os_mol)
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=subsys.get_dmat())
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        #Restricted Open Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="1e")
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.ROHF(self.os_mol)
        correct_dmat = test_scf.get_init_guess(key="1e")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="minao")
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.ROHF(self.os_mol)
        correct_dmat = test_scf.get_init_guess(key="minao")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="atom")
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.ROHF(self.os_mol)
        correct_dmat = test_scf.get_init_guess(key="atom")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))

        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="ft")
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        test_scf = scf.ROHF(self.os_mol)
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=subsys.get_dmat())
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_sr_scf.make_rdm1()))


    #@unittest.skip
    def test_hf_in_env_energy(self):

        # Closed shell
        hl_method = 'hf'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.RHF(self.cs_mol)
        true_e = true_scf.kernel()
        self.assertAlmostEqual(subsys_hl_e, true_e, delta=1e-10)

        # Unrestricted Open shell
        hl_method = 'hf'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_unrestricted=True)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.UHF(self.os_mol)
        true_e = true_scf.kernel()
        self.assertAlmostEqual(subsys_hl_e, true_e, delta=1e-10)

        # Restricted Open shell
        hl_method = 'hf'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.ROHF(self.os_mol)
        true_e = true_scf.kernel()
        self.assertAlmostEqual(subsys_hl_e, true_e, delta=1e-10)


    def test_dft_in_env_energy(self):

        # Closed shell
        hl_method = 'm06'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.RKS(self.cs_mol)
        true_scf.xc = 'm06'
        true_e = true_scf.kernel()
        self.assertAlmostEqual(subsys_hl_e, true_e, delta=1e-10)

        # Unrestricted Open shell
        hl_method = 'm06'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_unrestricted=True)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.UKS(self.os_mol)
        true_scf.xc = 'm06'
        true_e = true_scf.kernel()
        self.assertAlmostEqual(subsys_hl_e, true_e, delta=1e-10)

        # Restricted Open shell
        hl_method = 'm06'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.ROKS(self.os_mol)
        true_scf.xc = 'm06'
        true_e = true_scf.kernel()
        self.assertAlmostEqual(subsys_hl_e, true_e, delta=1e-10)

    def test_ccsd_in_env_energy(self):

        # Closed shell
        hl_method = 'ccsd'
        hl_dict = {'froz_core_orbs': 1}
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method, hl_dict=hl_dict)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.RHF(self.cs_mol)
        true_hf_e = true_scf.kernel()
        true_cc = cc.CCSD(true_scf)
        true_cc.frozen = 1
        true_cc_e = true_cc.kernel()[0]
        self.assertAlmostEqual(subsys_hl_e, true_hf_e + true_cc_e, delta=1e-10)

        # Unrestricted Open shell
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_unrestricted=True)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.UHF(self.os_mol)
        true_hf_e = true_scf.kernel()
        true_cc = cc.UCCSD(true_scf)
        true_cc_e = true_cc.kernel()[0]
        self.assertAlmostEqual(subsys_hl_e, true_hf_e + true_cc_e, delta=1e-10)

        # Restricted Open shell
        #hl_method = 'ccsd'
        #subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method)
        #subsys.init_density()
        #subsys_hl_e = subsys.get_hl_in_env_energy()
        #true_scf = scf.ROHF(self.os_mol)
        #true_hf_e = true_scf.kernel()
        #true_cc = cc.UCCSD(true_scf)
        #true_cc_e = true_cc.kernel()[0]
        #self.assertAlmostEqual(subsys_hl_e, true_hf_e + true_cc_e, delta=1e-10)

    def test_ccsdt_in_env_energy(self):

        # Closed shell
        hl_method = 'ccsd(t)'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.RHF(self.cs_mol)
        true_hf_e = true_scf.kernel()
        true_cc = cc.CCSD(true_scf)
        true_cc_e = true_cc.kernel()[0]
        true_t_e = ccsd_t.kernel(true_cc, true_cc.ao2mo())
        self.assertAlmostEqual(subsys_hl_e, true_hf_e + true_cc_e + true_t_e, delta=1e-10)

        # Open shell
        hl_method = 'ccsd(t)'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_unrestricted=True)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.UHF(self.os_mol)
        true_hf_e = true_scf.kernel()
        true_cc = cc.UCCSD(true_scf)
        true_cc_e = true_cc.kernel()[0]
        true_t_e = uccsd_t.kernel(true_cc, true_cc.ao2mo())
        self.assertAlmostEqual(subsys_hl_e, true_hf_e + true_cc_e + true_t_e, delta=1e-10)

    def test_mp_in_env_energy(self):

        # Closed shell
        hl_method = 'mp2'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.RHF(self.cs_mol)
        true_hf_e = true_scf.kernel()
        true_mp = mp.MP2(true_scf)
        true_mp_e = true_mp.kernel()[0]
        self.assertAlmostEqual(subsys_hl_e, true_hf_e + true_mp_e, delta=1e-10)

        # Open shell
        hl_method = 'mp2'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_unrestricted=True)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.UHF(self.os_mol)
        true_hf_e = true_scf.kernel()
        true_mp = mp.UMP2(true_scf)
        true_mp_e = true_mp.kernel()[0]
        self.assertAlmostEqual(subsys_hl_e, true_hf_e + true_mp_e, delta=1e-10)

    def test_casscf_in_env_energy(self):

        # Closed shell
        hl_method = 'cas[2,2]'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method)
        subsys.init_density()
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.RHF(self.cs_mol)
        true_hf_e = true_scf.kernel()
        true_casscf = mcscf.CASSCF(true_scf, 2, 2)
        true_casscf_e = true_casscf.kernel()[0]
        self.assertAlmostEqual(subsys_hl_e, true_casscf_e, delta=1e-8)

    def test_ci_in_env_energy(self):
        pass

    def test_dmrg_in_env_energy(self):
        pass

    def test_gw_in_env_energy(self):
        pass

    def test_hci_in_env_energy(self):
        pass

    def test_icmpspt_in_env_energy(self):
        pass

    def test_mrpt_in_env_energy(self):
        pass

    def test_shciscf_in_env_energy(self):
        pass

    def test_fci_in_env_energy(self):
        pass

    def test_fciqmc_in_env_energy(self):
        pass

    #@unittest.skip
    def test_fcidump_in_env_energy(self):

        #there is not a great way to test this. Pretty sure it's working. 
        # Closed shell
        t_file = tempfile.NamedTemporaryFile()
        hl_method = 'fcidump'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method, filename=t_file.name)
        subsys.init_density()
        subsys.chkfile_index = '0'
        subsys_hl_e = subsys.get_hl_in_env_energy()
        true_scf = scf.RHF(self.cs_mol)
        true_hf_e = true_scf.kernel()
        t_file = tempfile.NamedTemporaryFile()
        fcidump_filename = t_file.name
        tools.fcidump.from_scf(true_scf, fcidump_filename, tol=1e-200)

        with open(subsys.filename + '.fcidump', 'r') as fin:
            test_fcidump = fin.read()[:100].splitlines()
            test_fcidump += fin.read()[200:300].splitlines()
            test_fcidump += fin.read()[300:400].splitlines()
            test_fcidump += fin.read()[1000:1200].splitlines()
            test_fcidump += fin.read()[3000:3200].splitlines()

        with open(fcidump_filename, 'r') as fin:
            true_fcidump = fin.read()[:100].splitlines()
            true_fcidump += fin.read()[200:300].splitlines()
            true_fcidump += fin.read()[300:400].splitlines()
            true_fcidump += fin.read()[1000:1200].splitlines()
            true_fcidump += fin.read()[3000:3200].splitlines()

        self.assertEqual(test_fcidump[:4], true_fcidump[:4])
        for i in range(4, len(test_fcidump)):
            print (i)
            print (test_fcidump[i])
            print (true_fcidump[i])
            test_fci_val = float(test_fcidump[i].split()[0])
            true_fci_val = float(true_fcidump[i].split()[0])
            self.assertAlmostEqual(test_fci_val, true_fci_val)
