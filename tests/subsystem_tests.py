# A module to tests the methods of the Subsystem

import unittest
import os
import shutil
import re

from copy import copy

from qsome import cluster_subsystem, cluster_supersystem
from pyscf import gto, lib, scf, dft, cc

import numpy as np

class TestEnvSubsystemMethods(unittest.TestCase):

    def setUp(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
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
    def test_update_density(self):
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        dim0 = subsys.get_dmat().shape[0]
        dim1 = subsys.get_dmat().shape[1]
        new_dmat = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        subsys.update_density(new_dmat)
        new_dmat = new_dmat[0] + new_dmat[1]
        self.assertTrue(np.array_equal(subsys.get_dmat(), new_dmat))

    #@unittest.skip
    def test_update_subsys_fock(self):
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys_dmat = subsys.get_dmat()
        subsys.update_subsys_fock()
        sub_scf = dft.RKS(self.cs_mol)
        sub_scf.xc = self.env_method
        true_fock = sub_scf.get_fock(dm=subsys_dmat)
        true_fock = [true_fock, true_fock]
        self.assertTrue(np.allclose(subsys.subsys_fock, true_fock))

    #@unittest.skip
    def test_update_proj_pot(self):
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        dim0 = subsys.emb_pot[0].shape[0]
        dim1 = subsys.emb_pot[1].shape[1]
        proj_potent = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        subsys.update_proj_pot(proj_potent)
        self.assertTrue(np.array_equal(proj_potent, subsys.proj_pot))

    #@unittest.skip
    def test_update_emb_fock(self):
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        dim0 = subsys.emb_fock[0].shape[0]
        dim1 = subsys.emb_fock[1].shape[1]
        emb_fock = [np.random.rand(dim0, dim1), np.random.rand(dim0, dim1)]
        subsys.update_emb_fock(emb_fock)
        self.assertTrue(np.array_equal(emb_fock, subsys.emb_fock))
        true_emb_pot = [emb_fock[0] - subsys.subsys_fock[0],
                        emb_fock[1] - subsys.subsys_fock[1]]
        self.assertTrue(np.array_equal(true_emb_pot, subsys.emb_pot))

    #@unittest.skip
    def test_env_proj_e(self):

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
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
        subsys.update_proj_pot(proj_potent)
        proj_e = subsys.get_env_proj_e()
        self.assertEqual(test_proj_e, proj_e)

        # Unrestricted Open Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
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
        subsys.update_proj_pot(proj_potent)
        proj_e = subsys.get_env_proj_e()
        self.assertEqual(test_proj_e, proj_e)

    #@unittest.skip
    def test_env_embed_e(self):

        # Closed Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
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
        subsys.update_emb_fock(emb_fock)
        emb_e = subsys.get_env_emb_e()
        self.assertEqual(true_emb_e, emb_e)

        # Unrestricted Open Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.os_mol, self.env_method, unrestricted=True)
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
        subsys.update_emb_fock(emb_fock)
        embed_e = subsys.get_env_emb_e()
        self.assertEqual(true_emb_e, embed_e)

    #@unittest.skip
    def test_get_env_elec_energy(self):

        # Closed Shell
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
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
        
        # With just projeciton potential
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
        # Default test
        def_elec_e = subsys.get_env_elec_energy()
        sub_dmat = subsys.dmat
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

    #@unittest.skip
    def test_get_env_energy(self):

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
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

    #@unittest.skip
    def test_save_density(self):
        import tempfile
        from pyscf.tools import cubegen
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.diagonalize()
        test_ftmp = tempfile.NamedTemporaryFile()
        subsys.save_density_file(filename=test_ftmp.name)
        sub_dmat = subsys.get_dmat()
        true_ftmp = tempfile.NamedTemporaryFile()
        cubegen.density(self.cs_mol, true_ftmp.name, sub_dmat)

        with open(test_ftmp.name + '.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

    #@unittest.skip
    def test_save_orbs(self):
        import tempfile
        from pyscf.tools import molden
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.diagonalize()
        sub_mo_coeff = subsys.env_mo_coeff
        sub_mo_energy = subsys.env_mo_energy
        sub_mo_occ = subsys.env_mo_occ
        test_ftmp = tempfile.NamedTemporaryFile()
        subsys.save_orbital_file(filename=test_ftmp.name)
        true_ftmp = tempfile.NamedTemporaryFile()
        molden.from_mo(self.cs_mol, true_ftmp.name, sub_mo_coeff[0], ene=sub_mo_energy[0], occ=(sub_mo_occ[0] * 2.))

        with open(test_ftmp.name + '.molden', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data, true_den_data)

    #@unittest.skip
    def test_save_read_chkfile(self):
        import tempfile
        import h5py
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys.set_chkfile_index('0')
        subsys.diagonalize()
        test_ftmp = tempfile.NamedTemporaryFile()
        subsys.save_chkfile(filename=test_ftmp.name)

        with h5py.File(test_ftmp.name + '.hdf5', 'r') as hf:
            subsys_coeff = hf[f'subsystem:0/mo_coeff']
            sub_env_mo_coeff = subsys_coeff[:]
            subsys_occ = hf[f'subsystem:0/mo_occ']
            sub_env_mo_occ = subsys_occ[:]
            subsys_energy = hf[f'subsystem:0/mo_energy']
            sub_env_mo_energy = subsys_energy[:]

        self.assertTrue(np.array_equal(subsys.env_mo_coeff, sub_env_mo_coeff))
        self.assertTrue(np.array_equal(subsys.env_mo_occ, sub_env_mo_occ))
        self.assertTrue(np.array_equal(subsys.env_mo_energy, sub_env_mo_energy))

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
        subsys2.set_chkfile_index('0')
        subsys2.read_chkfile(filename=(test_ftmp.name+'.hdf5'))
        self.assertTrue(np.array_equal(subsys.env_mo_coeff, subsys2.env_mo_coeff))
        self.assertTrue(np.array_equal(subsys.env_mo_occ, subsys2.env_mo_occ))
        self.assertTrue(np.array_equal(subsys.env_mo_energy, subsys2.env_mo_energy))

    #@unittest.skip
    def test_diagonalize(self):
        # Closed Shell
        # Unsure how to test this with embedding potential or projection pot.
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)
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
        subsys.diagonalize()
        test_scf = dft.UKS(self.os_mol)
        test_scf.max_cycle = 1
        test_scf.xc = self.env_method
        test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(test_dmat[0], subsys.dmat[0]))
        self.assertTrue(np.allclose(test_dmat[1], subsys.dmat[1]))
    
    #Check this test for use.
    @unittest.skip
    def test_update_fock(self):

        # Closed Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'lda'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method)
        sub_dmat = subsys.dmat
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
        sub_dmat = subsys.dmat
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
        subsys_hl_e = subsys.hl_in_env_energy()
        test_scf = scf.RHF(self.cs_mol)
        correct_dmat = test_scf.get_init_guess(key="1e")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_scf.make_rdm1()))

        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="minao")
        subsys_hl_e = subsys.hl_in_env_energy()
        test_scf = scf.RHF(self.cs_mol)
        correct_dmat = test_scf.get_init_guess(key="minao")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_scf.make_rdm1()))

        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="atom")
        subsys_hl_e = subsys.hl_in_env_energy()
        test_scf = scf.RHF(self.cs_mol)
        correct_dmat = test_scf.get_init_guess(key="atom")
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=correct_dmat)
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_scf.make_rdm1()))

        #Use the embedded density as the hl guess.
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method, hl_conv=conv_param, hl_cycles=0, hl_initguess="ft")
        subsys_hl_e = subsys.hl_in_env_energy()
        test_scf = scf.RHF(self.cs_mol)
        test_scf.max_cycle = 0
        test_scf.kernel(dm0=subsys.get_dmat())
        correct_dmat = test_scf.make_rdm1()
        self.assertTrue(np.allclose(correct_dmat, subsys.hl_scf.make_rdm1()))

    #@unittest.skip
    def test_hf_in_env_energy(self):

        # Closed shell
        hl_method = 'hf'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method)
        subsys_hl_e = subsys.hl_in_env_energy()
        true_scf = scf.RHF(self.cs_mol)
        true_e = true_scf.kernel()
        self.assertAlmostEqual(subsys_hl_e, true_e, delta=1e-10)

        # Open shell
        hl_method = 'hf'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_unrestricted=True)
        subsys_hl_e = subsys.hl_in_env_energy()
        true_scf = scf.UHF(self.os_mol)
        true_e = true_scf.kernel()
        self.assertAlmostEqual(subsys_hl_e, true_e, delta=1e-10)


    def test_dft_in_env_energy(self):

        # Closed shell
        hl_method = 'm06'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method)
        subsys_hl_e = subsys.hl_in_env_energy()
        true_scf = scf.RKS(self.cs_mol)
        true_scf.xc = 'm06'
        true_e = true_scf.kernel()
        self.assertAlmostEqual(subsys_hl_e, true_e, delta=1e-10)

        # Open shell
        hl_method = 'm06'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_unrestricted=True)
        subsys_hl_e = subsys.hl_in_env_energy()
        true_scf = scf.UKS(self.os_mol)
        true_scf.xc = 'm06'
        true_e = true_scf.kernel()
        self.assertAlmostEqual(subsys_hl_e, true_e, delta=1e-10)

    def test_ccsd_in_env_energy(self):

        # Closed shell
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method)
        subsys_hl_e = subsys.hl_in_env_energy()
        true_scf = scf.RHF(self.cs_mol)
        true_hf_e = true_scf.kernel()
        true_cc = cc.CCSD(true_scf)
        true_cc_e = true_cc.kernel()[0]
        self.assertAlmostEqual(subsys_hl_e, true_hf_e + true_cc_e, delta=1e-10)

        # Open shell
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol, self.env_method, hl_method, hl_unrestricted=True)
        subsys_hl_e = subsys.hl_in_env_energy()
        true_scf = scf.UHF(self.os_mol)
        true_hf_e = true_scf.kernel()
        true_cc = cc.UCCSD(true_scf)
        true_cc_e = true_cc.kernel()[0]
        self.assertAlmostEqual(subsys_hl_e, true_hf_e + true_cc_e, delta=1e-10)

    def test_cas_in_env_energy(self):
        pass


    @unittest.skip
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

    @unittest.skip
    def test_hl_in_env_energy(self):
        # Closed Shell
        # Yet to test including embedding or projection potentials.
        mol = gto.Mole()
        mol.verbose = 3
        #O 0.0 0.0 0.0
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'lda'
        hl_method  = 'ccsd'
        conv_param = 1e-10
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method, hl_conv=conv_param)
        subsys_fock = subsys.env_scf.get_fock(dm=(subsys.dmat))
        subsys_hl_e = subsys.hl_in_env_energy()
        test_scf = scf.RHF(mol)
        hf = test_scf.kernel()
        from pyscf import cc
        cc_calc = cc.CCSD(test_scf)
        cc_calc.conv_tol = 1e-10
        cc_calc.conv_tol_normt = 1e-6
        test_hl_e = cc_calc.kernel()[0]
        self.assertAlmostEqual(subsys_hl_e, hf + test_hl_e, delta=1e-5)

        #Unrestricted Open Shell
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        mol.basis = '3-21g'
        mol.spin = 1
        mol.build()
        env_method = 'lda'
        hl_method  = 'uccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method, hl_unrestricted=True, unrestricted=True, hl_conv=conv_param)
        subsys_fock = subsys.env_scf.get_fock(dm=subsys.dmat)
        subsys_hl_e = subsys.hl_in_env_energy()
        test_scf = scf.UHF(mol)
        hf = test_scf.kernel()
        from pyscf import cc
        cc_calc = cc.UCCSD(test_scf)
        cc_calc.conv_tol = 1e-10
        test_hl_e = cc_calc.kernel()[0]
        self.assertAlmostEqual(subsys_hl_e, hf + test_hl_e, delta=1e-5)
