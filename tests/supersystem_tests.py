#A module to test the methods of the supersystem aside from the F&t Method.

#get_supersystem_energy
#get_emb_subsys_elec_energy
#correct_env_energy
#get_hl_energy
#correct_hl_energy
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
import scipy as sp

import tempfile

class TestClusterSuperSystemMethods(unittest.TestCase):

    def setUp(self):

        mol1 = gto.Mole()
        mol1.verbose = 3
        mol1.atom = '''
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  0.758602  0.000000  -0.504284'''
        mol1.basis = 'cc-pVDZ'
        mol1.build()
        self.cs_mol1 = mol1

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O  0.000000  10.000000  0.000000
        H  0.758602  10.000000  0.504284
        H  0.758602  10.000000  -0.504284'''
        mol2.basis = 'cc-pVDZ'
        mol2.build()
        self.cs_mol2 = mol2

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom = '''
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol3.basis = '3-21g'
        mol3.build()
        self.cs_mol3 = mol3

        mol4 = gto.Mole()
        mol4.verbose = 3
        mol4.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        '''
        mol4.basis = '3-21g'
        mol4.build()
        self.env_method = 'lda'
        self.cs_mol4 = mol4

        mol5 = gto.Mole()
        mol5.verbose = 3
        mol5.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:H 0. 0. 2.857
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0
        '''
        mol5.basis = '3-21g'
        mol5.build()
        self.cs_mol34 = mol4

        mol6 = gto.Mole()
        mol6.verbose = 3
        mol6.atom = '''
        He 3.0 0.0 0.0'''
        mol6.basis = '3-21g'
        mol6.build()
        self.cs_mol5 = mol6

        os_mol1 = gto.Mole()
        os_mol1.verbose = 3
        os_mol1.atom = '''
        Li 0.0 0.0 0.0
        '''
        os_mol1.basis = '3-21g'
        os_mol1.spin = 1
        os_mol1.build()
        self.os_mol1 = os_mol1

        os_mol2 = gto.Mole()
        os_mol2.verbose = 3
        os_mol2.atom = '''
        Li 0.0 0.0 0.0
        He 3.0 0.0 0.0
        '''
        os_mol2.basis = '3-21g'
        os_mol2.spin = 1
        os_mol2.build()
        self.os_mol16 = os_mol2

        os_mol3 = gto.Mole()
        os_mol3.verbose = 3
        os_mol3.atom = '''
        H 1.595 0.0 0.0'''
        os_mol3.basis = '3-21g'
        os_mol3.spin = -1
        os_mol3.build()
        self.os_mol2 = os_mol3

        os_mol4 = gto.Mole()
        os_mol4.verbose = 3
        os_mol4.atom = '''
        O 1.94 0.0 0.0'''
        os_mol4.basis = '3-21g'
        os_mol4.spin = 2
        os_mol4.build()
        self.os_mol3 = os_mol4

        mol8 = gto.Mole()
        mol8.verbose = 3
        mol8.atom = '''
        Li 0.0 0.0 0.0
        H 1.595 0.0 0.0'''
        mol8.basis = '3-21g'
        mol8.build()
        self.cs_mol13 = mol8

    @unittest.skip
    def test_init_densities(self):

        env_method = 'lda'
        hl_method = 'ccsd'
        #Closed Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol1, env_method, hl_method)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], env_method)
        supersystem.init_density()

        scf_obj = supersystem.fs_scf
        scf_obj.kernel()
        init_dmat = scf_obj.make_rdm1()
        self.assertTrue(np.allclose(init_dmat, supersystem.fs_dmat[0] + supersystem.fs_dmat[1]))
        self.assertTrue(np.allclose(init_dmat, supersystem.get_emb_dmat()))

        #Unrestricted
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, env_method, hl_method, unrestricted=True)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.os_mol2, env_method, unrestricted=True)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], env_method, fs_unrestricted=True)
        supersystem.init_density()

        scf_obj = supersystem.fs_scf
        scf_obj.kernel()
        init_dmat = scf_obj.make_rdm1()
        self.assertTrue(np.allclose(init_dmat, supersystem.fs_dmat))
       
        #Test this method.
        #true_emb_in_emb = [None, None]
        #true_emb_in_emb[0] = sp.linalg.block_diag(subsys.get_dmat()[0], subsys2.get_dmat()[0])
        #true_emb_in_emb[1] = sp.linalg.block_diag(subsys.get_dmat()[1], subsys2.get_dmat()[1])
        #self.assertTrue(np.array_equal(true_emb_in_emb, supersystem.get_emb_dmat()))

        #Restricted Open Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, env_method, hl_method)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], env_method)
        supersystem.init_density()

        scf_obj = supersystem.fs_scf
        scf_obj.kernel()
        init_dmat = scf_obj.make_rdm1()
        self.assertTrue(np.allclose(init_dmat, supersystem.fs_dmat))
        
        #Test this method.
        #true_emb_in_emb = [None, None]
        #true_emb_in_emb[0] = sp.linalg.block_diag(subsys.get_dmat()[0], subsys2.get_dmat()[0])
        #true_emb_in_emb[1] = sp.linalg.block_diag(subsys.get_dmat()[1], subsys2.get_dmat()[1])
        #self.assertTrue(np.array_equal(true_emb_in_emb, supersystem.get_emb_dmat()))

    @unittest.skip
    def test_get_emb_ext_pot(self):
        pass

    @unittest.skip
    def test_get_supersystem_energy(self):

        #Closed Shell
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol1, self.env_method, hl_method)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol2, self.env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')
        supersystem.init_density()
        supsystem_e = supersystem.get_supersystem_energy()

        test_scf = dft.RKS(supersystem.mol.copy())
        test_scf.xc = 'b3lyp'
        test_e = test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertAlmostEqual(test_e, supsystem_e)
        self.assertTrue(np.allclose(test_dmat, (supersystem.fs_dmat[0] + supersystem.fs_dmat[1])))

        # Unrestricted Open Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, hl_unrestricted=True, unrestricted=True)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol3, self.env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', fs_unrestricted=True)
        supersystem.init_density()
        supsystem_e = supersystem.get_supersystem_energy()

        test_scf = dft.UKS(supersystem.mol.copy())
        test_scf.xc = 'b3lyp'
        test_e = test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertAlmostEqual(test_e, supsystem_e)
        self.assertTrue(np.allclose(test_dmat[0], supersystem.fs_dmat[0]))
        self.assertTrue(np.allclose(test_dmat[1], supersystem.fs_dmat[1]))

        # Restricted Open Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol3, self.env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')
        supersystem.init_density()
        supsystem_e = supersystem.get_supersystem_energy()

        test_scf = dft.ROKS(supersystem.mol.copy())
        test_scf.xc = 'b3lyp'
        test_e = test_scf.kernel()
        test_dmat = test_scf.make_rdm1()
        self.assertAlmostEqual(test_e, supsystem_e)
        self.assertTrue(np.allclose(test_dmat[0], supersystem.fs_dmat[0]))
        self.assertTrue(np.allclose(test_dmat[1], supersystem.fs_dmat[1]))

    @unittest.skip
    def test_save_fs_density(self):
        from pyscf.tools import cubegen

        t_file = tempfile.NamedTemporaryFile()
        hl_method = 'ccsd'

        #Closed Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol1, self.env_method, hl_method, filename=t_file.name)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol2, self.env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        supersystem.save_fs_density_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        sup_dmat = supersystem.fs_dmat[0] + supersystem.fs_dmat[1]
        cubegen.density(supersystem.mol, true_ftmp.name, sup_dmat)

        with open(t_file.name + '_' + chkfile_index + '_fs.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        #Unrestricted
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name, unrestricted=True)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.os_mol2, self.env_method, filename=t_file.name, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name, fs_unrestricted=True)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        supersystem.save_fs_density_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        sup_dmat = supersystem.fs_dmat[0]
        cubegen.density(supersystem.mol, true_ftmp.name, sup_dmat)

        with open(t_file.name + '_' + chkfile_index + '_fs_alpha.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        sup_dmat = supersystem.fs_dmat[1]
        cubegen.density(supersystem.mol, true_ftmp.name, sup_dmat)

        with open(t_file.name + '_' + chkfile_index + '_fs_beta.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        #Restricted Open Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol3, self.env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        supersystem.save_fs_density_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        sup_dmat = supersystem.fs_dmat[0]
        cubegen.density(supersystem.mol, true_ftmp.name, sup_dmat)

        with open(t_file.name + '_' + chkfile_index + '_fs_alpha.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        sup_dmat = supersystem.fs_dmat[1]
        cubegen.density(supersystem.mol, true_ftmp.name, sup_dmat)

        with open(t_file.name + '_' + chkfile_index + '_fs_beta.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

    @unittest.skip
    def test_save_fs_spin_density(self):
        from pyscf.tools import cubegen

        t_file = tempfile.NamedTemporaryFile()
        hl_method = 'ccsd'

        #Unrestricted
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name, unrestricted=True)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.os_mol2, self.env_method, filename=t_file.name, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name, fs_unrestricted=True)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        supersystem.save_fs_spin_density_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        sup_dmat = supersystem.fs_dmat
        cubegen.density(supersystem.mol, true_ftmp.name, np.subtract(sup_dmat[0], sup_dmat[1]))

        with open(t_file.name + '_' + chkfile_index + '_fs_spinden.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        #Restricted Open Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol3, self.env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        supersystem.save_fs_spin_density_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        sup_dmat = supersystem.fs_dmat
        cubegen.density(supersystem.mol, true_ftmp.name, np.subtract(sup_dmat[0], sup_dmat[1]))

        with open(t_file.name + '_' + chkfile_index + '_fs_spinden.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

    @unittest.skip
    def test_save_fs_orbs(self):
        from pyscf.tools import molden
        t_file = tempfile.NamedTemporaryFile()
        hl_method = 'ccsd'

        #Closed Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol1, self.env_method, hl_method, filename=t_file.name)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol2, self.env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        sup_mo_coeff = supersystem.fs_scf.mo_coeff
        sup_mo_energy = supersystem.fs_scf.mo_energy
        sup_mo_occ = supersystem.fs_scf.mo_occ
        test_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        supersystem.save_fs_orbital_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        molden.from_mo(supersystem.mol, true_ftmp.name, sup_mo_coeff, ene=sup_mo_energy, occ=sup_mo_occ)

        with open(t_file.name + '_' + chkfile_index + '_fs.molden', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data, true_den_data)

        #Unrestricted
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name, unrestricted=True)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.os_mol2, self.env_method, filename=t_file.name, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name, fs_unrestricted=True)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        sup_mo_coeff = supersystem.fs_scf.mo_coeff
        sup_mo_energy = supersystem.fs_scf.mo_energy
        sup_mo_occ = supersystem.fs_scf.mo_occ
        supersystem.save_fs_orbital_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        molden.from_mo(supersystem.mol, true_ftmp.name, sup_mo_coeff[0], ene=sup_mo_energy[0], occ=sup_mo_occ[0], spin="Alpha")

        with open(t_file.name + '_' + chkfile_index + '_fs_alpha.molden', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data, true_den_data)

        molden.from_mo(supersystem.mol, true_ftmp.name, sup_mo_coeff[1], ene=sup_mo_energy[1], occ=sup_mo_occ[1], spin="Beta")

        with open(t_file.name + '_' + chkfile_index + '_fs_beta.molden', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data, true_den_data)

        #Restricted Open Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol3, self.env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        sup_mo_coeff = supersystem.fs_scf.mo_coeff
        sup_mo_energy = supersystem.fs_scf.mo_energy
        sup_mo_occ = supersystem.fs_scf.mo_occ
        supersystem.save_fs_orbital_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        molden.from_mo(supersystem.mol, true_ftmp.name, sup_mo_coeff, ene=sup_mo_energy, occ=sup_mo_occ)
        with open(t_file.name + '_' + chkfile_index + '_fs.molden', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()
        self.assertEqual(test_den_data, true_den_data)

    @unittest.skip
    def test_get_emb_subsys_elec_energy(self):

        # Closed Shell
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol1, self.env_method, hl_method)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol2, self.env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], self.env_method, ft_initguess='minao')
        supersystem.init_density()
        supersystem.update_fock()

        mf = dft.RKS(supersystem.mol)
        mf.xc = self.env_method
        mf_t_dmat = mf.get_init_guess(key='minao')
        mf_init_dmat = np.zeros_like(mf_t_dmat)

        #get energies of two embedded systems. 
        s2s = supersystem.sub2sup
        mf_init_dmat[np.ix_(s2s[0], s2s[0])] += subsys.get_dmat()
        mf_init_dmat[np.ix_(s2s[1], s2s[1])] += subsys2.get_dmat()
        mf_hcore = mf.get_hcore()
        mf_init_veff = mf.get_veff(dm=mf_init_dmat)

        dm_1 = mf_init_dmat[np.ix_(s2s[0], s2s[0])]
        hcore_1_emb = mf_hcore[np.ix_(s2s[0], s2s[0])]
        veff_1_emb = mf_init_veff[np.ix_(s2s[0], s2s[0])]
        mf_1 = dft.RKS(self.cs_mol1)
        mf_1.xc = self.env_method
        mf_1.grids = supersystem.fs_scf.grids
        hcore_1_emb = hcore_1_emb - mf_1.get_hcore()
        veff_1 = mf_1.get_veff(dm=dm_1)
        veff_1_emb = veff_1_emb - veff_1
        test_sub1_e = mf_1.energy_elec(dm=dm_1)[0] + np.einsum('ij,ji', hcore_1_emb, dm_1) + np.einsum('ij,ji', veff_1_emb, dm_1)

        dm_2 = mf_init_dmat[np.ix_(s2s[1], s2s[1])]
        hcore_2_emb = mf_hcore[np.ix_(s2s[1], s2s[1])]
        veff_2_emb = mf_init_veff[np.ix_(s2s[1], s2s[1])]
        mf_2 = dft.RKS(self.cs_mol2)
        mf_2.xc = self.env_method
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
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, unrestricted=True, hl_unrestricted=True)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.os_mol2, self.env_method, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], self.env_method, ft_initguess='minao')
        supersystem.init_density()

        supsystem_e = supersystem.get_supersystem_energy()
        mf = dft.UKS(supersystem.mol)
        mf.xc = self.env_method
        mf_t_dmat = mf.get_init_guess(key='minao')
        mf_init_dmat = np.zeros_like(mf_t_dmat)

        #get energies of two embedded systems. 
        s2s = supersystem.sub2sup
        mf_init_dmat[0][np.ix_(s2s[0], s2s[0])] += subsys.env_dmat[0] 
        mf_init_dmat[1][np.ix_(s2s[0], s2s[0])] += subsys.env_dmat[1] 
        mf_init_dmat[0][np.ix_(s2s[1], s2s[1])] += subsys2.env_dmat[0]
        mf_init_dmat[1][np.ix_(s2s[1], s2s[1])] += subsys2.env_dmat[1]
        mf_hcore = mf.get_hcore()
        mf_init_veff = mf.get_veff(dm=mf_init_dmat)

        dm_1 = [mf_init_dmat[0][np.ix_(s2s[0], s2s[0])], mf_init_dmat[1][np.ix_(s2s[0], s2s[0])]]
        hcore_1_emb = mf_hcore[np.ix_(s2s[0], s2s[0])]
        veff_1_emb = [mf_init_veff[0][np.ix_(s2s[0], s2s[0])], mf_init_veff[0][np.ix_(s2s[0], s2s[0])]]
        mf_1 = dft.UKS(self.os_mol1)
        mf_1.xc = self.env_method
        mf_1.grids = supersystem.fs_scf.grids
        hcore_1_emb = hcore_1_emb - mf_1.get_hcore()
        veff_1 = mf_1.get_veff(dm=dm_1)
        veff_1_emb = [veff_1_emb[0] - veff_1[0], veff_1_emb[1] - veff_1[1]]
        test_sub1_e = mf_1.energy_elec(dm=dm_1)[0] + np.einsum('ij,ji', hcore_1_emb, dm_1[0]) + np.einsum('ij,ji', hcore_1_emb, dm_1[1]) +  np.einsum('ij,ji', veff_1_emb[0], dm_1[0]) + np.einsum('ij,ji', veff_1_emb[1], dm_1[1])

        dm_2 = [mf_init_dmat[0][np.ix_(s2s[1], s2s[1])], mf_init_dmat[1][np.ix_(s2s[1], s2s[1])]]
        hcore_2_emb = mf_hcore[np.ix_(s2s[1], s2s[1])]
        veff_2_emb = [mf_init_veff[1][np.ix_(s2s[1], s2s[1])], mf_init_veff[1][np.ix_(s2s[1], s2s[1])]]
        mf_2 = dft.UKS(self.os_mol2)
        mf_2.xc = self.env_method
        mf_2.grids = supersystem.fs_scf.grids
        hcore_2_emb = hcore_2_emb - mf_2.get_hcore()
        veff_2 = mf_2.get_veff(dm=dm_2)
        veff_2_emb = [veff_2_emb[0] - veff_2[0], veff_2_emb[1] - veff_2[1]]
        test_sub2_e = mf_2.energy_elec(dm=dm_2)[0] + np.einsum('ij,ji', hcore_2_emb, dm_2[0]) + np.einsum('ij,ji', hcore_2_emb, dm_2[1]) +  np.einsum('ij,ji', veff_2_emb[0], dm_2[0]) + np.einsum('ij,ji', veff_2_emb[1], dm_2[1])


        sub1_e = supersystem.subsystems[0].get_env_elec_energy()
        sub2_e = supersystem.subsystems[1].get_env_elec_energy()
        self.assertAlmostEqual(test_sub1_e, sub1_e, delta=1e-8)
        self.assertAlmostEqual(test_sub2_e, sub2_e, delta=1e-8)

        # Restricted Open Shell
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol3, self.env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], self.env_method, ft_initguess='minao')
        supersystem.init_density()

        supsystem_e = supersystem.get_supersystem_energy()
        mf = dft.ROKS(supersystem.mol)
        mf.xc = self.env_method
        mf_t_dmat = mf.get_init_guess(key='minao')
        mf_init_dmat = np.zeros_like(mf_t_dmat)

        #get energies of two embedded systems. 
        s2s = supersystem.sub2sup
        mf_init_dmat[0][np.ix_(s2s[0], s2s[0])] += subsys.env_dmat[0] 
        mf_init_dmat[1][np.ix_(s2s[0], s2s[0])] += subsys.env_dmat[1] 
        mf_init_dmat[0][np.ix_(s2s[1], s2s[1])] += subsys2.env_dmat[0]
        mf_init_dmat[1][np.ix_(s2s[1], s2s[1])] += subsys2.env_dmat[1]
        mf_hcore = mf.get_hcore()
        mf_init_veff = mf.get_veff(dm=mf_init_dmat)

        dm_1 = [mf_init_dmat[0][np.ix_(s2s[0], s2s[0])], mf_init_dmat[1][np.ix_(s2s[0], s2s[0])]]
        hcore_1_emb = mf_hcore[np.ix_(s2s[0], s2s[0])]
        veff_1_emb = [mf_init_veff[0][np.ix_(s2s[0], s2s[0])], mf_init_veff[1][np.ix_(s2s[0], s2s[0])]]
        mf_1 = dft.ROKS(self.os_mol1)
        mf_1.xc = self.env_method
        mf_1.grids = supersystem.fs_scf.grids
        hcore_1_emb = hcore_1_emb - mf_1.get_hcore()
        veff_1 = mf_1.get_veff(dm=dm_1)
        veff_1_emb = [veff_1_emb[0] - veff_1[0], veff_1_emb[1] - veff_1[1]]
        test_sub1_e = mf_1.energy_elec(dm=dm_1)[0] + np.einsum('ij,ji', hcore_1_emb, dm_1[0]) + np.einsum('ij,ji', hcore_1_emb, dm_1[1]) +  np.einsum('ij,ji', veff_1_emb[0], dm_1[0]) + np.einsum('ij,ji', veff_1_emb[1], dm_1[1])

        dm_2 = [mf_init_dmat[0][np.ix_(s2s[1], s2s[1])], mf_init_dmat[1][np.ix_(s2s[1], s2s[1])]]
        hcore_2_emb = mf_hcore[np.ix_(s2s[1], s2s[1])]
        veff_2_emb = [mf_init_veff[0][np.ix_(s2s[1], s2s[1])], mf_init_veff[1][np.ix_(s2s[1], s2s[1])]]
        mf_2 = dft.ROKS(self.cs_mol3)
        mf_2.xc = self.env_method
        mf_2.grids = supersystem.fs_scf.grids
        hcore_2_emb = hcore_2_emb - mf_2.get_hcore()
        veff_2 = mf_2.get_veff(dm=dm_2)
        veff_2_emb = [veff_2_emb[0] - veff_2[0], veff_2_emb[1] - veff_2[1]]
        test_sub2_e = mf_2.energy_elec(dm=dm_2)[0] + np.einsum('ij,ji', hcore_2_emb, dm_2[0]) + np.einsum('ij,ji', hcore_2_emb, dm_2[1]) +  np.einsum('ij,ji', veff_2_emb[0], dm_2[0]) + np.einsum('ij,ji', veff_2_emb[1], dm_2[1])


        sub1_e = supersystem.subsystems[0].get_env_elec_energy()
        supersystem.subsystems[1].update_subsys_fock()
        sub2_e = supersystem.subsystems[1].get_env_elec_energy() #this part of the test doesn't work because negative spin.
        self.assertAlmostEqual(test_sub1_e, sub1_e, delta=1e-8)
        self.assertAlmostEqual(test_sub2_e, sub2_e, delta=1e-8)


    #@unittest.skip
    def test_get_hl_energy(self):
        pass
    #@unittest.skip
    def test_get_env_energy(self):
        pass

    #@unittest.skip
    def test_get_env_in_env_energy(self):
        pass

    #@unittest.skip
    def test_update_ro_fock(self):
        pass

    @unittest.skip
    def test_update_fock(self):

        # Closed Shell
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol1, self.env_method, hl_method)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol2, self.env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')
        supersystem.init_density()

        mf = scf.RKS(supersystem.mol)
        mf.xc = 'b3lyp'
        grids = dft.gen_grid.Grids(supersystem.mol)
        grids.build()
        mf.grids = grids
        mf_init_dmat = np.zeros_like(supersystem.get_emb_dmat())
        s2s = supersystem.sub2sup
        mf_init_dmat[np.ix_(s2s[0], s2s[0])] += subsys.get_dmat()
        mf_init_dmat[np.ix_(s2s[1], s2s[1])] += subsys2.get_dmat()
        mf_hcore = mf.get_hcore()
        mf_init_veff = mf.get_veff(dm=mf_init_dmat)
        full_fock = mf_hcore + mf_init_veff
        self.assertTrue(np.allclose(full_fock, supersystem.fock[0]))

        test_fock1 = full_fock[np.ix_(s2s[0], s2s[0])]
        test_fock2 = full_fock[np.ix_(s2s[1], s2s[1])]
        self.assertTrue(np.allclose(test_fock1, supersystem.subsystems[0].emb_fock[0]))
        self.assertTrue(np.allclose(test_fock2, supersystem.subsystems[1].emb_fock[0]))

        # Unrestricted Open Shell
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, unrestricted=True)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.os_mol2, self.env_method, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao', fs_unrestricted=True)
        supersystem.init_density()
        supersystem.update_fock()

        mf = dft.UKS(supersystem.mol.copy())
        mf.xc = 'm06'
        mf_t_dmat = mf.get_init_guess(key='minao')
        mf_init_dmat = np.zeros_like(mf_t_dmat)

        #get energies of two embedded systems. 
        s2s = supersystem.sub2sup
        mf_init_dmat[0][np.ix_(s2s[0], s2s[0])] += subsys.env_dmat[0]
        mf_init_dmat[0][np.ix_(s2s[1], s2s[1])] += subsys2.env_dmat[0]
        mf_init_dmat[1][np.ix_(s2s[0], s2s[0])] += subsys.env_dmat[1]
        mf_init_dmat[1][np.ix_(s2s[1], s2s[1])] += subsys2.env_dmat[1]
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

        # Restricted Open Shell
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol3, self.env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao')
        supersystem.init_density()
        supersystem.update_fock()

        mf = dft.ROKS(supersystem.mol.copy())
        mf.xc = 'm06'
        mf_t_dmat = mf.get_init_guess(key='minao')
        mf_init_dmat = np.zeros_like(mf_t_dmat)

        #get energies of two embedded systems. 
        s2s = supersystem.sub2sup
        mf_init_dmat[0][np.ix_(s2s[0], s2s[0])] += subsys.env_dmat[0]
        mf_init_dmat[0][np.ix_(s2s[1], s2s[1])] += subsys2.env_dmat[0]
        mf_init_dmat[1][np.ix_(s2s[0], s2s[0])] += subsys.env_dmat[1]
        mf_init_dmat[1][np.ix_(s2s[1], s2s[1])] += subsys2.env_dmat[1]
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

    @unittest.skip
    def test_update_proj_pot(self):
        #Only test I can think of is to just recreate the 
        #projection potential and check they are still equal.
        pass

    #@unittest.skip
    def test_save_read_chkfile(self):
        t_file = tempfile.NamedTemporaryFile()
        hl_method = 'ccsd'

        #Closed Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol1, self.env_method, hl_method, filename=t_file.name)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol2, self.env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supersystem.init_density()
        old_dmat = supersystem.fs_dmat
        old_sub1_dmat = supersystem.subsystems[0].get_dmat()
        old_sub2_dmat = supersystem.subsystems[1].get_dmat()
        supersystem2 = cluster_supersystem.ClusterSuperSystem([copy(subsys), copy(subsys2)], 'b3lyp', ft_initguess='readchk', filename=t_file.name)

        supersystem2.init_density()
        new_dmat = supersystem2.fs_dmat
        new_sub1_dmat = supersystem2.subsystems[0].get_dmat()
        new_sub2_dmat = supersystem2.subsystems[1].get_dmat()
        self.assertTrue(np.equal(old_dmat, new_dmat).all)
        self.assertTrue(np.equal(old_sub1_dmat, new_sub1_dmat).all)
        self.assertTrue(np.equal(old_sub2_dmat, new_sub2_dmat).all)

        #Unrestricted
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name, unrestricted=True)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.os_mol2, self.env_method, filename=t_file.name, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name, fs_unrestricted=True)
        supersystem.init_density()
        old_dmat = supersystem.fs_dmat
        old_sub1_dmat = supersystem.subsystems[0].get_dmat()
        old_sub2_dmat = supersystem.subsystems[1].get_dmat()
        supersystem2 = cluster_supersystem.ClusterSuperSystem([copy(subsys), copy(subsys2)], 'b3lyp', ft_initguess='readchk', filename=t_file.name, fs_unrestricted=True)

        supersystem2.init_density()
        new_dmat = supersystem2.fs_dmat
        new_sub1_dmat = supersystem2.subsystems[0].get_dmat()
        new_sub2_dmat = supersystem2.subsystems[1].get_dmat()
        self.assertTrue(np.equal(old_dmat, new_dmat).all)
        self.assertTrue(np.equal(old_sub1_dmat, new_sub1_dmat).all)
        self.assertTrue(np.equal(old_sub2_dmat, new_sub2_dmat).all)

        #Restricted Open Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name)

        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol3, self.env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supersystem.init_density()
        old_dmat = supersystem.fs_dmat
        old_sub1_dmat = supersystem.subsystems[0].get_dmat()
        old_sub2_dmat = supersystem.subsystems[1].get_dmat()
        supersystem2 = cluster_supersystem.ClusterSuperSystem([copy(subsys), copy(subsys2)], 'b3lyp', ft_initguess='readchk', filename=t_file.name)

        supersystem2.init_density()
        new_dmat = supersystem2.fs_dmat
        new_sub1_dmat = supersystem2.subsystems[0].get_dmat()
        new_sub2_dmat = supersystem2.subsystems[1].get_dmat()
        self.assertTrue(np.equal(old_dmat, new_dmat).all)
        self.assertTrue(np.equal(old_sub1_dmat, new_sub1_dmat).all)
        self.assertTrue(np.equal(old_sub2_dmat, new_sub2_dmat).all)

    #@unittest.skip
    def test_save_ft_density(self):
        from pyscf.tools import cubegen
        t_file = tempfile.NamedTemporaryFile()
        hl_method = 'ccsd'

        #Closed Shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol1, self.env_method, hl_method, filename=t_file.name)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol2, self.env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        supersystem.freeze_and_thaw()
        supersystem.save_ft_density_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        nS = supersystem.mol.nao_nr()
        dm_env = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        for i in range(len(supersystem.subsystems)):
            dm_env[0][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += supersystem.subsystems[i].env_dmat[0]
            dm_env[1][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += supersystem.subsystems[i].env_dmat[1]
        sup_dmat = dm_env[0] + dm_env[1]
        cubegen.density(supersystem.mol.copy(), true_ftmp.name, sup_dmat)

        with open(t_file.name + '_' + chkfile_index + '_ft.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        #Unrestricted
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name, unrestricted=True)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.os_mol2, self.env_method, filename=t_file.name, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name, fs_unrestricted=True)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        supersystem.freeze_and_thaw()
        supersystem.save_ft_density_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        nS = supersystem.mol.nao_nr()
        dm_env = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        for i in range(len(supersystem.subsystems)):
            dm_env[0][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += supersystem.subsystems[i].env_dmat[0]
            dm_env[1][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += supersystem.subsystems[i].env_dmat[1]
        cubegen.density(supersystem.mol.copy(), true_ftmp.name, dm_env[0])

        with open(t_file.name + '_' + chkfile_index + '_ft_alpha.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        cubegen.density(supersystem.mol.copy(), true_ftmp.name, dm_env[1])

        with open(t_file.name + '_' + chkfile_index + '_ft_beta.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        #Restricted open shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol3, self.env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        supersystem.freeze_and_thaw()
        supersystem.save_ft_density_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        nS = supersystem.mol.nao_nr()
        dm_env = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        for i in range(len(supersystem.subsystems)):
            dm_env[0][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += supersystem.subsystems[i].env_dmat[0]
            dm_env[1][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += supersystem.subsystems[i].env_dmat[1]
        cubegen.density(supersystem.mol.copy(), true_ftmp.name, dm_env[0])

        with open(t_file.name + '_' + chkfile_index + '_ft_alpha.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        cubegen.density(supersystem.mol.copy(), true_ftmp.name, dm_env[1])

        with open(t_file.name + '_' + chkfile_index + '_ft_beta.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

    #@unittest.skip
    def test_save_ft_spin_density(self):
        from pyscf.tools import cubegen
        hl_method = 'ccsd'
        t_file = tempfile.NamedTemporaryFile()

        #Unrestricted
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name, unrestricted=True)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.os_mol2, self.env_method, filename=t_file.name, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name, fs_unrestricted=True)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        supersystem.freeze_and_thaw()
        supersystem.save_ft_spin_density_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        nS = supersystem.mol.nao_nr()
        dm_env = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        for i in range(len(supersystem.subsystems)):
            dm_env[0][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += supersystem.subsystems[i].env_dmat[0]
            dm_env[1][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += supersystem.subsystems[i].env_dmat[1]
        cubegen.density(supersystem.mol.copy(), true_ftmp.name, np.subtract(dm_env[0], dm_env[1]))

        with open(t_file.name + '_' + chkfile_index + '_ft_spinden.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

        #Restricted open shell
        subsys = cluster_subsystem.ClusterHLSubSystem(self.os_mol1, self.env_method, hl_method, filename=t_file.name)
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol3, self.env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supersystem.init_density()
        supersystem.get_supersystem_energy()
        supersystem.freeze_and_thaw()
        supersystem.save_ft_spin_density_file()
        true_ftmp = tempfile.NamedTemporaryFile()
        chkfile_index = supersystem.chkfile_index
        nS = supersystem.mol.nao_nr()
        dm_env = [np.zeros((nS, nS)), np.zeros((nS, nS))]
        for i in range(len(supersystem.subsystems)):
            dm_env[0][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += supersystem.subsystems[i].env_dmat[0]
            dm_env[1][np.ix_(supersystem.sub2sup[i], supersystem.sub2sup[i])] += supersystem.subsystems[i].env_dmat[1]
        cubegen.density(supersystem.mol.copy(), true_ftmp.name, np.subtract(dm_env[0], dm_env[1]))

        with open(t_file.name + '_' + chkfile_index + '_ft_spinden.cube', 'r') as fin:
            test_den_data = fin.read()

        with open(true_ftmp.name, 'r') as fin:
            true_den_data = fin.read()

        self.assertEqual(test_den_data[99:], true_den_data[99:])

    #@unittest.skip
    def test_freeze_and_thaw(self):

        #Restricted closed shell
        #Supermolecular test.
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        C      0.7710806955  -0.0001048861   0.0000400510
        H      1.1560846512   0.8695663320  -0.5203105003
        H      1.1560491322   0.0161891484   1.0133671125
        H      1.1560865179  -0.8856013435  -0.4928324985
        ghost:C     -0.7713511096  -0.0001546299  -0.0000054393
        ghost:H     -1.1561315704   0.8855266211   0.4927506464 
        ghost:H     -1.1560645399  -0.0160685116  -1.0134290757 
        ghost:H     -1.1560647411  -0.8697976282   0.5205503870 
        '''
        mol.basis = 'cc-pVDZ'
        mol.charge = -1
        mol.build()
        env_method = 'pbe'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        ghost:C      0.7710806955  -0.0001048861   0.0000400510
        ghost:H      1.1560846512   0.8695663320  -0.5203105003
        ghost:H      1.1560491322   0.0161891484   1.0133671125
        ghost:H      1.1560865179  -0.8856013435  -0.4928324985
        C     -0.7713511096  -0.0001546299  -0.0000054393
        H     -1.1561315704   0.8855266211   0.4927506464 
        H     -1.1560645399  -0.0160685116  -1.0134290757 
        H     -1.1560647411  -0.8697976282   0.5205503870 
        '''
        mol2.basis = 'cc-pVDZ'
        mol2.charge = 1
        mol2.build()
        env_method = 'pbe'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'pbe', ft_cycles=25)
        supersystem.init_density()
        supersystem.freeze_and_thaw()
        supersystem.get_env_in_env_energy()
        supersystem.get_supersystem_energy()

        projector_energy = np.trace(np.dot(subsys.env_dmat[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.env_dmat[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.env_dmat[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.env_dmat[1], supersystem.proj_pot[1][1]))
        self.assertAlmostEqual(0.0, projector_energy, delta=1e-10)

        sup_e = supersystem.get_env_in_env_energy()
        test_e = supersystem.get_supersystem_energy()
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
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method)

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
        supersystem.init_density()
        supersystem.freeze_and_thaw()

        projector_energy = np.trace(np.dot(subsys.env_dmat[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.env_dmat[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.env_dmat[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.env_dmat[1], supersystem.proj_pot[1][1]))
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
        mf.kernel()
        test_dmat = mf.make_rdm1()
        test_e = mf.energy_tot()
        sup_e = supersystem.get_env_in_env_energy()
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
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method)

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
        supersystem.init_density()

        initial_projector_energy = 0.0
        initial_projector_energy = np.trace(np.dot(subsys.env_dmat[0], supersystem.proj_pot[0][0]))
        initial_projector_energy += np.trace(np.dot(subsys.env_dmat[1], supersystem.proj_pot[0][1]))
        initial_projector_energy += np.trace(np.dot(subsys2.env_dmat[0], supersystem.proj_pot[1][0]))
        initial_projector_energy += np.trace(np.dot(subsys2.env_dmat[1], supersystem.proj_pot[1][1]))

        supersystem.freeze_and_thaw()

        projector_energy = np.trace(np.dot(subsys.env_dmat[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.env_dmat[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.env_dmat[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.env_dmat[1], supersystem.proj_pot[1][1]))
        self.assertGreaterEqual(0.0, projector_energy-initial_projector_energy)

        # Unrestricted Open Shell
        ###Supermolecular test.
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        C     1.3026    9.2236   -0.0001     
        F     2.3785   10.1081    0.0001
        H     1.3873    8.5867   -0.8899    
        H     1.3873    8.5862    0.8894 
        ghost:C     -0.0002    9.9997    0.0001
        ghost:H     -0.8544    9.3128   -0.0001 
        ghost:H     -0.0683   10.6366    0.8880 
        ghost:H     -0.0683   10.6370   -0.8875 
        '''
        mol.basis = 'cc-pVDZ'
        mol.spin = 1
        mol.build()
        env_method = 'hf'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method, unrestricted=True, hl_unrestricted=True)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        ghost:C     1.3026    9.2236   -0.0001     
        ghost:F     2.3785   10.1081    0.0001
        ghost:H     1.3873    8.5867   -0.8899    
        ghost:H     1.3873    8.5862    0.8894 
        C     -0.0002    9.9997    0.0001
        H     -0.8544    9.3128   -0.0001 
        H     -0.0683   10.6366    0.8880 
        H     -0.0683   10.6370   -0.8875 
        '''
        mol2.basis = 'cc-pVDZ'
        mol2.spin = -1
        mol2.build()
        env_method = 'hf'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'hf', fs_unrestricted=True)
        supersystem.init_density()
        supersystem.freeze_and_thaw()

        projector_energy = np.trace(np.dot(subsys.get_dmat()[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.get_dmat()[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.get_dmat()[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.get_dmat()[1], supersystem.proj_pot[1][1]))
        self.assertAlmostEqual(0.0, projector_energy, delta=1e-15)

        test_dmat = supersystem.get_emb_dmat()
        true_dmat = supersystem.fs_scf.make_rdm1()
        test_e = supersystem.get_supersystem_energy()
        sup_e = supersystem.get_env_in_env_energy()
        self.assertAlmostEqual(test_e, sup_e, delta=1e-10) 

        ##Long distance test.
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0. 0. 0.
        O 1.13 0. 0.
        '''
        mol.basis = '3-21g'
        mol.spin = 2
        mol.build()
        nv_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method, unrestricted=True, hl_unrestricted=True)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 0. 100.0 0.0
        O 1.13 100.0 0.0
        '''
        mol2.basis = '3-21g'
        mol2.spin = 2
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='supmol', fs_cycles=1000, fs_unrestricted=True, ft_cycles=1000)
        supersystem.init_density()
        supersystem.freeze_and_thaw()

        projector_energy = np.trace(np.dot(subsys.get_dmat()[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.get_dmat()[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.get_dmat()[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.get_dmat()[1], supersystem.proj_pot[1][1]))
        self.assertAlmostEqual(0.0, projector_energy, delta=1e-14)


        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom ='''
        O 0. 0. 0.
        O 1.13 0. 0.
        O 0. 100.0 0.0
        O 1.13 100.0 0.0
        '''
        mol3.basis = '3-21g'
        mol3.spin = 4
        mol3.build()
        mf = dft.UKS(mol3)
        mf.xc = 'm06'
        mf.grids = supersystem.fs_scf.grids
        mf.max_cycle = 1000
        mf.kernel()
        test_dmat = mf.make_rdm1()
        test_e = mf.energy_tot()
        sup_e = supersystem.get_env_in_env_energy()
        self.assertAlmostEqual(test_e, sup_e, delta=1e-10)

        #Projection energy
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        Li 0. 0. 0.
        '''
        mol.basis = 'cc-pVDZ'
        mol.spin = 1
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method, unrestricted=True, hl_unrestricted=True)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        H 1.595 0.0 0.0
        '''
        mol2.basis = 'cc-pVDZ'
        mol2.spin = -1
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='supmol')
        supersystem.init_density()

        initial_projector_energy = 0.0
        initial_projector_energy = np.trace(np.dot(subsys.get_dmat()[0], supersystem.proj_pot[0][0]))
        initial_projector_energy += np.trace(np.dot(subsys.get_dmat()[1], supersystem.proj_pot[0][1]))
        initial_projector_energy += np.trace(np.dot(subsys2.get_dmat()[0], supersystem.proj_pot[1][0]))
        initial_projector_energy += np.trace(np.dot(subsys2.get_dmat()[1], supersystem.proj_pot[1][1]))

        supersystem.freeze_and_thaw()

        projector_energy = np.trace(np.dot(subsys.get_dmat()[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.get_dmat()[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.get_dmat()[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.get_dmat()[1], supersystem.proj_pot[1][1]))
        self.assertGreaterEqual(0.0, projector_energy-initial_projector_energy)

        #Localized Spin
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
            ghost.C                 -2.95182400   -0.40708300    0.00000200
            ghost.H                 -2.94830500   -1.05409800   -0.87975500
            ghost.H                 -2.94830300   -1.05410200    0.87975600
            ghost.H                 -3.88904200    0.14923700    0.00000400
            C                  0.74345800    0.66525400    0.00000100
            H                  0.75917800    1.30099000   -0.88423600
            H                  0.75918100    1.30098600    0.88424000
            C                 -0.44163200   -0.26055200    0.00000000
            H                 -0.39317700   -0.91474400   -0.87593000
            H                 -0.39317500   -0.91474700    0.87592800
            C                 -1.74853900    0.51173600    0.00000300
            H                 -1.78059900    1.17041100    0.87438300
            H                 -1.78060000    1.17041500   -0.87437400'''
        mol.basis = 'cc-pVDZ'
        mol.spin = 1
        mol.charge = -1
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method, unrestricted=True, hl_unrestricted=True)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
            C                 -2.95182400   -0.40708300    0.00000200
            H                 -2.94830500   -1.05409800   -0.87975500
            H                 -2.94830300   -1.05410200    0.87975600
            H                 -3.88904200    0.14923700    0.00000400
            ghost.C                  0.74345800    0.66525400    0.00000100
            ghost.H                  0.75917800    1.30099000   -0.88423600
            ghost.H                  0.75918100    1.30098600    0.88424000
            ghost.C                 -0.44163200   -0.26055200    0.00000000
            ghost.H                 -0.39317700   -0.91474400   -0.87593000
            ghost.H                 -0.39317500   -0.91474700    0.87592800
            ghost.C                 -1.74853900    0.51173600    0.00000300
            ghost.H                 -1.78059900    1.17041100    0.87438300
            ghost.H                 -1.78060000    1.17041500   -0.87437400'''
        mol2.basis = 'cc-pVDZ'
        mol2.charge = 1
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, unrestricted=True)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', fs_unrestricted=True)
        supersystem.init_density()
        supersystem.freeze_and_thaw()

        projector_energy = np.trace(np.dot(subsys.get_dmat()[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.get_dmat()[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.get_dmat()[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.get_dmat()[1], supersystem.proj_pot[1][1]))
        self.assertAlmostEqual(0.0, projector_energy, delta=1e-15)

        mol3 = gto.Mole()
        mol3.atom ='''
            C                  0.74345800    0.66525400    0.00000100
            H                  0.75917800    1.30099000   -0.88423600
            H                  0.75918100    1.30098600    0.88424000
            C                 -0.44163200   -0.26055200    0.00000000
            H                 -0.39317700   -0.91474400   -0.87593000
            H                 -0.39317500   -0.91474700    0.87592800
            C                 -1.74853900    0.51173600    0.00000300
            H                 -1.78059900    1.17041100    0.87438300
            H                 -1.78060000    1.17041500   -0.87437400
            C                 -2.95182400   -0.40708300    0.00000200
            H                 -2.94830500   -1.05409800   -0.87975500
            H                 -2.94830300   -1.05410200    0.87975600
            H                 -3.88904200    0.14923700    0.00000400
        '''
        mol3.basis = 'cc-pVDZ'
        mol3.spin = 1
        mol3.build()
        mf = dft.UKS(mol3)
        mf.xc = 'm06'
        grids = supersystem.fs_scf.grids
        mf.grids = grids
        mf.kernel()
        test_e = mf.energy_tot()
        sup_e = supersystem.get_env_in_env_energy()
        self.assertAlmostEqual(test_e, sup_e, delta=1e-10)

        # Restricted Open Shell

        #Localized Spin
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
            C                  0.74345800    0.66525400    0.00000100
            H                  0.75917800    1.30099000   -0.88423600
            H                  0.75918100    1.30098600    0.88424000
            ghost.C                 -0.44163200   -0.26055200    0.00000000
            ghost.H                 -0.39317700   -0.91474400   -0.87593000
            ghost.H                 -0.39317500   -0.91474700    0.87592800
            ghost.C                 -1.74853900    0.51173600    0.00000300
            ghost.H                 -1.78059900    1.17041100    0.87438300
            ghost.H                 -1.78060000    1.17041500   -0.87437400
            ghost.H                 -2.95182400   -0.40708300    0.00000200'''
        mol.basis = '6-31g'
        mol.spin = 1
        mol.charge = -1
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method, hl_unrestricted=True)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
            ghost.C                  0.74345800    0.66525400    0.00000100
            ghost.H                  0.75917800    1.30099000   -0.88423600
            ghost.H                  0.75918100    1.30098600    0.88424000
            C                 -0.44163200   -0.26055200    0.00000000
            H                 -0.39317700   -0.91474400   -0.87593000
            H                 -0.39317500   -0.91474700    0.87592800
            C                 -1.74853900    0.51173600    0.00000300
            H                 -1.78059900    1.17041100    0.87438300
            H                 -1.78060000    1.17041500   -0.87437400
            H                 -2.95182400   -0.40708300    0.00000200'''
        mol2.basis = '6-31g'
        mol2.charge = 1
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_cycles=200)
        supersystem.init_density()
        supersystem.freeze_and_thaw()

        projector_energy = np.trace(np.dot(subsys.get_dmat()[0], supersystem.proj_pot[0][0]))
        projector_energy += np.trace(np.dot(subsys.get_dmat()[1], supersystem.proj_pot[0][1]))
        projector_energy += np.trace(np.dot(subsys2.env_dmat[0], supersystem.proj_pot[1][0]))
        projector_energy += np.trace(np.dot(subsys2.env_dmat[1], supersystem.proj_pot[1][1]))
        self.assertAlmostEqual(0.0, projector_energy, delta=1e-15)

        mol3 = gto.Mole()
        mol3.atom ='''
            C                  0.74345800    0.66525400    0.00000100
            H                  0.75917800    1.30099000   -0.88423600
            H                  0.75918100    1.30098600    0.88424000
            C                 -0.44163200   -0.26055200    0.00000000
            H                 -0.39317700   -0.91474400   -0.87593000
            H                 -0.39317500   -0.91474700    0.87592800
            C                 -1.74853900    0.51173600    0.00000300
            H                 -1.78059900    1.17041100    0.87438300
            H                 -1.78060000    1.17041500   -0.87437400
            H                 -2.95182400   -0.40708300    0.00000200
        '''
        mol3.basis = '6-31g'
        mol3.spin = 1
        mol3.build()
        mf = dft.ROKS(mol3)
        mf.xc = 'm06'
        grids = supersystem.fs_scf.grids
        mf.grids = grids
        mf.kernel()
        test_e = mf.energy_tot()
        sup_e = supersystem.get_env_in_env_energy()
        self.assertAlmostEqual(test_e, sup_e, delta=1e-10)

        #Supermolecular test.
        #mol = gto.Mole()
        #mol.verbose = 3
        #mol.atom = '''
        #H      1.1851     -0.0039      0.9875
        #C      0.7516     -0.0225     -0.0209
        #H      1.1669      0.8330     -0.5693
        #H      1.1155     -0.9329     -0.5145
        #ghost:C     -0.7516      0.0225      0.0209
        #ghost:H     -1.1669     -0.8334      0.5687
        #ghost:H     -1.1157      0.9326      0.5151
        #ghost:H     -1.1850      0.0044     -0.9875
        #'''
        #mol.basis = 'cc-pVDZ'
        #mol.spin = 1
        #mol.build()
        #env_method = 'rohf'
        #active_method = 'ccsd'
        #subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method)

        #mol2 = gto.Mole()
        #mol2.verbose = 3
        #mol2.atom = '''
        #C     -0.7516      0.0225      0.0209
        #H     -1.1669     -0.8334      0.5687
        #H     -1.1157      0.9326      0.5151
        #H     -1.1850      0.0044     -0.9875
        #ghost:H      1.1851     -0.0039      0.9875
        #ghost:C      0.7516     -0.0225     -0.0209
        #ghost:H      1.1669      0.8330     -0.5693
        #ghost:H      1.1155     -0.9329     -0.5145
        #'''
        #mol2.basis = 'cc-pVDZ'
        #mol2.spin = -1
        #mol2.build()
        #env_method = 'rohf'
        #subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        #supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'hf', ft_conv=1e-10)
        #supersystem.init_density()
        #supersystem.freeze_and_thaw()

        #projector_energy = np.trace(np.dot(subsys.get_dmat()[0], supersystem.proj_pot[0][0]))
        #projector_energy += np.trace(np.dot(subsys.get_dmat()[1], supersystem.proj_pot[0][1]))
        #projector_energy += np.trace(np.dot(subsys2.get_dmat()[0], supersystem.proj_pot[1][0]))
        #projector_energy += np.trace(np.dot(subsys2.get_dmat()[1], supersystem.proj_pot[1][1]))
        #self.assertAlmostEqual(0.0, projector_energy, delta=1e-13)

        #mol3 = gto.Mole()
        #mol3.atom ='''
        #H      1.1851     -0.0039      0.9875
        #C      0.7516     -0.0225     -0.0209
        #H      1.1669      0.8330     -0.5693
        #H      1.1155     -0.9329     -0.5145
        #C     -0.7516      0.0225      0.0209
        #H     -1.1669     -0.8334      0.5687
        #H     -1.1157      0.9326      0.5151
        #H     -1.1850      0.0044     -0.9875
        #'''
        #mol3.basis = 'cc-pVDZ'
        #mol3.build()
        #mf = scf.RHF(mol3)
        ##mf.xc = 'm06'
        ##grids = supersystem.fs_scf.grids
        ##mf.grids = grids
        #mf.kernel()
        #test_e = mf.energy_tot()
        #sup_e = supersystem.get_env_in_env_energy()
        #self.assertAlmostEqual(test_e, sup_e, delta=1e-10)

        #mol = gto.Mole()
        #mol.verbose = 3
        #mol.atom = '''
        #H      1.1851     -0.0039      0.9875
        #C      0.7516     -0.0225     -0.0209
        #H      1.1669      0.8330     -0.5693
        #H      1.1155     -0.9329     -0.5145
        #ghost:C     -0.7516      0.0225      0.0209
        #ghost:H     -1.1669     -0.8334      0.5687
        #ghost:H     -1.1157      0.9326      0.5151
        #ghost:H     -1.1850      0.0044     -0.9875
        #'''
        #mol.basis = '3-21g'
        #mol.spin = 1
        ##mol.charge = -1
        #mol.build()
        #env_method = 'hf'
        #active_method = 'ccsd'
        #subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method, initguess='minao')

        #mol2 = gto.Mole()
        #mol2.verbose = 3
        #mol2.atom = '''
        #C     -0.7516      0.0225      0.0209
        #H     -1.1669     -0.8334      0.5687
        #H     -1.1157      0.9326      0.5151
        #H     -1.1850      0.0044     -0.9875
        #ghost:H      1.1851     -0.0039      0.9875
        #ghost:C      0.7516     -0.0225     -0.0209
        #ghost:H      1.1669      0.8330     -0.5693
        #ghost:H      1.1155     -0.9329     -0.5145
        #'''
        #mol2.basis = '3-21g'
        #mol2.spin = -1
        ##mol2.charge = 1
        #mol2.build()
        #env_method = 'hf'
        #subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, initguess='minao')
        #supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'hf', ft_cycles=100)
        #supersystem.init_density()
        #supersystem.freeze_and_thaw()

        #projector_energy = np.trace(np.dot(subsys.get_dmat()[0], supersystem.proj_pot[0][0]))
        #projector_energy += np.trace(np.dot(subsys.get_dmat()[1], supersystem.proj_pot[0][1]))
        #projector_energy += np.trace(np.dot(subsys2.get_dmat()[0], supersystem.proj_pot[1][0]))
        #projector_energy += np.trace(np.dot(subsys2.get_dmat()[1], supersystem.proj_pot[1][1]))
        #self.assertAlmostEqual(0.0, projector_energy, delta=1e-13)

        #mol3 = gto.Mole()
        #mol3.atom ='''
        #H      1.1851     -0.0039      0.9875
        #C      0.7516     -0.0225     -0.0209
        #H      1.1669      0.8330     -0.5693
        #H      1.1155     -0.9329     -0.5145
        #C     -0.7516      0.0225      0.0209
        #H     -1.1669     -0.8334      0.5687
        #H     -1.1157      0.9326      0.5151
        #H     -1.1850      0.0044     -0.9875
        #'''
        #mol3.basis = '3-21g'
        #mol3.build()
        #mf = scf.ROHF(mol3)
        ##mf.xc = 'm06'
        ##grids = supersystem.fs_scf.grids
        ##mf.grids = grids
        #mf.kernel()
        #test_e = mf.energy_tot()
        #sup_e = supersystem.get_env_in_env_energy()
        #self.assertAlmostEqual(test_e, sup_e, delta=1e-10)

        ##Long distance test.
        #mol = gto.Mole()
        #mol.verbose = 3
        #mol.atom = '''
        #O 0. 0. 0.
        #O 1.13 0. 0.
        #'''
        #mol.basis = 'cc-pVDZ'
        #mol.spin = 2
        #mol.build()
        #env_method = 'rohf'
        #active_method = 'ccsd'
        #subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method)

        #mol2 = gto.Mole()
        #mol2.verbose = 3
        #mol2.atom = '''
        #O 0. 100.0 0.0
        #O 1.13 100.0 0.0
        #'''
        #mol2.basis = 'cc-pVDZ'
        #mol2.spin = -2
        #mol2.build()
        #env_method = 'rohf'
        #subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        #supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'hf', ft_initguess='supmol')
        #supersystem.init_density()
        #supersystem.freeze_and_thaw()

        #projector_energy = np.trace(np.dot(subsys.get_dmat()[0], supersystem.proj_pot[0][0]))
        #projector_energy += np.trace(np.dot(subsys.get_dmat()[1], supersystem.proj_pot[0][1]))
        #projector_energy += np.trace(np.dot(subsys2.get_dmat()[0], supersystem.proj_pot[1][0]))
        #projector_energy += np.trace(np.dot(subsys2.get_dmat()[1], supersystem.proj_pot[1][1]))
        #self.assertAlmostEqual(0.0, projector_energy, delta=1e-14)
        #self.assertTrue(False)

        #mol3 = gto.Mole()
        #mol3.verbose = 3
        #mol3.atom ='''
        #Li 0. 0. 0.
        #Li 12.22 0.0 0.0
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
        #self.assertTrue(np.allclose(test_dmat, (supersystem.dmat[0] + supersystem.dmat[1]), atol=1e-6))
        #self.assertAlmostEqual(test_e, sup_e, delta=1e-10)

        #Projection energy
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
        #subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, active_method)

        #mol2 = gto.Mole()
        #mol2.verbose = 3
        #mol2.atom = '''
        #Li 1.595 0.0 0.0
        #'''
        #mol2.basis = '3-21g'
        #mol2.spin = -1
        #mol2.build()
        #env_method = 'm06'
        #subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        #supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='supmol', fs_unrestricted=True, ft_cycles=10)

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

    @unittest.skip
    def test_get_supersystem_nuc_grad(self):
        #Closed Shell
        t_file = tempfile.NamedTemporaryFile()
        mol = gto.Mole()
        #mol.verbose = 4
        mol.atom = '''
        H 0.758602  0.000000  0.504284
        H 0.758602  0.000000  -0.504284 
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'b3lyp'
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method, filename=t_file.name)

        mol2 = gto.Mole()
        #mol2.verbose = 4
        mol2.atom = '''
        O 0.0 0.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'b3lyp'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, filename=t_file.name)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', filename=t_file.name)
        supsystem_e = supersystem.get_supersystem_energy()
        supsystem_grad = supersystem.get_supersystem_nuc_grad()

        mol3 = gto.Mole()
        #mol3.verbose = 4
        mol3.atom = '''
        H 0.758602  0.000000  0.504284
        H 0.758602  0.000000  -0.504284 
        O 0.0 0.0 0.0
        '''
        mol3.basis = '3-21g'
        mol3.build()
        test_scf = dft.RKS(mol3)
        test_scf.xc = 'b3lyp'
        test_scf.conv_tol = supersystem.fs_conv
        grids = dft.gen_grid.Grids(mol3)
        grids.level = supersystem.grid_level
        grids.build()
        test_scf.grids = grids
        test_e = test_scf.kernel()
        test_grad = test_scf.nuc_grad_method()
        test_grad.kernel()
        #self.assertAlmostEqual(test_grad.grad(), supsystem_grad.grad())

    @unittest.skip
    def test_get_subsystem_nuc_grad(self):
        #Closed Shell
        mol = gto.Mole()
        #mol.verbose = 4
        mol.atom = '''
        O 0.0 0.0 0.0
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'b3lyp'
        hl_method = 'rhf'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method)

        mol2 = gto.Mole()
        #mol2.verbose = 4
        mol2.atom = '''
        O 1.208 0.0 0.0
        '''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'b3lyp'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', ft_cycles=50)
        supsystem_e = supersystem.get_supersystem_energy()
        supersystem.freeze_and_thaw()
        subsystem_grad = subsys.get_env_nuc_grad()
        subsys.hl_in_env_energy()
        subsys.get_hl_nuc_grad()
        supersystem_grad = supersystem.get_embedding_nuc_grad()

        mol3 = gto.Mole()
        #mol3.verbose = 4
        mol3.atom = '''
        H 0.758602  0.000000  0.504284
        H 0.758602  0.000000  -0.504284 
        O 0.0 0.0 0.0
        H 0.758602  30.000000  0.504284
        H 0.758602  30.000000  -0.504284 
        O 0.0 30.0 0.0
        '''
        mol3.basis = '3-21g'
        mol3.build()
        test_scf = dft.RKS(mol3)
        test_scf.xc = 'b3lyp'
        test_scf.conv_tol = supersystem.fs_conv
        grids = dft.gen_grid.Grids(mol3)
        grids.level = supersystem.grid_level
        grids.build()
        test_scf.grids = grids
        test_e = test_scf.kernel()
        test_grad = test_scf.nuc_grad_method()
        elec_grad = test_grad.grad_elec(test_scf.mo_energy, test_scf.mo_coeff, test_scf.mo_occ)
        #print (elec_grad)
        #self.assertAlmostEqual(test_grad.grad(), supsystem_grad.grad())

    @unittest.skip
    def test_correct_env_energy(self):
        pass

    @unittest.skip
    def test_correct_hl_energy(self):
        pass

    @unittest.skip
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
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method)

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
        supersystem.init_density()

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
                FDS[0] = np.dot( FAB[0], np.dot( supersystem.subsystems[B].env_dmat[0], SBA ))
                FDS[1] = np.dot( FAB[1], np.dot( supersystem.subsystems[B].env_dmat[1], SBA ))
                POp[0] += -1. * ( FDS[0] + FDS[0].transpose() ) 
                POp[1] += -1. * ( FDS[0] + FDS[0].transpose() )

            sub_pop_list[i] = POp

        self.assertTrue(np.allclose(sub_pop_list[0][0], supersystem.proj_pot[0][0]))
        self.assertTrue(np.allclose(sub_pop_list[0][1], supersystem.proj_pot[0][1]))
        self.assertTrue(np.allclose(sub_pop_list[1][0], supersystem.proj_pot[1][0]))
        self.assertTrue(np.allclose(sub_pop_list[1][1], supersystem.proj_pot[1][1]))

if __name__ == "__main__":
    unittest.main()
