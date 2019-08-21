#A module to test system inits.
# Daniel Graham


import unittest
import os
import shutil
import re

from copy import copy

from qsome import cluster_subsystem, cluster_supersystem
from pyscf import gto, lib, scf, dft

import numpy as np

class TestEnvSubsystem(unittest.TestCase):

    def test_basic_subsystem(self):
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

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'm06')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.env_smearsigma, 0)
        self.assertEqual(subsys.env_damp, 0)
        self.assertEqual(subsys.env_shift, 0)
        self.assertEqual(subsys.env_subcycles, 1)
        self.assertEqual(subsys.freeze, False)
        self.assertEqual(subsys.env_initguess, None)
        self.assertEqual(subsys.verbose, 3)

        #Check SCF object
        scf_obj = subsys.env_scf
        comp_scf_obj = dft.RKS(mol)
        self.assertEqual(type(scf_obj), type(comp_scf_obj))
        self.assertEqual(scf_obj.xc, 'm06')

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat))

    def test_unrestricted_subsystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, unrestricted=True)

        #Check SCF object
        scf_obj = subsys.env_scf
        comp_scf_obj = dft.UKS(mol)
        self.assertEqual(type(scf_obj), type(comp_scf_obj))
        self.assertEqual(scf_obj.xc, 'm06')

        #Check density
        self.assertEqual(subsys.env_method, 'm06')
        init_dmat = scf.uhf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat))

    def test_restrictedos_subsystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.spin = -2
        mol.build()
        env_method = 'm06'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method)

        #Check SCF object
        scf_obj = subsys.env_scf
        comp_scf_obj = dft.ROKS(mol)
        self.assertEqual(type(scf_obj), type(comp_scf_obj))
        self.assertEqual(scf_obj.xc, 'm06')

        #Check density
        self.assertEqual(subsys.env_method, 'm06')
        init_dmat = scf.rohf.init_guess_by_minao(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat))

    def test_custom_subsystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'b3lyp'

        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, 
            env_order=2, env_smearsigma=0.5, conv=1e-4, damp=1, shift=1, 
            subcycles=10, setfermi=1.0, diis=2, unrestricted=False,
            density_fitting=True, freeze=True, verbose=2, nproc=4, pmem=300,
            scrdir='/path/to/scratch/', save_orbs=True, save_density=True)

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'b3lyp')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.env_smearsigma, 0.5)
        self.assertEqual(subsys.env_damp, 1)
        self.assertEqual(subsys.env_shift, 1)
        self.assertEqual(subsys.env_subcycles, 10)
        self.assertEqual(subsys.freeze, True)
        self.assertEqual(subsys.env_initguess, None)
        self.assertEqual(subsys.verbose, 2)
        self.assertEqual(subsys.save_orbs, True)
        self.assertEqual(subsys.save_density, True)

        #Check SCF object
        scf_obj = subsys.env_scf
        comp_scf_obj = dft.RKS(mol)
        self.assertEqual(type(scf_obj), type(comp_scf_obj))
        self.assertEqual(scf_obj.xc, 'b3lyp')

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat))

    def test_ghost_subsystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost.H 0. 0.0 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method)

        self.assertEqual(subsys.mol.atom_coords()[3][0],0.)
        self.assertEqual(subsys.mol.atom_coords()[3][1],0.)
        self.assertAlmostEqual(subsys.mol.atom_coords()[3][2],5.39894754, delta=1e-8)

        self.assertEqual(subsys.mol.atom_charges()[0],8)
        self.assertEqual(subsys.mol.atom_charges()[1],1)
        self.assertEqual(subsys.mol.atom_charges()[2],1)
        self.assertEqual(subsys.mol.atom_charges()[3],0)

        self.assertEqual(subsys.mol._basis['ghost.H'],subsys.mol._basis['H'])

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat))

class TestActiveSubSystem(unittest.TestCase):

    def test_custom_obj_def(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'
        hl_method = 'ccsd'

        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method)

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'm06')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.env_smearsigma, 0)
        self.assertEqual(subsys.env_damp, 0)
        self.assertEqual(subsys.env_shift, 0)
        self.assertEqual(subsys.env_subcycles, 1)
        self.assertEqual(subsys.freeze, False)
        self.assertEqual(subsys.env_initguess, None)
        self.assertEqual(subsys.verbose, 3)

        self.assertEqual(subsys.hl_method, 'ccsd')
        self.assertEqual(subsys.cas_loc_orbs, False)
        self.assertEqual(subsys.cas_active_orbs, None)
        self.assertEqual(subsys.hl_conv, 1e-9)
        self.assertEqual(subsys.hl_grad, None)
        self.assertEqual(subsys.hl_cycles, 100)
        self.assertEqual(subsys.hl_damp, 0)
        self.assertEqual(subsys.hl_shift, 0)
        self.assertEqual(subsys.hl_initguess, None)

    def test_custom_obj_set(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'
        hl_method  = 'mp2'

        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, 
            hl_method, cas_loc_orbs=True, cas_active_orbs=[2,3,4,5],
            hl_conv=1e-9, hl_grad=1e-8, hl_cycles=2, hl_damp=0.1,
            hl_shift=0.001, hl_initguess='minao', env_smearsigma=0.5, damp=1,
            shift=1, subcycles=10, freeze=True, initguess='supmol',
            verbose=2, save_orbs=True, save_density=True, 
            hl_save_orbs=True, hl_save_density=True)

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'm06')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.env_smearsigma, 0.5)
        self.assertEqual(subsys.env_damp, 1)
        self.assertEqual(subsys.env_shift, 1)
        self.assertEqual(subsys.env_subcycles, 10)
        self.assertEqual(subsys.freeze, True)
        self.assertEqual(subsys.env_initguess, 'supmol')
        self.assertEqual(subsys.verbose, 2)
        self.assertEqual(subsys.save_orbs, True)
        self.assertEqual(subsys.save_density, True)

        # hl methods
        self.assertEqual(subsys.hl_method, 'mp2')
        self.assertEqual(subsys.cas_loc_orbs, True)
        self.assertEqual(subsys.cas_active_orbs, [2,3,4,5])
        self.assertEqual(subsys.hl_conv, 1e-9)
        self.assertEqual(subsys.hl_grad, 1e-8)
        self.assertEqual(subsys.hl_cycles, 2)
        self.assertEqual(subsys.hl_damp, 0.1)
        self.assertEqual(subsys.hl_shift, -1.001)
        self.assertEqual(subsys.hl_initguess, 'minao')
        self.assertEqual(subsys.hl_save_orbs, True)
        self.assertEqual(subsys.hl_save_density, True)


class TestExcitedSubSystem(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
        
class TestSuperSystem(unittest.TestCase):

    def test_basic_supersystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method)

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
        self.assertEqual(supersystem.fs_method, 'b3lyp')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_damp, 0)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.fs_cycles, None)
        self.assertEqual(supersystem.fs_conv, 1e-9)
        self.assertEqual(supersystem.fs_grad, None)
        self.assertEqual(supersystem.fs_damp, 0)
        self.assertEqual(supersystem.fs_shift, 0)
        self.assertEqual(supersystem.fs_smearsigma, 0)
        self.assertEqual(supersystem.fs_initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-7)
        self.assertEqual(supersystem.fs_verbose, 3)
        #self.assertEqual(supersystem.analysis, False)
        #self.assertEqual(supersystem.debug, False)

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat))

        init_dmat = scf.get_init_guess(mol2)
        self.assertTrue(np.array_equal(init_dmat, subsys2.dmat))

        init_dmat = scf.get_init_guess(gto.mole.conc_mol(mol, mol2))
        self.assertTrue(np.allclose(init_dmat, supersystem.dmat[0] + supersystem.dmat[1]))
        

    def test_custom_supersystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'm06'
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 1.0 0.0 0.0
        O 3.0 0.0 0.0'''
        mol2.basis = 'aug-cc-pVDZ'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2],
                          'b3lyp', ft_proj_oper='huzfermi', ft_cycles=2, 
                          ft_conv=1e-1, ft_grad=1e-4, ft_diis=3, 
                          ft_initguess='1e', ft_updatefock=1,
                          ft_damp=0.5, fs_cycles=3, fs_conv=2, fs_grad=4, 
                          fs_damp=1, fs_shift=2.1, fs_smearsigma=0.1, 
                          fs_initguess='atom', fs_grid_level=2, fs_rhocutoff=1e-2,
                          fs_verbose=1, fs_save_orbs=True, fs_save_density=True, 
                          compare_density=True, ft_save_density=True, 
                          ft_save_orbs=True)

        self.assertEqual(supersystem.fs_method, 'b3lyp')
        self.assertEqual(supersystem.proj_oper, 'huzfermi')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 2)
        self.assertEqual(supersystem.ft_conv, 1e-1)
        self.assertEqual(supersystem.ft_grad, 1e-4)
        self.assertEqual(supersystem.ft_damp, 0.5)
        self.assertIsInstance(supersystem.ft_diis[0], lib.diis.DIIS)
        self.assertIsInstance(supersystem.ft_diis[1], lib.diis.DIIS)
        self.assertEqual(supersystem.ft_setfermi, -0.1)
        self.assertEqual(supersystem.ft_initguess, '1e')
        self.assertEqual(supersystem.ft_updatefock, 1)

        self.assertEqual(supersystem.fs_cycles, 3)
        self.assertEqual(supersystem.fs_conv, 2)
        self.assertEqual(supersystem.fs_grad, 4)
        self.assertEqual(supersystem.fs_damp, 1)
        self.assertEqual(supersystem.fs_shift, 2.1)
        self.assertEqual(supersystem.fs_smearsigma, 0.1)
        self.assertEqual(supersystem.fs_initguess, 'atom')
        self.assertEqual(supersystem.grid_level, 2)
        self.assertEqual(supersystem.rho_cutoff, 1e-2)
        self.assertEqual(supersystem.verbose, 1)
        self.assertEqual(supersystem.analysis, True)
        self.assertEqual(supersystem.debug, True)
        self.assertEqual(supersystem.compare_density, True)
        self.assertEqual(supersystem.fs_save_orbs, True)
        self.assertEqual(supersystem.fs_save_density, True)
        self.assertEqual(supersystem.ft_save_orbs, True)
        self.assertEqual(supersystem.ft_save_density, True)

        #Check density
        init_dmat = scf.get_init_guess(mol, '1e')
        self.assertTrue(np.allclose(init_dmat, subsys.dmat[0] + subsys.dmat[1]))

        init_dmat = scf.get_init_guess(mol2, '1e')
        self.assertTrue(np.allclose(init_dmat, subsys2.dmat[0] + subsys2.dmat[1]))

        init_dmat2 = scf.hf.init_guess_by_atom(gto.mole.conc_mol(mol, mol2))
        self.assertTrue(np.allclose(init_dmat2, supersystem.dmat[0] + supersystem.dmat[1]))

    def test_readchk_supersystem(self):
        #Delete the saved check file
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        He 1.0 20.0 0.0
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='readchk')
        self.assertEqual(supersystem.fs_method, 'm06')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_initguess, 'readchk')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.fs_conv, 1e-9)
        self.assertEqual(supersystem.fs_grad, None)
        self.assertEqual(supersystem.fs_damp, 0)
        self.assertEqual(supersystem.fs_shift, 0)
        self.assertEqual(supersystem.fs_smearsigma, 0)
        self.assertEqual(supersystem.fs_initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-7)
        self.assertEqual(supersystem.fs_verbose, 3)
        #self.assertEqual(supersystem.analysis, False)

        #Check density
        old_sup_dmat = copy(supersystem.dmat)
        old_sub1_dmat = copy(subsys.dmat)
        old_sub2_dmat = copy(subsys2.dmat)

        #This isn't working because the save doesn't have mo_coeffs or mo_orbitals to store. Only a density fragment. 
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='readchk')
        self.assertTrue(np.allclose(old_sup_dmat, supersystem.dmat))
        self.assertTrue(np.allclose(old_sub1_dmat, subsys.dmat))
        self.assertTrue(np.allclose(old_sub2_dmat, subsys2.dmat))

    def test_partialghost_supersystem(self):
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
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')
        self.assertEqual(supersystem.fs_method, 'b3lyp')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.fs_cycles, None)
        self.assertEqual(supersystem.fs_conv, 1e-9)
        self.assertEqual(supersystem.fs_grad, None)
        self.assertEqual(supersystem.fs_damp, 0)
        self.assertEqual(supersystem.fs_shift, 0)
        self.assertEqual(supersystem.fs_smearsigma, 0)
        self.assertEqual(supersystem.fs_initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-7)
        self.assertEqual(supersystem.fs_verbose, 3)
        #self.assertEqual(supersystem.analysis, False)
        #self.assertEqual(supersystem.debug, False)

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat))

        init_dmat = scf.get_init_guess(mol2)
        self.assertTrue(np.array_equal(init_dmat, subsys2.dmat))

        #Check concat mols
        self.assertAlmostEqual(supersystem.mol.atom_coords()[2][0], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[2][1], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[2][2], 5.39894754, delta=1e-8)
        self.assertEqual(len(supersystem.mol.atom_coords()), 5)
        self.assertEqual(supersystem.mol._basis['ghost:H'], supersystem.mol._basis['H'])

    def test_totalghost_supersystem(self):
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
        ghost:He 2.0 20.0 0.0
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')
        self.assertEqual(supersystem.fs_method, 'b3lyp')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.fs_conv, 1e-9)
        self.assertEqual(supersystem.fs_grad, None)
        self.assertEqual(supersystem.fs_damp, 0)
        self.assertEqual(supersystem.fs_shift, 0)
        self.assertEqual(supersystem.fs_smearsigma, 0)
        self.assertEqual(supersystem.fs_initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-7)
        self.assertEqual(supersystem.fs_verbose, 3)
        #self.assertEqual(supersystem.analysis, False)
        #self.assertEqual(supersystem.debug, False)

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat))

        init_dmat = scf.get_init_guess(mol2)
        self.assertTrue(np.array_equal(init_dmat, subsys2.dmat))

        #Check concat mols
        self.assertAlmostEqual(supersystem.mol.atom_coords()[2][0], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[2][1], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[2][2], 5.39894754, delta=1e-8)
        self.assertEqual(supersystem.mol._basis['ghost:H'], supersystem.mol._basis['H'])
        self.assertAlmostEqual(supersystem.mol.atom_coords()[3][0], 3.77945225, delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[3][1], 37.79452249, delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[3][2], 0.0, delta=1e-8)
        self.assertEqual(len(supersystem.mol.atom_coords()), 6)
        self.assertEqual(supersystem.mol._basis['ghost:He-1'], supersystem.mol._basis['He-1'])

        
    def test_ovlpghost_supersystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost:He 1.0 20.0 0.0
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
        ghost:H 0. 2.757 2.857
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao')
        self.assertEqual(supersystem.fs_method, 'b3lyp')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.fs_cycles, None)
        self.assertEqual(supersystem.fs_conv, 1e-9)
        self.assertEqual(supersystem.fs_grad, None)
        self.assertEqual(supersystem.fs_damp, 0)
        self.assertEqual(supersystem.fs_shift, 0)
        self.assertEqual(supersystem.fs_smearsigma, 0)
        self.assertEqual(supersystem.fs_initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-7)
        self.assertEqual(supersystem.fs_verbose, 3)
        #self.assertEqual(supersystem.analysis, False)
        #self.assertEqual(supersystem.debug, False)

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat))

        init_dmat = scf.get_init_guess(mol2)
        self.assertTrue(np.array_equal(init_dmat, subsys2.dmat))

        #Check concat mols
        self.assertAlmostEqual(supersystem.mol.atom_coords()[2][0], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[2][1], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[2][2], 5.39894754, delta=1e-8)
        self.assertEqual(len(supersystem.mol.atom_coords()), 5)
        self.assertEqual(supersystem.mol._basis['ghost:H'], supersystem.mol._basis['H'])

if __name__ == "__main__":
    unittest.main()
