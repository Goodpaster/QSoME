#A module to test system inits.
# Daniel Graham


import unittest
import os
import shutil
import re

from copy import copy

from qsome import cluster_subsystem, cluster_supersystem
from pyscf import gto, lib, scf

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

        self.assertEqual(subsys.smearsigma, 0)
        self.assertEqual(subsys.damp, 0)
        self.assertEqual(subsys.shift, 0)
        self.assertEqual(subsys.subcycles, 1)
        self.assertEqual(subsys.freeze, False)
        self.assertEqual(subsys.initguess, None)
        self.assertEqual(subsys.grid_level, 4)
        self.assertEqual(subsys.rho_cutoff, 1e-7)
        self.assertEqual(subsys.verbose, 3)
        self.assertEqual(subsys.analysis, False)
        self.assertEqual(subsys.debug, False)

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat[0] + subsys.dmat[1]))

    def test_unrestricted_subsystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'um06'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method)

        self.assertEqual(subsys.env_method, 'um06')
        init_dmat = scf.uhf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat))

    def test_closed_subsystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        env_method = 'rom06'
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method)

        self.assertEqual(subsys.env_method, 'rom06')
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
        env_method = 'm06'

        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, env_method, 
            smearsigma=0.5, damp=1, shift=1, subcycles=10, freeze=True, 
            initguess='supmol', grid_level=1, rhocutoff=1e-1, verbose=2, 
            analysis=True, debug=True)

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'm06')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.smearsigma, 0.5)
        self.assertEqual(subsys.damp, 1)
        self.assertEqual(subsys.shift, 1)
        self.assertEqual(subsys.subcycles, 10)
        self.assertEqual(subsys.freeze, True)
        self.assertEqual(subsys.initguess, 'supmol')
        self.assertEqual(subsys.grid_level, 1)
        self.assertEqual(subsys.rho_cutoff, 1e-1)
        self.assertEqual(subsys.verbose, 2)
        self.assertEqual(subsys.analysis, True)
        self.assertEqual(subsys.debug, True)

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat[0] + subsys.dmat[1]))

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

        self.assertEqual(subsys.mol.basis['ghost.H'],subsys.mol.basis['H'])

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat[0] + subsys.dmat[1]))

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
        active_method = 'ccsd'

        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'm06')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.smearsigma, 0)
        self.assertEqual(subsys.damp, 0)
        self.assertEqual(subsys.shift, 0)
        self.assertEqual(subsys.subcycles, 1)
        self.assertEqual(subsys.freeze, False)
        self.assertEqual(subsys.initguess, None)
        self.assertEqual(subsys.grid_level, 4)
        self.assertEqual(subsys.rho_cutoff, 1e-7)
        self.assertEqual(subsys.verbose, 3)
        self.assertEqual(subsys.analysis, False)
        self.assertEqual(subsys.debug, False)

        # active methods
        self.assertEqual(subsys.active_method, 'ccsd')
        self.assertEqual(subsys.localize_orbitals, False)
        self.assertEqual(subsys.active_orbs, None)
        self.assertEqual(subsys.active_conv, 1e-9)
        self.assertEqual(subsys.active_grad, None)
        self.assertEqual(subsys.active_cycles, 100)
        self.assertEqual(subsys.active_damp, 0)
        self.assertEqual(subsys.active_shift, 0)

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
        active_method  = 'mp2'

        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, 
            active_method, localize_orbitals=True, active_orbs=[2,3,4,5],
            active_conv=1e-9, active_grad=1e-8, active_cycles=2, 
            active_damp=0.1, active_shift=0.001, smearsigma=0.5, damp=1, 
            shift=1, subcycles=10, freeze=True, initguess='supmol', grid_level=1, 
            rhocutoff=1e-3, verbose=2, analysis=True, debug=True)

        self.assertEqual(subsys.mol, mol)
        self.assertEqual(subsys.env_method, 'm06')
        self.assertEqual(subsys.filename, os.getcwd() + '/temp.inp')

        self.assertEqual(subsys.smearsigma, 0.5)
        self.assertEqual(subsys.damp, 1)
        self.assertEqual(subsys.shift, 1)
        self.assertEqual(subsys.subcycles, 10)
        self.assertEqual(subsys.freeze, True)
        self.assertEqual(subsys.initguess, 'supmol')
        self.assertEqual(subsys.grid_level, 1)
        self.assertEqual(subsys.rho_cutoff, 1e-3)
        self.assertEqual(subsys.verbose, 2)
        self.assertEqual(subsys.analysis, True)

        # active methods
        self.assertEqual(subsys.active_method, 'mp2')
        self.assertEqual(subsys.localize_orbitals, True)
        self.assertEqual(subsys.active_orbs, [2,3,4,5])
        self.assertEqual(subsys.active_conv, 1e-9)
        self.assertEqual(subsys.active_grad, 1e-8)
        self.assertEqual(subsys.active_cycles, 2)
        self.assertEqual(subsys.active_damp, 0.1)
        self.assertEqual(subsys.active_shift, 0.001)


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

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'rb3lyp', ft_initguess='minao')
        self.assertEqual(supersystem.ct_method, 'rb3lyp')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_setfermi, None)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.cycles, 100)
        self.assertEqual(supersystem.conv, 1e-9)
        self.assertEqual(supersystem.grad, None)
        self.assertEqual(supersystem.damp, 0)
        self.assertEqual(supersystem.shift, 0)
        self.assertEqual(supersystem.smearsigma, 0)
        self.assertEqual(supersystem.initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-7)
        self.assertEqual(supersystem.verbose, 3)
        self.assertEqual(supersystem.analysis, False)
        self.assertEqual(supersystem.debug, False)

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat[0] + subsys.dmat[1]))

        init_dmat = scf.get_init_guess(mol2)
        self.assertTrue(np.array_equal(init_dmat, subsys2.dmat[0] + subsys2.dmat[1]))

        init_dmat = scf.get_init_guess(gto.mole.conc_mol(mol, mol2))
        self.assertTrue(np.array_equal(init_dmat, supersystem.dmat[0] + supersystem.dmat[1]))
        

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
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

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
                          'rb3lyp', proj_oper='huzfermi', ft_cycles=2, 
                          ft_conv=1e-1, ft_grad=1e-4, ft_diis=3, 
                          ft_setfermi=-0.1, ft_initguess='1e', ft_updatefock=1, 
                          cycles=3, conv=2, grad=4, damp=1, shift=2.1, 
                          smearsigma=0.1, initguess='atom', 
                          grid_level=2, rhocutoff=1e-2, verbose=1, 
                          analysis=True, debug=True)

        self.assertEqual(supersystem.ct_method, 'rb3lyp')
        self.assertEqual(supersystem.proj_oper, 'huzfermi')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 2)
        self.assertEqual(supersystem.ft_conv, 1e-1)
        self.assertEqual(supersystem.ft_grad, 1e-4)
        self.assertIsInstance(supersystem.ft_diis[0], lib.diis.DIIS)
        self.assertIsInstance(supersystem.ft_diis[1], lib.diis.DIIS)
        self.assertEqual(supersystem.ft_setfermi, -0.1)
        self.assertEqual(supersystem.ft_initguess, '1e')
        self.assertEqual(supersystem.ft_updatefock, 1)

        self.assertEqual(supersystem.cycles, 3)
        self.assertEqual(supersystem.conv, 2)
        self.assertEqual(supersystem.grad, 4)
        self.assertEqual(supersystem.damp, 1)
        self.assertEqual(supersystem.shift, 2.1)
        self.assertEqual(supersystem.smearsigma, 0.1)
        self.assertEqual(supersystem.initguess, 'atom')
        self.assertEqual(supersystem.grid_level, 2)
        self.assertEqual(supersystem.rho_cutoff, 1e-2)
        self.assertEqual(supersystem.verbose, 1)
        self.assertEqual(supersystem.analysis, True)
        self.assertEqual(supersystem.debug, True)

        #Check density
        init_dmat = scf.get_init_guess(mol, '1e')
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat[0] + subsys.dmat[1]))

        init_dmat = scf.get_init_guess(mol2, '1e')
        self.assertTrue(np.array_equal(init_dmat, subsys2.dmat[0] + subsys2.dmat[1]))

        init_dmat = scf.hf.init_guess_by_atom(gto.mole.conc_mol(mol, mol2))
        self.assertTrue(np.array_equal(init_dmat, supersystem.dmat[0] + supersystem.dmat[1]))

    def test_readchk_supersystem(self):
        #Delete the saved check file
        os.system('rm temp.hdf5')
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        He 1.0 20.0 0.0
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='readchk')
        self.assertEqual(supersystem.ct_method, 'm06')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_setfermi, None)
        self.assertEqual(supersystem.ft_initguess, 'readchk')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.cycles, 100)
        self.assertEqual(supersystem.conv, 1e-9)
        self.assertEqual(supersystem.grad, None)
        self.assertEqual(supersystem.damp, 0)
        self.assertEqual(supersystem.shift, 0)
        self.assertEqual(supersystem.smearsigma, 0)
        self.assertEqual(supersystem.initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-7)
        self.assertEqual(supersystem.verbose, 3)
        self.assertEqual(supersystem.analysis, False)
        self.assertEqual(supersystem.debug, False)

        #Check density
        old_sup_dmat = copy(supersystem.dmat)
        old_sub1_dmat = copy(subsys.dmat)
        old_sub2_dmat = copy(subsys2.dmat)

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
        ghost.H 0. 0. 2.857
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

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'rb3lyp', ft_initguess='minao')
        self.assertEqual(supersystem.ct_method, 'rb3lyp')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_setfermi, None)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.cycles, 100)
        self.assertEqual(supersystem.conv, 1e-9)
        self.assertEqual(supersystem.grad, None)
        self.assertEqual(supersystem.damp, 0)
        self.assertEqual(supersystem.shift, 0)
        self.assertEqual(supersystem.smearsigma, 0)
        self.assertEqual(supersystem.initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-7)
        self.assertEqual(supersystem.verbose, 3)
        self.assertEqual(supersystem.analysis, False)
        self.assertEqual(supersystem.debug, False)

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat[0] + subsys.dmat[1]))

        init_dmat = scf.get_init_guess(mol2)
        self.assertTrue(np.array_equal(init_dmat, subsys2.dmat[0] + subsys2.dmat[1]))

        #Check concat mols
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-1][0], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-1][1], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-1][2], 5.39894754, delta=1e-8)
        self.assertEqual(len(supersystem.mol.atom_coords()), 5)
        self.assertEqual(supersystem.mol.basis['ghost.H:1'], supersystem.mol.basis['H'])

    def test_totalghost_supersystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost.H 0. 0. 2.857
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        ghost.He 2.0 20.0 0.0
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'rb3lyp', ft_initguess='minao')
        self.assertEqual(supersystem.ct_method, 'rb3lyp')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_setfermi, None)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.cycles, 100)
        self.assertEqual(supersystem.conv, 1e-9)
        self.assertEqual(supersystem.grad, None)
        self.assertEqual(supersystem.damp, 0)
        self.assertEqual(supersystem.shift, 0)
        self.assertEqual(supersystem.smearsigma, 0)
        self.assertEqual(supersystem.initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-7)
        self.assertEqual(supersystem.verbose, 3)
        self.assertEqual(supersystem.analysis, False)
        self.assertEqual(supersystem.debug, False)

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat[0] + subsys.dmat[1]))

        init_dmat = scf.get_init_guess(mol2)
        self.assertTrue(np.array_equal(init_dmat, subsys2.dmat[0] + subsys2.dmat[1]))

        #Check concat mols
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-2][0], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-2][1], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-2][2], 5.39894754, delta=1e-8)
        self.assertEqual(supersystem.mol.basis['ghost.H:1'], supersystem.mol.basis['H'])
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-1][0], 3.77945225, delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-1][1], 37.79452249, delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-1][2], 0.0, delta=1e-8)
        self.assertEqual(len(supersystem.mol.atom_coords()), 6)
        self.assertEqual(supersystem.mol.basis['ghost.He:2'], supersystem.mol.basis['He:1'])

        
    def test_ovlpghost_supersystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        ghost.He 1.0 20.0 0.0
        ghost.H 0. 0. 2.857
        '''
        mol.basis = '3-21g'
        mol.build()
        env_method = 'm06'
        active_method = 'ccsd'
        subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        ghost.H 0. 2.757 2.857
        He 1.0 20.0 0.0
        He 3.0 20.0 0.0'''
        mol2.basis = '3-21g'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'rb3lyp', ft_initguess='minao')
        self.assertEqual(supersystem.ct_method, 'rb3lyp')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.filename, os.getcwd()+"/temp.inp")
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_setfermi, None)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.cycles, 100)
        self.assertEqual(supersystem.conv, 1e-9)
        self.assertEqual(supersystem.grad, None)
        self.assertEqual(supersystem.damp, 0)
        self.assertEqual(supersystem.shift, 0)
        self.assertEqual(supersystem.smearsigma, 0)
        self.assertEqual(supersystem.initguess, None)
        self.assertEqual(supersystem.grid_level, 4)
        self.assertEqual(supersystem.rho_cutoff, 1e-7)
        self.assertEqual(supersystem.verbose, 3)
        self.assertEqual(supersystem.analysis, False)
        self.assertEqual(supersystem.debug, False)

        #Check density
        init_dmat = scf.get_init_guess(mol)
        self.assertTrue(np.array_equal(init_dmat, subsys.dmat[0] + subsys.dmat[1]))

        init_dmat = scf.get_init_guess(mol2)
        self.assertTrue(np.array_equal(init_dmat, subsys2.dmat[0] + subsys2.dmat[1]))

        #Check concat mols
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-1][0], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-1][1], 0., delta=1e-8)
        self.assertAlmostEqual(supersystem.mol.atom_coords()[-1][2], 5.39894754, delta=1e-8)
        self.assertEqual(len(supersystem.mol.atom_coords()), 5)
        self.assertEqual(supersystem.mol.basis['ghost.H:2'], supersystem.mol.basis['H'])

if __name__ == "__main__":
    unittest.main()
