#A module to test system inits.
# Daniel Graham


import unittest
import os
import shutil
import re

from copy import copy

from qsome import cluster_subsystem, cluster_supersystem, interaction_mediator
from pyscf import gto, lib, scf, dft

import numpy as np

import tempfile

class TestEnvSubsystem(unittest.TestCase):

    def setUp(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. 0.7586 0.5043
        H 0. 0.7586 -0.5043'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        self.cs_mol = mol

        self.env_method = 'm06'
        self.env_method2 = 'b3lyp'

    def test_basic_subsystem(self):

        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method)

        self.assertEqual(subsys.mol, self.cs_mol)
        self.assertEqual(subsys.env_method, self.env_method)
        self.assertEqual(subsys.env_order, 1)

        self.assertEqual(subsys.env_smearsigma, 0.)
        self.assertEqual(subsys.env_damp, 0.)
        self.assertEqual(subsys.env_shift, 0.)
        self.assertEqual(subsys.env_subcycles, 1)
        self.assertEqual(subsys.freeze, False)
        self.assertEqual(subsys.env_initguess, None)
        self.assertEqual(subsys.verbose, 3)

        #Check SCF object
        scf_obj = subsys.env_scf
        comp_scf_obj = dft.RKS(self.cs_mol)
        self.assertEqual(type(scf_obj), type(comp_scf_obj))
        self.assertEqual(scf_obj.xc, 'm06')

    def test_unrestricted_subsystem(self):
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method, unrestricted=True)
        subsys.init_density()

        #Check SCF object
        scf_obj = subsys.env_scf
        comp_scf_obj = dft.UKS(self.cs_mol)
        self.assertEqual(type(scf_obj), type(comp_scf_obj))
        self.assertEqual(scf_obj.xc, 'm06')
        self.assertEqual(subsys.env_method, self.env_method)

    def test_restrictedos_subsystem(self):
        mol = self.cs_mol.copy()
        mol.spin = 2
        mol.build()
        subsys = cluster_subsystem.ClusterEnvSubSystem(mol, self.env_method)
        subsys.init_density()

        #Check SCF object
        scf_obj = subsys.env_scf
        comp_scf_obj = dft.ROKS(mol)
        self.assertEqual(type(scf_obj), type(comp_scf_obj))
        self.assertEqual(scf_obj.xc, 'm06')

    def test_density_fitting_subsystem(self):
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method2, density_fitting=True)

        #Check SCF object
        scf_obj = subsys.env_scf
        comp_scf_obj = dft.RKS(self.cs_mol).density_fit()
        from pyscf.df.df_jk import _DFHF
        self.assertTrue(isinstance(scf_obj, _DFHF))

    def test_custom_subsystem(self):

        t_file = tempfile.NamedTemporaryFile()
        subsys = cluster_subsystem.ClusterEnvSubSystem(self.cs_mol, self.env_method,
            env_order=2, env_smearsigma=0.5, damp=1, shift=1, subcycles=10,
            diis=2, unrestricted=False, density_fitting=True, freeze=True,
            verbose=2, nproc=4, pmem=300, scrdir='/path/to/scratch/',
            save_orbs=True, save_density=True, filename=t_file.name)

        self.assertEqual(subsys.mol, self.cs_mol)
        self.assertEqual(subsys.env_method, self.env_method)
        self.assertEqual(subsys.filename, t_file.name)

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
        comp_scf_obj = dft.RKS(self.cs_mol).density_fit()
        from pyscf.df.df_jk import _DFHF
        self.assertTrue(isinstance(scf_obj, _DFHF))
        self.assertEqual(scf_obj.xc, self.env_method)

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

        self.assertEqual(subsys.mol._basis['GHOST-H'],subsys.mol._basis['H'])

class TestActiveSubSystem(unittest.TestCase):

    def setUp(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol.basis = 'aug-cc-pVDZ'
        mol.build()
        self.cs_mol = mol

        self.env_method = 'm06'
        self.env_method2 = 'b3lyp'

    def test_basic_subsystem(self):
        hl_method = 'ccsd'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, hl_method)

        self.assertEqual(subsys.mol, self.cs_mol)
        self.assertEqual(subsys.env_method, self.env_method)

        self.assertEqual(subsys.env_smearsigma, 0)
        self.assertEqual(subsys.env_damp, 0)
        self.assertEqual(subsys.env_shift, 0)
        self.assertEqual(subsys.env_subcycles, 1)
        self.assertEqual(subsys.freeze, False)
        self.assertEqual(subsys.env_initguess, None)
        self.assertEqual(subsys.verbose, 3)

        self.assertEqual(subsys.hl_method, 'ccsd')
        self.assertEqual(subsys.hl_conv, None)
        self.assertEqual(subsys.hl_grad, None)
        self.assertEqual(subsys.hl_cycles, None)
        self.assertEqual(subsys.hl_damp, 0)
        self.assertEqual(subsys.hl_shift, 0)
        self.assertEqual(subsys.hl_initguess, None)

    def test_custom_obj_set(self):
        t_file = tempfile.NamedTemporaryFile()
        hl_method  = 'caspt2'
        subsys = cluster_subsystem.ClusterHLSubSystem(self.cs_mol, self.env_method, 
            hl_method, hl_conv=1e-9, hl_grad=1e-8, hl_cycles=2, hl_damp=0.1,
            hl_shift=0.001, hl_initguess='minao', env_smearsigma=0.5, damp=1,
            shift=1, subcycles=10, freeze=True, initguess='supmol',
            verbose=2, save_orbs=True, save_density=True, 
            hl_save_orbs=True, hl_save_density=True, filename=t_file.name, hl_dict={'active_orbs':[2,3,4,5], 'loc_orbs':True})

        self.assertEqual(subsys.mol, self.cs_mol)
        self.assertEqual(subsys.env_method, self.env_method)
        self.assertEqual(subsys.filename, t_file.name)

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
        self.assertEqual(subsys.hl_method, 'caspt2')
        self.assertEqual(subsys.cas_loc_orbs, True)
        self.assertEqual(subsys.cas_active_orbs, [2,3,4,5])
        self.assertEqual(subsys.hl_conv, 1e-9)
        self.assertEqual(subsys.hl_grad, 1e-8)
        self.assertEqual(subsys.hl_cycles, 2)
        self.assertEqual(subsys.hl_damp, 0.1)
        self.assertEqual(subsys.hl_shift, 0.001)
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

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao')
        self.assertEqual(supersystem.fs_method, 'm06')
        self.assertEqual(supersystem.proj_oper, 'huz')
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_damp, 0)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.fs_cycles, None)
        self.assertEqual(supersystem.fs_conv, None)
        self.assertEqual(supersystem.fs_grad, None)
        self.assertEqual(supersystem.fs_damp, 0)
        self.assertEqual(supersystem.fs_shift, 0)
        self.assertEqual(supersystem.fs_smearsigma, 0)
        self.assertEqual(supersystem.fs_initguess, None)
        self.assertEqual(supersystem.grid_level, None)
        self.assertEqual(supersystem.rho_cutoff, None)
        self.assertEqual(supersystem.fs_verbose, None)
        #self.assertEqual(supersystem.analysis, False)
        #self.assertEqual(supersystem.debug, False)

        #Check SCF object
        scf_obj = supersystem.fs_scf
        comp_scf_obj = dft.RKS(gto.mole.conc_mol(mol, mol2))
        self.assertEqual(type(scf_obj), type(comp_scf_obj))
        print (comp_scf_obj.mol._basis)
        print (supersystem.mol._basis)
        self.assertTrue(gto.same_mol(comp_scf_obj.mol, supersystem.mol, cmp_basis=False))
        self.assertEqual(scf_obj.xc, 'm06')

    def test_unrestricted_supersystem(self):
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
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao', fs_unrestricted=True, chkfile_index=0)

        #Check SCF object
        scf_obj = supersystem.fs_scf
        comp_scf_obj = dft.UKS(gto.mole.conc_mol(mol, mol2))
        self.assertEqual(type(scf_obj), type(comp_scf_obj))
        self.assertEqual(scf_obj.xc, 'm06')

    def test_hsros_supersystem(self):
        mol = gto.Mole()
        mol.verbose = 3
        mol.atom = '''
        H 0. -2.757 2.857
        H 0. 2.757 2.857
        '''
        mol.basis = '3-21g'
        mol.spin = 2
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
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao')

        #Check SCF object
        scf_obj = supersystem.fs_scf
        comp_scf_obj = dft.ROKS(gto.mole.conc_mol(mol, mol2))
        self.assertEqual(type(scf_obj), type(comp_scf_obj))
        self.assertEqual(scf_obj.xc, 'm06')

    def test_density_fitting_supersystem(self):
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
        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='minao', fs_density_fitting=True)

        #Check SCF object
        scf_obj = supersystem.fs_scf
        from pyscf.df.df_jk import _DFHF
        self.assertTrue(isinstance(scf_obj, _DFHF))
        self.assertEqual(scf_obj.xc, 'm06')

    def test_custom_supersystem(self):
        t_file = tempfile.NamedTemporaryFile()
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
        hl_dict = {'froz_orbs': 10}
        subsys = cluster_subsystem.ClusterHLSubSystem(mol, env_method, hl_method, filename=t_file.name, hl_dict=hl_dict)

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 1.0 0.0 0.0
        O 3.0 0.0 0.0'''
        mol2.basis = 'aug-cc-pVDZ'
        mol2.build()
        env_method = 'm06'
        subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method, filename=t_file.name)

        supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2],
                          'b3lyp', ft_proj_oper='huzfermi', ft_cycles=2, 
                          ft_conv=1e-1, ft_grad=1e-4, ft_diis=3, 
                          ft_initguess='1e', ft_updatefock=1, ft_setfermi=-0.1,
                          ft_damp=0.5, fs_cycles=3, fs_conv=2, fs_grad=4, 
                          fs_damp=1, fs_shift=2.1, fs_smearsigma=0.1, 
                          fs_initguess='atom', fs_grid_level=2, fs_rhocutoff=1e-2,
                          fs_verbose=1, fs_save_orbs=True, fs_save_density=True, 
                          compare_density=True, ft_save_density=True, 
                          ft_save_orbs=True, filename=t_file.name)

        self.assertEqual(supersystem.fs_method, 'b3lyp')
        self.assertEqual(supersystem.proj_oper, 'huzfermi')
        self.assertEqual(supersystem.filename, t_file.name)
        self.assertEqual(supersystem.ft_cycles, 2)
        self.assertEqual(supersystem.ft_conv, 1e-1)
        self.assertEqual(supersystem.ft_grad, 1e-4)
        self.assertEqual(supersystem.ft_damp, 0.5)
        self.assertIsInstance(supersystem.ft_diis, lib.diis.DIIS)
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
        self.assertEqual(supersystem.fs_verbose, 1)
        #self.assertEqual(supersystem.analysis, True)
        #self.assertEqual(supersystem.debug, True)
        self.assertEqual(supersystem.compare_density, True)
        self.assertEqual(supersystem.fs_save_orbs, True)
        self.assertEqual(supersystem.fs_save_density, True)
        self.assertEqual(supersystem.ft_save_orbs, True)
        self.assertEqual(supersystem.ft_save_density, True)


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
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.fs_cycles, None)
        self.assertEqual(supersystem.fs_conv, None)
        self.assertEqual(supersystem.fs_grad, None)
        self.assertEqual(supersystem.fs_damp, 0)
        self.assertEqual(supersystem.fs_shift, 0)
        self.assertEqual(supersystem.fs_smearsigma, 0)
        self.assertEqual(supersystem.fs_initguess, None)
        self.assertEqual(supersystem.grid_level, None)
        self.assertEqual(supersystem.rho_cutoff, None)
        self.assertEqual(supersystem.fs_verbose, None)
        #self.assertEqual(supersystem.analysis, False)
        #self.assertEqual(supersystem.debug, False)

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
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.fs_conv, None)
        self.assertEqual(supersystem.fs_grad, None)
        self.assertEqual(supersystem.fs_damp, 0)
        self.assertEqual(supersystem.fs_shift, 0)
        self.assertEqual(supersystem.fs_smearsigma, 0)
        self.assertEqual(supersystem.fs_initguess, None)
        self.assertEqual(supersystem.grid_level, None)
        self.assertEqual(supersystem.rho_cutoff, None)
        self.assertEqual(supersystem.fs_verbose, None)
        #self.assertEqual(supersystem.analysis, False)
        #self.assertEqual(supersystem.debug, False)

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
        self.assertEqual(supersystem.ft_cycles, 100)
        self.assertEqual(supersystem.ft_conv, 1e-8)
        self.assertEqual(supersystem.ft_grad, None)
        self.assertEqual(supersystem.ft_initguess, 'minao')
        self.assertEqual(supersystem.ft_updatefock, 0)

        self.assertEqual(supersystem.fs_cycles, None)
        self.assertEqual(supersystem.fs_conv, None)
        self.assertEqual(supersystem.fs_grad, None)
        self.assertEqual(supersystem.fs_damp, 0)
        self.assertEqual(supersystem.fs_shift, 0)
        self.assertEqual(supersystem.fs_smearsigma, 0)
        self.assertEqual(supersystem.fs_initguess, None)
        self.assertEqual(supersystem.grid_level, None)
        self.assertEqual(supersystem.rho_cutoff, None)
        self.assertEqual(supersystem.fs_verbose, None)
        #self.assertEqual(supersystem.analysis, False)
        #self.assertEqual(supersystem.debug, False)

class TestInteractionMediator(unittest.TestCase):

    def setUp(self):

        mol1 = gto.Mole()
        mol1.atom = '''
        O 10.0 0.0 0.0
        H 10. 0.7586 0.5043
        H 10. 0.7586 -0.5043'''
        mol1.basis = 'sto-3g'
        mol1.build()
        self.sub1 = cluster_subsystem.ClusterEnvSubSystem(mol1, 'lda', env_order=1)

        mol2 = gto.Mole()
        mol2.atom = '''
        O 20.0 0.0 0.0
        H 20. 0.7586 0.5043
        H 20. 0.7586 -0.5043'''
        mol2.basis = 'sto-3g'
        mol2.build()
        self.sub2 = cluster_subsystem.ClusterEnvSubSystem(mol2, 'm06', env_order=2)

        mol3 = gto.Mole()
        mol3.atom = '''
        O 0.0 0.0 0.0
        H 0. 0.7586 0.5043
        H 0. 0.7586 -0.5043'''
        mol3.basis = 'sto-3g'
        mol3.build()
        self.sub3 = cluster_subsystem.ClusterHLSubSystem(mol3, 'm06', 'rhf', env_order=2)

        mol4 = gto.Mole()
        mol4.verbose=3
        mol4.atom = '''
        O 30.0 0.0 0.0
        H 30. 0.7586 0.5043
        H 30. 0.7586 -0.5043'''
        mol4.basis = 'sto-3g'
        mol4.build()
        self.sub4 = cluster_subsystem.ClusterEnvSubSystem(mol4, 'lda', env_order=2)

        sup1_alt_sub_mol = gto.Mole()
        sup1_alt_sub_mol.verbose = 3
        sup1_alt_sub_mol.atom = '''
        O 0.0 0.0 0.0
        H 0. 0.7586 0.5043
        H 0. 0.7586 -0.5043
        O 20.0 0.0 0.0
        H 20. 0.7586 0.5043
        H 20. 0.7586 -0.5043'''
        sup1_alt_sub_mol.basis = 'sto-3g'
        sup1_alt_sub_mol.build()
        self.sup1_alt_sub = cluster_subsystem.ClusterEnvSubSystem(sup1_alt_sub_mol, 'lda', env_order=2)

    def test_basic_interaction_mediator(self):
        subsystems = [self.sub2, self.sub3]
        mediator = interaction_mediator.InteractionMediator(subsystems)
        self.assertEqual(len(mediator.supersystems), 1)

        supersystem_1 = cluster_supersystem.ClusterSuperSystem(subsystems, 'm06') 
        for i in range(len(mediator.supersystems)):
            test = mediator.supersystems[i]
            self.assertEqual(test.fs_method, supersystem_1.fs_method)
            self.assertEqual(test.proj_oper, supersystem_1.proj_oper)
            self.assertEqual(test.ft_cycles, supersystem_1.ft_cycles)
            self.assertEqual(test.ft_conv, supersystem_1.ft_conv)
            self.assertEqual(test.ft_grad, supersystem_1.ft_grad)
            self.assertEqual(test.ft_damp, supersystem_1.ft_damp)
            self.assertEqual(test.ft_initguess, supersystem_1.ft_initguess)
            self.assertEqual(test.ft_updatefock, supersystem_1.ft_updatefock)

            self.assertEqual(test.fs_cycles, supersystem_1.fs_cycles)
            self.assertEqual(test.fs_conv, supersystem_1.fs_conv)
            self.assertEqual(test.fs_grad, supersystem_1.fs_grad)
            self.assertEqual(test.fs_damp, supersystem_1.fs_damp)
            self.assertEqual(test.fs_shift, supersystem_1.fs_shift)
            self.assertEqual(test.fs_smearsigma, supersystem_1.fs_smearsigma)
            self.assertEqual(test.fs_initguess, supersystem_1.fs_initguess)
            self.assertEqual(test.grid_level, supersystem_1.grid_level)
            self.assertEqual(test.rho_cutoff, supersystem_1.rho_cutoff)
            self.assertEqual(test.fs_verbose, supersystem_1.fs_verbose)

    def test_three_system_interaction_mediator(self):
        subsystems = [self.sub1, self.sub2, self.sub3]
        mediator = interaction_mediator.InteractionMediator(subsystems)
        self.assertEqual(len(mediator.supersystems), 2)
        sub_1 = [self.sub2, self.sub4]
        sub_2 = [self.sub1, self.sub3]
        supersystem_1 = cluster_supersystem.ClusterSuperSystem(sub_1, 'lda') 
        supersystem_2 = cluster_supersystem.ClusterSuperSystem(sub_2, 'm06') 
        sup_list = [supersystem_1, supersystem_2]
        for i in range(len(mediator.supersystems)):
            test = mediator.supersystems[i]
            self.assertEqual(test.fs_method, sup_list[i].fs_method)
            self.assertEqual(test.proj_oper, sup_list[i].proj_oper)
            self.assertEqual(test.ft_cycles, sup_list[i].ft_cycles)
            self.assertEqual(test.ft_conv, sup_list[i].ft_conv)
            self.assertEqual(test.ft_grad, sup_list[i].ft_grad)
            self.assertEqual(test.ft_damp, sup_list[i].ft_damp)
            self.assertEqual(test.ft_initguess, sup_list[i].ft_initguess)
            self.assertEqual(test.ft_updatefock, sup_list[i].ft_updatefock)

            self.assertEqual(test.fs_cycles, sup_list[i].fs_cycles)
            self.assertEqual(test.fs_conv, sup_list[i].fs_conv)
            self.assertEqual(test.fs_grad, sup_list[i].fs_grad)
            self.assertEqual(test.fs_damp, sup_list[i].fs_damp)
            self.assertEqual(test.fs_shift, sup_list[i].fs_shift)
            self.assertEqual(test.fs_smearsigma, sup_list[i].fs_smearsigma)
            self.assertEqual(test.fs_initguess, sup_list[i].fs_initguess)
            self.assertEqual(test.grid_level, sup_list[i].grid_level)
            self.assertEqual(test.rho_cutoff, sup_list[i].rho_cutoff)
            self.assertEqual(test.fs_verbose, sup_list[i].fs_verbose)

    def test_unrestricted_interaction_mediator(self):
        pass

    def test_custom_interaction_mediator(self):
        pass

if __name__ == "__main__":
    unittest.main()
