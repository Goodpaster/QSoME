# Tests for the input reader object
# Daniel Graham


import unittest
import os
import shutil
import re

from qsome import inp_reader
from pyscf import gto


def_filename = "default.inp"
default_str = """
subsystem
He    0.0000    0.0000    0.0000
hl_method_num 1
end

subsystem
C    2.0000    0.0000    0.0000
end

subsystem
C    2.0000    2.0000    0.0000
end

subsystem
C    2.0000    2.0000    2.0000
end

env_method_settings
 env_method pbe
end

hl_method_settings
 hl_order 1
 hl_method rhf
end

basis
 default 3-21g
end

"""

explicit_filename = "explicit.inp"
explicit_str = """
subsystem
He    0.0000    0.0000    0.0000
basis
 default 3-21g
end
env_method_num 3
hl_method_num 1
hl_method_settings
 initguess atom
 spin 2
 conv 1e-3
 grad 1e-2
 cycles 100
 damp .8
 shift 2.
 use_ext openmolcas
 unrestricted
 compress_approx
 density_fitting
 cas_settings
  loc_orbs
  cas_initguess rhf
  active_orbs [1,2,3]
  avas ['1d']
 end
end
end

subsystem
He-1    2.0000    0.0000    0.0000
He    2.0000    2.0000    0.0000
He    2.0000    2.0000    2.0000
charge +2
basis
 He sto-3g
 He-1 6-31g
end
env_method_num 3
hl_method_num 2
end

subsystem
C    4.0000    0.0000    0.0000
C    4.0000    2.0000    0.0000
C    4.0000    2.0000    2.0000
ghost:C    4.0000    4.0000    2.0000
unit bohr
ecp
 default lanl2dz
end
env_method_num 2
env_method_settings
 smearsigma 0.1
 initguess atom
 conv 1e-3
 damp 0.5
 shift 1.
 subcycles 10
 setfermi 1.2
 diis 1
 density_fitting
 freeze
 save_orbs
 save_density
end
end

subsystem
H-1    8.0000    0.0000    0.0000
H    8.0000    2.0000    0.0000
H    8.0000    2.0000    2.0000
spin 3
env_method_num 3
env_method_settings
 unrestricted
end
end

subsystem
H-1    10.0000    0.0000    0.0000
H    10.0000    2.0000    0.0000
H    10.0000    2.0000    2.0000
spin 3
env_method_num 1
end


env_method_settings
 env_order 1
 env_method lda
 smearsigma 0.1
 initguess supmol
 conv 1e-1
 grad 1e4
 cycles 1
 damp 0.0001
 shift 20.0
 diis 4
 grid 7
 rhocutoff 1.
 verbose 10
 unrestricted
 density_fitting
 compare_density
 save_orbs
 save_density
 embed_settings
  cycles 100
  subcycles 2
  conv 1e-4
  grad 1e4
  damp 0.1
  diis 1
  setfermi 3.
  updatefock 1
  initguess submol
  unrestricted
  save_orbs
  save_density
  mu 1e-5
 end
end

env_method_settings
 env_order 2
 env_method pbe
end

env_method_settings
 env_order 3
 env_method pbe
end

hl_method_settings
 hl_order 1
 hl_method cas[2,2]
 initguess minao
 spin 3
 conv 1.
 grad 1.
 cycles 3
 damp 0.5
 shift 3.
 use_ext bagel
 cas_settings
  cas_initguess ci
 end
end

hl_method_settings
 hl_order 2
 hl_method rhf
end

unit angstrom
basis
 default cc-pVDZ
 H-1 cc-pVTZ
 H aug-cc-pVDZ
end

ppmem 2500
nproc 3
scrdir /path/to/scratch/dir
"""
temp_inp_dir = "/temp_input/"

class TestKwargCreation(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+explicit_filename, "w") as f:
            f.write(explicit_str)

    def test_default_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        correct_env_kwargs = {'env_order': 1, 
                              'env_method': 'pbe', 
                              'filename': path + def_filename}
        correct_hl_kwargs = {'hl_order': 1, 'hl_method': 'rhf'}
        correct_supersystem_kwargs = {'env_order': 1,
                                      'fs_method': 'pbe',
                                      'filename': path + def_filename}
        self.assertEqual(len(in_obj.env_subsystem_kwargs), 4)
        self.assertEqual(len(in_obj.hl_subsystem_kwargs), 1)
        self.assertEqual(len(in_obj.supersystem_kwargs), 1)
        for n in in_obj.env_subsystem_kwargs:
            self.assertDictEqual(n, correct_env_kwargs)
        for n in in_obj.hl_subsystem_kwargs:
            self.assertDictEqual(n, correct_hl_kwargs)
        for n in in_obj.supersystem_kwargs:
            self.assertDictEqual(n, correct_supersystem_kwargs)

    def test_explicit_inp(self):
        path = os.getcwd() + temp_inp_dir 
        in_obj = inp_reader.InpReader(path + explicit_filename)
        correct_env_kwargs_1 = {'env_order': 3,
                                'env_method': 'pbe', 
                                'filename': path + explicit_filename, 
                                'ppmem': 2500, 
                                'nproc': 3, 
                                'scrdir': '/path/to/scratch/dir'}
        correct_env_kwargs_2 = {'env_order': 3,
                                'env_method': 'pbe', 
                                'filename': path + explicit_filename, 
                                'ppmem': 2500, 
                                'nproc': 3, 
                                'scrdir': '/path/to/scratch/dir'}
        correct_env_kwargs_3 = {'env_order': 2,
                                'env_method': 'pbe',
                                'env_smearsigma': 0.1,
                                'initguess': 'atom',
                                'conv': 1e-3,
                                'damp': 0.5,
                                'shift': 1.,
                                'subcycles': 10,
                                'setfermi': 1.2,
                                'diis': 1,
                                'density_fitting': True,
                                'freeze': True,
                                'save_orbs': True,
                                'save_density': True,
                                'filename': path + explicit_filename, 
                                'ppmem': 2500, 
                                'nproc': 3, 
                                'scrdir': '/path/to/scratch/dir'}
        correct_env_kwargs_4 = {'env_order': 3,
                                'env_method': 'pbe', 
                                'unrestricted': True,
                                'filename': path + explicit_filename, 
                                'ppmem': 2500, 
                                'nproc': 3, 
                                'scrdir': '/path/to/scratch/dir'}
        correct_env_kwargs_5 = {'env_order': 1,
                                'env_method': 'lda', 
                                'subcycles': 2,
                                'unrestricted': True,
                                'filename': path + explicit_filename, 
                                'ppmem': 2500, 
                                'nproc': 3, 
                                'scrdir': '/path/to/scratch/dir'}
        env_list = [correct_env_kwargs_1, correct_env_kwargs_2, correct_env_kwargs_3, correct_env_kwargs_4, correct_env_kwargs_5]
        correct_hl_kwargs_1 = {'hl_order': 1,
                               'hl_method': 'cas[2,2]',
                               'hl_initguess': 'atom',
                               'hl_spin': 2,
                               'hl_conv': 1e-3,
                               'hl_grad': 1e-2,
                               'hl_cycles': 100,
                               'hl_damp': 0.8,
                               'hl_shift': 2.,
                               'hl_ext': 'openmolcas',
                               'hl_unrestricted': True,
                               'hl_compress_approx': True,
                               'hl_density_fitting': True,                                   'cas_loc_orbs': True,
                               'cas_initguess': 'rhf',
                               'cas_active_orbs': [1,2,3],
                               'cas_avas':['1d']}
                                
        correct_hl_kwargs_2 = {'hl_order': 2,
                               'hl_method': 'rhf'}
        hl_list = [correct_hl_kwargs_1, correct_hl_kwargs_2]
        correct_sup_kwargs_1 = {'env_order': 1,
                                'fs_method': 'lda',
                                'fs_smearsigma': 0.1,
                                'fs_initguess': 'supmol',
                                'fs_conv': 1e-1,
                                'fs_grad': 1e4,
                                'fs_cycles': 1,
                                'fs_damp': 0.0001,
                                'fs_shift': 20.0,
                                'fs_diis': 4,
                                'fs_grid_level': 7,
                                'fs_rhocutoff': 1.,
                                'fs_verbose': 10,
                                'fs_unrestricted': True,
                                'fs_density_fitting': True,
                                'compare_density': True,
                                'fs_save_orbs': True,
                                'fs_save_density': True,
                                'ft_cycles': 100,
                                'ft_conv': 1e-4,
                                'ft_grad': 1e4,
                                'ft_damp': 0.1,
                                'ft_diis': 1,
                                'ft_setfermi': 3.,
                                'ft_updatefock': 1,
                                'ft_initguess': 'submol',
                                'ft_unrestricted': True,
                                'ft_save_orbs': True,
                                'ft_save_density': True,
                                'ft_proj_oper': 1e-5,
                                'filename': path + explicit_filename, 
                                'ppmem': 2500, 
                                'nproc': 3, 
                                'scrdir': '/path/to/scratch/dir'}
        correct_sup_kwargs_2 = {'env_order': 2,
                                'fs_method': 'pbe',
                                'filename': path + explicit_filename, 
                                'ppmem': 2500, 
                                'nproc': 3, 
                                'scrdir': '/path/to/scratch/dir'}
        correct_sup_kwargs_3 = {'env_order': 3,
                                'fs_method': 'pbe',
                                'filename': path + explicit_filename, 
                                'ppmem': 2500, 
                                'nproc': 3, 
                                'scrdir': '/path/to/scratch/dir'}
        sup_list = [correct_sup_kwargs_1, correct_sup_kwargs_2, correct_sup_kwargs_3]
        self.assertEqual(len(in_obj.env_subsystem_kwargs), 5)
        self.assertEqual(len(in_obj.hl_subsystem_kwargs), 2)
        self.assertEqual(len(in_obj.supersystem_kwargs), 3)
        for i in range(len(in_obj.env_subsystem_kwargs)):
            test = in_obj.env_subsystem_kwargs[i]
            self.assertDictEqual(test, env_list[i])
            print (f"ENV SUBSYSTEM {i} GOOD")
        for i in range(len(in_obj.hl_subsystem_kwargs)):
            test = in_obj.hl_subsystem_kwargs[i]
            self.assertDictEqual(test, hl_list[i])
            print (f"HL SUBSYSTEM {i} GOOD")
        for i in range(len(in_obj.supersystem_kwargs)):
            test = in_obj.supersystem_kwargs[i]
            self.assertDictEqual(test, sup_list[i])
            print (f"SUPERSYSTEM {i} GOOD")
         
    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestMolCreation(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+explicit_filename, "w") as f:
            f.write(explicit_str)

    def test_default(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)

        correct_mol1 = gto.M()
        correct_mol1.atom = '''
            He    0.0000    0.0000    0.0000'''
        correct_mol1.basis = '3-21g'
        correct_mol1.build()

        correct_mol2 = gto.M()
        correct_mol2.atom = '''
            C    2.0000    0.0000    0.0000'''
        correct_mol2.basis = '3-21g'
        correct_mol2.build()

        correct_mol3 = gto.M()
        correct_mol3.atom = '''
            C    2.0000    2.0000    0.0000'''
        correct_mol3.basis = '3-21g'
        correct_mol3.build()

        correct_mol4 = gto.M()
        correct_mol4.atom = '''
            C    2.0000    2.0000    2.0000'''
        correct_mol4.basis = '3-21g'
        correct_mol4.build()

        corr_mol_list = [correct_mol1, correct_mol2, correct_mol3, correct_mol4]
        self.assertEqual(len(in_obj.subsys_mols), 4)
        for i in range(len(in_obj.subsys_mols)):
            test = in_obj.subsys_mols[i]
            corr = corr_mol_list[i]
            self.assertListEqual(test._atom, corr._atom) 
            self.assertDictEqual(test._basis, corr._basis) 
        
    def test_explicit(self):
        path = os.getcwd() + temp_inp_dir 
        in_obj = inp_reader.InpReader(path + explicit_filename)

        correct_mol1 = gto.M()
        correct_mol1.atom = '''
            He    0.0000    0.0000    0.0000'''
        correct_mol1.basis = '3-21g'
        correct_mol1.build()

        correct_mol2 = gto.M()
        correct_mol2.atom = '''
            He-1    2.0000    0.0000    0.0000
            He    2.0000    2.0000    0.0000
            He    2.0000    2.0000    2.0000'''
        correct_mol2.charge = 2
        correct_mol2.basis = {'He' : 'sto-3g', 'He-1': '6-31g'}
        correct_mol2.build()

        correct_mol3 = gto.M()
        correct_mol3.atom = '''
            C    4.0000    0.0000    0.0000
            C    4.0000    2.0000    0.0000
            C    4.0000    2.0000    2.0000
            ghost:C    4.0000    4.0000    2.0000'''
        correct_mol3.basis = 'cc-pVDZ'
        correct_mol3.ecp = 'lanl2dz'
        correct_mol3.unit = 'bohr'
        correct_mol3.build()

        correct_mol4 = gto.M()
        correct_mol4.atom = '''
            H-1    8.0000    0.0000    0.0000
            H    8.0000    2.0000    0.0000
            H    8.0000    2.0000    2.0000'''
        correct_mol4.basis = {'H' : 'aug-cc-pVDZ', 'H-1': 'cc-pVTZ'}
        correct_mol4.spin = 3
        correct_mol4.build()

        correct_mol5 = gto.M()
        correct_mol5.atom = '''
            H-1    10.0000    0.0000    0.0000
            H    10.0000    2.0000    0.0000
            H    10.0000    2.0000    2.0000'''
        correct_mol5.basis = {'H' : 'aug-cc-pVDZ', 'H-1': 'cc-pVTZ'}
        correct_mol5.spin = 3
        correct_mol5.build()

        corr_mol_list = [correct_mol1, correct_mol2, correct_mol3, correct_mol4, correct_mol5]
        self.assertEqual(len(in_obj.subsys_mols), 5)
        for i in range(len(in_obj.subsys_mols)):
            test = in_obj.subsys_mols[i]
            corr = corr_mol_list[i]
            self.assertListEqual(test._atom, corr._atom) 
            self.assertEqual(test.charge, corr.charge) 
            self.assertEqual(test.spin, corr.spin) 
            for k in test._atom:
                self.assertListEqual(test._basis[k[0]], corr._basis[k[0]]) 
            for k in test._ecp.keys():
                self.assertListEqual(test._ecp[k], corr._ecp[k]) 
        
    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

if __name__ == "__main__":
    unittest.main()
