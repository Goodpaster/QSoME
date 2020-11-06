# Tests for the input reader object
# Daniel Graham


import unittest
import os
import shutil
import re

import input_reader
from input_reader import helpers as ir_helpers
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

bad_file_format = "bad_formatting.inp"
bad_file_format_str = """
Throw an error"""

env_settings_filename = 'env_settings.inp'
env_settings_str_replace = """
env_method_settings
 env_method lda
 smearsigma 0.1
 initguess submol
 conv 1e-6
 grad 1e-8
 cycles 100
 damp 0.2
 shift 1.4
 diis 1
 grid 4
 rhocutoff 1e-1
 verbose 3
 unrestricted
 density_fitting
 compare_density
 save_orbs
 save_density
 save_spin_density
 embed_settings
  cycles 12
  subcycles 2
  basis_tau 0.2
  conv 1e-4
  grad 1e-2
  damp 0.5
  diis 3
  setfermi 0.35
  updatefock 3
  updateproj 2
  initguess readchk
  save_orbs
  save_density
  save_spin_density
  huzfermi
 end
end"""
env_settings_str = default_str.replace("""
env_method_settings
 env_method pbe
end""", env_settings_str_replace)

hl_settings_filename = 'hl_settings.inp'
hl_settings_str_replace = """
hl_method_settings
 hl_order 1
 hl_method caspt2[10,10]
 initguess submol
 spin 4
 conv 1e-12
 grad 1e-14
 cycles 100
 damp 10.
 shift 12.1
 compress_approx
 unrestricted
 density_fitting
 save_orbs
 save_density
 save_spin_density
 use_ext openmolcas
 cas_settings
  loc_orbs
  cas_initguess rhf
  active_orbs 3,4,5,6,7
  avas 1d
 end
end"""

hl_settings_str = default_str.replace("""
hl_method_settings
 hl_order 1
 hl_method rhf
end""", hl_settings_str_replace)

subsys_env_set_filename = 'subsys_env_set.inp'
subsys_env_set_str_replace = """
env_method_settings
 smearsigma 0.2
 initguess readchk
 conv 1e-20
 damp 12.2
 shift 2.3
 subcycles 10
 setfermi 1.112
 diis 6
 unrestricted
 density_fitting
 freeze 
 save_orbs
 save_density
 save_spin_density
end"""

subsys_env_set_str = env_settings_str.replace("""
subsystem
C    2.0000    0.0000    0.0000
end""", """
subsystem
C    2.0000    0.0000    0.0000
""" + subsys_env_set_str_replace + """
end""")


subsys_hl_set_filename = 'subsys_hl_set.inp'
subsys_hl_set_str_replace = """
hl_method_settings
 initguess minao
 spin 4
 conv 1e-3
 grad 1e1
 cycles 4
 damp 0.001
 shift 0.002
 use_ext molpro
 unrestricted
 compress_approx
 density_fitting
 save_orbs
 save_density
 save_spin_density
 cas_settings
  cas_initguess ci
  active_orbs 5,6,7,8,9
  avas 3d
 end
end
"""

subsys_hl_set_str = hl_settings_str.replace("""
He    0.0000    0.0000    0.0000
hl_method_num 1
""", """
He    0.0000    0.0000    0.0000
hl_method_num 1""" + subsys_hl_set_str_replace)

multi_env_filename = 'multi_env.inp'
multi_env_str = """
subsystem
He    0.0000    0.0000    0.0000
hl_method_num 1
env_method_num 1
end

subsystem
C    2.0000    0.0000    0.0000
env_method_num 1
end

subsystem
C    2.0000    2.0000    0.0000
env_method_num 2
end

subsystem
C    2.0000    2.0000    2.0000
env_method_num 2
end

env_method_settings
 env_order 1
 env_method pbe
end

env_method_settings
 env_order 2
 env_method lda
end

hl_method_settings
 hl_order 1
 hl_method rhf
end

basis
 default 3-21g
end

"""

specify_basis_filename = 'specify_basis.inp'
specify_basis_str = """
subsystem
He    0.0000    0.0000    0.0000
hl_method_num 1
end

subsystem
C    2.0000    0.0000    0.0000
end

subsystem
C-1    2.0000    2.0000    0.0000
end

subsystem
C-1    2.0000    2.0000    2.0000
basis
 default cc-pVTZ
end
end

env_method_settings
 env_method pbe
end

hl_method_settings
 hl_order 1
 hl_method rhf
end

basis
 C sto-3g
 C-1 cc-pVDZ
 default 3-21g
end

"""

specify_ecp_filename = 'specify_ecp.inp'
specify_ecp_str = """
subsystem
He    0.0000    0.0000    0.0000
hl_method_num 1
end

subsystem
Fe    2.0000    0.0000    0.0000
ecp
 default lanl2dz
end
end

subsystem
Fe-1    2.0000    2.0000    0.0000
ecp
 default bfd-pp
end
end

subsystem
Fe-1    2.0000    2.0000    2.0000
end

env_method_settings
 env_method pbe
end

ecp
 Fe-1 lanl08
 default lanl2tz
end

hl_method_settings
 hl_order 1
 hl_method rhf
end

basis
 default 3-21g
end

"""

spin_charge_filename = 'spin_charge.inp'
spin_charge_str = """
subsystem
He    0.0000    0.0000    0.0000
hl_method_num 1
spin 1
charge -1
end

subsystem
C    2.0000    0.0000    0.0000
spin 2
end

subsystem
C    2.0000    2.0000    0.0000
charge -1
spin -1
end

subsystem
C    2.0000    2.0000    2.0000
end

env_method_settings
 env_method pbe
 spin 2
 charge -2
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
  active_orbs 1,2,3
  avas 1d
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
Fe    4.0000    0.0000    0.0000
Fe    4.0000    2.0000    0.0000
Fe    4.0000    2.0000    2.0000
ghost:Fe    4.0000    4.0000    2.0000
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
 save_spin_density
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
 save_spin_density
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
 save_spin_density
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

ghostlink_filename = "ghostlink_tests.inp"
ghostlink_str = """
subsystem
He    0.0000    0.0000    0.0000
He    1.0000    0.0000    0.0000
hl_method_num 1
addlinkbasis
end

subsystem
C    2.0000    0.0000    0.0000
end

subsystem
C    2.0000    2.0000    0.0000
addlinkbasis
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

excited_filename = "excited.inp"
excited_str = """
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
 excited
 excited_settings
  conv 1e-9
  nroots 4
 end
 embed_settings
  excited_relax
  excited_settings
   conv 1e-9
   nroots 4
  end
 end
end

hl_method_settings
 hl_order 1
 hl_method ccsd
 excited
 excited_settings
  nroots 4
 end
end

basis
 default 3-21g
end
"""

temp_inp_dir = "/temp_input/"
class TestGenerateInputReaderObject(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir
        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)
        self.def_filename_path = path + def_filename
        with open(self.def_filename_path, "w") as f:
            f.write(default_str)

        self.bad_file_format_path = path + bad_file_format
        with open(self.bad_file_format_path, "w") as f:
            f.write(bad_file_format_str)

    def test_read_filename_and_return_inp_object(self):
        test_obj = inp_reader.read_input(self.def_filename_path)
        self.assertIsInstance(test_obj, ir_helpers.Namespace)

    def test_assert_file_not_found(self):
        with self.assertRaises(ir_helpers.ReaderError):
            inp_reader.read_input("")

    def test_assert_bad_format(self):
        with self.assertRaises(ir_helpers.ReaderError):
            inp_reader.read_input(self.bad_file_format_path)

    def test_namespace_params(self):
        test_obj = inp_reader.read_input(self.def_filename_path)
        print (test_obj)
        self.assertTrue(False)

    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    
        
class TestKwargCreation(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+env_settings_filename, "w") as f:
            f.write(env_settings_str)

        with open(path+hl_settings_filename, "w") as f:
            f.write(hl_settings_str)

        with open(path+subsys_env_set_filename, "w") as f:
            f.write(subsys_env_set_str)
            
        with open(path+subsys_hl_set_filename, "w") as f:
            f.write(subsys_hl_set_str)

        with open(path+multi_env_filename, "w") as f:
            f.write(multi_env_str)

        with open(path+explicit_filename, "w") as f:
            f.write(explicit_str)

        with open(path+excited_filename, "w") as f:
            f.write(excited_str)



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
        self.assertEqual(len(in_obj.hl_subsystem_kwargs), 4)
        self.assertEqual(len(in_obj.supersystem_kwargs), 1)
        for n in in_obj.env_subsystem_kwargs:
            self.assertDictEqual(n, correct_env_kwargs)
        for i, n in enumerate(in_obj.hl_subsystem_kwargs):
            if i == 0:
                self.assertDictEqual(n, correct_hl_kwargs)
            else:
                self.assertIsNone(n)
        for n in in_obj.supersystem_kwargs:
            self.assertDictEqual(n, correct_supersystem_kwargs)

    def test_environ_settings(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + env_settings_filename)
        correct_supersystem_kwargs = {'env_order': 1,
                                      'fs_method': 'lda',
                                      'fs_smearsigma': 0.1,
                                      'fs_initguess': 'submol',
                                      'fs_conv': 1e-6,
                                      'fs_grad': 1e-8,
                                      'fs_cycles': 100,
                                      'fs_damp': 0.2,
                                      'fs_shift': 1.4,
                                      'fs_diis': 1,
                                      'fs_grid_level': 4,
                                      'fs_rhocutoff': 1e-1,
                                      'fs_verbose': 3,
                                      'fs_unrestricted': True,
                                      'fs_density_fitting': True,
                                      'compare_density': True,
                                      'fs_save_orbs': True,
                                      'fs_save_density': True,
                                      'fs_save_spin_density': True,
                                      'ft_cycles': 12,
                                      'ft_subcycles': 2,
                                      'ft_basis_tau': 0.2,
                                      'ft_conv': 1e-4,
                                      'ft_grad': 1e-2,
                                      'ft_damp': 0.5,
                                      'ft_diis': 3,
                                      'ft_setfermi': 0.35,
                                      'ft_updatefock': 3,
                                      'ft_initguess': 'readchk',
                                      'ft_save_orbs': True,
                                      'ft_save_density': True,
                                      'ft_save_spin_density': True,
                                      'ft_proj_oper': 'huzfermi',
                                      'filename': path + env_settings_filename}
        self.assertEqual(len(in_obj.supersystem_kwargs), 1)
        for n in in_obj.supersystem_kwargs:
            self.assertDictEqual(n, correct_supersystem_kwargs)

    def test_hl_settings(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + hl_settings_filename)
        correct_hl_kwargs = {'hl_order': 1,
                             'hl_method': 'caspt2[10,10]',
                             'hl_initguess': 'submol',
                             'hl_conv': 1e-12,
                             'hl_spin': 4,
                             'hl_grad': 1e-14,
                             'hl_cycles': 100,
                             'hl_damp': 10.,
                             'hl_shift': 12.1,
                             'hl_ext': 'openmolcas',
                             'hl_unrestricted': True,
                             'hl_compress_approx': True,
                             'hl_density_fitting': True,
                             'hl_save_orbs': True,
                             'hl_save_density': True,
                             'hl_save_spin_density': True,
                             'hl_dict': {'loc_orbs': True, 'cas_initguess': 'rhf', 'active_orbs': [3,4,5,6,7], 'avas': ['1d']}
                             }
        self.assertEqual(len(in_obj.hl_subsystem_kwargs), 4)
        for i in range(len(in_obj.hl_subsystem_kwargs)):
            n = in_obj.hl_subsystem_kwargs[i]
            if i == 0:
                self.assertDictEqual(n, correct_hl_kwargs)
            else:
                self.assertEqual(n, None)

    def test_subsys_env_settings(self):
        path = os.getcwd() + temp_inp_dir 
        in_obj = inp_reader.InpReader(path + subsys_env_set_filename)
        correct_env_kwargs_1 = {'env_order': 1,
                                'env_method': 'lda',
                                'env_smearsigma': 0.2,
                                'initguess': 'readchk',
                                'conv': 1e-20,
                                'damp': 12.2,
                                'shift': 2.3,
                                'subcycles': 10,
                                'setfermi': 1.112,
                                'diis': 6,
                                'unrestricted': True,
                                'density_fitting': True,
                                'freeze': True,
                                'save_orbs':True,
                                'save_density':True,
                                'save_spin_density':True,
                                'filename': path + subsys_env_set_filename}
        self.assertDictEqual(in_obj.env_subsystem_kwargs[1], correct_env_kwargs_1)

    def test_subsys_hl_settings(self):
        path = os.getcwd() + temp_inp_dir 
        in_obj = inp_reader.InpReader(path + subsys_hl_set_filename)
        correct_hl_kwargs_1 = {'hl_order': 1,
                               'hl_method': 'caspt2[10,10]',
                               'hl_initguess': 'minao',
                               'hl_spin': 4,
                               'hl_conv': 1e-3,
                               'hl_grad': 1e1,
                               'hl_cycles': 4,
                               'hl_damp': 0.001,
                               'hl_shift': 0.002,
                               'hl_ext': 'molpro',
                               'hl_unrestricted': True,
                               'hl_compress_approx': True,
                               'hl_density_fitting': True,
                               'hl_save_orbs': True,
                               'hl_save_density': True,
                               'hl_save_spin_density': True,
                               'hl_dict': {'cas_initguess': 'ci', 'active_orbs': [5,6,7,8,9], 'avas': ['3d']}
                               }
        self.assertDictEqual(in_obj.hl_subsystem_kwargs[0], correct_hl_kwargs_1)

    def test_multi_env(self):
        path = os.getcwd() + temp_inp_dir 
        in_obj = inp_reader.InpReader(path + multi_env_filename)
        correct_env_kwargs_1 = {'env_order': 1,
                                'fs_method': 'pbe', 
                                'filename': path + multi_env_filename}
        correct_env_kwargs_2 = {'env_order': 2,
                                'fs_method': 'lda', 
                                'filename': path + multi_env_filename}

        self.assertDictEqual(in_obj.supersystem_kwargs[0], correct_env_kwargs_1)
        self.assertDictEqual(in_obj.supersystem_kwargs[1], correct_env_kwargs_2)
        self.assertEqual(in_obj.env_subsystem_kwargs[0]['env_order'], 1)
        self.assertEqual(in_obj.env_subsystem_kwargs[1]['env_order'], 1)
        self.assertEqual(in_obj.env_subsystem_kwargs[2]['env_order'], 2)
        self.assertEqual(in_obj.env_subsystem_kwargs[3]['env_order'], 2)

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
                                'save_spin_density': True,
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
                               'hl_density_fitting': True,
                               'hl_dict': {'loc_orbs': True, 'cas_initguess': 'rhf', 'active_orbs': [1,2,3], 'avas':['1d']}
                               }
        correct_hl_kwargs_2 = {'hl_order': 2,
                               'hl_method': 'rhf',
                               'hl_save_spin_density': True}
        hl_list = [correct_hl_kwargs_1, correct_hl_kwargs_2, None, None, None]
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
                                'fs_save_spin_density': True,
                                'ft_cycles': 100,
                                'ft_subcycles': 2,
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
        self.assertEqual(len(in_obj.hl_subsystem_kwargs), 5)
        self.assertEqual(len(in_obj.supersystem_kwargs), 3)
        for i in range(len(in_obj.env_subsystem_kwargs)):
            test = in_obj.env_subsystem_kwargs[i]
            self.assertDictEqual(test, env_list[i])
            print (f"ENV SUBSYSTEM {i} GOOD")
        for i in range(len(in_obj.hl_subsystem_kwargs)):

            test = in_obj.hl_subsystem_kwargs[i]
            self.assertEqual(test, hl_list[i])
            print (f"HL SUBSYSTEM {i} GOOD")
        for i in range(len(in_obj.supersystem_kwargs)):
            test = in_obj.supersystem_kwargs[i]
            self.assertDictEqual(test, sup_list[i])
            print (f"SUPERSYSTEM {i} GOOD")

    def test_excited_kwargs(self):
        path = os.getcwd() + temp_inp_dir
        in_obj = inp_reader.InpReader(path + excited_filename)
        correct_env_kwargs = {'env_order': 1, 
                              'env_method': 'pbe',
                              'filename': path + excited_filename}
        correct_hl_kwargs = {'hl_order': 1,
                             'hl_method': 'ccsd',
                             'hl_excited': True,
                             'excited_dict': {'nroots': 4}}
        correct_supersystem_kwargs = {'env_order': 1,
                                      'fs_method': 'pbe',
                                      'fs_excited_dict': {'conv': 1e-9, 'nroots': 4},
                                      'fs_excited': True,
                                      'ft_excited_dict': {'conv': 1e-9, 'nroots': 4},
                                      'ft_excited_relax': True,
                                      'filename': path + excited_filename}
        self.assertEqual(len(in_obj.env_subsystem_kwargs), 4)
        self.assertEqual(len(in_obj.hl_subsystem_kwargs), 4)
        self.assertEqual(len(in_obj.supersystem_kwargs), 1)
        for n in in_obj.env_subsystem_kwargs:
            self.assertDictEqual(n, correct_env_kwargs)
        for i, n in enumerate(in_obj.hl_subsystem_kwargs):
            if i == 0:
                self.assertDictEqual(n, correct_hl_kwargs)
            else:
                self.assertIsNone(n)
        for n in in_obj.supersystem_kwargs:
            self.assertDictEqual(n, correct_supersystem_kwargs)
         
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

        with open(path+specify_basis_filename, "w") as f:
            f.write(specify_basis_str)

        with open(path+specify_ecp_filename, "w") as f:
            f.write(specify_ecp_str)

        with open(path+spin_charge_filename, "w") as f:
            f.write(spin_charge_str)

        with open(path+explicit_filename, "w") as f:
            f.write(explicit_str)

        with open(path+ghostlink_filename, "w") as f:
            f.write(ghostlink_str)

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

    def test_def_basis(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + specify_basis_filename)
        self.maxDiff = None

        correct_mol1 = gto.M()
        correct_mol1.atom = '''
            He    0.0000    0.0000    0.0000'''
        correct_mol1.basis = '3-21g'
        correct_mol1.build()

        correct_mol2 = gto.M()
        correct_mol2.atom = '''
            C    2.0000    0.0000    0.0000'''
        correct_mol2.basis = 'sto-3g'
        correct_mol2.build()

        correct_mol3 = gto.M()
        correct_mol3.atom = '''
            C-1    2.0000    2.0000    0.0000'''
        correct_mol3.basis = 'cc-pVDZ'
        correct_mol3.build()

        correct_mol4 = gto.M()
        correct_mol4.atom = '''
            C-1    2.0000    2.0000    2.0000'''
        correct_mol4.basis = 'cc-pVDZ'
        correct_mol4.build()

        corr_mol_list = [correct_mol1, correct_mol2, correct_mol3, correct_mol4]
        self.assertEqual(len(in_obj.subsys_mols), 4)
        for i in range(len(in_obj.subsys_mols)):
            test = in_obj.subsys_mols[i]
            corr = corr_mol_list[i]
            self.assertListEqual(test._atom, corr._atom) 
            for k in test._atom:
                self.assertListEqual(test._basis[k[0]], corr._basis[k[0]]) 

    def test_def_ecp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + specify_ecp_filename)
        self.maxDiff = None

        correct_mol1 = gto.M()
        correct_mol1.atom = '''
            He    0.0000    0.0000    0.0000'''
        correct_mol1.basis = '3-21g'
        correct_mol1.ecp = 'lanl2tz'
        correct_mol1.build()

        correct_mol2 = gto.M()
        correct_mol2.atom = '''
            Fe    2.0000    0.0000    0.0000'''
        correct_mol2.basis = '3-21g'
        correct_mol2.ecp = 'lanl2dz'
        correct_mol2.build()

        correct_mol3 = gto.M()
        correct_mol3.atom = '''
            Fe-1    2.0000    2.0000    0.0000'''
        correct_mol3.basis = '3-21g'
        correct_mol3.ecp = 'lanl08'
        correct_mol3.build()

        correct_mol4 = gto.M()
        correct_mol4.atom = '''
            Fe-1    2.0000    2.0000    2.0000'''
        correct_mol4.basis = '3-21g'
        correct_mol4.ecp = 'lanl08'
        correct_mol4.build()

        corr_mol_list = [correct_mol1, correct_mol2, correct_mol3, correct_mol4]
        self.assertEqual(len(in_obj.subsys_mols), 4)
        for i in range(len(in_obj.subsys_mols)):
            test = in_obj.subsys_mols[i]
            corr = corr_mol_list[i]
            self.assertListEqual(test._atom, corr._atom) 
            for k in corr._ecp.keys():
                self.assertListEqual(test._ecp[k], corr._ecp[k]) 

    def test_spin_charge(self):
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
            Fe    4.0000    0.0000    0.0000
            Fe    4.0000    2.0000    0.0000
            Fe    4.0000    2.0000    2.0000
            ghost:Fe    4.0000    4.0000    2.0000'''
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

    def test_ghostlink(self):
        path = os.getcwd() + temp_inp_dir
        in_obj = inp_reader.InpReader(path + ghostlink_filename)

        correct_mol1 = gto.M()
        correct_mol1.atom = '''
            He    0.0000    0.0000    0.0000
            He    1.0000    0.0000    0.0000
            GHOST-H 1.0000  0.0000    0.0000
            GHOST-H 1.5000  0.0000    0.0000'''
        correct_mol1.basis = '3-21g'
        correct_mol1.build()

        correct_mol2 = gto.M()
        correct_mol2.atom = '''
            C    2.0000    0.0000    0.0000'''
        correct_mol2.basis = '3-21g'
        correct_mol2.build()

        correct_mol3 = gto.M()
        correct_mol3.atom = '''
            C    2.0000    2.0000    0.0000
            GHOST-H 2.0000 1.0000    0.0000
            GHOST-H 2.0000 2.0000    1.0000'''
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
            self.assertEqual(len(test._atom), len(corr._atom))
            for i,atom in enumerate(test._atom):
                self.assertEqual(test._atom[i][0], corr._atom[i][0])
                self.assertAlmostEqual(test._atom[i][1][0], corr._atom[i][1][0])
                self.assertAlmostEqual(test._atom[i][1][1], corr._atom[i][1][1])
                self.assertAlmostEqual(test._atom[i][1][2], corr._atom[i][1][2])
                self.assertListEqual(test._basis[atom[0]], corr._basis[atom[0]])
        
    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

if __name__ == "__main__":
    unittest.main()
