# Tests for the input reader object
# Daniel Graham


import unittest
import os
import shutil
import re

from qsome import inp_reader
#from pyscf import gto


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
  avas [1d]
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
 default 
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
 shift 20
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
        inp = in_obj.inp
        print (inp.env_subsystem_kwargs)
        print (inp.hl_subsystem_kwargs)
        print (inp.supersystem_method_kwargs)
        self.assertTrue(False)


    def test_explicit_inp(self):
        pass
         
    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    

class TestMolCreation(unittest.TestCase):
    pass
