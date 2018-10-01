#A module to test system inits.
# Daniel Graham


import unittest
import os
import shutil
import re

from qsome import inp_reader, cluster_subssytem, cluster_supersystem
from pyscf import gto

def_filename = "default.inp"
default_str = """
subsystem
He    0.0000    0.0000    0.0000
end

subsystem
He    2.0000    0.0000    0.0000
end

embed
 env_method pbe
 huzinaga
end

basis 3-21g
active_method hf
"""

exp_set_filename = 'exp_set.inp'
exp_set_str = """
subsystem
C    0.0000    0.0000    0.0000
charge -1
spin 1
basis aug-cc-pVDZ
smearsigma 0.1
unit angstrom
initguess minao
damp 0.2
shift 0.1
subcycles 4
end

subsystem
C    2.0000    0.0000    0.0000
charge +2
spin -2
smearsigma 0.01
unit bohr
freeze
initguess supmol
damp 0.1
shift 0.2
end

embed
 env_method pbe
 huzfermi
 cycles 10
 conv 1e-4
 grad 1e-4
 diis 2
 update_fock 2
 initguess atom
 setfermi -4
end

ct_method u
basis 3-21g
ct_settings
 conv 1e-3
 grad 1e-2
 cycles 300
 damp 0.1
 shift 0.3
 smearsigma 0.2
 initguess 1e
 includeghost
end

active_method caspt2[2,2]
cas_settings
 localize_orbitals
 active_orbs [4,5]
end
 
active_settings
 conv 1e-10
 grad 1e-11
 cycles 500
 damp 0.1
 shift 0.2
 smearsigma 0.1
 initguess readchk
end

grid 5
verbose 1
gencube test1
compden test2
analysis
debug
"""

class TestSuperSystem(unittest.TestCase):
    def setUp(self):
        pass
    def test_from_inp(self):
        pass
    def test_custom_obj(self):
        pass
    def tearDown(self):
        pass
class TestEnvSubSystem(unittest.TestCase):
    def setUp(self):
        with open(path+def_filename, "w") as f:
            f.write(default_str)

        with open(path+exp_set_filename, "w") as f:
            f.write(exp_set_str)

    def test_from_inp(self):
        subsystems = []
        in_obj = inp_reader.InpReader(path + def_filename)
        for i in range(len(in_obj.subsys_mols)):
            mol = in_obj.subsys_mols[i]
            env_method = in_obj.env_subsystem_kwargs[i].pop('env_method')
            env_kwargs = in_obj.env_subsystem_kwargs[i]
            subsystems.append(ClusterEnvSubSystem(mol, env_method, env_kwargs))

    def test_custom_obj(self):
        pass
    def tearDown(self):
        pass
class TestActiveSubSystem(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
class TestExcitedSubSystem(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
