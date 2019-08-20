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
temp_inp_dir = "/temp_input/"

class TestInputReader(unittest.TestCase):

    def setUp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way

        if os.path.isdir(path):
            shutil.rmtree(path)    
        os.mkdir(path)

        with open(path+def_filename, "w") as f:
            f.write(default_str)

    def test_default_inp(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way
        in_obj = inp_reader.InpReader(path + def_filename)
        inp = in_obj.inp

        #Check atoms
        self.assertEqual(inp.subsystem[0].atoms[0].group(1), 'He')
        self.assertEqual(float(inp.subsystem[0].atoms[0].group(2)), 0.0)
        self.assertEqual(float(inp.subsystem[0].atoms[0].group(3)), 0.0)
        self.assertEqual(float(inp.subsystem[0].atoms[0].group(4)), 0.0)

        self.assertEqual(inp.subsystem[1].atoms[0].group(1), 'C')
        self.assertEqual(float(inp.subsystem[1].atoms[0].group(2)), 2.0)
        self.assertEqual(float(inp.subsystem[1].atoms[0].group(3)), 0.0)
        self.assertEqual(float(inp.subsystem[1].atoms[0].group(4)), 0.0)

        #Check operator and methods
        self.assertEqual(inp.env_method_settings[0].env_method, 'pbe')
        self.assertEqual(inp.hl_method_settings[0].hl_method, 'rhf')
        self.assertEqual(inp.basis.basis_def[0].group(0), 'default 3-21g')
        self.assertTrue(False)
         
    def tearDown(self):
        path = os.getcwd() + temp_inp_dir   #Maybe a better way.
        if os.path.isdir(path):
            shutil.rmtree(path)    
