#Tests the helper functions.
#TODO TEST ECP fxns
# Daniel Graham


import unittest
import os
import shutil
import re

from qsome import helpers
from pyscf import gto

class TestConcatMols(unittest.TestCase):
    def setUp(self):
        self.mol1 = gto.M()
        self.mol1.atom = """
        H   -1.06  0.0   0.0
        C   0.0   0.0   0.0
        """
        self.mol1.basis = 'sto-3g'
        self.mol1.spin = 1
        self.mol1.build()

        self.mol2 = gto.M()
        self.mol2.atom = """
        C   1.2   0.0   0.0
        H   2.26  0.0   0.0
        """
        self.mol2.basis = '6-311g'
        self.mol2.spin = -1
        self.mol2.build()

        self.mol3 = gto.M()
        self.mol3.atom = """
        O   -1.16   0.0   0.0
        """
        self.mol3.basis = 'sto-3g'
        self.mol3.build()

        self.mol4 = gto.M()
        self.mol4.atom = """
        C   0.0   0.0   0.0
        """
        self.mol4.basis = '6-311g'
        self.mol4.build()

        self.mol5 = gto.M()
        self.mol5.atom = """
        O   1.16   0.0   0.0
        """
        self.mol5.basis = 'sto-3g'
        self.mol5.build()

        self.mol6 = gto.M()
        self.mol6.atom = """
        ghost.C  0.0  0.0  0.0
        C   1.2   0.0   0.0
        H   2.26  0.0   0.0
        """
        self.mol6.basis = 'cc-pVDZ'
        self.mol6.spin = 1
        self.mol6.build()

        self.mol7 = gto.M()
        self.mol7.atom = """
        ghost.C 2.0 0.0 0.0
        O    0.0   1.16   0.0
        """
        self.mol7.basis = {'default': 'aug-cc-pVDZ', 'O': 'cc-pVTZ'}
        self.mol7.build()

        self.mol8 = gto.M()
        self.mol8.atom = """
        C 0.0 0.0 0.0
        ghost.C 2.0 0.0 0.0
        """
        self.mol8.basis = {'default': 'cc-pVDZ'}
        self.mol8.build()

        #self.mol9 = gto.M()
        #self.mol9.atom = """
        #H   -1.06  0.0   0.0
        #C   0.0   0.0   0.0
        #"""
        #self.mol9.basis = 'sto-3g'
        #self.mol9.ecp = {}
        #self.
        #self.mol9.spin = 1
        #self.mol9.build()

    def test_basic(self):
        test_mol_12 = helpers.concat_mols([self.mol1, self.mol2])

        corr_mol12 = gto.M()
        corr_mol12.atom = """
        H-0   -1.06  0.0   0.0
        C-0   0.0   0.0   0.0
        C-1   1.2   0.0   0.0
        H-1   2.26  0.0   0.0
        """
        corr_mol12.spin = 0
        corr_mol12.basis = {'C-0': 'sto-3g', 'H-0': 'sto-3g', 'C-1': '6-311g', 'H-1': '6-311g'}
        corr_mol12.build()

        self.assertEqual(test_mol_12.spin, corr_mol12.spin)
        self.assertTrue(gto.same_mol(test_mol_12, corr_mol12))
        self.assertTrue(gto.same_mol(test_mol_12, corr_mol12))

        test_mol_345 = helpers.concat_mols([self.mol3, self.mol4, self.mol5])

        corr_mol345 = gto.M()
        corr_mol345.atom = """
        O-0   -1.16   0.0   0.0
        C-1   0.0   0.0   0.0
        O-2   1.16   0.0   0.0
        """
        corr_mol345.basis = {'O-0': 'sto-3g', 'C-1': '6-311g', 'O-2': 'sto-3g'}
        corr_mol345.build()

        self.assertTrue(gto.same_basis_set(test_mol_345, corr_mol345))
        self.assertTrue(gto.same_mol(test_mol_345, corr_mol345))

    def test_remove_ovlp(self):
        test_mol_16 = helpers.concat_mols([self.mol1, self.mol6])

        corr_mol16 = gto.M()
        corr_mol16.atom = """
        H-0   -1.06  0.0   0.0
        C-0   0.0   0.0   0.0
        C-1   1.2   0.0   0.0
        H-1   2.26  0.0   0.0
        """
        corr_mol16.basis = {'C-0': 'sto-3g', 'H-0': 'sto-3g', 'C-1': '6-311g', 'H-1': '6-311g'}

        with self.assertRaises(AssertionError):
            test_mol_166 = helpers.concat_mols([test_mol_16, self.mol6])

        test_mol_378 = helpers.concat_mols([self.mol3, self.mol7, self.mol8])
        corr_mol378 = gto.M()
        corr_mol378.atom = """
        O-0   -1.16   0.0   0.0
        GHOST-C-1 2.0 0.0 0.0
        O-1    0.0   1.16   0.0
        C-2    0.0   0.0   0.0
        """
        corr_mol378.basis = {'O-0': 'sto-3g', 'O-1': 'cc-pVTZ', 'GHOST-C-1': 'aug-cc-pVDZ', 'C-2': 'cc-pVDZ'}
        corr_mol378.build()
        self.assertTrue(gto.same_basis_set(test_mol_378, corr_mol378))
        self.assertTrue(gto.same_mol(test_mol_378, corr_mol378))
        
    def test_explicit_ecp(self):
        pass

if __name__ == "__main__":
    unittest.main()
