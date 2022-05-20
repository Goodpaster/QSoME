import unittest
from qsome.subsystem import qm_subsystem
from pyscf import gto, scf, dft, tools

import numpy as np
import tempfile

class TestQMSubsystemMethods(unittest.TestCase):

    def setUp(self):
        cs_mol = gto.Mole()
        cs_mol.verbose = 3
        cs_mol.atom = '''
        O 0.0 0.0 0.0
        H 0.758602 0.00 0.504284
        H 0.758602 0.00 -0.504284'''
        cs_mol.basis = '3-21g'
        cs_mol.build()
        self.cs_hf = scf.RHF(cs_mol)
        self.cs_dft = dft.RKS(cs_mol)
        self.cs_dft.xc = 'lda'


        os_mol = gto.Mole()
        os_mol.verbose = 3
        os_mol.atom = '''
        Li 0.0 0.0 0.0
        '''
        os_mol.basis = '3-21g'
        os_mol.spin = 1
        os_mol.build()

        self.os_hf = scf.UHF(os_mol)
        self.os_dft = dft.UKS(os_mol)
        self.os_dft.xc = 'lda'

