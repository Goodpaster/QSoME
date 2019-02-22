# A module to tests the custom pyscf methods.

import unittest
import os
import shutil
import re

from copy import copy

from qsome import cluster_subsystem, cluster_supersystem
from pyscf import gto, lib, scf, dft

import numpy as np

class TestCustomPYSCF(unittest.TestCase):

    def test_rhf_get_fock(self):
        pass
    def test_rhf_energy_elec(self):
        pass

    def test_rohf_get_fock(self):
        pass
    def test_rohf_energy_elec(self):
        pass

    def test_uhf_get_fock(self):
        pass
    def test_uhf_energy_elec(self):
        pass

    def test_rks_energy_elec(self):
        pass
    def test_uks_energy_elec(self):
        pass

    def test_exc_uks(self):
        pass
