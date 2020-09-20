# A module to tests the methods of the molpro

import unittest
import os
import shutil
import re

from copy import copy

from qsome import cluster_subsystem, cluster_supersystem
from pyscf import gto, lib, scf, dft

import numpy as np

class TestCalculateMolpro(unittest.TestCase):

    def test_ShPyscf2Molpro(self):
        pass
    def test_write_h0(self):
        pass
    def test_pyscf2molpro_geom(self):
        pass
    def test_convert_basis_to_molpro(self):
        pass
    def test_pyscf2molpro_basis(self):
        pass
    def test_generate_molpro_input(self):
        pass
    def test_molpro_energy(self):
        pass
