# Tests for the input reader object
# Daniel Graham


import unittest
import os
import shutil
import re

from qsome.cluster_subsystem import ClusterEnvSubSystem, ClusterHLSubSystem
from qsome.cluster_supersystem import ClusterSuperSystem
from qsome.interaction_mediator import InteractionMediator
from pyscf import gto




class TestSetup(unittest.TestCase):

    def test_simple_subsystems(self):

        mol1 = gto.Mole()
        mol1.verbose = 3
        mol1.atom = '''
        O 0.0 0.0 0.0
        H 0. -2.757 2.857
        H 0. 2.757 2.857'''
        mol1.basis = 'sto-3g'
        mol1.build()

        mol2 = gto.Mole()
        mol2.verbose = 3
        mol2.atom = '''
        O 1.0 0.0 0.0
        H 1. -2.757 2.857
        H 1. 2.757 2.857'''
        mol2.basis = '3-21g'
        mol2.build()

        mol3 = gto.Mole()
        mol3.verbose = 3
        mol3.atom = '''
        O 2.0 0.0 0.0
        H 2. -2.757 2.857
        H 2. 2.757 2.857'''
        mol3.basis = '6-31g'
        mol3.build()

#        mol1 = gto.Mole()
#        mol1.verbose = 3
#        mol1.atom = '''
#        O 0.0 0.0 0.0
#        H 0. -2.757 2.857
#        H 0. 2.757 2.857'''
#        mol1.basis = 'aug-cc-pVDZ'
#        mol1.build()
#
#        mol2 = gto.Mole()
#        mol2.verbose = 3
#        mol2.atom = '''
#        O 1.0 0.0 0.0
#        H 1. -2.757 2.857
#        H 1. 2.757 2.857'''
#        mol2.basis = 'cc-pVDZ'
#        mol2.build()
#
#        mol3 = gto.Mole()
#        mol3.verbose = 3
#        mol3.atom = '''
#        O 2.0 0.0 0.0
#        H 2. -2.757 2.857
#        H 2. 2.757 2.857'''
#        mol3.basis = 'cc-pVDZ'
#        mol3.build()
#
        mol4 = gto.Mole()
        mol4.verbose = 3
        mol4.atom = '''
        O 3.0 0.0 0.0
        H 3. -2.757 2.857
        H 3. 2.757 2.857'''
        mol4.basis = 'aug-cc-pVTZ'
        mol4.build()

        mol5 = gto.Mole()
        mol5.verbose = 3
        mol5.atom = '''
        O 4.0 0.0 0.0
        H 4. -2.757 2.857
        H 4. 2.757 2.857'''
        mol5.basis = '3-21g'
        mol5.build()

        sub1 = ClusterEnvSubSystem(mol1, 'lda', env_order=1)
        sub2 = ClusterEnvSubSystem(mol2, 'lda', env_order=1)
        sub3 = ClusterEnvSubSystem(mol3, 'pbe', env_order=2)
        sub4 = ClusterEnvSubSystem(mol4, 'm06', env_order=3)
        sub5 = ClusterHLSubSystem(mol5, 'm06', 'rhf', env_order=3)
        subsystems = [sub1, sub2, sub3, sub4, sub5]
        mediator = InteractionMediator(subsystems)

if __name__ == "__main__":
    unittest.main()
