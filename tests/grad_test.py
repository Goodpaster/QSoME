# A test to be tried later.


#import unittest
#import os
#import shutil
#import re
#
#from copy import copy
#
#from qsome import cluster_subsystem, cluster_supersystem
#from pyscf import gto, lib, scf, dft
#import numpy as np
#
##Closed Shell
#mol = gto.Mole()
##mol.verbose = 4
#mol.atom = '''
#He 0.0 0.0 0.0
#'''
#mol.basis = 'cc-pVDZ'
#mol.build()
#env_method = 'b3lyp'
#active_method = 'rhf'
#subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)
#
#mol2 = gto.Mole()
##mol2.verbose = 4
#mol2.atom = '''
#He 2.510 0.0 0.0
#'''
#mol2.basis = 'cc-pVDZ'
#mol2.build()
#env_method = 'b3lyp'
#subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
#supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'b3lyp', ft_initguess='minao', ft_cycles=50)
#supsystem_e = supersystem.get_supersystem_energy()
#supersystem.freeze_and_thaw()
#subsystem_grad = subsys.get_env_nuc_grad()
#subsys.active_in_env_energy()
#subsys.get_active_nuc_grad()
#supersystem_grad = supersystem.get_embedding_nuc_grad()
#total_energy = supsystem_e - supersystem.subsystems[0].env_energy + supersystem.subsystems[0].active_energy
#print (total_energy)
#
#
