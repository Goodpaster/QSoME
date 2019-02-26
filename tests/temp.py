from qsome import cluster_subsystem, cluster_supersystem
from pyscf import gto, lib, scf, dft

mol = gto.Mole()
mol.verbose = 3
mol.atom = '''
C          1.32733        0.65646       -0.00000
H          1.31466        1.29233       -0.88425
H          1.31464        1.29232        0.88424
Cl         2.91844       -0.18529        0.00001
C          0.18538       -0.32225       -0.00002
H          0.26448       -0.97314       -0.87599
H          0.26448       -0.97316        0.87594
C         -1.15407        0.38973       -0.00001
H         -1.22228        1.04790        0.87504
H         -1.22230        1.04790       -0.87506
C         -2.32898       -0.56797       -0.00000
H         -2.25617       -1.22432       -0.87362
H         -2.25615       -1.22432        0.87362
'''
mol.basis = 'sto-3g'
mol.charge = -1
mol.build()
env_method = 'm06'
active_method = 'ccsd'
subsys = cluster_subsystem.ClusterActiveSubSystem(mol, env_method, active_method)

mol2 = gto.Mole()
mol2.verbose = 3
mol2.atom = '''
C         -3.66168        0.14952        0.00001
H         -4.50033       -0.54680        0.00004
H         -3.76119        0.78917        0.87984
H         -3.76123        0.78914       -0.87983
'''
mol2.basis = 'sto-3g'
mol2.charge = 1
mol2.build()
env_method = 'm06'
subsys2 = cluster_subsystem.ClusterEnvSubSystem(mol2, env_method)
supersystem = cluster_supersystem.ClusterSuperSystem([subsys, subsys2], 'm06', ft_initguess='localsup')
