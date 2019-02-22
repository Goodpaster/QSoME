#Parameters which generate good results fastest.
import numpy as np
from pyscf import gto, scf, cc
from pyscf.cc import ccsd_t
from qsome import cluster_subsystem, cluster_supersystem

sub1_react = '''
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

sub2_react = '''
C         -3.66168        0.14952        0.00001
H         -4.50033       -0.54680        0.00004
H         -3.76119        0.78917        0.87984
H         -3.76123        0.78914       -0.87983
'''

sub1_trans = '''
C         -1.40554        0.58881       -0.01889
H         -2.26382        0.96377        0.50578
H         -1.41299        0.68348       -1.08814
Cl        -2.16538       -1.46213       -0.10012
F         -0.98381        2.61999       -0.26484
C         -0.12057        0.38814        0.70776
H          0.10969        1.34494        1.18299
H         -0.25082       -0.36818        1.48815
C          1.01165       -0.00388       -0.21483
H          0.74024       -0.92339       -0.74823
H          1.12521        0.78460       -0.96808
C          2.32405       -0.20742        0.51295
H          2.57757        0.70882        1.05856
H          2.19751       -0.98641        1.27430
'''

sub2_trans = '''
C          3.45731       -0.58142       -0.42016
H          4.40140       -0.73248        0.10838
H          3.22542       -1.50333       -0.96012
H          3.61494        0.19921       -1.16898
'''

basis_to_use = 'cc-pVDZ'
dft_method = 'm06'
active_method = 'ccsd(t)'

sub1_react_mol = gto.Mole()
sub1_react_mol.atom = sub1_react
sub1_react_mol.basis = basis_to_use
sub1_react_mol.charge = -1
sub1_react_mol.build()

sub2_react_mol = gto.Mole()
sub2_react_mol.atom = sub2_react
sub2_react_mol.basis = basis_to_use
sub2_react_mol.charge = 1
sub2_react_mol.build()

sub1_trans_mol = gto.Mole()
sub1_trans_mol.atom = sub1_trans
sub1_trans_mol.basis = basis_to_use
sub1_trans_mol.charge = -2
sub1_trans_mol.build()

sub2_trans_mol = gto.Mole()
sub2_trans_mol.atom = sub2_trans
sub2_trans_mol.basis = basis_to_use
sub2_trans_mol.charge = 1
sub2_trans_mol.build()

sub1_react = cluster_subsystem.ClusterActiveSubSystem(sub1_react_mol, dft_method, active_method)
sub2_react = cluster_subsystem.ClusterEnvSubSystem(sub2_react_mol, dft_method)
sup_react = cluster_supersystem.ClusterSuperSystem([sub1_react, sub2_react], dft_method)
sup_react.freeze_and_thaw()
sup_react.get_active_energy()
sup_react_energy = sup_react.get_supersystem_energy()
react_energy = sup_react_energy - sup_react.subsystems[0].env_energy + sup_react.subsystems[0].active_energy


sub1_trans = cluster_subsystem.ClusterActiveSubSystem(sub1_trans_mol, dft_method, active_method)
sub2_trans = cluster_subsystem.ClusterEnvSubSystem(sub2_trans_mol, dft_method)
sup_trans = cluster_supersystem.ClusterSuperSystem([sub1_trans, sub2_trans], dft_method)
sup_trans.freeze_and_thaw()
sup_trans.get_active_energy()
sup_trans_energy = sup_trans.get_supersystem_energy()
trans_energy = sup_trans_energy - sup_trans.subsystems[0].env_energy + sup_trans.subsystems[0].active_energy


