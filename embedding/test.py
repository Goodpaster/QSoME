from embedding import framework
from embedding import subsystem
from pyscf import gto, scf, dft


mol1 = gto.M()
mol1.atom = '''
O 0. 0. 0.'''
mol1.build()

mol2 = gto.M()
mol2.atom = '''
He 1. 0. 0.'''
mol2.build()

mf1 = scf.RHF(mol1)
mf2 = dft.RKS(mol2)
mf2.xc = 'pbe'

sub1 = qm_subsystem.QMSubsystem(mf1)
sub2 = qm_subsystem.QMSubsystem(mf2)

sub_list = [sub1, [sub2]]

mediator_list = [None]
oniom_obj = oniom_framework.ONIOM_Framework(sub_list, mediator_list)
oniom_obj.kernel()
