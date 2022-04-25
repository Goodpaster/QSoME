import numpy as np
from qsome import cluster_subsystem, cluster_supersystem, helpers
from pyscf import gto, ao2mo
cs_mol1 = gto.M()
cs_mol1.atom = '''
N    1.7030   29.2921   -0.4884
'''
cs_mol1.basis = '3-21g'
cs_mol1.charge = -3
cs_mol1.build()

cs_mol2 = gto.M()
cs_mol2.atom = '''
F    0.1341   29.9216    0.1245
F    1.1937   28.0518    0.1245
F    2.2823   29.9034    0.1243
'''
cs_mol2.basis = '3-21g'
cs_mol2.charge = 3
cs_mol2.build()

env_method = 'hf'
hl_method = 'ccsd'
subsys1 = cluster_subsystem.ClusterHLSubSystem(cs_mol1, env_method, hl_method)
env_subsys1 = cluster_subsystem.ClusterEnvSubSystem(cs_mol2, env_method)
sup_mol11 = helpers.concat_mols([cs_mol1, cs_mol2])
fs_scf_obj_1 = helpers.gen_scf_obj(sup_mol11, env_method)
supersystem = cluster_supersystem.ClusterSuperSystem([subsys1, env_subsys1], env_method, fs_scf_obj_1)
supersystem.init_density()
supersystem.freeze_and_thaw()

subsys1_mo_coeff = np.zeros_like(supersystem.get_emb_dmat())
subsys2_mo_coeff = np.zeros_like(supersystem.get_emb_dmat())
s2s = supersystem.sub2sup
subsys1_mo_coeff[np.ix_(s2s[0], s2s[0])] += subsys1.env_mo_coeff[0]
subsys2_mo_coeff[np.ix_(s2s[1], s2s[1])] += env_subsys1.env_mo_coeff[0]

eri = supersystem.mol.intor('int2e')
print (eri.shape)
#test_mo = np.einsum('mnzs,mi,nj,zk,sl->ijkl', eri, subsys1_mo_coeff, subsys2_mo_coeff, subsys1_mo_coeff, subsys1_mo_coeff)

mo_2e = ao2mo.kernel(supersystem.mol, (subsys1_mo_coeff, subsys2_mo_coeff, subsys1_mo_coeff, subsys1_mo_coeff), compact=False)
mo_2e = mo_2e.reshape(eri.shape)

#print (test_mo.shape)
print (mo_2e.shape)


#Calculate X term in MO basis

#Calculate X term in AO basis
