from pyscf import gto, scf
import numpy as np


cs_mol = gto.M()
cs_mol.atom = '''
N    1.7030   29.2921   -0.4884
F    0.1341   29.9216    0.1245
F    1.1937   28.0518    0.1245
F    2.2823   29.9034    0.1243
'''
cs_mol.basis = '3-21g'
cs_mol.build()

mf = scf.RHF(cs_mol)
mf.kernel()
mf_grad = mf.nuc_grad_method()
hcore_deriv = mf_grad.hcore_generator(cs_mol)
hcore_ao_grad = hcore_deriv(0)
aoslices = cs_mol.aoslice_by_atom()
p0,p1 = aoslices[0,2:]


veff_ao = mf_grad.get_veff(cs_mol, mf.make_rdm1())
veff_ao_atm = np.zeros_like(veff_ao)
veff_ao_atm[:,p0:p1] += veff_ao[:,p0:p1]
veff_ao_atm[:,:,p0:p1] += veff_ao[:,:,p0:p1]

x_dir_diff = 0.000001
coord_0 = cs_mol.atom_coords()
coord_0[0][0] = coord_0[0][0] - x_dir_diff
mol0 = cs_mol.set_geom_(coord_0, 'B', inplace=False)
mol0.build()

mf0 = scf.RHF(mol0)
mf0.kernel()
j0 = mf0.get_jk(mol0, mf0.make_rdm1())[0]
veff0 = mf0.get_veff(mol0, mf0.make_rdm1())
smat0 = mf0.get_ovlp()

coord_2 = cs_mol.atom_coords()
coord_2[0][0] = coord_2[0][0] + x_dir_diff
mol2 = cs_mol.set_geom_(coord_2, 'B', inplace=False)
mol2.build()

mf2 = scf.RHF(mol2)
mf2.kernel()
j2 = mf2.get_jk(mol2, mf2.make_rdm1())[0]
veff2 = mf2.get_veff(mol2, mf2.make_rdm1())
smat2 = mf2.get_ovlp()

hcore_num_diff = (mf2.get_hcore() - mf0.get_hcore())/(x_dir_diff*2.)
dmat_num_diff = (mf2.make_rdm1() - mf0.make_rdm1())/(x_dir_diff*2.)
j_num_diff = (j2-j0)/(x_dir_diff*2.)
veff_num_diff = (veff2-veff0)/(x_dir_diff*2.)
ovlp_num_diff = (smat2-smat0)/(x_dir_diff*2.)

test_ovlp_diff = np.zeros_like(ovlp_num_diff)
ovlp_grad = mf_grad.get_ovlp()[0]
test_ovlp_diff[p0:p1] += ovlp_grad[p0:p1]
test_ovlp_diff[:,p0:p1] -= ovlp_grad[:,p0:p1]
print (ovlp_num_diff)
print (test_ovlp_diff)
print (np.max(np.abs(ovlp_num_diff - test_ovlp_diff)))
print (x)


j_num_noao = j_num_diff - mf.get_jk(cs_mol, dmat_num_diff)[0]
print (j_num_noao)
ao_deriv = cs_mol.intor('int2e_ip1')[0]
full_ao = np.zeros_like(ao_deriv)
print (full_ao.shape)
full_ao[p0:p1] += ao_deriv[p0:p1]
full_ao += full_ao.transpose(1,0,2,3)
full_ao += full_ao.transpose(2,3,0,1)
full_ao *= -1.
full_ao2 = (mol2.intor('int2e') - mol0.intor('int2e'))/(x_dir_diff*2.)
test_j = mf_grad.get_jk(cs_mol, mf.make_rdm1())[0]
dmat_ao_term = np.einsum('ijkl,kl->ij', full_ao, mf.make_rdm1())
print (dmat_ao_term)
print (test_j[0])

veff_noao = veff_num_diff - mf.get_veff(cs_mol, dmat_num_diff)

#test get j
j_true = mf.get_jk(cs_mol, mf.make_rdm1())[0]
k_true = mf.get_jk(cs_mol, mf.make_rdm1())[1]

ao = cs_mol.intor('int2e')
j_test = np.einsum('ijkl,kl-> ij', ao, mf.make_rdm1())
k_test = np.einsum('ikjl,kl-> ij', ao, mf.make_rdm1())
