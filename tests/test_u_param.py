from pyscf import gto, scf, hessian
import numpy as np

#THE PAPER IS ABOUT SPIN-ORBITALS not CLOSED SHELL ORBITALS.

mol = gto.M()
mol.atom = '''
O 0., 0., -0.00001
H 0., -0.757, 0.587
H 0., 0.757, 0.587
'''
mol.basis = '3-21g'
mol.build()
mf = scf.UHF(mol)
mf.kernel()

ao_ints_1 = mol.intor('int2e')
ao_ovlp_1 = mol.intor('int1e_ovlp')
dm0 = mf.make_rdm1()

mol = gto.M()
mol.atom = '''
O 0., 0., 0.00001
H 0., -0.757, 0.587
H 0., 0.757, 0.587
'''
mol.basis = '3-21g'
mol.build()
mf = scf.UHF(mol)
mf.kernel()
ao_ints_2 = mol.intor('int2e')
ao_ovlp_2 = mol.intor('int1e_ovlp')

ao_diff = ((ao_ints_2-ao_ints_1)/0.00002) * .529177210903
dm_diff = ((mf.make_rdm1()-dm0)/0.00002) * .529177210903

mol = gto.M()
mol.atom = '''
O 0., 0., 0.
H 0., -0.757, 0.587
H 0., 0.757, 0.587
'''
mol.basis = '3-21g'
mol.build()
mf = scf.UHF(mol)
mf.kernel()

hessobj = mf.Hessian()

h1ao = hessobj.make_h1(mf.mo_coeff, mf.mo_occ)
mo1, mo_e1 = hessobj.solve_mo1(mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1ao)

mocc = [None, None]
mocc[0] = mf.mo_coeff[0][:,mf.mo_occ[0]>0]
mocc[1] = mf.mo_coeff[1][:,mf.mo_occ[1]>0]
dm1 = [None, None]
print (mo1[0][0].shape)
print (mocc[0].shape)
print (c)
dm1[0] = np.einsum('ypi,qi->ypq', mo1[0][0], mocc[0])
dm1[0] += dm1[0].transpose(0,2,1)
dm1[0] = dm1[0][2]
dm1[1] = np.einsum('ypi,qi->ypq', mo1[1][0], mocc[1])
dm1[1] += dm1[1].transpose(0,2,1)
dm1[1] = dm1[1][2]
occ_orbs = [None, None]
occ_orbs[0] = np.count_nonzero(mf.mo_occ[0])
occ_orbs[1] = np.count_nonzero(mf.mo_occ[1])
dm1_mo = [None, None]
dm1_mo[0] = np.dot(np.dot(mf.mo_coeff[0].T, np.einsum('ni,nm,mj->ij', mf.get_ovlp(), dm1[0], mf.get_ovlp())), mf.mo_coeff[0])
dm1_mo[1] = np.dot(np.dot(mf.mo_coeff[1].T, np.einsum('ni,nm,mj->ij', mf.get_ovlp(), dm1[1], mf.get_ovlp())), mf.mo_coeff[1])

#print (dm1)
#print (dm_diff)
#print (np.max(np.abs(dm1-dm_diff)))

#For the paper it uses 2nmo*2nmo where upper left block is alpha, lower right is beta 
#And off diagonal is ab and ba.

#VV part of dm_mo
nmo = mol.nao_nr()
dm2_mo = np.zeros((nmo*2, nmo*2))

#OO part of dm_mo
aoslices = mol.aoslice_by_atom()
O_p0, O_p1 = aoslices [0,2:]
nuc_grad_obj = mf.nuc_grad_method()
smat_ao = nuc_grad_obj.get_ovlp()
smat_ao_o = np.zeros((3,nmo, nmo))
smat_ao_o[:,O_p0:O_p1] += smat_ao[:,O_p0:O_p1]
smat_ao_o[:,:,O_p0:O_p1] += smat_ao[:,O_p0:O_p1].transpose(0,2,1)
smat_mo_o = [None, None]
smat_mo_o[0] = np.einsum('qm,mn,np->qp', mf.mo_coeff[0].T, smat_ao_o[2], mocc[0])
dm2_mo[:occ_orbs[0],:occ_orbs[0]] -= smat_mo_o[0][:occ_orbs[0],:occ_orbs[0]]
smat_mo_o[1] = np.einsum('qm,mn,np->qp', mf.mo_coeff[1].T, smat_ao_o[2], mocc[1])
dm2_mo[nmo:nmo+occ_orbs[1],nmo:nmo+occ_orbs[1]] -= smat_mo_o[1][:occ_orbs[1],:occ_orbs[1]]

#OV Part of P

#Calculate A
vir_orbs = [None, None]
vir_orbs[0] = nmo-occ_orbs[0]
vir_orbs[1] = nmo-occ_orbs[1]
a_dim = (occ_orbs[0] + occ_orbs[1]) * (vir_orbs[0] + vir_orbs[1])
a_mat = np.zeros((a_dim, a_dim))
ao_2int = mol.intor('int2e')
from pyscf import ao2mo
int2e_mo = [None, None]
int2e_mo_aa = ao2mo.kernel(ao_2int, mf.mo_coeff[0])
int2e_mo_bb = ao2mo.kernel(ao_2int, mf.mo_coeff[1])
int2e_mo_ab = ao2mo.kernel(ao_2int, (mf.mo_coeff[0],mf.mo_coeff[0],mf.mo_coeff[1],mf.mo_coeff[1]), compact=False)
int2e_mo_ab = int2e_mo_ab.reshape([nmo,nmo,nmo,nmo])
int2e_mo_ba = ao2mo.kernel(ao_2int, (mf.mo_coeff[1],mf.mo_coeff[1],mf.mo_coeff[0],mf.mo_coeff[0]), compact=False)
int2e_mo_ba = int2e_mo_ab.reshape([nmo,nmo,nmo,nmo])

#A few other terms.
for ai in range(a_dim):
    for bj in range(a_dim):
        a_val = ai//(occ_orbs[0] + occ_orbs[1])
        i_val = ai - ((occ_orbs[0] + occ_orbs[1]) * a_val)
        b_val = bj // (occ_orbs[0] + occ_orbs[1])
        j_val = bj - ((occ_orbs[0] + occ_orbs[1]) * b_val)
        #AA block
        if a_val < vir_orbs[0] and i_val < occ_orbs[0] and b_val < vir_orbs[0] and j_val < occ_orbs[0]:
            a_index = a_val + occ_orbs[0]
            b_index = b_val + occ_orbs[0]
            int2e_1 = int2e_mo_aa[a_index,b_index,i_val,j_val] - int2e_mo_aa[a_index,b_index,j_val,i_val]
            int2e_2 = int2e_mo_aa[a_index,j_val,i_val,b_index] - int2e_mo_aa[a_index,j_val,b_index,i_val]
            denom = mf.mo_energy[0][i_val] - mf.mo_energy[0][a_index]
            a_mat[ai,bj] = (int2e_1 + int2e_2)/denom
        #BB block
        elif a_val >= vir_orbs[0] and i_val >= occ_orbs[0] and b_val >= vir_orbs[0] and j_val >= occ_orbs[0]:
            a_index = a_val-vir_orbs[0] + occ_orbs[1]
            b_index = b_val-vir_orbs[0] + occ_orbs[1]
            i_index = i_val-occ_orbs[0]
            j_index = j_val-occ_orbs[0]
            int2e_1 = int2e_mo_bb[a_index,b_index,i_index,j_index] - int2e_mo_bb[a_index,b_index,j_index,i_index]
            int2e_2 = int2e_mo_bb[a_index,j_index,i_index,b_index] - int2e_mo_bb[a_index,j_index,b_index,i_index]
            denom = mf.mo_energy[1][i_index] - mf.mo_energy[1][a_index]
            a_mat[ai,bj] = (int2e_1 + int2e_2)/denom
        #AB block
        elif a_val < vir_orbs[0] and i_val < occ_orbs[0] and b_val >= vir_orbs[0] and j_val >= occ_orbs[0]:
            a_index = a_val + occ_orbs[0]
            b_index = b_val-vir_orbs[0] + occ_orbs[1]
            i_index = i_val
            j_index = j_val-occ_orbs[0]
            int2e_1 = int2e_mo_ab[a_index,b_index,i_index,j_index] - int2e_mo_ab[a_index,b_index,j_index,i_index]
            int2e_2 = int2e_mo_ab[a_index,j_index,i_index,b_index] - int2e_mo_ab[a_index,j_index,b_index,i_index]
            denom = mf.mo_energy[0][i_index] - mf.mo_energy[0][a_index]
            a_mat[ai,bj] = (int2e_1 + int2e_2)/denom

        ##BA Block
        elif a_val >= vir_orbs[0] and i_val >= occ_orbs[0] and b_val < vir_orbs[0] and j_val < occ_orbs[0]:
            a_index = a_val-vir_orbs[0] + occ_orbs[1]
            b_index = b_val + occ_orbs[1]
            i_index = i_val-occ_orbs[0]
            j_index = j_val
            int2e_1 = int2e_mo_ab[a_index,b_index,i_index,j_index] - int2e_mo_ab[a_index,b_index,j_index,i_index]
            int2e_2 = int2e_mo_ab[a_index,j_index,i_index,b_index] - int2e_mo_ab[a_index,j_index,b_index,i_index]
            denom = mf.mo_energy[1][i_index] - mf.mo_energy[1][a_index]
            a_mat[ai,bj] = (int2e_1 + int2e_2)/denom



##Terms for B
b_vec = np.zeros((a_dim))
hcore_deriv = nuc_grad_obj.hcore_generator(mol)
h1ao = hcore_deriv(0)[2]
##h1mo_vo = np.dot(mf.mo_coeff.T, np.dot(h1ao, mocc))
h1mo = [None, None]
h1mo[0] = np.dot(mf.mo_coeff[0].T, np.dot(h1ao, mf.mo_coeff[0]))
h1mo[1] = np.dot(mf.mo_coeff[1].T, np.dot(h1ao, mf.mo_coeff[1]))

##S term 2e For some reason this isn't in the pyscf version.
##s1_mo = np.dot(mf.mo_coeff.T, np.dot(smat_ao_o[2], mf.mo_coeff))
##print (s1_mo)
#int2e_mo_occ = int2e_mo[occ_orbs:,:occ_orbs,:occ_orbs,:occ_orbs]
smat_mo_o_occ = [None, None]
smat_mo_o_occ[0] = smat_mo_o[0][:occ_orbs[0], :occ_orbs[0]]
smat_mo_o_occ[1] = smat_mo_o[1][:occ_orbs[1], :occ_orbs[1]]
sterm = np.zeros((a_dim))
for ai in range(a_dim):
    a_val = ai//(occ_orbs[0] + occ_orbs[1])
    i_val = ai - ((occ_orbs[0] + occ_orbs[1]) * a_val)
    #AA
    if a_val < vir_orbs[0] and i_val < occ_orbs[0]:
        a_index = a_val + occ_orbs[0]
        i_index = i_val
        q = h1mo[0][a_index,i_index]
        q -= smat_mo_o[0] * mf.mo_energy[0][i_index]
        denom = mf.mo_energy[0][i_index] - mf.mo_energy[0][a_index]

    #BB
    elif a_val >= vir_orbs[0] and i_val >= occ_orbs[0]:
        a_index = a_val-vir_orbs[0] + occ_orbs[1]
        i_index = i_val - occ_orbs[0]
        denom = mf.mo_energy[1][i_index] - mf.mo_energy[1][a_index]
    #AB
    #BA



##s_term = np.einsum('alik,kl', int2e_mo_occ, smat_mo_o[:occ_orbs, :occ_orbs])
##s_term -= np.einsum('alki,kl', int2e_mo_occ, smat_mo_o[:occ_orbs,:occ_orbs])
#
##Ao grad term
#ao_grad_term = np.zeros((vir_orbs, occ_orbs))
#ao_2int_grad = mol.intor('int2e_ip1')[2]
##ao_2int_grad += ao_2int_grad.transpose(0,2,1,3,4) + ao_2int_grad.transpose(0,4,3,1,2) + ao_2int_grad.transpose(0,3,4,2,1)
#oxygen_2int_grad = np.zeros_like(ao_2int_grad)
#oxygen_2int_grad[O_p0:O_p1] += ao_2int_grad[O_p0:O_p1]
#oxygen_2int_grad += oxygen_2int_grad.transpose(1,0,2,3) + oxygen_2int_grad.transpose(2,3,0,1) + oxygen_2int_grad.transpose(2,3,1,0)
#dm0 = mf.make_rdm1()
#for a in range(vir_orbs):
#    for i in range(occ_orbs):
#        for m in range(nmo):
#            for n in range(nmo):
#                for l in range(nmo):
#                    for s in range(nmo):
#                        a_true = a + occ_orbs
#                        #ao_grad_term[a,i] += mf.mo_coeff.T[m,a_true] * mf.mo_coeff[n,i] * dm0[l,s] * (oxygen_2int_grad[m,l,n,s] - oxygen_2int_grad[l,m,n,s])
#                        ao_grad_term[a,i] += mf.mo_coeff.T[m,a_true] * mf.mo_coeff[l,i] * dm0[n,s] * (oxygen_2int_grad[m,l,n,s] - oxygen_2int_grad[m,s,n,l])
#
#
##Test 2electron integrals for A
#true_j = mf.get_j()
#true_k = mf.get_k()
#true_dm = mf.make_rdm1()
#test_j = np.zeros_like(true_j)
#test_k = np.zeros_like(true_k)
##print (np.max(mf.get_veff() - (true_j*2. - true_k)*.5))
#for m in range(nmo):
#    for n in range(nmo):
#        for l in range(nmo):
#            for s in range(nmo):
#                test_j[m,n] += ao_2int[m,n,l,s]*true_dm[l,s]
#                test_k[m,n] += ao_2int[m,s,l,n]*true_dm[l,s]
#
##print (np.max(true_j - test_j))
##print (np.max(true_k - test_k))
#true_2e_pot = mf.get_veff()
#test_2e_pot = np.zeros_like(true_2e_pot)
#for m in range(nmo):
#    for n in range(nmo):
#        for l in range(nmo):
#            for s in range(nmo):
#                test_2e_pot[m,n] += true_dm[l,s] * (2.*ao_2int[m,n,l,s] - ao_2int[m,s,l,n])
#
##print (np.max(true_2e_pot))
##print (np.max(test_2e_pot))
##print (np.max(true_2e_pot-test_2e_pot* 0.5))
#
#mat_size = (vir_orbs*occ_orbs)
#for n in range(mat_size):
#    for m in range(mat_size):
#        a = n // occ_orbs
#        i = n - (a * occ_orbs)
#        b = m // occ_orbs
#        j = m - (b*occ_orbs)
#        a_index = a + occ_orbs
#        b_index = b + occ_orbs
#        denom = mf.mo_energy[i] - mf.mo_energy[a_index]
#        #Calculate Q
#        q = h1mo[a_index,i] - smat_mo_o[a_index,i]*mf.mo_energy[i] - sterm[a,i] + ao_grad_term[a, i]
#        a_num = (int2e_mo[a_index,b_index,i,j] - int2e_mo[a_index,j,i,b_index]) + (int2e_mo[a_index,j,i,b_index] - int2e_mo[a_index,b_index,i,j])
#        a_mat[n,m] = a_num/denom
#        b_vec[n] += q/denom
#
#u = np.dot(np.linalg.inv(np.identity(a_mat.shape[0]) - a_mat), b_vec)
#u = u.reshape(vir_orbs, occ_orbs)
##print (u)
#for n in range(mat_size):
#    a = n // occ_orbs
#    i = n - (a * occ_orbs)
#    a_index = a + occ_orbs
#    dm2_mo[i, a_index] = u[a,i]
#    dm2_mo[a_index, i] = u.T[i,a]
##c1 = np.einsum('mi,nm->ni', u, mf.mo_coeff)
##grad_mocc = c1[:,mf.mo_occ>0]
##mocc = mf.mo_coeff[:,mf.mo_occ>0]

#print (np.max(np.abs(dm1_mo[0][:occ_orbs[0],:occ_orbs[0]]-dm2_mo[:occ_orbs[0],:occ_orbs[0]])))
#print (np.max(np.abs(dm1_mo[1][:occ_orbs[1],:occ_orbs[1]]-dm2_mo[nmo:nmo+occ_orbs[1],nmo:nmo+occ_orbs[1]])))


print (np.max(np.abs(dm1_mo[occ_orbs:,:occ_orbs]+dm2_mo[occ_orbs:,:occ_orbs])))
print (np.max(np.abs(dm1_mo[occ_orbs:,:occ_orbs]-dm2_mo[occ_orbs:,:occ_orbs])))
#dm2 = np.einsum('rs,mr,ns->mn', dm2_mo, mf.mo_coeff.T, mf.mo_coeff)
#dm2 += dm2.transpose()
#print (dm1)
#print (dm2_mo)

