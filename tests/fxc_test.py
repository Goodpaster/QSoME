# A test to be tried later.
from copy import copy
from pyscf import gto, lib, scf, dft
import numpy as np

mol_total = gto.Mole()
mol_total.atom = '''
C  0.5000  0.2500  2.8660
C  0.0000  0.7500  3.7320
C  0.0000  0.7500  2.0000
H  0.0000  0.2131  4.0420
H  0.0000  1.0600  4.2690
H  0.0000  1.2869  3.4220
H  0.0000  1.2869  2.3100
H  0.0000  1.0600  1.4631
H  0.0000  0.2131  1.6900
'''
mol_total.basis = '3-21g'
mol_total.spin = 2
mol_total.build()
super_dft = dft.UKS(mol_total)
super_dft.xc = 'lda'
super_dft.kernel()
dm = super_dft.make_rdm1()
ni = super_dft._numint
#Text nr_fxc method
vmat = ni.nr_fxc(mol_total, super_dft.grids, 'lda', dm, dm, spin=2, verbose=9)

print (vmat[0] * 3.) #Why is this off by a factor of 3?
vhf = super_dft.get_veff()
#print (vhf.vj.shape)
print (vhf[0] - vhf.vj)
#print (vhf[1] - vhf.vj)
#nao,nmo = super_dft.mo_coeff[0].shape
#rho0, vxc, fxc = ni.cache_xc_kernel(mol_total, super_dft.grids, super_dft.xc, super_dft.mo_coeff, super_dft.mo_occ, mol_total.spin)
#print (len(rho0))
#print (len(vxc))
#print (len(fxc))
