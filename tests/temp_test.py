import numpy as np
from pyscf import gto, dft

mol = gto.M()
mol.atom = '''
C  0.0000  0.2500  2.8660
C  0.0000  0.7500  3.7320
C  0.0000  0.7500  2.0000
H  0.0000  0.2131  4.0420
H  0.0000  1.0600  4.2690
H  0.0000  1.2869  3.4220
H  0.0000  1.2869  2.3100
H  0.0000  1.0600  1.4631
H  0.0000  0.2131  1.6900
'''

mol.basis = '3-21g'
mol.spin = 2
mol.build()
mf = dft.UKS(mol)
mf.xc = 'b3lyp'
mf.kernel()
vxc = mf.get_veff()
dm = mf.make_rdm1()
vhf = (np.trace(np.dot(dm[0], vxc[0])))
vhf += (np.trace(np.dot(dm[1], vxc[1])))
vj0 = np.trace(np.dot(dm[0], vxc.vj))
vk0 = np.trace(np.dot(dm[0], vxc.vk[0]))
vj1 = np.trace(np.dot(dm[1], vxc.vj))
vk1 = np.trace(np.dot(dm[1], vxc.vk[1]))
print (vxc.ecoul)
print (0.5 * (vj0 + vj1))
print (vxc.exc)
print (vhf - (vj0 + vj1))
print (vk0 + vk1)
print (vhf - (vj0 + vj1) + vk0 + vk1)
print (vxc.ecoul + vxc.exc)
#print (np.trace(np.dot(dm[0], vxc[0])) + np.trace(np.dot(dm[1], vxc[1])))
