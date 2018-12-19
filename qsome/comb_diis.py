#Implement EDIIS+DIIS and ADIIS+DIIS
#By Daniel Graham

import numpy as np
from pyscf.scf import diis as scf_diis
from pyscf.lib import diis as lib_diis

class EDIIS_DIIS():

    '''combined method found in J.Chem. Phys. 137, 054110(2012)'''
    def __init__(self, mf):
        self.ediis = scf_diis.EDIIS()
        self.cdiis = scf_diis.DIIS(mf)
        self.minimum_err = 2.0

    def update(self, s, d, f, mf, h1e, vhf):
        cdiis_fock = self.cdiis.update(s,d,f)
        ediis_fock = self.ediis.update(s,d,f,mf,h1e,vhf)

        cdiis_err_vec = scf_diis.get_err_vec(s,d,f)
        cdiis_err = np.max(np.abs(cdiis_err_vec))
        if cdiis_err < self.minimum_err:
            self.minimum_err = cdiis_err
        pct_change = ((cdiis_err - self.minimum_err) / self.minimum_err) * 100
        if cdiis_err > 1.0 or pct_change >= 10:
            return ediis_fock
        elif cdiis_err < 1e-3:
            return cdiis_fock
        else:
            return ((10 * cdiis_err) * ediis_fock) + ((1 - (10 * cdiis_err)) * cdiis_fock)

class ADIIS_DIIS():

    def __init__(self, mf):
        self.adiis = scf_diis.ADIIS()
        self.cdiis = scf_diis.DIIS(mf)
        self.minimum_err = 2.0

    def update(self, s, d, f, mf, h1e, vhf):
        cdiis_fock = self.cdiis.update(s,d,f)
        adiis_fock = self.adiis.update(s,d,f,mf,h1e,vhf)

        cdiis_err_vec = scf_diis.get_err_vec(s,d,f)
        cdiis_err = np.max(np.abs(cdiis_err_vec))
        if cdiis_err < self.minimum_err:
            self.minimum_err = cdiis_err
        pct_change = ((cdiis_err - self.minimum_err) * 100) / (self.minimum_err * 100)
        if cdiis_err > 1.0 or pct_change >= 10:
            return adiis_fock
        elif cdiis_err < 1e-3:
            return cdiis_fock
        else:
            return ((10 * cdiis_err) * adiis_fock) + ((1 - (10 * cdiis_err)) * cdiis_fock)
        #if cdiis_err < 0.1: 
        #    return cdiis_fock
        #else:
        #    return adiis_fock

