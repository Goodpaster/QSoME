#Implement EDIIS+DIIS and ADIIS+DIIS
#By Daniel Graham

import numpy as np
from pyscf.scf import diis as scf_diis
from pyscf.lib import diis as lib_diis
from pyscf.lib import logger

class EDIIS(scf_diis.EDIIS):

    def update(self, s, d, f, emb_energy_elec):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = np.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = np.zeros(shape, dtype=f.dtype)
            self._buffer['etot'] = np.zeros(self.space)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f
        self._buffer['etot'][self._head] = emb_energy_elec
        self._head += 1

        ds = self._buffer['dm' ]
        fs = self._buffer['fock']
        es = self._buffer['etot']
        etot, c = scf_diis.ediis_minimize(es, ds, fs)
        logger.debug1(self, 'E %s  diis-c %s', etot, c)
        fock = np.einsum('i,i...pq->...pq', c, fs)
        return fock

class ADIIS(scf_diis.ADIIS):

    def update(self, s, d, f, emb_energy_elec):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = np.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = np.zeros(shape, dtype=f.dtype)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f

        ds = self._buffer['dm' ]
        fs = self._buffer['fock']
        fun, c = scf_diis.adiis_minimize(ds, fs, self._head)
        if self.verbose >= logger.DEBUG1:
            etot = emb_energy_elec
            logger.debug1(self, 'E %s  diis-c %s ', etot, c)
        fock = np.einsum('i,i...pq->...pq', c, fs)
        self._head += 1
        return fock

class EDIIS_DIIS():

    '''combined method found in J.Chem. Phys. 137, 054110(2012)'''
    def __init__(self):
        self.ediis = EDIIS()
        self.diis = lib_diis.DIIS()
        self.minimum_err = 2.0

    def update(self, s, d, f, emb_elec_energy):
        self.diis_fock = self.diis.update(f)
        self.ediis_fock = self.ediis.update(s,d,f, emb_elec_energy)

    def mix(self):
        diis_err_vec = self.diis.get_err_vec(self.diis._head-1)
        diis_err = np.max(np.abs(diis_err_vec))
        if diis_err < self.minimum_err:
            self.minimum_err = diis_err
        pct_change = ((diis_err - self.minimum_err) / self.minimum_err) * 100
        if diis_err > 1.0 or pct_change >= 10:
            return self.ediis_fock
        elif diis_err < 1e-3:
            return self.diis_fock
        else:
            return ((10 * diis_err) * self.ediis_fock) + ((1 - (10 * diis_err)) * self.diis_fock)

class ADIIS_DIIS():

    def __init__(self):
        self.adiis = ADIIS()
        self.diis = lib_diis.DIIS()
        self.minimum_err = 2.0

    def update(self, s, d, f, emb_elec_energy):
        self.diis_fock = self.diis.update(f)
        self.adiis_fock = self.adiis.update(s,d,f,emb_elec_energy)

    def mix(self):
        diis_err_vec = self.diis.get_err_vec(self.diis._head-1)
        diis_err = np.max(np.abs(diis_err_vec))
        if diis_err < self.minimum_err:
            self.minimum_err = diis_err
        pct_change = ((diis_err - self.minimum_err) * 100) / (self.minimum_err * 100)
        if diis_err > 1.0 or pct_change >= 10:
            return self.adiis_fock
        elif diis_err < 1e-3:
            return self.diis_fock
        else:
            return ((10 * diis_err) * self.adiis_fock) + ((1 - (10 * diis_err)) * self.diis_fock)

class EDIIS_CDIIS():

    '''combined method found in J.Chem. Phys. 137, 054110(2012)'''
    def __init__(self):
        self.ediis = EDIIS()
        self.cdiis = scf_diis.DIIS()
        self.minimum_err = 2.0

    def update(self, s, d, f, emb_elec_energy):
        cdiis_fock = self.cdiis.update(s,d,f)
        ediis_fock = self.ediis.update(s,d,f, emb_elec_energy)

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

class ADIIS_CDIIS():

    def __init__(self):
        self.adiis = ADIIS()
        self.cdiis = scf_diis.DIIS()
        self.minimum_err = 2.0

    def update(self, s, d, f, emb_elec_energy):
        cdiis_fock = self.cdiis.update(s,d,f)
        adiis_fock = self.adiis.update(s,d,f,emb_elec_energy)

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

