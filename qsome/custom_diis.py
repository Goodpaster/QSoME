#Implement EDIIS+DIIS and ADIIS+DIIS
#By Daniel Graham

import scipy
import numpy as np
from pyscf import lib, scf


DEBUG = True

class EDIIS(scf.diis.EDIIS):

    def update(self, s, d, f, elec_e):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = np.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = np.zeros(shape, dtype=f.dtype)
            self._buffer['etot'] = np.zeros(self.space)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f
        self._buffer['etot'][self._head] = elec_e
        self._head += 1

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        es = self._buffer['etot']
        etot, c = ediis_minimize(es, ds, fs)
        lib.logger.debug1(self, 'E %s  diis-c %s', etot, c)
        fock = np.einsum('i,i...pq->...pq', c, fs)
        return fock

def ediis_minimize(es, ds, fs):
    nx = es.size
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = np.einsum('inpq,jnqp->ij', ds, fs).real
    diag = df.diagonal()
    df = diag[:,None] + diag - df - df.T

    def costf(x):
        c = x**2 / (x**2).sum()
        return np.einsum('i,i', c, es) - np.einsum('i,ij,j', c, df, c)

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = es - 2*np.einsum('i,ik->k', c, df)
        cx = np.diag(x*x2sum) - np.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return np.einsum('k,kn->n', fc, cx)

    if DEBUG:
        x0 = np.random.random(nx)
        dfx0 = np.zeros_like(x0)
        for i in range(nx):
            x1 = x0.copy()
            x1[i] += 1e-4
            dfx0[i] = (costf(x1) - costf(x0))*1e4
            print((dfx0 - grad(x0)) / dfx0)

    #res = scipy.optimize.minimize(costf, np.ones(nx), method='BFGS',
    #                              jac=grad, tol=1e-9)
    res = scipy.optimize.minimize(costf, np.ones(nx), method='BFGS', tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()


class ADIIS(scf.diis.ADIIS):

    def update(self, s, d, f, elec_e):
        if self._head >= self.space:
            self._head = 0
        if not self._buffer:
            shape = (self.space,) + f.shape
            self._buffer['dm'  ] = np.zeros(shape, dtype=f.dtype)
            self._buffer['fock'] = np.zeros(shape, dtype=f.dtype)
        self._buffer['dm'  ][self._head] = d
        self._buffer['fock'][self._head] = f

        ds = self._buffer['dm'  ]
        fs = self._buffer['fock']
        fun, c = adiis_minimize(ds, fs, self._head)
        if self.verbose >= lib.logger.DEBUG1:
            etot = elec_e + fun
            lib.logger.debug1(self, 'E %s  diis-c %s', etot, c)
        fock = np.einsum('i,i...pq->...pq', c, fs)
        self._head += 1
        return fock

def adiis_minimize(ds, fs, idnewest):
    nx = ds.shape[0]
    nao = ds.shape[-1]
    ds = ds.reshape(nx,-1,nao,nao)
    fs = fs.reshape(nx,-1,nao,nao)
    df = np.einsum('inpq,jnqp->ij', ds, fs).real
    d_fn = df[:,idnewest]
    dn_f = df[idnewest]
    dn_fn = df[idnewest,idnewest]
    dd_fn = d_fn - dn_fn
    df = df - d_fn[:,None] - dn_f + dn_fn

    def costf(x):
        c = x**2 / (x**2).sum()
        return (np.einsum('i,i', c, dd_fn) * 2 +
                np.einsum('i,ij,j', c, df, c))

    #def grad(x):
    #    x2sum = (x**2).sum()
    #    c = x**2 / x2sum
    #    fc = es - 2*np.einsum('i,ik->k', c, df)
    #    cx = np.diag(x*x2sum) - np.einsum('k,n->kn', x**2, x)
    #    cx *= 2/x2sum**2
    #    return np.einsum('k,kn->n', fc, cx)

    #if DEBUG:
    #    x0 = np.random.random(nx)
    #    dfx0 = np.zeros_like(x0)
    #    for i in range(nx):
    #        x1 = x0.copy()
    #        x1[i] += 1e-4
    #        dfx0[i] = (costf(x1) - costf(x0))*1e4
    #        print((dfx0 - grad(x0)) / dfx0)

    #res = scipy.optimize.minimize(costf, np.ones(nx), method='BFGS',
    #                              jac=grad, tol=1e-9)
    res = scipy.optimize.minimize(costf, np.ones(nx), method='BFGS', tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()

class DIIS_CDIIS():

    def __init__(self):
        self.diis = lib.diis.DIIS()
        self.cdiis = scf.diis.DIIS()
        self.minimum_err = 2.0

    def update(self, s, d, f, elec_e=None):
        diis_fock = self.diis.update(f)
        cdiis_fock = self.cdiis.update(s,d,f)

        cdiis_err_vec = scf.diis.get_err_vec(s,d,f)
        cdiis_err = np.max(np.abs(cdiis_err_vec))
        if cdiis_err < self.minimum_err:
            self.minimum_err = cdiis_err
        pct_change = ((cdiis_err - self.minimum_err) / self.minimum_err) * 100
        print (cdiis_err)
        print (pct_change)
        if cdiis_err > 1.0 or pct_change >= 10:
            print ("here1")
            return diis_fock
        elif cdiis_err < 1e-3:
            return cdiis_fock
            print ("here2")
        else:
            print ("here3")
            return ((10 * cdiis_err) * diis_fock) + ((1 - (10 * cdiis_err)) * cdiis_fock)




class EDIIS_DIIS():

    '''combined method found in J.Chem. Phys. 137, 054110(2012)'''
    def __init__(self, env_obj):
        self.env_obj = env_obj
        self.mf = env_obj.env_scf
        self.ediis = scf_diis.EDIIS()
        self.cdiis = scf_diis.DIIS(self.mf)
        self.minimum_err = 2.0

    def custom_ediis_update(self, d, f):
        if self.ediis._head >= self.ediis.space:
            self.ediis._head = 0
        if not self.ediis._buffer:
            shape = (self.ediis.space,) + f.shape
            self.ediis._buffer['dm'  ] = np.zeros(shape, dtype=f.dtype)
            self.ediis._buffer['fock'] = np.zeros(shape, dtype=f.dtype)
            self.ediis._buffer['etot'] = np.zeros(self.ediis.space)
        self.ediis._buffer['dm'  ][self.ediis._head] = d
        self.ediis._buffer['fock'][self.ediis._head] = f
        self.ediis._buffer['etot'][self.ediis._head] = self.env_obj.get_env_elec_energy(dmat=d)
        self.ediis._head += 1

        ds = self.ediis._buffer['dm'  ]
        fs = self.ediis._buffer['fock']
        es = self.ediis._buffer['etot']
        etot, c = scf_diis.ediis_minimize(es, ds, fs)
        lib.logger.debug1(self.ediis, 'E %s  diis-c %s', etot, c)
        fock = np.einsum('i,i...pq->...pq', c, fs)
        return fock

    def update(self, d, f):
        s = self.mf.get_ovlp()
        cdiis_fock = self.cdiis.update(s,d,f)
        ediis_fock = self.custom_ediis_update(d,f)

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

    def __init__(self, env_obj):
        self.adiis = scf_diis.ADIIS()
        self.cdiis = scf_diis.DIIS(env_obj)
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

