from pyscf import lo, gto
import numpy as np
from integrals import dist2, concatenate_mols

cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

NTYPE = np.int
ctypedef np.int_t NTYPE_t

cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
def localize(inp, mSup, mSCF, Dmat, Smat):
    """Localize the functions unto one fragment."""

    cdef int i, j
    s2s = inp.sub2sup

    CS = mSup.mo_coeff
    nOrbs = mSup.mol.nelec[0]
    if mSup.mol._ecp is None: mSup.mol._ecp = ''
    CL = lo.PM(mSup.mol, CS[:, :nOrbs]).kernel()
    mA = [ copy_mol(mSCF[i].mol, ghost=False) for i in range(inp.nsubsys) ]
    mS = [ gto.getints('cint1e_ovlp_sph', mA[i]._atm, mA[i]._bas, mA[i]._env) for i in range(inp.nsubsys) ]
    nssl = [ None for i in range(inp.nsubsys) ]
    for i in range(inp.nsubsys):
        nssl[i] = np.zeros(mA[i].natm, dtype=int)
        for j in range(mA[i].natm):
            ib = np.where(mA[i]._bas.transpose()[0] == j)[0].min()
            ie = np.where(mA[i]._bas.transpose()[0] == j)[0].max()
            ir = mA[i].nao_nr_range(ib, ie + 1)
            ir = ir[1] - ir[0]
            nssl[i][j] = ir

        if nssl[i].sum() != mA[i].nao_nr():
            print 'ERROR: naos not equal!'

    mAB = mSup.mol
    nsl = np.zeros(mAB.natm, dtype=int)
    for i in range(mAB.natm):
        ib = np.where(mAB._bas.transpose()[0] == i)[0].min()
        ie = np.where(mAB._bas.transpose()[0] == i)[0].max()
        ir = mAB.nao_nr_range(ib, ie + 1)
        ir = ir[1] - ir[0]
        nsl[i] = ir

    if nsl.sum() != mAB.nao_nr():
        print 'ERROR: naos not equal!'
    sub2sup = [ None for i in range(inp.nsubsys) ]
    for i in range(inp.nsubsys):
        sub2sup[i] = np.zeros(mA[i].nao_nr(), dtype=int)
        for a in range(mA[i].natm):
            match = False
            for b in range(mAB.natm):
                d = dist2(mA[i].atom_coord(a), mAB.atom_coord(b))
                if d < 0.001:
                    match = True
                    ia = nssl[i][0:a].sum()
                    ja = ia + nssl[i][a]
                    ib = nsl[0:b].sum()
                    jb = ib + nsl[b]
                    sub2sup[i][ia:ja] = range(ib, jb) 

            if not match:
                print 'ERROR: I did not find an atom match!'

    CL = CL.transpose()
    osplit = np.zeros(len(CL))
    for i in range(len(CL)):
        temp = np.zeros(inp.nsubsys)
        for j in range(1):
            dl = CL[i][sub2sup[j]]
            dl = np.outer(dl, dl)
            temp[j] = np.trace(np.dot(dl, mS[j]))

        osplit[i] = abs(temp[0])

    osplit2 = np.zeros(len(osplit), dtype=NTYPE)
    for i in range(mA[0].nelec[0]):
        a = np.where(osplit == osplit.max())[0]
        osplit2[a] = 1
        osplit[a] = 0

    Dnew = [ np.zeros_like(Dmat[i]) for i in range(inp.nsubsys) ]
    for i in range(len(CL)):
        if osplit2[i]:
            dl = CL[i][inp.sub2sup[0]]
            dl = np.outer(dl, dl)
            Dnew[0] += dl * 2.0
        else:
            dl = CL[i][inp.sub2sup[1]]
            dl = np.outer(dl, dl)
            Dnew[1] += dl * 2.0

    for i in range(inp.nsubsys):
        elec = mSCF[i].mol.nelectron
        S = Smat[np.ix_(s2s[i], s2s[i])]
        atel = np.trace(np.dot(Dnew[i], S))
        print 'Localized electrons in subsys {0}: {1:7.3f} / {2:3.0f}'.format(i + 1, atel, elec)

    return Dnew

