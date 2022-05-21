#A class to do DFT embedding

from pyscf import lib
from embedding.framework import oniom_framework
import numpy as np



def get_embed_dmat(subsys1, subsys2, comb_subsys=None, s2s=None):
    if comb_subsys is None:
        comb_subsys = oniom_framework.combine_subsystems(subsys1, subsys2)
    if s2s is None:
        s2s = oniom_framework.gen_sub2sup(comb_subsys.mol, [subsys1, subsys2])

    emb_dmat = np.zeros_like(comb_subsys.make_rdm1()) #This only works if given an rdm1
    if emb_dmat.ndim > 2:
        emb_dmat[np.ix_([0,1], s2s[0], s2s[0])] += subsys1.make_rdm1()
        emb_dmat[np.ix_([0,1], s2s[1], s2s[1])] += subsys2.make_rdm1()

    else:
        emb_dmat[np.ix_(s2s[0], s2s[0])] += subsys1.make_rdm1()
        emb_dmat[np.ix_(s2s[1], s2s[1])] += subsys2.make_rdm1()

    return emb_dmat

def get_embed_pot(subsys1, subsys2, comb_subsys=None, s2s=None, proj_oper=None, nake=None):

    if comb_subsys is None:
        comb_subsys = oniom_framework.combine_subsystems(subsys1, subsys2)
    if s2s is None:
        s2s = oniom_framework.gen_sub2sup(comb_subsys.mol, [subsys1, subsys2])

    emb_dmat = get_embed_dmat(subsys1, subsys2, comb_subsys, s2s)
    
    emb_fock = comb_subsys.mf_obj.get_fock(dm=emb_dmat)

    s1_proj = None
    s2_proj = None

    if proj_oper.lower() in ('huz', 'huzinaga'):
        if emb_fock.ndim > 2:
            s1_proj = np.zeros_like(subsys1.make_rdm1())
            s1_fock_ab = emb_fock[np.ix_([0,1], s2s[0], s2s[1])]
            s1_smat_ba = comb_subsys.mf_obj.get_ovlp()[np.ix_(s2s[1], s2s[0])]
            fock_den_smat = [None, None]
            fock_den_smat[0] = np.dot(s1_fock_ab[0], np.dot(subsys2.make_rdm1()[0], s1_smat_ba))
            fock_den_smat[1] = np.dot(s1_fock_ab[1], np.dot(subsys2.make_rdm1()[1], s1_smat_ba))
            s1_proj[0] += -1. * (fock_den_smat[0] + fock_den_smat[0].T)
            s1_proj[1] += -1. * (fock_den_smat[1] + fock_den_smat[1].T)

            s2_proj = np.zeros_like(subsys2.make_rdm1())
            s2_fock_ab = emb_fock[np.ix_([0,1], s2s[1], s2s[0])]
            s2_smat_ba = comb_subsys.mf_obj.get_ovlp()[np.ix_(s2s[0], s2s[1])]
            fock_den_smat = [None, None]
            fock_den_smat[0] = np.dot(s2_fock_ab[0], np.dot(subsys1.make_rdm1()[0], s2_smat_ba))
            fock_den_smat[1] = np.dot(s2_fock_ab[1], np.dot(subsys1.make_rdm1()[1], s2_smat_ba))
            s2_proj[0] += -1. * (fock_den_smat[0] + fock_den_smat[0].T)
            s2_proj[1] += -1. * (fock_den_smat[1] + fock_den_smat[1].T)
        else:
            s1_fock_ab = emb_fock[np.ix_(s2s[0], s2s[1])]
            s1_smat_ba = comb_subsys.mf_obj.get_ovlp()[np.ix_(s2s[1], s2s[0])]
            fock_den_smat = np.dot(s1_fock_ab, np.dot(subsys2.make_rdm1(), s1_smat_ba))
            s1_proj = -.5 * (fock_den_smat + fock_den_smat.T)

            s2_fock_ab = emb_fock[np.ix_(s2s[1], s2s[0])]
            s2_smat_ba = comb_subsys.mf_obj.get_ovlp()[np.ix_(s2s[0], s2s[1])]
            fock_den_smat = np.dot(s2_fock_ab, np.dot(subsys1.make_rdm1(), s2_smat_ba))
            s2_proj = -.5 * (fock_den_smat + fock_den_smat.T)

    if emb_fock.ndim > 2:
        s1_emb_fock = emb_fock[np.ix_([0,1], s2s[0], s2s[0])]
        s2_emb_fock = emb_fock[np.ix_([0,1], s2s[1], s2s[1])]

    else:
        s1_emb_fock = emb_fock[np.ix_(s2s[0], s2s[0])]
        s2_emb_fock = emb_fock[np.ix_(s2s[1], s2s[1])]
    return [lib.tag_array(s1_emb_fock, proj_pot=s1_proj), lib.tag_array(s2_emb_fock, proj_pot=s2_proj)]


class DFTEmbedding:

    def __init__(self, proj_oper=None, nake=None):
        self.comb_subsys = None
        self.s2s = None
        self.proj_oper = proj_oper
        self.nake = nake

    def get_embed_pot(self, subsys1, subsys2):
        return get_embed_pot(subsys1, subsys2, self.comb_subsys, self.s2s, self.proj_oper, self.nake)
