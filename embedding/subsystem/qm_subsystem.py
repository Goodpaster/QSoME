#Defines a quantum mechanical subsystem object.

from pyscf import scf

def get_hcore(mol, emb_pot=None, proj_pot=None):
    return scf.get_hcore(mol) + emb_pot + proj_pot

def get_fock(*args, **kwargs, emb_fock):
    return emb_fock


class QMSubsystem:

    def __init__(qm_obj):
        self.qm_obj = qm_obj
        self.mf_obj = None
        if issubclass(type(self.qm_obj), scf.hf.SCF):
            self.mf_obj = self.qm_obj
        else:
            self.mf_obj = self.qm_obj._scf
        self.proj_pot = 0.
        self.emb_pot = 0.
        self.mf_obj.get_hcore = get_hcore(self.mf_obj.mol, self.emb_pot, self.proj_pot)

    def relax(cycles, emb_fock=None):
        if emb_fock: #This should reduce computational time by not calculating the subsystem fock multiple times.
            self.mf_obj.get_fock = get_fock(*args, **kwargs, emb_fock)
        else: #Return get fock to normal operation.
            pass

        return self.mf_obj.scf(cycles)
    def make_rdm1():
        return self.mf_obj.make_rdm1()
    def energy_tot():
        return self.mf_obj.energy_tot()

    def hl_energy():
        pass
