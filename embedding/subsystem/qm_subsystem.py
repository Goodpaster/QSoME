#Defines a quantum mechanical subsystem object.

from pyscf import scf

class QMSubsystem:

    def __init__(self, qm_obj):
        self.qm_obj = qm_obj
        self.mf_obj = None
        if issubclass(type(self.qm_obj), scf.hf.SCF):
            self.mf_obj = self.qm_obj
        elif (type(self.qm_obj) is str):
            print ('external method')
        else:
            self.mf_obj = self.qm_obj._scf
        self.mol = self.mf_obj.mol
        self.proj_pot = 0.
        self.emb_pot = 0.
        self.emb_fock = 0.
        self.subsys_get_fock = self.mf_obj.get_fock
        self.mf_obj.get_hcore = lambda *args: scf.hf.get_hcore(self.mol) + self.emb_pot + self.proj_pot

    def use_emb_fock(self):
        #Overwrites the get_fock method to just return the embedded fock. This speeds up execution to avoid calculating multiple fock matrices twice.
        self.mf_obj.get_fock = lambda *args, **kwargs: self.emb_fock

    def reset_fock(self):
        #Resets the get_fock method to be just for the subsystem not an embedded fock matrix.
        self.mf_obj.get_fock = self.subsys_get_fock

    def relax(cycles, emb_fock=None):
        if emb_fock: #This should reduce computational time by not calculating the subsystem fock multiple times.
            pass
        else: #Return get fock to normal operation.
            pass

        return self.mf_obj.scf(cycles)
    def make_rdm1(self):
        return self.mf_obj.make_rdm1()
    def energy_tot(self):
        return self.mf_obj.energy_tot()

    def hl_energy(self):
        pass

    def kernel(self):
        if issubclass(type(self.qm_obj), scf.hf.SCF):
            return self.qm_obj.scf()
        elif (type(self.qm_obj) is str):
            print ('external method')
        else:
            print ('init')
            self.mf_obj.kernel()
