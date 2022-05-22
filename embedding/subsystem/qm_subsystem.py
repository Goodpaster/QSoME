#Defines a quantum mechanical subsystem object.

from pyscf import scf

class QMSubsystem:

    def __init__(self, qm_obj, init_guess='sup'):
        self.qm_obj = qm_obj
        self.mf_obj = None
        if issubclass(type(self.qm_obj), scf.hf.SCF):
            self.mf_obj = self.qm_obj
        elif (type(self.qm_obj) is str):
            print ('external method')
        else:
            self.mf_obj = self.qm_obj._scf
        self.mol = self.mf_obj.mol
        self.conv_tol = self.mf_obj.conv_tol
        self.max_cycle = self.mf_obj.max_cycle
        self.subcycle = 1
        self.proj_pot = 0.
        self.emb_pot = 0.
        self.emb_fock = 0.
        self.init_dmat = None
        self.init_guess = init_guess
        self.subsys_get_fock = self.mf_obj.get_fock
        self.mf_obj.get_hcore = lambda *args: scf.hf.get_hcore(self.mol) + self.emb_pot + self.proj_pot

    def use_emb_fock(self):
        #Overwrites the get_fock method to just return the embedded fock. This speeds up execution to avoid calculating multiple fock matrices twice.
        self.mf_obj.get_fock = lambda *args, **kwargs: self.emb_fock + self.proj_pot

    def reset_fock(self):
        #Resets the get_fock method to be just for the subsystem not an embedded fock matrix.
        self.mf_obj.get_fock = self.subsys_get_fock

    def relax(self, subcycle=None):
        if subcycle is None:
            subcycle = self.subcycle
        prev_max_cycle = self.qm_obj.max_cycle
        prev_verbose = self.qm_obj.verbose
        self.qm_obj.verbose = 0
        self.qm_obj.max_cycle = subcycle
        out_val = self.qm_obj.scf(dm0=self.make_rdm1())
        self.qm_obj.verbose = prev_verbose
        self.qm_obj.max_cycle = prev_max_cycle
        return out_val


    def init_den(self, sup_dmat=None):
        if self.init_guess is 'sup' and sup_dmat is not None:
            self.init_dmat = sup_dmat
        else:
            self.init_dmat = self.mf_obj.get_init_guess()


    def make_rdm1(self):
        if self.mf_obj.mo_coeff is None:
            return self.init_dmat
        return self.mf_obj.make_rdm1()

    def energy_tot(self):
        return self.qm_obj.energy_tot()

    def kernel(self):
        if issubclass(type(self.qm_obj), scf.hf.SCF):
            return self.qm_obj.scf(dm0=self.make_rdm1())
        elif (type(self.qm_obj) is str):
            print ('external method')
        else:
            print ('init')
            self.mf_obj.kernel()
            #initialize the qm object again with the new mf_obj
