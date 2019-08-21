


class InteractionMediator:

    def __init__(self, supersystems, filename=None, verbose=3, 
                 analysis=False, nproc=None, pmem=None):

        self.supersystems = supersystems
        self.filename = filename
        self.nproc = nproc
        self.pmem = pmem
        self.scrdir = scrdir

    def do_embedding(self):
        pass

    def get_embedding_energy(self):
        pass

    
