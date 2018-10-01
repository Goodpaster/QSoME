# A method to define all cluster supsystem objects
# Daniel Graham

import subsystem
class ClusterEnvSubSystem(subsystem.Subsystem):

    def __init__(self, mol, env_method, filename=None, smearsigma=0, damp=0, 
                 shift=0, subcycles=1, freeze=False, initguess='minao',
                 grid=4, verbose=4, analysis=False, debug=False)

        self.mol = mol
        self.env_method = env_method

        #Check if none
        self.filename = filename

        self.smearsigma = smearsigma
        self.damp = damp
        self.shift = shift

        # diagonalization subcycles
        self.subcycles = subcycles 

        self.freeze = freeze #Whether to freeze during freeze and thaw or not.

        self.initguess = initguess

        self.grid = grid
        self.verbose = verbose
        self.analysis = analysis
        self.debug = debug

class ClusterActiveSubSystem(ClusterEnvSubSystem):

    def __init__(self, active_method, localize_orbitals=False, active_orbs=None,
                 conv=1e-8, grad=1e-8, cycles=100, damp=0, shift=0,
                 smearsigma=0, initguess='minao')A
        super().__init__()

class ClusterExcitedSubSystem(ClusterActiveSubSystem):

    def__init__(self):
        super().__init__()

