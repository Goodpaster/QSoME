# A method to define a cluster supersystem
# Daniel Graham


class ClusterSuperSystem(SuperSystem):

    def __init__(self, subsystems, ct_method, proj_oper='huz', filename=None,
                 ft_cycles=100, ft_conv=1e-8, ft_grad=1e-8, ft_diis=1, 
                 ft_setfermi=0, ft_initguess='minao', ft_updatefock=0, 
                 cycles=100, conv=1e-8, grad=1e-8, damp=0, shift=0, 
                 smearsigma=0, initguess='minao', includeghost=False, 
                 grid=4, verbose=3, analysis=False, debug=False):

        self.subsystems = subsystems
        self.ct_method = ct_method
        self.proj_oper = proj_oper

        #Check if none
        self.filename = filename

        # freeze and thaw settings
        self.ft_cycles = ft_cycles
        self.ft_conv = ft_conv
        self.ft_grad = ft_grad
        self.ft_diis = ft_diis
        self.ft_setfermi = ft_setfermi
        self.ft_initguess = ft_initguess
        self.ft_updatefock = ft_updatefock

        # charge transfer settings
        self.cycles = cycles
        self.conv = conv
        self.grad = grad
        self.damp = damp
        self.shift = shift
        self.smearsigma = smearsigma
        self.initguess = initguess
        self.includeghost = includeghost

        # general system settings
        self.grid = grid
        self.verbose = verbose
        self.analysis = analysis
        self.debug = debug

        self.init_density()

    def init_density(self):
        pass
