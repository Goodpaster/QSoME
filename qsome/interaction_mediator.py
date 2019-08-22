


from qsome.cluster_subsystem import ClusterEnvSubSystem, ClusterHLSubSystem
from qsome.cluster_supersystem import ClusterSuperSystem
from qsome import cluster_supersystem

class InteractionMediator:

    def __init__(self, subsystems, supersystem_kwargs=None, filename=None, verbose=3, 
                 analysis=False, nproc=None, pmem=None, scrdir=None):

        self.subsystems = subsystems
        self.supersystem_kwargs = supersystem_kwargs
        self.filename = filename
        self.nproc = nproc
        self.pmem = pmem
        self.scrdir = scrdir
        self.supersystems = self.gen_supersystems()

    def gen_supersystems(self):
        sup_kwargs = self.supersystem_kwargs
        supersystems = []
        sorted_subs = sorted(self.subsystems, key=lambda x: x.env_order)
        while len(sorted_subs) > 0:
            curr_order = sorted_subs[0].env_order
            curr_method = sorted_subs[0].env_method
            curr_sup_kwargs = {}
            if sup_kwargs is not None:
                match_sup_kwargs = [x for x in sup_kwargs if x.env_num == curr_order]
                assert len(match_sup_kwargs) < 2,'Ambigious supersystem settings'
                curr_sup_kwargs = match_sup_kwargs[0]
            higher_order_subs = [x for x in sorted_subs if x.env_order > curr_order]
            sub_list = []
            while len(sorted_subs) > 0 and sorted_subs[0].env_order == curr_order:
                sub_list.append(sorted_subs.pop(0))
            if len(higher_order_subs) > 0:
                combined_subs = self.combine_subsystems(higher_order_subs, curr_method)
                sub_list.append(combined_subs)
            supersystem = ClusterSuperSystem(sub_list, curr_method, env_order=curr_order, curr_sup_kwargs)
            supersystems.append(supersystem)

        return supersystems


    def combine_subsystems(self, subsystems, env_method, kwargs=None):
        mol = cluster_supersystem.concat_mols(subsystems)
        sub_order = subsystems[0].env_order
        sub_unrestricted = False
        for sub in subsystems:
            if sub.unrestricted:
                sub_unrestricted = True
        #Need a way of specifying the settings of the implied subsystem somehow.
        return ClusterEnvSubSystem(mol, env_method, env_order=sub_order, unrestricted=sub_unrestricted)


    def do_embedding(self):
        #Do freeze and thaw for all supersystems, passing along the potential. Upon freezing and thawing all supersystems do the high level calculation.
        ext_potential = [0., 0.]
        for i in range(len(self.supersystems)):
            curr_sup = self.supersystems[i]
            curr_sup.update_ext_pot(ext_potential)
            curr_sup.freeze_and_thaw()
            new_ext_pot = curr_sup.get_ext_pot() #This method formats the external potential for the next super system

    def get_embedding_energy(self):
        #Add all the components together to get an energy summary.
        pass

    
