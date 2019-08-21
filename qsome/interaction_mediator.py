


class InteractionMediator:

    def __init__(self, subsystems, supersystem_kwargs=None, filename=None, verbose=3, 
                 analysis=False, nproc=None, pmem=None):

        self.subsystems = subsystems
        self.filename = filename
        self.nproc = nproc
        self.pmem = pmem
        self.scrdir = scrdir
        self.supersystems = self.gen_supersystems(supersystem_kwargs)

    def gen_supersystems(self, sup_kwargs=None):
        supersystems = []
        sorted_subs = sorted(self.subsystems, key=lambda x: x.env_order)
        while len(sorted_subs) > 0:
            curr_order = sorted_subs[0].env_order
            curr_method = sorted_subs[0].env_method
            #match_sup_kwargs = [x for x in sup_kwargs if x.env_num == curr_order]
            #assert len(match_sup_kwargs) < 2,'Ambigious supersystem settings'
            #assert len(match_sup_kwargs) > 0,'Missing supersystem settings'
            #curr_sup_kwargs = match_sup_kwargs[0]
            higher_order_subs = [x for x in sorted_subs if x.env_order > curr_order]
            combined_subs = self.combine_subsystems(higher_order_subs, curr_method)
            sub_list = []
            while sorted_subs[0].env_order == curr_order:
                sub_list.append(sorted_subs.pop(0))
            sub_list.append(combined_subs)
            supersystem = cluster_supersystems.ClusterSuperSystem(sub_list, curr_method, env_order=curr_order)
            supersystems.append(supersystem)

        return supersystems




    def combine_subsystems(self, subsystems, env_method, kwargs=None):
        mol = cluster_supersystem.concat_mols(subsystems)
        sub_order = subsystems[0].env_order
        return cluster_subsystem.ClusterEnvSubSystem(mol, env_method, env_order=sub_order)


    def do_embedding(self):
        pass

    def get_embedding_energy(self):
        pass

    
