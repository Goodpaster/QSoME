


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
            sub_sup_kwargs = {}
            if sup_kwargs is not None:
                match_sup_kwargs = [x for x in sup_kwargs if x['env_order'] == curr_order]
                sub_sup_kwargs = [x for x in sup_kwargs if x['env_order'] == (curr_order + 1)]
                assert len(match_sup_kwargs) < 2,'Ambigious supersystem settings'
                curr_sup_kwargs = match_sup_kwargs[0]
            higher_order_subs = [x for x in sorted_subs if x.env_order > curr_order]
            sub_list = []
            while len(sorted_subs) > 0 and sorted_subs[0].env_order == curr_order:
                sub_list.append(sorted_subs.pop(0))
            if len(higher_order_subs) > 0:
                if len(sub_sup_kwargs) > 0:
                    combined_subs = self.combine_subsystems(higher_order_subs, curr_method, fs_kwargs=sub_sup_kwargs[0])
                else:
                    combined_subs = self.combine_subsystems(higher_order_subs, curr_method)

                sub_list.append(combined_subs)
            curr_sup_kwargs['env_order'] = curr_order
            supersystem = ClusterSuperSystem(sub_list, curr_method, **curr_sup_kwargs)
            supersystems.append(supersystem)

        return supersystems


    def combine_subsystems(self, subsystems, env_method, fs_kwargs=None):
        mol = cluster_supersystem.concat_mols(subsystems)
        sub_order = subsystems[0].env_order
        sub_unrestricted = False
        if fs_kwargs is not None:
            if 'fs_unrestricted' in fs_kwargs.keys():
                sub_unrestricted = fs_kwargs['fs_unrestricted']
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
            ext_potential = new_ext_pot

    def get_emb_energy(self):
        #Add all the components together to get an energy summary.
        #E_tot = E_sup1 - Esup1.subsystems[-1] + E_sup2 - Esup2.subsystems[-1] ... - EsupN.subsystems_env_energy + EsupN.subsystems_hl_energy
        E_tot = 0
        for i in range(len(self.supersystems) - 1):
            sup = self.supersystems[i]
            sup_e = sup.get_supersystem_energy()
            sup_e = sup.get_env_energy()
            sub_e = sup.subsystems[-1].env_energy
            print (f"Supersystem {i + 1} Energy: {sup_e}")
            print (f"Higher level subsystem Energy: {sub_e}")
            E_tot += sup_e - sub_e

        sup = self.supersystems[-1]
        sup_e = sup.get_supersystem_energy()
        sup_e = sup.get_active_energy()
        sup_e = sup.get_env_energy()
        E_tot += sup_e
        for sub in sup.subsystems:
            if isinstance(sub, cluster_subsystem.ClusterHLSubSystem):
                E_tot -= sub.env_energy
                E_tot += sub.active_energy

    print("".center(80, '*'))
    print(f"Total Embedding Energy:     {E_tot}")
    print("".center(80,'*'))

        pass

    
