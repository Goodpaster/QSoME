


class InteractionMediator:

    def __init__(self, subsystems, env_emb_settings, filename=None, verbose=3, 
                 analysis=False, nproc=None, pmem=None, scr_dir=None):
        pass

    def gen_sub2sup(self, mol=None, subsystems=None):
        pass 

    def concat_mols(self, mols=None, mol_ordering=None):

        if mols is None or mol_ordering is None:
            mols = []
            mol_ordering = []
            for subsystem in self.subsystems:
                mols.append(gto.mol.copy(subsystem.mol))
                mol_ordering.append(subsystem.env_order)
        # Raise too few subsystem error
        assert (len(mols) < 2),"Must have more than 1 subsystem"

        fs_mols = []
        while len(mols) > 0:
            mol1 = gto.mol.copy(mols[0])
            min_order = mol_ordering[0]
            for m in range(1, len(mols)):
                mol2 = gto.mol.copy(mols[m])
                for j in range(mol2.natm):
                    old_name = mol2.atom_symbol(j) 
                    new_name = mol2.atom_symbol(j) + '-' + str(m)
                    mol2._atom[j] = (new_name, mol2._atom[j][1])
                    if old_name in mol2._basis.keys():
                        mol2._basis[new_name] = mol2._basis.pop(old_name)
                    if old_name in mol2.ecp.keys():
                        mol2.ecp[new_name] = mol2.ecp.pop(old_name)
                        mol2._ecp[new_name] = mol2._ecp.pop(old_name)
                        mol2._atm, mol2._ecpbas, mol2._env = mol2.make_ecp_env(mol2._atm, mol2._ecp, mol2._env)
                mol1 = gto.mole.conc_mol(mol1, mol2)
                if mol_ordering[m] < min_order:
                    min_order = mol_ordering[m]

            #Remove overlapping ghost atoms.
            # I still think there is a better way.
            def remove_overlap(atom_list):
                added_already = {}
                no_dup = []
                for i in range(len(atom_list)):
                    coord_tuple = tuple(atom_list[i][1])
                    atom_name = atom_list[i][0]
                    if not 'ghost' in atom_name:
                        no_dup.append(atom_list[i])
                        if coord_tuple in added_already:
                            dup_index = added_already[coord_tuple]
                            if not 'ghost' in no_dup[dup_index][0]:
                                print ("OVERLAPPING ATOMS")
                            else:
                                del no_dup[added_already[coord_tuple]]
                                for key in added_already.keys():
                                    if added_already[key] > added_already[coord_tuple]:
                                        added_already[key] -= 1

                        added_already[coord_tuple] = (len(no_dup) - 1)
                    else:
                        if coord_tuple in added_already:
                            pass 
                        else:
                            no_dup.append(atom_list[i])
                            added_already[coord_tuple] = (len(no_dup) - 1)
                return no_dup

            mol1._atom = remove_overlap(mol1._atom)    
            mol1.atom = mol1._atom
            mol1.unit = 'bohr'
            mol1.build(basis=mol1._basis)
            fs_mols.append(mol1)

            new_mols = []
            new_mol_ordering = []
            for n in range(len(mols)):
                if mol_ordering[n] != min_order:
                    new_mols.append(mols[n])
                    new_mol_ordering.append(mol_ordering[n])
            mols = new_mols
            mol_ordering = new_mol_ordering

        return fs_mols
                
    def init_fs_scf(self, fs_mol_list=None, fs_setting_list=None):
        
        if fs_mol_list is None:
            fs_mol_list = self.fs_mol_list
        if fs_setting_list is None:
            fs_setting_list = self.fs_setting_list

        for m in range(len(fs_mol_list)):
            if self.pmem is not None:
               fs_mol_list[m].max_memory = self.pmem
            
        

