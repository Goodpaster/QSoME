#Sets up the oniom framework object.
#Would we ever want two model subsystems?
from pyscf import gto
import numpy as np

from embed import subsystems
from copy import deepcopy as copy

#Performs resursive concatenation to create one mol object.
def concat_mols(mol_list):
    """Concatenates Mole objects into one Mole.

    Parameters
    ----------
    mol_list : list
        List of Mole objects to concatenate into single Mole.
    """

    assert (len(mol_list) > 1), "Must have more than 1 mol object"

    total_spin = 0
    for i, mol_obj in enumerate(mol_list):
        mol1 = gto.mole.copy(mol_obj)
        total_spin += mol1.spin
        mol1.unit = 'bohr'
        atom_coords = mol1.atom_coords()
        mol1.atom = []
        uniq_atoms = set()
        for j, coord in enumerate(atom_coords):
            a_symb = mol1.atom_symbol(j) + '-' + str(i)
            uniq_atoms.add(a_symb)
            mol1.atom.append([a_symb, tuple(coord)])

        if isinstance(mol1.basis, (str, tuple, list)):
            new_basis = dict(((a, mol1.basis) for a in uniq_atoms))
        elif isinstance(mol1.basis, dict):
            old_basis = copy.copy(mol1.basis)
            new_basis = {}
            if 'default' in old_basis:
                default_basis = old_basis['default']
                new_basis = dict(((a, default_basis) for a in uniq_atoms))
                del old_basis['default']
            for atom_symb in old_basis:
                new_symb = atom_symb + '-' + str(i)
                new_basis[new_symb] = old_basis[atom_symb]
        else:
            new_basis = mol1.basis
        mol1.basis = gto.format_basis(new_basis)

        if mol1.ecp:
            if isinstance(mol1.ecp, str):
                new_ecp = dict(((a, str(mol1.ecp)) for a in uniq_atoms))
            elif isinstance(mol1.ecp, dict):
                old_ecp = copy.copy(mol1.ecp)
                if 'default' in old_ecp:
                    default_ecp = old_ecp['default']
                    new_ecp = dict(((a, default_ecp) for a in uniq_atoms))
                    del old_ecp['default']
                for atom_symb in old_ecp:
                    new_symb = atom_symb + '-' + str(i)
                    new_ecp[new_symb] = old_ecp[atom_symb]
            else:
                new_ecp = mol1.ecp
            mol1.ecp = gto.format_ecp(new_ecp)

        mol1.build()
        if i == 0:
            mol2 = gto.mole.copy(mol1)
        else:
            new_mol = gto.mole.conc_mol(mol2, mol1)
            new_mol.build()
            mol2 = new_mol

    conc_mol = mol2
    conc_mol.spin = total_spin 
    #Remove overlapping ghost atoms.
    def __remove_overlap_ghost(mol):
    
        int_dist = gto.inter_distance(mol)
        same_coords = np.argwhere(int_dist == 0.0)
        remove_list = []
        while len(same_coords) > 0:
            curr_coord = same_coords[-1]
            if curr_coord[0] == curr_coord[1]:
                same_coords = same_coords[:-1]
            else:
                atom1 = mol.atom_symbol(curr_coord[0]).lower()
                atom2 = mol.atom_symbol(curr_coord[1]).lower()
                assert ("ghost" in atom1 or "ghost" in atom2),\
                        f"{atom1.capitalize()} and {atom2.capitalize()} are overlapping!"
                if "ghost" in atom2:
                    remove_list.append(curr_coord[1])
                else:
                    remove_list.append(curr_coord[0])
                same_coords = same_coords[:-1]
                inverse_coords = (curr_coord[1], curr_coord[0])
                inverse_index = np.argwhere((same_coords == inverse_coords).all(axis=1))
                same_coords = np.delete(same_coords, inverse_index, axis=0)
    
        for remove_index in sorted(remove_list, reverse=True):
            mol.atom.pop(remove_index)
        mol.build()
        return mol

    final_mol = __remove_overlap_ghost(conc_mol)
    final_mol = conc_mol
    return final_mol

def collapse_subsystems(model_subsystems):
    #Returns a mol object consisting of all model subsystem mols combined.
    if type(model_subsystems[-1]) is list:
        mol1 = collapse_subsystems(model_subsystems[-1])
        return concat_mols(model_subsystems[0], mol1)
    elif len(model_subsystems) == 1:
        return model_subsystems[0].mol
    else:
        return concat_mols(model_subsystems)

def create_model_subsys(env_subsys, model_subsystems):
    #First create model mol object.
    model_mol = collapse_subsystems(model_subsystems)
    #create a subsystem object that uses model mol but the same other parameters as env_subsys.
    if issubclass(type(env_subsys.qm_obj), scf.hf.SCF):
        qm_obj = copy(env_subsys.qm_obj)
        qm_obj.mol = model_mol
    elif (type(env_subsys.qm_obj) is str):
        print ('external method')
    else:
        scf_obj = copy(env_subsys.mf_obj)
        scf_obj.mol = model_mol
        qm_obj = copy(env_subsys.qm_obj)
        qm_obj.__init__(scf_obj)
    model_subsys = subsystems.qm_subsystem.QMSubsystem(qm_obj)
    return model_subsys

def combine_subsystems(subsys1, subsys2):
    return create_model_subsys(subsys1, [subsys1, subsys2])

class ONIOM_Framework:

    def __init__(self, subsystems, embed_mediators):

        self.subsystems = subsystems
        self.embed_mediators = embed_mediators


    def get_energy(self):
        #Returns the total energy of th esystem.
        pass

    def kernel(self):

        total_energy = 0.
        subsys_list = self.subsystems
        external_potential = None
        mediators_list = copy(self.embed_mediators)

        #This iterative loop does the embedding. Goes one level deeper every time.
        while type(subsys_list[-1]) is list:
            env_subsys = subsys_list[0]
            model_subsys_list = subsys_list[-1]
            model_subsys = create_model_subsys(env_subsys, model_subsys_list)

            #do full system calculation
            comb_system = combine_subsystems(env_subsys, model_subsys)
            if external_potential is None:
                external_potential = np.zeros_like(comb_system.mf_scf.mo_coeff)
            comb_system.embed_pot = external_potential
            total_energy += comb_system.kernel()

            #do embedding
            embed_mediator = mediators_list.pop()
            ft_err = 1.
            ft_iter = 0
            while (ft_err > env_subsys.conv_tol and ft_iter < env_subsys.max_cycle):
                ft_err = 0.
                ft_iter += 1
                old_dmat = [env_subsys.make_rdm1(), model_subsys.make_rdm1()]
                #set the fock emb for each subsys
                e_terms = embed_mediator.get_embed_pot(env_subsys, model_subsys)
                
                #diagonalize subsys
                env_subsys.relax()
                model_subsys.relax()
                new_dmat = [env_subsys.make_rdm1(), model_subsys.make_rdm1()]

                ft_err += np.max(np.abs(new_dmat[0] - old_dmat[0]))
                ft_err += np.max(np.abs(new_dmat[1] - old_dmat[1]))

            #update the external potential

            subsys_list = subsys_list[-1]

        #do final energy calculation of highest level system

        return total_energy
