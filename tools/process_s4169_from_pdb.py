"""
This script process s4169 dataset into npz with mutation position information
without dependency on torchdrug.
"""
import os, sys
from gemnet.training.rotamer_utils import RotamerBase
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from rdkit import Chem
import pandas as pd
from torch.utils.data import Dataset
from gemnet.training.rotamer_utils import RotamerBase, RotamorResidueConfig

residue2id = {"GLY": 0, "ALA": 1, "SER": 2, "PRO": 3, "VAL": 4, "THR": 5, "CYS": 6, "ILE": 7, "LEU": 8,
                "ASN": 9, "ASP": 10, "GLN": 11, "LYS": 12, "GLU": 13, "MET": 14, "HIS": 15, "PHE": 16,
                "ARG": 17, "TYR": 18, "TRP": 19}
id2residue = {v:k for k,v in residue2id.items()}
atom_name2id = {"C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
                "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
                "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
                "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
                "OH": 33, "OXT": 34, "SD": 35, "SG": 36, "UNK": 37}
id2residue = {v: k for k, v in residue2id.items()}
id2residue_symbol = {0: "G", 1: "A", 2: "S", 3: "P", 4: "V", 5: "T", 6: "C", 7: "I", 8: "L", 9: "N",
                        10: "D", 11: "Q", 12: "K", 13: "E", 14: "M", 15: "H", 16: "F", 17: "R", 18: "Y", 19: "W"}
symbol2residueid = {v:k for k,v in id2residue_symbol.items()}
id2atom_name = {v: k for k, v in atom_name2id.items()}


## utils functions ##
def concat_dict(data_dict):
    """
    concat a data dict where the value of each key 
    is list of np.array or just a list.
    """
    data_dict_processed = defaultdict(list)
    for k, v in data_dict.items():
        if isinstance(v[0], np.ndarray) and len(v[0].shape):
            data_dict_processed[k] = np.concatenate(v)
        else:
            data_dict_processed[k] = np.stack(v)
    return data_dict_processed

def compute_atom_position(atom):
    """
    Atom position in the molecular conformation.
    Return 3D position if available, otherwise 2D position is returned.

    Note it takes much time to compute the conformation for large molecules.
    """
    mol = atom.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return [pos.x, pos.y, pos.z]

def dict2numpy(data):
    return {k: np.array(v) for k, v  in data.items()}

class MutationChangeStructure(Dataset):
    """
    A light-weight Mutation Change dataset (could be used to generate .npz) 
    with dependency on rdkit
    The file structure is:
        {path}/
            train/
                data.csv
                data/
                    xxx.pdb
            val/
            test/

    """
    sub_folders = ["train", "val", "test"]

    protein_fields = ["id", "E", "num_residue", "N", "mutation_position"]
    residue_fields = ["residue_names", "residue_types_origin", "chain_ids", "residue_types", "relative_chain_ids"]
    atom_fields = ["F", "residue_ids", "atom_names", "atom_ids", "Z", "R", "atomic_numbers"]

    data_fields = protein_fields + residue_fields + atom_fields


    chain_dict = {chr(65 + i): i for i in range(26)}

    def __init__(self, path):
        self.path = path
        ## data place holder
        self.wild_type_data = defaultdict(list)
        self.mutant_data = defaultdict(list)
        self.data_frames = list()
        sub_paths = [os.path.join(self.path, sub_folder) for sub_folder in self.sub_folders]
        for sub_path in sub_paths:
            data_frame = pd.read_csv(os.path.join(sub_path, "data.csv"))
            pdb_path = os.path.join(sub_path, "data")
            wild_type_data_dict, mutant_data_dict = self.load_data(data_frame, pdb_path)
            self.data_frames.append(data_frame)

            for key in wild_type_data_dict.keys():
                self.wild_type_data[key].extend(wild_type_data_dict[key])
                self.mutant_data[key].extend(mutant_data_dict[key])

        ## concat dict
        self.wild_type_data, self.mutant_data = concat_dict(self.wild_type_data), concat_dict(self.mutant_data)
        ## to variadic
        self.wild_type_data['id'] = np.arange(len(self.wild_type_data['N']))
        self.mutant_data['id'] = np.arange(len(self.mutant_data['N']))
        shape_dict = {k:v.shape for k,v in self.wild_type_data.items()}
        print(shape_dict)
        self.wild_type_data, self.mutant_data = self.to_variadic(self.wild_type_data), self.to_variadic(self.mutant_data)


    def load_data(self, data_frame, pdb_path):
        """
        load the data based on a data frame and a pdb list.
        
        Returned:
            wild_type_data_dict: (dict of list)
        """
        wild_type_data_dict = defaultdict(list)
        mutant_data_dict = defaultdict(list)

        for idx in tqdm(range(len(data_frame))):
            data_info = data_frame.iloc[idx, :]
            ## get data from a row ##
            wild_type_file = data_info["wt_protein"]
            mutant_file = data_info["mt_protein"]
            ddG = float(data_info["ddG"])
            mutation_info = data_info["mutation"]
            residue_before = id2residue[symbol2residueid[mutation_info[0]]]
            residue_after = id2residue[symbol2residueid[mutation_info[-1]]]
            mutation_chain_id = self.chain_dict[mutation_info[1]]
            mutation_position = int(mutation_info[2:-1]) - 1
            chain_A, chain_B = data_info["chain_a"], data_info["chain_b"]
            #########################

            ## load wild type and mutant data from pdb ##
            wild_type_path = os.path.join(pdb_path, wild_type_file)
            mutant_path = os.path.join(pdb_path, mutant_file)
            wild_type_data = self.from_pdb(wild_type_path)
            mutant_data = self.from_pdb(mutant_path)
            #############################################

            ## attach relative chain id, mutation position, add random force and target ddG ##
            def assign_relative_chain_id(chain_name, chain_A, chain_B):
                if chain_name in chain_A:
                    return 0
                elif chain_name in chain_B:
                    return 1
                else:
                    return -1

            def assign_mutation(data_dict, target_chain_id, mutation_position):
                try:
                    return np.where(np.array(data_dict["chain_ids"])==target_chain_id)[0][mutation_position]
                except:
                    return -1

            wild_type_data["relative_chain_ids"] = [assign_relative_chain_id(x, chain_A, chain_B) for x in wild_type_data["chain_names"]]
            wild_type_data["mutation_position"] = assign_mutation(wild_type_data, mutation_chain_id, mutation_position)
            wild_type_data["F"] = np.random.randn(wild_type_data["N"], 3).tolist()
            wild_type_data["E"] = ddG


            mutant_data["relative_chain_ids"] = [assign_relative_chain_id(x, chain_A, chain_B) for x in mutant_data["chain_names"]]
            mutant_data["mutation_position"] = assign_mutation(mutant_data, mutation_chain_id, mutation_position)
            mutant_data["F"] = np.random.randn(wild_type_data["N"], 3).tolist()
            mutant_data["E"] = ddG
            ########################################################################

            try:
                assert residue_before == wild_type_data['residue_names'][wild_type_data['mutation_position']]
                assert residue_after == mutant_data['residue_names'][mutant_data['mutation_position']]
            except:
                residue_types_wt = wild_type_data['residue_types_origin']
                residue_types_mt = mutant_data['residue_types_origin']
                correct_positions = np.where(np.array(residue_types_wt) != np.array(residue_types_mt))[0]
                if len(correct_positions) > 1:
                    print("multiple mutation founded in this data: {}".format(data_info))
                correct_position = correct_positions[0]
                

                print(f"row {idx} with wrong mutation position of {int(mutation_info[2:-1]) - 1}, the correct one is {correct_position}")
                wild_type_data['mutation_position'] = correct_position
                mutant_data['mutation_position'] = correct_position

            
            #####################
            wild_type_data = dict2numpy(wild_type_data)
            mutant_data = dict2numpy(mutant_data)
            for k, v in wild_type_data.items():
                wild_type_data_dict[k].append(v)
            for k, v in mutant_data.items():
                mutant_data_dict[k].append(v)


        return wild_type_data_dict, mutant_data_dict

    def from_pdb(self, pdb_file):
        if not os.path.exists(pdb_file):
            raise FileNotFoundError("No such file `%s`" % pdb_file)
        mol = Chem.MolFromPDBFile(pdb_file, sanitize=False)
        if mol is None:
            return None
        return self.from_mol(mol)

    def from_mol(self, mol):
        """
        This function is the core loading function. 
        The data would be returned as a dict:
        {   
            "N": [1, ] number of atom
            "num_residue": [1, ] number of residue

            "R": [N_atom, 3] atom position,
            "Z": [N_atom, ] atom type in residue,
            "residue_ids": [N_atom, ] indicate the residue ids the atom belongs to
            "atom_names": [N_atom, ] (str) atom names
            "atom_ids": [N_atom, ] (int) atom indices in a residue

            "residue_names": [N_residue, ] (str) residue name (one letter)
            "residue_types_origin": [N_residue, ] (int) 20 letters residue types index
            "residue_types": [N_residue, ] (int) rotamer residue type index (18 residues)
            "chain_ids": [N_residue, ] (int) chain index 
            "chain_names"" [N_residue, ] (str) chain name
        }
        """
        ## protein level fields ##
        N = 0
        num_residue = 0

        ## atom level fields ##
        R = list()
        Z = list()
        atomic_numbers = list()
        residue_ids = list()
        atom_names = list()
        atom_ids = list()

        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]

        ## residue level fields ##
        lst_residue = -1
        lst_residue_type = -1
        lst_residue_atom_names = set()
        
        residues = list()
        residue_types = list()
        residue_names = list()
        residue_types_origin = list()

        chain_ids = list()
        chain_names = list()
        atom_index_in_residue = 0
        for a_idx, atom in enumerate(atoms):
            ## only get atom information for residues that are recognizable
            residue = atom.GetPDBResidueInfo()
            type = residue.GetResidueName().strip()
            if type in residue2id:
                N += 1
                R.append(compute_atom_position(atom))
                atomic_numbers.append(atom.GetAtomicNum())
            
                name = residue.GetName().strip()

                atom_names.append(name)
                Z.append(atom_name2id.get(name, atom_name2id["UNK"]))

                if residue.GetResidueNumber() != lst_residue or type != lst_residue_type or \
                    name in lst_residue_atom_names:
                    atom_index_in_residue = 1 if a_idx==1 else 0
                    num_residue += 1
                    lst_residue = residue.GetResidueNumber()
                    lst_residue_type = type
                    lst_residue_atom_names = set()

                    chain_names.append(residue.GetChainId())
                    chain_ids.append(self.chain_dict.get(residue.GetChainId(), -1))
                    residues.append(residue)

                ## atom ids: here we just use inter residue index
                atom_ids.append(atom_index_in_residue)
                atom_index_in_residue+=1
                
                residue_ids.append(num_residue - 1)
                lst_residue_atom_names.add(name)

        if len(residues) != num_residue:
            raise ValueError("Number of residues doesn't match with number of alpha carbons")

        for i, residue in enumerate(residues):
            residue_name = residue.GetResidueName()
            residue_names.append(residue_name)
            residue_type = residue2id[residue_name]
            residue_types_origin.append(residue_type)
        
        ## convert 20 letters residue type to rotamer ##
        residue_types = [RotamerBase.residue2id.get(r.lower(), 18) for r in residue_names]

        ## pack to dict
        data_dict = dict(N=N, num_residue=num_residue, 
                        R=R, Z=Z, atomic_numbers=atomic_numbers,
                        residue_ids=residue_ids, atom_names=atom_names, atom_ids=atom_ids,  
                        residue_types=residue_types, residue_names=residue_names, 
                        residue_types_origin=residue_types_origin, chain_ids=chain_ids, 
                        chain_names = chain_names)

        return data_dict

    def to_variadic(self, data_dict):
        """
        convert the data dict to variadict by 
        adding index offset on index attributes in atom level and residue level

        In this case, `residue_ids` and `mutation_position` should be converted.
        
        """
        residue_ids = data_dict["residue_ids"] # [N_atom, ]
        N = data_dict["N"] # [N_protein, ]
        num_residue = data_dict["num_residue"] # [N_protein, ]
        offset = num_residue.cumsum(0) - num_residue # [N_protein, ]
        atom_level_offset = np.repeat(offset, N) # [N_atom, ]

        residue_ids += atom_level_offset
        data_dict["residue_ids"] = residue_ids

        mutation_position = data_dict["mutation_position"] # [N_protein, ]
        mutation_position += offset
        data_dict['mutation_position'] = mutation_position
        return data_dict

    def save_npz(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.savez(os.path.join(output_path, "wild_type.npz"), **self.wild_type_data)
        np.savez(os.path.join(output_path, "mutants.npz"), **self.mutant_data) 

    def __len__(self):
        return len(self.wild_type_data['N'])

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int64, np.int32)):
            idx = [idx]
        if isinstance(idx, tuple):
            idx = list(idx)
        if isinstance(idx, slice):
            idx = np.arange(idx.start, min(idx.stop, len(self)), idx.step)

        protein_index = idx
        residue_num = self.wild_type_data['num_residue']
        N = self.wild_type_data['N']
        residue_start = residue_num.cumsum(0) - residue_num
        residue_index = np.arange(residue_start, residue_start + residue_num[idx])
        atom_start = N.cumsum(0) - N
        atom_index = np.arange(atom_start, atom_start+N)

        wild_type_data = defaultdict(list)
        mutant_data = defaultdict(list)
        for key in self.protein_fields:
            wild_type_data[key] = self.wild_type_data[key][idx]
            mutant_data[key] = self.mutant_data[key][idx]
        for key in self.residue_fields:
            wild_type_data[key] = self.wild_type_data[key][residue_index]
            mutant_data[key] = self.mutant_data[key][residue_index]
        for key in self.atom_fields:
            wild_type_data[key] = self.wild_type_data[key][atom_index]
            mutant_data[key] = self.mutant_data[key][atom_index]
        
        return wild_type_data, mutant_data
         

if __name__ == "__main__":
    data = MutationChangeStructure("./legacy/S4169/")
    data.save_npz("./data/s4169/")