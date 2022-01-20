import os, sys
import dill
from tqdm import tqdm
import random
import numpy as np
import torch

src_path = "./final_raw_dips/raw/"
tgt_path = "./outputs/"
if not os.path.exists(tgt_path):
    os.mkdir(tgt_path)

# Canonical data list
amino_acids = ["arg", "asp", "asn", "cys", "glu", "gln", "his", "ile", "leu", "lys", 
                "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val"]
amino_acid_dict = {aa: i for i, aa in enumerate(amino_acids)}

atom_name2id = {"C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
                "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
                "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
                "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
                "OH": 33, "OXT": 34, "SD": 35, "SG": 36, "UNK": 37}

# Get npz data
protein_fields = ["id", "E", "num_residue", "N"]
residue_fields = ["residue_names", "residue_types", "chain_ids", "is_interface"]
atom_fields = ["F", "residue_ids", "atom_names", "atom_ids", "Z", "R"]
all_fields = protein_fields + residue_fields + atom_fields

split_len = 12000
_split_id = 0
_sample_id = 0
_residue_id = 0

data_dict = {field: [] for field in all_fields}
for folder_name in sorted(os.listdir(src_path)):
    folder = os.path.join(src_path, folder_name)
    if not os.path.isdir(folder):
        continue

    folder_listing = tqdm(sorted(os.listdir(folder)))
    for filename in folder_listing:
        if not filename.endswith(".dill"):
            continue

        file = os.path.join(folder, filename)
        try:
            with open(file, "rb") as fin:
                data = dill.load(fin)
                chain1 = data.df0
                chain2 = data.df1
                positive_pairs = data.pos_idx
                positive_chain1, positive_chain2 = positive_pairs.T

                # prepare protein-level attributes
                chain1_residue_ids = list({int(res) for res in chain1.residue})
                chain2_residue_ids = list({int(res) for res in chain2.residue})

                # prepare residue-level attributes
                chain1_N = chain1.atom_name == "N"
                chain1_residue_names = [res.lower() for res in chain1.resname[chain1_N]]
                chain1_residue_types = [amino_acid_dict.get(res, len(amino_acids)) for res in chain1_residue_names]
                chain2_N = chain2.atom_name == "N"
                chain2_residue_names = [res.lower() for res in chain2.resname[chain2_N]]
                chain2_residue_types = [amino_acid_dict.get(res, len(amino_acids)) for res in chain2_residue_names]
                chain1_ca_ids = np.arange(len(chain1))[chain1.atom_name == "CA"].tolist()
                chain2_ca_ids = np.arange(len(chain2))[chain2.atom_name == "CA"].tolist()

                # prepare atom-level attributes
                chain1_atom2residue = torch.tensor([int(res) for res in chain1.residue], dtype=torch.long)
                chain1_atom2residue += max(0, -chain1_atom2residue.min())
                chain1_atom_cnt = torch.zeros(chain1_atom2residue.max() + 1, dtype=torch.long).scatter_add_(
                    0, chain1_atom2residue, torch.ones(len(chain1), dtype=torch.long))
                chain1_atom_cnt = chain1_atom_cnt[chain1_atom_cnt != 0]
                chain1_residue_ids_ = torch.repeat_interleave(torch.arange(_residue_id, _residue_id + len(chain1_residue_names)), chain1_atom_cnt)
                _residue_id += len(chain1_residue_names)

                chain2_atom2residue = torch.tensor([int(res) for res in chain2.residue], dtype=torch.long)
                chain2_atom2residue += max(0, -chain2_atom2residue.min())
                chain2_atom_cnt = torch.zeros(chain2_atom2residue.max() + 1, dtype=torch.long).scatter_add_(
                    0, chain2_atom2residue, torch.ones(len(chain2), dtype=torch.long))
                chain2_atom_cnt = chain2_atom_cnt[chain2_atom_cnt != 0]
                chain2_residue_ids_ = torch.repeat_interleave(torch.arange(_residue_id, _residue_id + len(chain2_residue_names)), chain2_atom_cnt)
                _residue_id += len(chain2_residue_names)

                chain1_atom_positions = np.stack([np.array(chain1.x), np.array(chain1.y), np.array(chain1.z)], axis=1)
                chain2_atom_positions = np.stack([np.array(chain2.x), np.array(chain2.y), np.array(chain2.z)], axis=1)

                # append protein-level attributes
                data_dict["id"].append(_sample_id)
                _sample_id += 1
                data_dict["E"].append(random.random())
                data_dict["num_residue"].append(len(chain1_residue_ids) + len(chain2_residue_ids))
                data_dict["N"].append(len(chain1) + len(chain2))

                # append residue-level attributes
                data_dict["residue_names"] += chain1_residue_names
                data_dict["residue_names"] += chain2_residue_names
                data_dict["residue_types"] += chain1_residue_types
                data_dict["residue_types"] += chain2_residue_types
                data_dict["chain_ids"] += [0] * len(chain1_residue_names)
                data_dict["chain_ids"] += [1] * len(chain2_residue_names)
                data_dict["is_interface"] += [ca_id in positive_chain1 for ca_id in chain1_ca_ids]
                data_dict["is_interface"] += [ca_id in positive_chain2 for ca_id in chain2_ca_ids]

                # append atom-level attributes
                data_dict["F"] += np.random.rand(len(chain1) + len(chain2), 3).tolist()
                data_dict["residue_ids"] += chain1_residue_ids_.tolist()
                for cnt in chain1_atom_cnt:
                    data_dict["atom_ids"] += list(range(cnt))
                data_dict["residue_ids"] += chain2_residue_ids_.tolist()
                for cnt in chain2_atom_cnt:
                    data_dict["atom_ids"] += list(range(cnt))
                data_dict["atom_names"] += list(chain1.atom_name)
                data_dict["atom_names"] += list(chain2.atom_name)
                data_dict["R"] += chain1_atom_positions.tolist()
                data_dict["R"] += chain2_atom_positions.tolist()
                data_dict["Z"] += [atom_name2id.get(atom_name, atom_name2id["UNK"]) for atom_name in chain1.atom_name]
                data_dict["Z"] += [atom_name2id.get(atom_name, atom_name2id["UNK"]) for atom_name in chain2.atom_name]
        except:
            continue

        if _sample_id >= split_len:
            data_dict = {k: np.array(v) for k, v in data_dict.items()}
            print(data_dict["E"].shape)
            print(data_dict["N"].shape)
            print(data_dict["residue_names"].shape)
            print(data_dict["residue_types"].shape)
            print(data_dict["F"].shape)
            print(data_dict["residue_ids"].shape)
            print(data_dict["atom_ids"].shape)
            print(data_dict["R"].shape)
            print(data_dict["Z"].shape)

            save_name = os.path.join(tgt_path, "DIPS_split_{}.npz".format(_split_id))
            np.savez(save_name, **data_dict)
            print("Save to: ", save_name)
            _split_id += 1
            _sample_id = 0
            _residue_id = 0
            data_dict = {field: [] for field in all_fields}
