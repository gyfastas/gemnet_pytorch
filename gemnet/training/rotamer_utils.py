import os, sys
import collections
import numpy as np
import pandas as pd


QuadrantData = collections.namedtuple(
    "QuadrantData", ["chimeans", "chisigmas", "probs", "meanprobs", "cumprobs", "exists", "rotinds"])


class RotamerBase:
    """
    Class for processing rotamer library, sampling rotamers and performing side chain rotation.

    Parameters
    ----------
        library_path: str
            Directory of rotamer library.
    """

    amino_acids = ["arg", "asp", "asn", "cys", "glu", "gln", "his", "ile", "leu", "lys",
                   "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val"]
    # The atoms constituting Chis; atom orders in a residue: [N, CA, C, O, CB, ...]
    chis_atoms = {"arg": [[0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
                  "asp": [[0, 1, 4, 5], [1, 4, 5, 6]],
                  "asn": [[0, 1, 4, 5], [1, 4, 5, 6]],
                  "cys": [[0, 1, 4, 5]],
                  "glu": [[0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7]],
                  "gln": [[0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7]],
                  "his": [[0, 1, 4, 5], [1, 4, 5, 6]],
                  "ile": [[0, 1, 4, 5], [1, 4, 5, 7]],
                  "leu": [[0, 1, 4, 5], [1, 4, 5, 6]],
                  "lys": [[0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
                  "met": [[0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7]],
                  "phe": [[0, 1, 4, 5], [1, 4, 5, 6]],
                  "pro": [[0, 1, 4, 5], [1, 4, 5, 6]],
                  "ser": [[0, 1, 4, 5]],
                  "thr": [[0, 1, 4, 5]],
                  "trp": [[0, 1, 4, 5], [1, 4, 5, 6]],
                  "tyr": [[0, 1, 4, 5], [1, 4, 5, 6]],
                  "val": [[0, 1, 4, 5]]}

    def __init__(self, library_path):
        self.library_path = library_path
        database = self.load_rotamor_library(library_path)
        self.database = self.process_raw_database(database)

    def load_rotamor_library(self, library_path):
        database = {}
        columns = collections.OrderedDict()
        columns["T"] = np.str
        columns["Phi"] = np.int64
        columns["Psi"] = np.int64
        columns["Count"] = np.int64
        columns["r1"] = np.int64
        columns["r2"] = np.int64
        columns["r3"] = np.int64
        columns["r4"] = np.int64
        columns["Probabil"] = np.float64
        columns["chi1Val"] = np.float64
        columns["chi2Val"] = np.float64
        columns["chi3Val"] = np.float64
        columns["chi4Val"] = np.float64
        columns["chi1Sig"] = np.float64
        columns["chi2Sig"] = np.float64
        columns["chi3Sig"] = np.float64
        columns["chi4Sig"] = np.float64

        for amino_acid in self.amino_acids:
            database[amino_acid] = pd.read_csv(
                os.path.join(library_path, "ExtendedOpt1-5/{}.bbdep.rotamers.lib".format(amino_acid)),
                names=list(columns.keys()), dtype=columns, comment="#", delim_whitespace=True, engine="c")
        return database

    def discrete_angle_to_bucket(self, angle):
        assert isinstance(angle, int)
        assert angle % 10 == 0
        assert -180 <= angle < 180
        bucket = (angle + 180) // 10
        return bucket

    def get_rotind(self, r1, r2, r3, r4):
        rotind = 1000000 * r1 + 10000 * r2 + 100 * r3 + r4
        return rotind

    def process_raw_database(self, database):
        processed_database = {}
        for amino_acid in self.amino_acids:
            dataframe = database[amino_acid]
            bucketed_data = [[{} for _1 in range(36)] for _2 in range(36)]
            rows = dataframe.to_dict("records")
            for row in rows:
                phi, psi = row["Phi"], row["Psi"]
                wraparound = False
                if phi == 180:
                    wraparound = True
                    phi = -180
                if psi == 180:
                    wraparound = True
                    psi = -180
                phi_bucket = self.discrete_angle_to_bucket(phi)
                psi_bucket = self.discrete_angle_to_bucket(psi)

                rotind = self.get_rotind(row["r1"], row["r2"], row["r3"], row["r4"])
                chi_means = np.array([row[f"chi{i}Val"] for i in range(1, 5)])
                chi_sigmas = np.array([row[f"chi{i}Sig"] for i in range(1, 5)])
                probability = row["Probabil"]
                bucket = bucketed_data[phi_bucket][psi_bucket]
                bucket_data = (chi_means, chi_sigmas, probability)
                if wraparound:
                    assert ((bucket[rotind][0] == bucket_data[0]).all()
                            and (bucket[rotind][1] == bucket_data[1]).all() and (bucket[rotind][2] == bucket_data[2]))
                else:
                    bucket[rotind] = bucket_data

            quadrant_data = [[None for _1 in range(36)] for _2 in range(36)]
            for lower_phi_bucket in range(36):
                for lower_psi_bucket in range(36):
                    upper_phi_bucket = (lower_phi_bucket + 1) % 36
                    upper_psi_bucket = (lower_psi_bucket + 1) % 36
                    quadrants = [bucketed_data[lower_phi_bucket][lower_psi_bucket],
                                 bucketed_data[upper_phi_bucket][lower_psi_bucket],
                                 bucketed_data[lower_phi_bucket][upper_psi_bucket],
                                 bucketed_data[upper_phi_bucket][upper_psi_bucket]]

                    rotinds = np.array(
                        sorted(set().union(*[set(quadrant.keys()) for quadrant in quadrants])), dtype=np.int)
                    assert len(rotinds) > 0
                    exists = np.zeros((len(rotinds), 4), dtype=np.bool)
                    probabilities = np.zeros((len(rotinds), 4), dtype=np.float64)
                    chi_means = np.zeros((len(rotinds), 4, 4), dtype=np.float64)
                    chi_sigmas = np.zeros((len(rotinds), 4, 4), dtype=np.float64)
                    for i, rotind in enumerate(rotinds):
                        for qid, quadrant in enumerate(quadrants):
                            if rotind not in quadrant:
                                continue
                            quadrant_chi_means, quadrant_chi_sigmas, quadrant_probability = quadrant[rotind]
                            exists[i, qid] = True
                            probabilities[i, qid] = quadrant_probability
                            chi_means[i, qid] = quadrant_chi_means
                            chi_sigmas[i, qid] = quadrant_chi_sigmas
                    mean_probabilities = probabilities.mean(1)
                    order = np.argsort(-mean_probabilities, kind="stable")
                    mean_probabilities = mean_probabilities[order]
                    cum_probabilities = np.cumsum(mean_probabilities)
                    assert np.abs(cum_probabilities[-1] - 1) < 1e-5

                    quadrant_data[lower_phi_bucket][lower_psi_bucket] = QuadrantData(
                        chimeans=chi_means[order], chisigmas=chi_sigmas[order], probs=probabilities[order],
                        exists=exists[order], rotinds=rotinds[order], meanprobs=mean_probabilities,
                        cumprobs=cum_probabilities)

            processed_database[amino_acid] = quadrant_data
        return processed_database

    def compute_dihedral(self, p0, p1, p2, p3, eps=1e-8):
        p23 = p3 - p2
        p12 = p2 - p1
        p01 = p1 - p0
        n1 = np.cross(p01, p12)
        n2 = np.cross(p12, p23)
        n1 = n1 / (np.linalg.norm(n1, axis=1, keepdims=True) + eps)
        n2 = n2 / (np.linalg.norm(n2, axis=1, keepdims=True) + eps)

        sin = (np.cross(n1, n2) * p12 / (np.linalg.norm(p12, axis=1, keepdims=True) + eps)).sum(1)
        cos = (n1 * n2).sum(1)
        angle = np.arctan2(sin, cos)
        angle = angle / np.pi * 180
        return angle

    def quadrant_data_and_interpolated_weights(self, phi, psi, name):
        lower_phi, lower_psi = int(phi // 10) * 10, int(psi // 10) * 10
        upper_phi, upper_psi = lower_phi + 10, lower_psi + 10
        lower_phi_bucket = self.discrete_angle_to_bucket(lower_phi)
        lower_psi_bucket = self.discrete_angle_to_bucket(lower_psi)
        quadrant_data = self.database[name][lower_phi_bucket][lower_psi_bucket]

        weights = np.array([(10 - (phi - lower_phi)) * (10 - (psi - lower_psi)),
                            (10 - (upper_phi - phi)) * (10 - (psi - lower_psi)),
                            (10 - (phi - lower_phi)) * (10 - (upper_psi - psi)),
                            (10 - (upper_phi - phi)) * (10 - (upper_psi - psi))])
        sum_existing_weights = (weights[np.newaxis, :] * quadrant_data.exists).sum(1)
        effective_weights = weights[np.newaxis, :] / sum_existing_weights[:, np.newaxis]
        return quadrant_data, effective_weights

    def sample_chis(self, phi, psi, name):
        quadrant_data, weights = self.quadrant_data_and_interpolated_weights(phi, psi, name)
        chi_means = quadrant_data.chimeans
        chi_sigmas = quadrant_data.chisigmas
        cum_probabilities = quadrant_data.cumprobs
        sample_index = np.random.randint(len(cum_probabilities), size=1)[0]

        quadrant = np.random.choice(4, p=weights[sample_index])
        chi_mean = chi_means[sample_index, quadrant]
        chi_sigma = chi_sigmas[sample_index, quadrant]
        chis = chi_mean + np.random.randn(4) * chi_sigma
        for _ in range(2):
            chis[chis >= 180] = chis[chis >= 180] - 360
            chis[chis < -180] = chis[chis < -180] + 360
        if name == "pro":
            chis[2] = 0
        return chis

    def rotate_side_chain(self, chis, target_chis, atom_positions, atom_ids, all_chi_atoms, eps=1e-8):
        chis = chis / 180. * np.pi
        target_chis = target_chis / 180. * np.pi
        for chi_id, chi_atoms in enumerate(all_chi_atoms):
            atom_1, atom_2, atom_3, atom_4 = chi_atoms
            atom_2_position = atom_positions[atom_ids == atom_2]
            atom_3_position = atom_positions[atom_ids == atom_3]
            axis = atom_3_position - atom_2_position
            axis_normalize = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + eps)
            chi = chis[chi_id]
            target_chi = target_chis[chi_id]
            rotate_angle = target_chi - chi

            # Rotate all subsequent atoms by the rotation angle
            rotate_atoms = atom_ids >= atom_4
            rotate_atoms_position = atom_positions[rotate_atoms] - atom_2_position[np.newaxis, :]
            parallel_component = (rotate_atoms_position * axis_normalize[np.newaxis, :]).sum(axis=1, keepdims=True) \
                                 * axis_normalize[np.newaxis, :]
            perpendicular_component = rotate_atoms_position - parallel_component
            perpendicular_component_norm = np.linalg.norm(perpendicular_component, axis=1, keepdims=True) + eps
            perpendicular_component_normalize = perpendicular_component / perpendicular_component_norm
            normal_vector = np.cross(axis_normalize[np.newaxis, :], perpendicular_component_normalize)
            transformed_atoms_position = perpendicular_component * np.cos(rotate_angle) + \
                                         normal_vector * perpendicular_component_norm * np.sin(rotate_angle) + \
                                         parallel_component + atom_2_position[np.newaxis, :]
            transformed_atoms_position[np.isnan(transformed_atoms_position)] = 10000
            atom_positions[rotate_atoms] = transformed_atoms_position
        return atom_positions
