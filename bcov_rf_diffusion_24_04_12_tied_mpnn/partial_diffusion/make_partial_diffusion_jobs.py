import sys
import os
import numpy as np
import math
import pyrosetta
import argparse

def initialize_pyrosetta():
    pyrosetta.init("-beta -mute all")

def get_parser():
    parser = argparse.ArgumentParser(description="Process some pdb files and residue interfaces and output diffusion job.")
    parser.add_argument("pdb", help="The path to the PDB file.")
    parser.add_argument("--manual_interface", type=str, help="Comma-separated list of manual interface residues (e.g., 23,24,25).")
    return parser

def parse_interface_arg(interface_str):
    return list(map(int, interface_str.split(',')))

def print_diffusion_job(pdb, contigA, contigB):
    print(f"/home/drhicks1/for_/for_kejia/motif_diffusion/bcov_hacked_motif.sh {pdb} {contigA} {contigB} /net/databases/diffusion/models/hotspot_models/BFF_4.pt")
    print(f"/home/drhicks1/for_/for_kejia/motif_diffusion/bcov_hacked_motif.sh {pdb} {contigA} {contigB} /net/databases/diffusion/models/hotspot_models/BFF_6.pt")
    print(f"/home/drhicks1/for_/for_kejia/motif_diffusion/bcov_hacked_motif.sh {pdb} {contigA} {contigB} /net/databases/diffusion/models/hotspot_models/BFF_9.pt")

def coordinate_frame(xyz1, xyz2, xyz3):
    a = xyz1; b = xyz2; c = xyz3

    e1 = a - b
    e1 /= np.linalg.norm(e1)

    e3 = np.cross(e1, c - b)
    e3_norm = np.linalg.norm(e3)

    if e3_norm == 0:
        raise ValueError('atom1, atom2, and atom3 are colinear!')

    e3 /= e3_norm

    e2 = np.cross(e3, e1)

    matrix = np.array([[e1[0], e2[0], e3[0]],
                       [e1[1], e2[1], e3[1]],
                       [e1[2], e2[2], e3[2]]])
    vector = a

    return matrix, vector

def cbeta_from_ca_c_n(ca_xyz, c_xyz, n_xyz):
    ## if problems, check angle (110.35) and length ()
    epsilon = 0.00001
    matrix, vector = coordinate_frame(ca_xyz, c_xyz, n_xyz)
    phi, theta, d = math.radians(122.0), math.radians(180 - 110 + epsilon), 1.5
    v1_xyz = matrix @ np.array([d * math.cos(theta), d * math.sin(theta) * math.cos(phi), d * math.sin(theta) * math.sin(phi)]) + vector
    return pyrosetta.rosetta.numeric.xyzVector_double_t(v1_xyz[0], v1_xyz[1], v1_xyz[2])

def res1_pointed_at_res2(res1, res2, angle_cutoff, dist_cutoff):
    import math
    is_pointed_at = False
    dist_squared = dist_cutoff * dist_cutoff

    if res1.type().has("CB") and res2.type().has("CB"):
        CA_position = res1.atom("CA").xyz()
        CB_position = res1.atom("CB").xyz()
        dest_position = res2.atom("CB").xyz()

    elif res1.type().has("CB") and res2.type().has("CA") and res2.type().has("C") and res2.type().has("N"):
        CA_position = res1.atom("CA").xyz()
        CB_position = res1.atom("CB").xyz()
        dest_position = cbeta_from_ca_c_n(res2.atom("CA").xyz(), res2.atom("C").xyz(), res2.atom("N").xyz())

    elif res1.type().has("CA") and res1.type().has("C") and res1.type().has("N") and res2.type().has("CB"):
        CA_position = res1.atom("CA").xyz()
        CB_position = cbeta_from_ca_c_n(res1.atom("CA").xyz(), res1.atom("C").xyz(), res1.atom("N").xyz())
        dest_position = res2.atom("CB").xyz()

    elif res1.type().has("CA") and res1.type().has("C") and res1.type().has("N") and res2.type().has("CA") and res2.type().has("C") and res2.type().has("N"):
        CA_position = res1.atom("CA").xyz()
        CB_position = cbeta_from_ca_c_n(res1.atom("CA").xyz(), res1.atom("C").xyz(), res1.atom("N").xyz())
        CB_position2 = cbeta_from_ca_c_n(res2.atom("CA").xyz(), res2.atom("C").xyz(), res2.atom("N").xyz())
        dest_position = CB_position2

    else:
        print("currently only works for protein residues with CB and/or CA")
        return is_pointed_at

    cbvector = CB_position - CA_position
    res1_vector = cbvector.normalize()

    # see if residues are close enough
    if CB_position.distance_squared( dest_position ) <= dist_squared:
        base_to_dest = (dest_position - CB_position).normalize()
        r1_dot_r2 = pyrosetta.rosetta.numeric.dot(res1_vector, base_to_dest)
        costheta = math.cos(math.radians(angle_cutoff))
        if r1_dot_r2 > costheta:
            is_pointed_at = True
    return is_pointed_at

def get_interface_by_vector(pose, residue_set1, residue_set2, vector_angle_cutoff=75.0, vector_dist_cutoff=9.0):
    vector_to_return = []

    # returns all resi1 pointing at resi2
    new_residue_set1 = []
    for i in range(len(residue_set1)):
        new_residue_set1.append([residue_set1[i], pose.residue(residue_set1[i])])
    new_residue_set2 = []
    for i in range(len(residue_set2)):
        new_residue_set2.append([residue_set2[i], pose.residue(residue_set2[i])])

    new_residue_set1
    new_residue_set2
    for resi in new_residue_set1:
        for resj in new_residue_set2:
            if resi[0] in vector_to_return:
                continue
            if resi[0] == resj[0]:
                continue
            if res1_pointed_at_res2(resi[1], resj[1], vector_angle_cutoff, vector_dist_cutoff):
                #print("{} pointed at {}".format(resi[0], resj[0]))
                vector_to_return.append(resi[0])

    return vector_to_return

def select_surface(pose):
    surface_residues = []
    shallow_atoms = pyrosetta.rosetta.core.scoring.atomic_depth.atoms_deeper_than(pose, 1.5, True, 3.5, False, 0.25)
    for i in range(1, pose.size()+1):
        resi = pose.residue(i)
        shallow_bools = shallow_atoms(i)
        for atom_i in range(1, resi.natoms()+1):
            if resi.atom_is_backbone(atom_i):
                continue
            if resi.atom_is_hydrogen(atom_i):
                continue
            if shallow_bools[atom_i]:
                surface_residues.append(i)
                break
    return surface_residues

def get_close_contacts(pose, residue_list1, residue_list2, threshold=5.0):
    close_contacts = []
    dist_squared = threshold * threshold

    for resi1 in residue_list1:
        if resi1 in close_contacts:
            continue
        for resi2 in residue_list2:
            res1 = pose.residue(resi1)
            res2 = pose.residue(resi2)
            for at_r1 in range(1, res1.natoms() + 1):
                if resi1 in close_contacts:
                    continue
                for at_r2 in range(1, res2.natoms() + 1):
                    if res1.atom(at_r1).xyz().distance_squared(res2.atom(at_r2).xyz()) <= dist_squared:
                        close_contacts.append(resi1)
                        break

    return close_contacts

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def range_blocks(nums):
    tups = ranges(nums)
    blocks = []
    for tup in tups:
        blocks.append(list(range(tup[0], tup[1]+1)))
    return blocks

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    initialize_pyrosetta()
    pose = pyrosetta.pose_from_pdb(args.pdb)
    
    poseA = pose.split_by_chain()[1]
    poseB = pose.split_by_chain()[2]
    
    chain1_range = list(range(pose.chain_begin(1), pose.chain_end(1) + 1))
    chain2_range = list(range(pose.chain_begin(2), pose.chain_end(2) + 1))
    
    if args.manual_interface:
        interface_res = parse_interface_arg(args.manual_interface)
    else:
        interface_res = get_interface_by_vector(pose, chain1_range, chain2_range)
        surface_res = select_surface(pose)
        close_contacts = get_close_contacts(pose, chain1_range, chain2_range[-17:]) #DRH hardcoded to only take the last 17 of the peptide
        interface_res = [x for x in interface_res if x in close_contacts]
        interface_res = [x for x in interface_res if x not in surface_res]
        interface_res = list(set(interface_res))
        interface_res.sort()

    
    combinations = [[0]]
    for combination in combinations:
        #Add some logic to remove interface residues from the inpaint
        inpaint_range = list(range(pose.chain_begin(1), pose.chain_end(1)+1))
        # print(f"fuse_range {fuse_range}")
        inpaint_resi = [x for x in inpaint_range if x not in interface_res]
        # print(f"fuse_resi {fuse_resi}")
        inpaint_blocks = range_blocks(inpaint_resi)
        # print(f"fuse_blocks {fuse_blocks}")
        inpaint_blocks = [x + ["paint"] for x in inpaint_blocks]
        # print(f"fuse_blocks {fuse_blocks}")
        not_inpaint_resi = [x for x in inpaint_range if x in interface_res]
        not_inpaint_blocks = range_blocks(not_inpaint_resi)
        # print(f"not_fuse_blocks {not_fuse_blocks}")
        not_inpaint_blocks = [x + ["not"] for x in not_inpaint_blocks]
        # print(f"not_fuse_blocks {not_fuse_blocks}")
        all_blocks = inpaint_blocks+not_inpaint_blocks
        all_blocks = sorted(all_blocks, key=lambda x: x[0])

        binder_len = poseA.size()
        contigA=""
        for block in all_blocks:
            if block[-1] == "paint":
                pad = 0
                for resi in combination:
                    if resi in block:
                        pad += 1
                binder_len += pad
                contigA = contigA + f"{block[-2]-block[0]+1+pad},"
            else:
                contigA = contigA + f"A{block[0]}-{block[-2]},"
        contigA = contigA[:-1]
        contigB = f"B1-{poseB.size()}"
    
    print_diffusion_job(args.pdb, contigA, contigB)

if __name__ == "__main__":
    main()
