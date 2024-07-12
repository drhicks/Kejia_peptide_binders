def check_fully_satisfied_bidenates(pose, bidentates, bidentate_resid):
    this_residue = pose.residue(bidentate_resid)
    sc_heavy_atms = {}
    hpol_map = {}
    for hpol in this_residue.Hpol_index():
        if not this_residue.atom_is_backbone():
            hpol_map[hpol] = False

    for atom_id in range(1, len(this_residue.atoms()) + 1):
        atm_name = this_residue.atom_name(atom_id).strip()
        if 'H' not in atm_name:
            if 'N' in atm_name or 'O' in atm_name:
                if not this_residue.atom_is_backbone(atom_id):
                    hb_group = [atm_name]
                    for hpol in hpol_map:
                        if not hpol_map[hpol] and hpol in this_residue.bonded_neighbor(atom_id):
                            hb_group.append(this_residue.atom_name(hpol).strip())
                            hpol_map[hpol] = True
                    sc_heavy_atms[tuple(hb_group)] = False

    for hb in bidentates[bidentate_resid]:
        for hb_i in [0, 2]:
            if hb[hb_i] == bidentate_resid:
                for hb_group in sc_heavy_atms:
                    if hb[hb_i + 1] in hb_group:
                        sc_heavy_atms[hb_group] = True

    return all(sc_heavy_atms.values())

def simple_hbond_finder(pose, reslist1, reslist2, delta_HA=3., delta_theta=30.,
                        reslist1_atom_type=['bb', 'sc'], reslist2_atom_type=['bb', 'sc'], verbose=False):
    """Find simple hydrogen bonds between two residue lists."""
    def _get_don_accpt_residue_atom_indices(pose, reslist, atom_type=['bb', 'sc']):
        don_list, accpt_list = [], []
        for resid in reslist:
            bb_list = [x for x in range(1, pose.residue(resid).natoms() + 1) if pose.residue(resid).atom_is_backbone(x)]
            for atomid in pose.residue(resid).Hpol_index():
                if ('bb' not in atom_type and atomid in bb_list) or ('sc' not in atom_type and atomid not in bb_list):
                    continue
                don_list.append((resid, atomid, pose.residue(resid).atom_base(atomid)))
            for atomid in pose.residue(resid).accpt_pos():
                if ('bb' not in atom_type and atomid in bb_list) or ('sc' not in atom_type and atomid not in bb_list):
                    continue
                accpt_list.append((resid, atomid, pose.residue(resid).atom_base(atomid)))
        return don_list, accpt_list

    reslist1_don, reslist1_accpt = _get_don_accpt_residue_atom_indices(pose, reslist1, reslist1_atom_type)
    reslist2_don, reslist2_accpt = _get_don_accpt_residue_atom_indices(pose, reslist2, reslist2_atom_type)

    hbonds = []

    for don in reslist1_don:
        for accpt in reslist2_accpt:
            dist = pyrosetta.rosetta.numeric.xyzVector_double_t.distance(
                pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            angle = pyrosetta.rosetta.numeric.angle_degrees_double(
                pose.residue(don[0]).xyz(don[2]), pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            if verbose:
                logging.info(f'{don}, {pose.residue(don[0]).atom_name(don[1]).strip()}, {accpt}, '
                             f'{pose.residue(accpt[0]).atom_name(accpt[1]).strip()}, {dist}, {angle}')
            if dist <= delta_HA and angle >= 180 - delta_theta:
                hbonds.append((don[0], don[1], accpt[0], accpt[1], dist, angle, 'don-accpt'))
    for don in reslist2_don:
        for accpt in reslist1_accpt:
            dist = pyrosetta.rosetta.numeric.xyzVector_double_t.distance(
                pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            angle = pyrosetta.rosetta.numeric.angle_degrees_double(
                pose.residue(don[0]).xyz(don[2]), pose.residue(don[0]).xyz(don[1]), pose.residue(accpt[0]).xyz(accpt[1]))
            if verbose:
                logging.info(f'{don}, {pose.residue(don[0]).atom_name(don[1]).strip()}, {accpt}, '
                             f'{pose.residue(accpt[0]).atom_name(accpt[1]).strip()}, {dist}, {angle}')
            if dist <= delta_HA and angle >= 180 - delta_theta:
                hbonds.append((accpt[0], accpt[1], don[0], don[1], dist, angle, 'accpt-don'))
    return hbonds

def find_potential_bidentate_hbond(pose, reslist1, reslist2, delta_HA=3., delta_theta=30., reslist1_atom_type=['sc'], reslist2_atom_type=['bb']):
    hbonds = simple_hbond_finder(pose, reslist1, reslist2, delta_HA=delta_HA, delta_theta=delta_theta, reslist1_atom_type=reslist1_atom_type, reslist2_atom_type=reslist2_atom_type)
    hbonds_by_sc_res = {}
    for hb in hbonds:
        if hb[0] not in hbonds_by_sc_res:
            hbonds_by_sc_res[hb[0]] = []
        if hb[6] == 'don-accpt':
            hbonds_by_sc_res[hb[0]].append((hb[0], pose.residue(hb[0]).atom_name(hb[1]).strip(), hb[2], pose.residue(hb[2]).atom_name(hb[3]).strip(), hb[4], hb[5]))
        elif hb[6] == 'accpt-don':
            hbonds_by_sc_res[hb[0]].append((hb[2], pose.residue(hb[2]).atom_name(hb[3]).strip(), hb[0], pose.residue(hb[0]).atom_name(hb[1]).strip(), hb[4], hb[5]))
        else:
            logging.error(f'Incorrect donor acceptor info: {hb[6]}')

    bidentate_hbonds_by_sc_res = {res: hbonds_by_sc_res[res] for res in hbonds_by_sc_res if len(hbonds_by_sc_res[res]) > 1}
    return bidentate_hbonds_by_sc_res

