import pandas as pd

def remove_constant_columns(df):
    """
    Removes columns from a Dataframe which only have 1 unique value.
    This is to clean up DFs loaded from compiled_metrics.csv, which contain a
    column for every config setting.
    """
    for col in df.columns:
        if len(df[col].astype(str).drop_duplicates())==1: # all the same
            df = df.drop(col, axis=1)
    return df


def stack_df_mpnn(df, merge_cols, stack_cols):
    ''' 
    Rearranges MPNN-related metrics (e.g. `<metric>_mpnn`) so that they are
    additional data rows with regular metric names (e.g. `<metric>`) with a
    column `mpnn` indicating that the row represents metrics of an
    mpnn-designed example. This is to allow certain plotting options with
    seaborn.
    '''
    tmp1 = df[merge_cols+stack_cols]
    tmp1['mpnn'] = 'no'
    tmp2 = df[merge_cols+[col+'_mpnn' for col in stack_cols]].rename(columns={col+'_mpnn':col for col in stack_cols})
    tmp2['mpnn'] = 'yes'
    return tmp1.append(tmp2)

def calc_success_rate(df,
                      columns,
                      thresholds = [(10,1,1),(10,1,0.8),(10,1,0.6),(10,1,0.4),
                                    (15,3,1),(15,3,0.8),(15,3,0.6),(15,3,0.4)],
                      wide=False):
    '''
    Calculates success rate at different pAE, motif RMSD, and TM-score thresholds

    Inputs:

        df : DataFrame loaded from compiled_metrics.csv. Should contain TM-score clusters
        columns : list of column names you want to group on in order to compute success rate
        thresholds : list of triples with thresholds for [(af2_pae_mean, contig_rmsd_af2, tm_score)]
        wide : If True, output DataFrame will have "wide" format

    Outputs:
        counts : DataFrame with success rates

    '''
    df['count']=1
    df['tm_cluster_1.00'] = df['name']
    total = df.groupby(columns).sum().reset_index()[columns+['count']].rename(columns={'count':'total'})

    df_s = []
    for pae_thresh, rmsd_thresh, tm_thresh in thresholds:
        tmp = df.drop_duplicates(columns+[f'tm_cluster_{tm_thresh:.2f}'])
        ct = tmp[(tmp['af2_pae_mean']<pae_thresh)
                & (tmp['contig_rmsd_af2']<rmsd_thresh)].groupby(columns).sum()\
             .reset_index()[columns+['count']]
        ct = total.merge(ct, on=columns,how='outer')
        ct['pae_thresh'] = pae_thresh
        ct['rmsd_thresh'] = rmsd_thresh
        ct['tm_thresh'] = tm_thresh
        df_s.append(ct)

    success = pd.concat(df_s)
    success.loc[success['count'].isnull(),'count'] = 0
    success['success_rate'] = success['count']/success['total']

    if wide:
        success_wide = success[columns].drop_duplicates()
        for pae_thresh, rmsd_thresh, tm_thresh in thresholds:
            tmp = success[(success['pae_thresh']==pae_thresh) &
                          (success['rmsd_thresh']==rmsd_thresh) &
                          (success['tm_thresh']==tm_thresh)]
            tmp2 = tmp[columns]
            tmp2[f'f_{pae_thresh}_{rmsd_thresh}@{tm_thresh}'] = tmp['success_rate']
            success_wide = success_wide.merge(tmp2, on=columns)
        return success_wide

    return success
