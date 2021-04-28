import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
import pickle
import pkg_resources
import os
from numpy.random import default_rng
from tqdm import tqdm


def load_regulons(levels=['A', 'B', 'C', 'D', 'E'], organism='Human', commercial=False):
    """
    Loads DoRothEA's regulons.
    
    Parameters
    ----------
    levels
        List of confidence levels to use. A regulons are the most confident, E the least.
    organism
        String determining which organism to use. Only `Human` and `Mouse` are supported.
    commercial
        Whether to use the academic or commercial version. 
    
    Returns
    -------
    DataFrame containing the relationships between gene targets (rows) and their TFs (columns). 

    Examples
    --------
    >>> import dorothea
    >>> regulons = dorothea.load_regulons(levels=['A'], organism='Human', commercial=False)
    """
    # Get package path
    path = 'data'
    fname = 'dorothea_'
    
    if commercial:
        fname = 'c_' + fname
    if organism == "Human":
        fname = fname + 'hs'
    elif organism == "Mouse":
        fname = fname + 'mm'
    else:
        raise("Wrong organism name. Please specify 'Human' or 'Mouse'.")
    fname = fname + '.pkl'
    path = pkg_resources.resource_filename(__name__, os.path.join(path, fname))
    
    # Open pickle object
    df = pickle.load(open(path, "rb" ))
    
    #Filter by levels of confidence
    df = df[df['confidence'].isin(levels)]
    
    # Transform to binary dataframe
    dorothea_df = df.pivot(index='target', columns='tf', values='mor')
    
    # Set nans to 0
    dorothea_df[np.isnan(dorothea_df)] = 0
    
    return dorothea_df

def extract(adata, obsm_key='dorothea'):
    """
    Generates a new AnnData object with TF activities stored in `.obsm` instead of gene expression. 
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    obsm_key
        `.osbm` key where TF activities are stored.
    
    Returns
    -------
    AnnData object with TF activities
    """
    obsm = adata.obsm
    obs = adata.obs
    df = adata.obsm[obsm_key]
    var = pd.DataFrame(index=df.columns)
    tf_adata = AnnData(np.array(df), obs=obs, var=var, obsm=obsm)
    return tf_adata
    

def process_input(data, use_raw=False):
    """
    Processes different input types so that they can be used downstream. 
    
    Parameters
    ----------
    data
        Annotated data matrix or DataFrame
    use_raw
        If data is an AnnData object, whether to use values stored in `.raw`.
    
    Returns
    -------
    genes : list of genes names
    samples : list of sample names
    X : gene expression matrix
    """
    if isinstance(data, AnnData):
        if not use_raw:
            genes = np.array(data.var.index)
            idx = np.argsort(genes)
            genes = genes[idx]
            samples = data.obs.index
            X = data.X[:,idx]
        else:
            genes = np.array(data.raw.var.index)
            idx = np.argsort(genes)
            genes = genes[idx]
            samples= data.raw.obs_names
            X = data.raw.X[:,idx]
    elif isinstance(data, pd.DataFrame):
        genes = np.array(data.columns)
        idx = np.argsort(genes)
        genes = genes[idx]
        samples = data.index
        X = np.array(data)[:,idx]
    else:
        raise ValueError('Input must be AnnData or pandas DataFrame.')
    return genes, samples, X

def mean_expr(X, R):
    # Run matrix mult
    tf_act = np.asarray(X.dot(R))
    return tf_act


def run(data, regnet, center=True, num_perm=0, norm=True, scale=True, scale_axis=0, inplace=True, use_raw=False, obsm_key='dorothea', min_size=5):
    """
    Runs TF activity prediction from gene expression using DoRothEA's regulons.
    
    Parameters
    ----------
    data
        Annotated data matrix or DataFrame.
    regnet
        Regulon network in DataFrame format.
    center
        Whether to center gene expression by cell/sample.
    num_perm
        Number of permutations to calculate p-vals of random activities.
    norm
        Whether to normalize activities per regulon size to correct for large regulons.
    scale
        Whether to scale the final activities.
    scale_axis
        0 to scale per feature, 1 to scale per cell/sample.
    inplace
        If `data` is an AnnData object, whether to update `data` or return a DataFrame.
    use_raw
        If data is an AnnData object, whether to use values stored in `.raw`.
    obsm_key
        `.osbm` key where TF activities will be stored.
    min_size
        TFs with regulons with less targets than `min_size` will be ignored.
    
    Returns
    -------
    Returns a DataFrame with TF activities or adds it to the `.obsm` key 'dorothea' 
    of the input AnnData object, depending on `inplace` and input data type.
    """
    # Get genes, samples/tfs and matrices from data and regnet
    x_genes, x_samples, X = process_input(data, use_raw=use_raw)

    assert len(x_genes) == len(set(x_genes)), 'Gene names are not unique'
    
    # Center gene expresison by cell
    if center:
        X = X - np.mean(X, axis=1).reshape(-1,1)

    # Sort targets (rows) alphabetically
    regnet = regnet.sort_index()
    r_targets, r_tfs = regnet.index, regnet.columns

    assert len(r_targets) == len(set(r_targets)), 'regnet target names are not unique'
    assert len(r_tfs) == len(set(r_tfs)), 'regnet tf names are not unique'

    # Subset by common genes
    common_genes = np.sort(list(set(r_targets) & set(x_genes)))

    target_fraction = len(common_genes) / len(r_targets)
    assert target_fraction > .05, f'Too few ({len(common_genes)}) target genes found. \
    Make sure you are using the correct organism.'

    print(f'{len(common_genes)} targets found')

    idx_x = np.searchsorted(x_genes, common_genes)
    X = X[:,idx_x]
    R = regnet.loc[common_genes].values
    
    # Check min size and filter
    msk_size = np.sum(R != 0, axis=0) < min_size
    num_small_reg = np.sum(msk_size)
    if num_small_reg > 0:
        print(f'{num_small_reg} TFs with < {min_size} targets')
        R[:, msk_size] = 0

    # Run matrix mult
    estimate = mean_expr(X, R)
    
    # Permutations
    if num_perm > 0:
        pvals = np.zeros(estimate.shape)
        for i in tqdm(range(num_perm)):
            perm = mean_expr(X, default_rng(seed=i).permutation(R))
            pvals += np.abs(perm) > np.abs(estimate)
        pvals = pvals / num_perm
        pvals[pvals == 0] = 1/num_perm
    else:
        pvals = np.full(estimate.shape, 0.1)
    
    # Normalize by num edges
    if norm:
        norm = np.sum(np.abs(R), axis=0)
        norm[norm == 0] = 1
        estimate = estimate / norm

    # Weight estimate by pvals
    tf_act = estimate * -np.log10(pvals)
    
    # Scale output
    if scale:
        std = np.std(tf_act, ddof=1, axis=scale_axis)
        std[std == 0] = 1
        mean = np.mean(tf_act, axis=scale_axis)
        if scale_axis == 0:
            tf_act = (tf_act - mean) / std
        elif scale_axis == 1:
            tf_act = (tf_act - mean.reshape(-1,1)) / std.reshape(-1,1)

    # Store in df
    result = pd.DataFrame(tf_act, columns=r_tfs, index=x_samples)

    if isinstance(data, AnnData) and inplace:
        # Update AnnData object
        data.obsm[obsm_key] = result
    else:
        # Return dataframe object
        data = result
        inplace = False

    return data if not inplace else None

def rank_tfs_groups(adata, groupby, group, reference='all', obsm_key='dorothea'):
    """
    Runs Wilcoxon rank-sum test between one group and a reference group.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    group
        Group or list of groups to compare.
    reference
        Reference group or list of reference groups to use as reference.
    obsm_key
         `.osbm` key to use to extract TF activities.
    
    Returns
    -------
    DataFrame with changes in TF activity between groups.
    """
    from scipy.stats import ranksums
    from statsmodels.stats.multitest import multipletests

    # Get TF activites
    adata = extract(adata, obsm_key=obsm_key)
    
    # Get tf names
    features = adata.var.index.values

    # Generate mask for group samples
    if isinstance(group, str):
        g_msk = (adata.obs[groupby] == group).values
    else:
        cond_lst = [(adata.obs[groupby] == grp).values for grp in group]
        g_msk = np.sum(cond_lst, axis=0).astype(bool)
        group = ', '.join(group)

    # Generate mask for reference samples
    if reference == 'all':
        ref_msk = ~g_msk
    elif isinstance(reference, str):
        ref_msk = (adata.obs[groupby] == reference).values
    else:
        cond_lst = [(adata.obs[groupby] == ref).values for ref in reference]
        ref_msk = np.sum(cond_lst, axis=0).astype(bool)
        reference = ', '.join(reference)
        
    assert np.sum(g_msk) > 0, 'No group samples found'
    assert np.sum(ref_msk) > 0, 'No reference samples found'

    # Wilcoxon rank-sum test 
    results = []
    for i in np.arange(len(features)):
        stat, pval = ranksums(adata.X[g_msk,i], adata.X[ref_msk,i])
        results.append([features[i], group, reference, stat, pval])

    # Tranform to df
    results = pd.DataFrame(
        results, 
        columns=['name', 'group', 'reference', 'statistic', 'pval']
    ).set_index('name')
    
    # Correct pvalues by FDR
    results[np.isnan(results['pval'])] = 1
    _, pvals_adj, _, _ = multipletests(
        results['pval'].values, alpha=0.05, method='fdr_bh'
    )
    results['pval_adj'] = pvals_adj
    
    # Sort by statistic
    results = results.sort_values('statistic', ascending=False)
    return results
