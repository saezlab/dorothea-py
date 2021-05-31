import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
from anndata import AnnData
import pickle
import pkg_resources
import os
from numpy.random import default_rng
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


"""TF activity prediction in Python"""


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
    

def process_input(data, use_raw=False, use_hvg=False):
    """
    Processes different input types so that they can be used downstream. 
    
    Parameters
    ----------
    data
        Annotated data matrix or DataFrame
    use_raw
        If data is an AnnData object, whether to use values stored in `.raw`.
    use_hvg
        If data is an AnnData object, whether to only use high variable genes.
    
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
            if use_hvg:
                hvg_msk = data.var.loc[genes].highly_variable
                X = X[:,hvg_msk]
                genes = genes[hvg_msk]
        else:
            genes = np.array(data.raw.var.index)
            idx = np.argsort(genes)
            genes = genes[idx]
            samples= data.raw.obs_names
            X = data.raw.X[:,idx]
            if use_hvg:
                hvg_msk = data.raw.var.loc[genes].highly_variable
                X = X[:,hvg_msk]
                genes = genes[hvg_msk]
    elif isinstance(data, pd.DataFrame):
        genes = np.array(data.columns)
        idx = np.argsort(genes)
        genes = genes[idx]
        samples = data.index
        X = np.array(data)[:,idx]
    else:
        raise ValueError('Input must be AnnData or pandas DataFrame.')
    return genes, samples, csr_matrix(X)

def dot_mult(X, R):
    # Run matrix mult
    tf_act = np.asarray(X.dot(R))
    return tf_act

def scale_arr(X, scale_axis):
    std = np.std(X, ddof=1, axis=scale_axis)
    std[std == 0] = 1
    mean = np.mean(X, axis=scale_axis)
    if scale_axis == 0:
        X = (X - mean) / std
    elif scale_axis == 1:
            X = (X - mean.reshape(-1,1)) / std.reshape(-1,1)
    return X


def center_arr(X):
    X = X.copy()
    sums = np.squeeze(X.sum(1).A)
    counts = np.diff(X.tocsr().indptr)
    means = sums/counts
    X.data -= np.repeat(means, counts)
    return X


def run(data, regnet, center=True, num_perm=0, norm=True, scale=True, scale_axis=0, inplace=True, 
        use_raw=False, use_hvg=False, obsm_key='dorothea', min_size=5):
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
    use_hvg
        If data is an AnnData object, whether to only use high variable genes.
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
    x_genes, x_samples, X = process_input(data, use_raw=use_raw, use_hvg=use_hvg)

    assert len(x_genes) == len(set(x_genes)), 'Gene names are not unique'
    
    # Center gene expresison by cell
    if center:
        X = center_arr(X)

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
    estimate = dot_mult(X, R)
    
    # Permutations
    if num_perm > 0:
        pvals = np.zeros(estimate.shape)
        for i in tqdm(range(num_perm)):
            perm = dot_mult(X, default_rng(seed=i).permutation(R))
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
        tf_act = scale_arr(tf_act, scale_axis)

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
        mc = np.mean(adata.X[g_msk,i]) - np.mean(adata.X[ref_msk,i])
        results.append([features[i], group, reference, stat, mc, pval])

    # Tranform to df
    results = pd.DataFrame(
        results, 
        columns=['name', 'group', 'reference', 'statistic', 'meanchange', 'pval']
    ).set_index('name')
    
    # Correct pvalues by FDR
    results[np.isnan(results['pval'])] = 1
    _, pvals_adj, _, _ = multipletests(
        results['pval'].values, alpha=0.05, method='fdr_bh'
    )
    results['pval_adj'] = pvals_adj
    
    # Sort by statistic
    results = results.sort_values('meanchange', ascending=False)
    return results


def check_regulon(adata, regnet, tf, groupby, use_raw=False, use_hvg=False, figsize=(12,6), 
                  cmap='rocket', show=None, return_fig=None):
    """
    Plots a heatmap with the expression of target genes for a given TF.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    regnet
        Regulon network in DataFrame format.
    tf
        Name of TF.
    groupby
        The key of the observations grouping to consider.
    use_raw
        If data is an AnnData object, whether to use values stored in `.raw`.
    use_hvg
        If data is an AnnData object, whether to only use high variable genes.
    figsize
        Size of the figure.
    cmap
        Color map to use.
    show
        Show the plot, do not return axis.
    return_fig
        Return the matplotlib figure.
    Returns
    -------
    Heatmap figure.
    """
    # Get genes, samples/tfs and matrices from data and regnet
    x_genes, x_samples, X = process_input(adata, use_raw=use_raw, use_hvg=use_hvg)

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

    idx_x = np.searchsorted(x_genes, common_genes)
    X = X[:,idx_x]
    R = regnet.loc[common_genes].values
    R = R[:,list(r_tfs).index(tf)]

    X = X[:,R!=0]
    common_genes = common_genes[R!=0]
    
    sort_genes = np.argsort(np.mean(X*-1,axis=0)).flat
    X = X[:,sort_genes]
    common_genes = common_genes[sort_genes]
    
    groups = np.unique(adata.obs[groupby])
    fig, axes = plt.subplots(len(groups), 1, 
                             gridspec_kw={'hspace': 0.05}, 
                             sharex=True,
                             figsize=figsize
                            )
    fig.suptitle(tf, fontsize=16)
    axes = axes.flatten()
    max_n = np.max(X)
    min_n = np.min(X)
    i = 1
    X = pd.DataFrame(X.A, columns=common_genes)
    for group,ax in zip(groups, axes):
        msk = (adata.obs[groupby] == group).values
        if i == len(groups):
            sns.heatmap(X.loc[msk], cbar=True,
                        yticklabels='', ax=ax, vmin=min_n, vmax=max_n,
                        cbar_kws = {"shrink": .70}, cmap=cmap
                       )
        else:
            sns.heatmap(X.loc[msk], cbar=True,
                        yticklabels='', ax=ax, vmin=min_n, vmax=max_n,
                        cbar_kws = {"shrink": .70}, cmap=cmap
                       )
            ax.axes.xaxis.set_visible(False)
        ax.set_ylabel(group, rotation='horizontal', ha='right')
        i += 1
    if return_fig is True:
        return fig
    if show is False:
        return axes
    plt.show()