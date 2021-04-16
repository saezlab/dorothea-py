import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
import pickle
import pkg_resources
import os


def load_regulons(levels=['A', 'B', 'C', 'D', 'E'], organism='Human', commercial=False):
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

def extract(data, obsm_key='dorothea'):
    obsm = data.obsm
    obs = data.obs
    df = data.obsm['dorothea']
    var = pd.DataFrame(index=df.columns)
    tadata = AnnData(np.array(df), obs=obs, var=var, obsm=obsm)
    return tadata
    

def process_input(data, use_raw=False):
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
        genes = np.array(df.columns)
        idx = np.argsort(genes)
        genes = genes[idx]
        samples = df.index
        X = np.array(df)[:,idx]
    else:
        raise ValueError('Input must be AnnData or pandas DataFrame.')
    return genes, samples, X


def run(data, regnet, center=True, scale=True, inplace=True, norm=True, use_raw=False):
    # Get genes, samples/tfs and matrices from data and regnet
    x_genes, x_samples, X = process_input(data, use_raw=use_raw)

    assert len(x_genes) == len(set(x_genes)), 'Gene names are not unique'

    if X.shape[0] <= 1 and (center or scale):
        raise ValueError('If there is only one observation no centering nor scaling can be performed.')

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

    if center:
        X = X - np.mean(X, axis=0)

    # Run matrix mult
    result = np.asarray(X.dot(R))
    
    # Normalize by num edges
    if norm:
        norm = np.sum(np.abs(R), axis=0)
        norm[norm == 0] = 1
        result = result / norm

    if scale:
        std = np.std(result, ddof=1, axis=0)
        std[std == 0] = 1
        result = (result - np.mean(result, axis=0)) / std

    # Remove nans
    result[np.isnan(result)] = 0

    # Store in df
    result = pd.DataFrame(result, columns=r_tfs, index=x_samples)

    if isinstance(data, AnnData) and inplace:
        # Update AnnData object
        data.obsm['dorothea'] = result
    else:
        # Return dataframe object
        data = result

    return data if not inplace else None
