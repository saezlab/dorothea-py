import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
import pickle
import pkg_resources


def load_regulons(levels=['A', 'B', 'C', 'D', 'E'], organism='Human'):
    # Get package path
    if organism == "Human":
        path = pkg_resources.resource_filename(__name__, 'data/dorothea_hs.pkl')
    elif organism == "Mouse":
        path = pkg_resources.resource_filename(__name__, 'data/dorothea_mm.pkl')
    else:
        raise("Wrong organism name. Please specify 'Human' or 'Mouse'.")
    
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
    

def process_input(data):
    if isinstance(data, AnnData):
        if data.raw is None:
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

def run(data, regnet, center=True, scale=True, inplace=True):
    # Get genes, samples/tfs and matrices from data and regnet
    x_genes, x_samples, X = process_input(data)
    if X.shape[0] <= 1 and (center or scale):
        raise ValueError('If there is only one observation no centering nor scaling can be performed.')
    r_genes, r_tfs, R = np.sort(regnet.index), regnet.columns, np.array(regnet)

    # Subset by common genes
    common_genes = np.sort(list(set(r_genes) & set(x_genes)))
    map_x = np.searchsorted(x_genes, common_genes)
    map_r = np.searchsorted(r_genes, common_genes)
    X = X[:,map_x]
    R = R[map_r]

    if center:
        X = X - np.mean(X, axis=0)

    # Run matrix mult
    result = np.asarray(X.dot(R))

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