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
    genes = np.unique(df.target)
    tfs = np.unique(df.tf)
    map_genes = {gene:i for i,gene in enumerate(genes)}
    map_tfs = {tf:i for i,tf in enumerate(tfs)}
    
    dorothea_df = np.zeros((len(genes), len(tfs)))
    for index, row in df.iterrows():
        tf = row['tf']
        gene = row['target']
        mor = int(row['mor'])
        dorothea_df[map_genes[gene], map_tfs[tf]] = mor
        
    dorothea_df = pd.DataFrame(dorothea_df, columns=tfs, index=genes)
    
    return dorothea_df

def match(x, table):
    """Returns a vector of the positions of (first) matches of 
    its first argument in its second"""
    table = list(table)
    m = [table.index(i) for i in x]
    return np.array(m)


def InferTFact(tf_m, exp_v):
    """Computes the activity of all TFs for a given cell"""
    TINY = 1.0e-20
    # Each row is a TF and a exp
    n_repeat, df = tf_m.shape
    # Repeat exp for each tf
    exp_m = np.repeat([exp_v], n_repeat, axis=0)
    # Compute lm
    cov = np.cov(tf_m, exp_m, bias=1)
    ssxm, ssym = np.split(np.diag(cov), 2)
    ssxym = np.diag(cov, k=len(tf_m))
    # Compute R value
    r = ssxym / np.sqrt(ssxm * ssym)
    # Compute t-value = TF activity
    tf_act = r * np.sqrt(df / ((1.0 - r + TINY)*(1.0 + r + TINY)))
    return tf_act


def run_scira(data, regnet, norm='c', inplace=True, scale=True):
    """This function is a wrapper to run SCIRA using regulons."""
    # Transform to df if AnnData object is given
    if isinstance(data, AnnData):
        if data.raw is None:
            df = pd.DataFrame(np.transpose(data.X), index=data.var.index, 
                                   columns=data.obs.index)

        else:
            df = pd.DataFrame(np.transpose(data.raw.X.toarray()), index=data.raw.var.index, 
                                   columns=data.raw.obs_names)

    # Get intersection of genes between expr data and the given regnet
    common_v = sorted(set(df.index.values) & set(regnet.index.values))
    map1_idx = match(common_v, df.index.values)
    map2_idx = match(common_v, regnet.index.values)

    if norm == "c":
        # Centering
        ndata = np.array(df)[map1_idx,]
        ndata = ndata - np.mean(ndata, axis=1, keepdims=True)
    else:
        ndata = np.array(data)[map1_idx,]
       
    # Order, filter and transpose (each row is tf and exp vector)
    nregnet = np.array(regnet)[map2_idx,].T
    ndata = ndata.T
    
    # Compute TF activities
    result = np.array([InferTFact(nregnet, expr_v) for expr_v in ndata])
    
    # Set nans to 0
    result[np.isnan(result)] = 0.0
    
    if scale:
        std = np.std(result, ddof=1, axis=0)
        std[std == 0] = 1
        result = (result - np.mean(result, axis=0)) / std
    
    # Store in df
    result = pd.DataFrame(result, columns=regnet.columns, index=df.columns)

    if isinstance(data, AnnData) and inplace:
        # Update AnnData object
        data.obsm['dorothea'] = result
    else:
        # Return dataframe object
        data = result

    return data if not inplace else None