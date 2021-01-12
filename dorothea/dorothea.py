import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import scanpy as sc
from anndata import AnnData
import itertools
import multiprocessing
import pickle
import pkg_resources


def load_regulons(levels=['A', 'B', 'C', 'D', 'E']):
    # Get package path
    path = pkg_resources.resource_filename(__name__, 'data/dorothea_hs.pkl')
    
    # Open pickle object
    df = pickle.load(open(path, "rb" ))
    
    #Filter by levels of confidence
    df = df[df['confidence'].isin(levels)]
    
    # Transform to binary dataframe
    genes = np.unique(df.target)
    tfs = np.unique(df.tf)
    map_genes = {gene:i for i,gene in enumerate(genes)}
    map_tfs = {tf:i for i,tf in enumerate(tfs)}
    
    dorothea_hs = np.zeros((len(genes), len(tfs)))
    for index, row in df.iterrows():
        tf = row['tf']
        gene = row['target']
        mor = int(row['mor'])
        dorothea_hs[map_genes[gene], map_tfs[tf]] = mor
        
    dorothea_hs = pd.DataFrame(dorothea_hs, columns=tfs, index=genes)
    
    return dorothea_hs

def match(x, table):
    """Returns a vector of the positions of (first) matches of 
    its first argument in its second"""
    table = list(table)
    m = [table.index(i) for i in x]
    return np.array(m)

def InferTFact(tf_v, expr_v):
    """Computes the TF activity by buidling a lm where x is the 
    regulon and y is the observed expression"""
    # Build lm
    slope, _, _, _, std_err = stats.linregress(x=tf_v, y=expr_v)
    # t-stat is the TF act, which is equal to slope/std_err
    tf_act = slope/std_err
    return tf_act

def run_scira(data, regnet, norm = None):
    """This function is a wrapper to run SCIRA using regulons."""
    # Transform to df if AnnData object is given
    if isinstance(data, AnnData):
        data = pd.DataFrame(np.transpose(data.X), index=data.var.index, 
                               columns=data.obs.index)

    # Get intersection of genes between expr data and the given regnet
    common_v = set(data.index.values) & set(regnet.index.values)
    map1_idx = match(common_v, data.index.values)
    map2_idx = match(common_v, regnet.index.values)

    if norm == "c":
        # Centering
        ndata = np.array(data)[map1_idx,] - np.matrix(np.mean(np.array(data)[map1_idx,], axis=1)).transpose()
    elif norm == "z":
        # Compute sd_v per gene
        sd_v = np.std(np.array(data)[map1_idx,], axis=1)
        # Check genes where std > 0
        nz_idx = np.where(sd_v > 0)[0]
        z_idx = np.where(sd_v == 0)[0]
        ndata = np.array(data)[map1_idx,]
        # Z-score normalize
        ndata[nz_idx,] = (np.array(data)[map1_idx[nz_idx],] - 
                         np.matrix(np.mean(np.array(data)[map1_idx[nz_idx],], 
                                           axis=1)).transpose()) / np.matrix(sd_v[nz_idx]).transpose()
        ndata[z_idx,] = 0
    else:
        ndata = np.array(data)[map1_idx,]
       
    # Order and filter regnet
    nregnet = np.array(regnet)[map2_idx,]
    
    # Compute TF activity and generate a new AnnData object
    tf_data = AnnData(np.array([[InferTFact(tf_v, expr_v) for tf_v in nregnet.T] for expr_v in ndata.T]))
    tf_data.X[np.isnan(tf_data.X)] = 0.0
    tf_data.obs.index = data.columns
    tf_data.var.index = regnet.columns
    
    return tf_data