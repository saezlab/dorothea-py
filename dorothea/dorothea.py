import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import scanpy as sc
from anndata import AnnData
import itertools
import multiprocessing

def match(x, table):
    table = list(table)
    m = [table.index(i) for i in x]
    return np.array(m)

def InferTFact(tf_v, expr_v):
    # Build lm
    slope, _, _, _, std_err = stats.linregress(x=tf_v, y=expr_v)
    # t-stat is the TF act, which is equal to slope/std_err
    tf_act = slope/std_err
    return tf_act

def InferTFactPRL(idx):
    exp_v = ndata[:,idx]
    act_v = [InferTFact(tf_v, exp_v) for tf_v in nregnet]
    return act_v


def run_scira(data, regnet, norm = "z", njobs=4):
    # Transform to df if AnnData object is given
    if isinstance(data, AnnData):
        data = pd.DataFrame(np.transpose(data.X.toarray()), index=data.var.index, 
                               columns=data.obs.index)

    # Get intersection of genes between expr data and the given regnet
    common_v = set(data.index.values) & set(regnet.index.values)
    map1_idx = match(common_v, data.index.values)
    map2_idx = match(common_v, regnet.index.values)

    # Centering
    ndata = np.array(data)[map1_idx,] - np.matrix(np.mean(np.array(data)[map1_idx,], axis=1)).transpose()

    if norm == "z":
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
    
    #it = itertools.product(np.array(regnet)[map2_idx,].T, ndata.T)
    #tf_acts = np.array([InferTFact(regnet,exp) for regnet,exp in it]).reshape(data.shape[1], regnet.shape[1])
    nregnet = np.array(regnet)[map2_idx,]
    
    tf_data = AnnData(np.array([[InferTFact(tf_v, expr_v) for tf_v in nregnet.T] for expr_v in ndata.T]))
    tf_data.obs.index = data.columns
    tf_data.var.index = regnet.columns
    
    return tf_data