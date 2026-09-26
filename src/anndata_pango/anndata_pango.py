
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
from scipy.sparse import csr_matrix

from .utils_anndata_pango import (transform_dataframe_dtype)

# register to AnnData's accessor
@ad.register_anndata_namespace("pango")
class Pango_Accessor:

    # init the accessor
    def __init__(self,adata: ad.AnnData):
        self._adata = adata

    # save the adata obj
    def save(self,save_path: str):

        # load the whole adata to memory
        adata = self._adata.to_memory()

        # transform the adata.X
        if adata.X is None:
            adata.X = csr_matrix((adata.n_obs, adata.n_vars), dtype='int8')

        # transform the str to object
        adata.obs_names = adata.obs_names.astype(object)
        adata.var_names = adata.var_names.astype(object)

        # transform the pd.dataframe
        adata.obs = transform_dataframe_dtype(adata.obs)
        adata.var = transform_dataframe_dtype(adata.var)
        for key, value in adata.uns.items():
            if isinstance(value, pd.DataFrame):
                adata.uns[key] = transform_dataframe_dtype(value)

        # write the adata obj to h5 system
        adata.write_h5ad(save_path)

        return None

def load_10x_h5_pango(h5_path: str):
    # load expression matrix from 10x_h5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adata = sc.read_10x_h5(h5_path)
        adata.var_names_make_unique()

    # write counts to layers and empty adata.X
    adata.layers["counts"] = adata.X.copy()
    adata.X = None

    # write barcode to adata.obs
    adata.obs["barcode"] = adata.obs_names.to_numpy().copy()

    return adata