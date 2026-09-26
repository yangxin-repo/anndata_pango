
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import warnings
from scipy.sparse import csr_matrix

from .utils_anndata_pango import (transform_dataframe_dtype)
from .image_pango.image_pango import Image_Pango

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

    def load_source_image(self,
                          tissue_position_path:str,
                          image_path:str,
                          microns_per_pxl:float,
                          scalefactor:float=1,
                          array_row:str="array_row",
                          array_col:str="array_col",
                          pxl_row:str = "pxl_row_in_fullres",
                          pxl_col:str = "pxl_col_in_fullres"):

        # load the tissue position information
        ti_po_df = pd.read_parquet(tissue_position_path)
        ti_po_df["pxl_row"] = ti_po_df[pxl_row]
        ti_po_df["pxl_col"] = ti_po_df[pxl_col]
        ti_po_df["array_row"] = ti_po_df[array_row]
        ti_po_df["array_col"] = ti_po_df[array_col]

        # create the Image_Pango class
        self._adata.uns["source_image"] = Image_Pango(image_path=image_path,
                                                      microns_per_pxl=microns_per_pxl)

        # add the registration information
        self._adata.uns["source_image"].add_coordinate_mapping(coordinate_mapping=ti_po_df[["barcode","array_row","array_col","pxl_row","pxl_col"]],
                                                               scalefactor=scalefactor,
                                                               mapping_key="barcode_registration")

        # extract the tissue part
        self._adata.uns["source_image"].extract_mapping_area(mapping_key="barcode_registration")

        return None

    def load_tissue_position(self,
                             tissue_position_path:str):

        # load the tissue position dataframe
        ti_po_df = pd.read_parquet(tissue_position_path).set_index("barcode")

        # write the spatial to obsm
        self._adata.obsm["spatial"] = (pd.DataFrame(index=self._adata.obs_names)
                                       .join(ti_po_df[["array_col","array_row"]])
                                       .to_numpy())

        return None

    def get_stardist_labels(self,
                            model_path:str):
        
        # resize the source image to adapt the stardist model
        self._adata.uns["source_image"].resize_source_image(microns_per_pixel=0.3)

        # load pixel matrix
        self._adata.uns["source_image"].load_pixel_matrix(normalize=True)

        # fit the model
        self._adata.uns["source_image"].fit_stardist_models(model_path=model_path)

        # expand the label to 3 pixel surrounding
        self._adata.uns["source_image"].expand_mask_labels(mask_label="stardist_label")

        # resample the label to barcode
        self._adata.uns["source_image"].resample_mask_labels(mask_label="stardist_label_expanded",
                                                             mapping_key="barcode_registration")

        # write the label to obs
        self._adata.obs = self._adata.obs.join(self._adata.uns["source_image"]
                                               .get_mapping_coordinate(mapping_key="stardist_label_expanded"))

        return None

    def aggregate_labels(self,
                         label:str,
                         except_labels:list):
        
        # transform the label to category
        self._adata.obs[label] = (np.char.add(f"{label}_",
                                             self._adata.obs["stardist_label"].astype(str).to_xarray()))
        
        except_labels = [f"{label}_{lab}" for lab in except_labels]

        # drop the except_labels
        self._adata = self._adata[~(self._adata.obs[label]
                                    .isin(except_labels))].copy()

        # extract the spatial for aggregate
        spatial_df = (pd.DataFrame(self._adata.obsm["spatial"],
                                  index=self._adata.obs_names,
                                  columns=["array_row","array_col"]).
                                  join(self._adata.obs[label]))

        # aggregate stardist labels
        self._adata = sc.get.aggregate(self._adata,
                                       by=label,
                                       func="sum",
                                       layer="counts")

        # aggregate spatial coordinate and write to adata
        spatial_df = (spatial_df
                      .groupby(by=label,
                               observed=True)[['array_row', 'array_col']]
                      .mean())
        self._adata.obsm["spatial"] = (pd.DataFrame(index=self._adata.obs_names)
                                       .join(spatial_df)
                                       .to_numpy())

        # # write sum as counts in csr_matrix dtype
        self._adata.layers["counts"] = csr_matrix(self._adata.layers["sum"])
        del self._adata.layers["sum"]

        # calculate the QC metrics
        self._adata.var["mt"] = self._adata.var_names.str.startswith("mt-")
        sc.pp.calculate_qc_metrics(adata=self._adata,
                                   percent_top=None,
                                   qc_vars=["mt"],
                                   log1p=True,
                                   inplace=True,
                                   layer="counts")

        return self._adata

    def add_obs_to_coordinate_mapping(self,
                                      mapping_key:str,
                                      obs_cols:list):
        
        self._adata.uns["source_image"].add_labels_to_coordinate_mapping(by="barcode",
                                                                         mapping_key=mapping_key,
                                                                         label_df=self._adata.obs[["barcode"]+obs_cols])

        return None

    def plot_labels_with_source_image(self,
                                      mapping_key:str,
                                      mask_label:str):

        self._adata.uns["source_image"].show_labels(mapping_key=mapping_key,
                                                    mask_label=mask_label)

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