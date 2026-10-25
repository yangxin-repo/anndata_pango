
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import scvi
import warnings
from scipy.sparse import csr_matrix
from anndata import AnnData
from scvi.external import SysVI

from .utils_anndata_pango import (transform_dataframe_dtype)
from .image_pango.image_pango import Image_Pango

# scvi settings
scvi.settings.dl_num_workers = 2
scvi.settings.dl_persistent_workers = True

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
                          image_path:str,
                          microns_per_pxl:float,
                          tissue_position_path:str=None,
                          scalefactor:float=1,
                          array_row:str="array_row",
                          array_col:str="array_col",
                          pxl_row:str = "pxl_row_in_fullres",
                          pxl_col:str = "pxl_col_in_fullres",
                          platform:str = "10x"):

        if platform == "10x" and tissue_position_path is not None:

            # load the tissue position information
            ti_po_df = pd.read_parquet(tissue_position_path)
            ti_po_df["pxl_row"] = ti_po_df[pxl_row]
            ti_po_df["pxl_col"] = ti_po_df[pxl_col]
            ti_po_df["array_row"] = ti_po_df[array_row]
            ti_po_df["array_col"] = ti_po_df[array_col]
            ti_po_df.index = ti_po_df["barcode"]

            # write spatial to obsm
            self._adata.obsm["spatial"] = (pd.DataFrame(index=self._adata.obs_names)
                                           .join(ti_po_df[["array_col","array_row"]])
                                           .to_numpy())

            # create the Image_Pango class
            self._adata.uns["source_image"] = Image_Pango(image_path=image_path,
                                                          microns_per_pxl=microns_per_pxl)

            # add the registration information
            self._adata.uns["source_image"].add_coordinate_mapping(coordinate_mapping=ti_po_df[["barcode","array_row","array_col","pxl_row","pxl_col"]],
                                                                   scalefactor=scalefactor,
                                                                   mapping_key="barcode_registration")

            # extract the tissue part
            self._adata.uns["source_image"].extract_mapping_area(mapping_key="barcode_registration")

        if platform == "stereo":

            # load the tissue position information
            ti_po_df = self._adata.obs[["x","y"]].copy()
            ti_po_df["barcode"] = self._adata.obs_names.to_numpy()
            ti_po_df[["array_row","array_col"]] = ti_po_df[["x","y"]].copy()
            ti_po_df[["pxl_row","pxl_col"]] = ti_po_df[["x","y"]].copy()
            ti_po_df.index = ti_po_df["barcode"]

            # write spatial to obsm
            self._adata.obsm["spatial"] = (pd.DataFrame(index=self._adata.obs_names)
                                           .join(ti_po_df[["array_col","array_row"]])
                                           .to_numpy())
            
            # create the Image_Pango class
            self._adata.uns["source_image"] = Image_Pango(image_path=image_path,
                                                          microns_per_pxl=microns_per_pxl)

            # add the registration information
            self._adata.uns["source_image"].add_coordinate_mapping(coordinate_mapping=ti_po_df[["barcode","array_row","array_col","pxl_row","pxl_col"]],
                                                                   scalefactor=scalefactor,
                                                                   mapping_key="barcode_registration")

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
                            model_path:str=None,
                            microns_per_pixel:float=0.3,
                            platform:str="10x",
                            model_name:str = "2D_versatile_he",
                            resize_source_image:bool=True):
        
        # resize the source image to adapt the stardist model
        if resize_source_image:
            self._adata.uns["source_image"].resize_source_image(microns_per_pixel=microns_per_pixel)

        # load pixel matrix
        self._adata.uns["source_image"].load_pixel_matrix(normalize=True,
                                                          platform=platform)

        # fit the model
        self._adata.uns["source_image"].fit_stardist_models(model_path=model_path,
                                                            platform=platform,
                                                            model_name=model_name)

        # expand the label to 3 pixel surrounding
        self._adata.uns["source_image"].expand_mask_labels(mask_label="stardist_label")

        # resample the label to barcode
        self._adata.uns["source_image"].resample_mask_labels(mask_label="stardist_label",
                                                             mapping_key="barcode_registration")
        self._adata.uns["source_image"].resample_mask_labels(mask_label="stardist_label_expanded",
                                                             mapping_key="barcode_registration")

        # write the label to obs
        self._adata.obs = self._adata.obs.join(self._adata.uns["source_image"]
                                               .get_mapping_coordinate(mapping_key="barcode_registration")
                                               .set_index("barcode"))

        return None

    def aggregate_labels (self,
                         label:str,
                         except_labels:list,
                         platform:str=None,
                         chip_series:str=None) -> AnnData:
        
        # transform the label to category
        self._adata.obs[label] = (np.char.add(f"{label}_",
                                             self._adata.obs[label].astype(str).to_xarray()))
        
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
        if 'counts' not in self._adata.layers:
            self._adata.layers["counts"] = self._adata.X.copy()

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

        # add additional information
        if platform is not None:
            self._adata.obs["platform"] = platform
        if chip_series is not None:
            self._adata.obs["chip_series"] = chip_series

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
                                      mask_label:str,
                                      mask_label_expanded:str=None):

        self._adata.uns["source_image"].show_labels(mapping_key=mapping_key,
                                                    mask_label=mask_label,
                                                    mask_label_expanded=mask_label_expanded)

        return None

    def fit_scvi_models(self,
                        batch_key:str="batch",
                        categorical_covariate_keys:list=None,
                        continuous_covariate_keys:list=None):
        
        # set the random seed for reproducibility
        scvi.settings.seed = 2026

        # select the HCGs
        scvi.data.poisson_gene_selection(self._adata,
                                         n_top_genes=4000,
                                         batch_key=batch_key,
                                         subset=True,
                                         layer="counts")

        # register adata to scvi
        self._adata.X = self._adata.layers["counts"].copy()
        scvi.model.SCVI.setup_anndata(self._adata,
                                      batch_key=batch_key,
                                      categorical_covariate_keys=categorical_covariate_keys,
                                      continuous_covariate_keys=continuous_covariate_keys)

        # build the scvi model
        model = scvi.model.SCVI(self._adata,
                                n_latent=50,
                                n_layers=2,
                                n_hidden=128,
                                gene_likelihood="zinb")

        # train the model
        model.train(max_epochs=200,
                    batch_size=128 ,
                    train_size=0.9,
                    early_stopping=True,
                    check_val_every_n_epoch=10)

        # write the latent to adata
        self._adata.obsm["X_scvi"] = model.get_latent_representation()

        return model

    def fit_sysVI_models(self,
                        batch_key:str="batch",
                        categorical_covariate_keys:list=None,
                        continuous_covariate_keys:list=None):

        # set the random seed for reproducibility
        scvi.settings.seed = 2026

        # select the HCGs
        self._adata.X = self._adata.layers["log1p"].copy()
        sc.pp.highly_variable_genes(self._adata,
                                    n_top_genes=4000,
                                    batch_key=batch_key,
                                    subset=True)

        # register adata to sysVI
        SysVI.setup_anndata(self._adata,
                            batch_key=batch_key,  
                            categorical_covariate_keys=categorical_covariate_keys,
                            continuous_covariate_keys=continuous_covariate_keys)

        # build the sysVI model
        model = SysVI(self._adata)

        # train the model
        model.train(max_epochs=200,
                    check_val_every_n_epoch=1,
                    plan_kwargs={"z_distance_cycle_weight": 2})

        # write the latent to adata
        self._adata.obsm["X_sysVI"] = model.get_latent_representation()

        return model

    def build_graph_related_space(self,
                                  use_rep:str="X_scvi",
                                  n_neighbors:int=30,
                                  n_pcs:int=50):

        # build the neighbor graph
        sc.pp.neighbors(self._adata,
                        use_rep=use_rep,
                        n_neighbors=n_neighbors,
                        n_pcs=n_pcs)

        # build the UMAP and t-SNE space
        sc.tl.umap(self._adata, 
                   min_dist=0.8)
        sc.tl.tsne(self._adata, 
                   use_rep="X_scvi")

        # build the leiden clustering
        sc.tl.leiden(self._adata, 
                     resolution=1, 
                     key_added="leiden", 
                     flavor="igraph",
                     n_iterations=-1)

        
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

def concat_pango(adata_list:list,
                 batch_key:list,
                 save_var:list=None,
                 normalize:bool=False) -> AnnData:

    # get the save var colnumns
    ad_num = len(adata_list)
    if save_var is None:
        save_var=[[] for i in range(ad_num)]
    else:
         var_num = len(save_var)
         if ad_num > var_num:
             save_var.extend([[] for _ in range(ad_num-var_num)])
         elif ad_num <= var_num:
             save_var = save_var[:ad_num]

    for ad_obj,var_col in zip(adata_list,save_var):
        ad_obj.var = ad_obj.var[var_col].copy()

    # normalize the adata list
    if normalize:
        for ad_obj in adata_list:
            ad_obj.X = ad_obj.layers["counts"].copy()
            
            sc.pp.filter_cells(ad_obj, 
                               min_counts=1)
            sc.pp.filter_genes(ad_obj, 
                               min_cells=1)
            sc.pp.normalize_total(ad_obj, 
                                  target_sum=1e4)
            sc.pp.log1p(ad_obj)
            ad_obj.layers["log1p"] = ad_obj.X.copy()

    # concat adata list
    adata_comb = ad.concat(adata_list,
                           join="inner",
                           merge="first",
                           label="batch",
                           keys=batch_key,
                           index_unique="_")

    return adata_comb