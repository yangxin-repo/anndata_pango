
import gc
import warnings
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import spatialleiden as sl
import squidpy as sq
from scipy.sparse import csr_matrix
from scvi.external import SysVI

from .image_pango.image_pango import Image_Pango
from .utils_anndata_pango import (
    convert_gene_symbols_to_ensembl_ids_pango,
    map_cell_type_pango,
    transform_dataframe_dtype,
)

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
    def save(self,
             save_path: str) -> None:

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
                          tissue_position_path:str|None = None,
                          array_col:str|None = None,
                          array_row:str|None = None,
                          pxl_row:str|None = None,
                          pxl_col:str|None = None,
                          barcode:str|None = None,
                          scalefactor:float=1,
                          platform:Literal["10x","stereo"]="10x") -> None:

        if platform == "10x":

            if  tissue_position_path is None:
                raise ValueError("Please provide the tissue_position_path for 10x platform.")
            else:
                # verify array_row,array_col,pxl_row,pxl_col
                if array_row is None:
                    array_row = ["array_row"]
                else:
                    array_row = [array_row]
                    
                if array_col is None:
                    array_col = ["array_col"]
                else:
                    array_col = [array_col]
                    
                if pxl_row is None:
                    pxl_row = ["pxl_row_in_fullres"]
                else:
                    pxl_row = [pxl_row]
                if pxl_col is None:
                    pxl_col = ["pxl_col_in_fullres"]
                else:
                    pxl_col = [pxl_col] 
                    
                if barcode is None:
                    barcode = ["barcode"]
                else:
                    barcode = [barcode]
                    
                # load the tissue position information
                try:
                    ti_po_df = pd.read_parquet(tissue_position_path)
                    ti_po_df["pxl_row"] = (ti_po_df[pxl_row]
                                           .to_numpy()
                                           .flatten()
                                           .astype(np.float32))
                    ti_po_df["pxl_col"] = (ti_po_df[pxl_col]
                                           .to_numpy()
                                           .flatten()
                                           .astype(np.float32))
                    ti_po_df["array_row"] = (ti_po_df[array_row]
                                             .to_numpy()
                                             .flatten()
                                             .astype(np.float32))
                    ti_po_df["array_col"] = (ti_po_df[array_col]
                                             .to_numpy()
                                             .flatten()
                                             .astype(np.float32))
                    ti_po_df.index = (ti_po_df[barcode]
                                      .to_numpy()
                                      .flatten()
                                      .astype(str))
                except Exception as exc:
                    raise ValueError("Columns not found in tissue position dataframe for 10x platform.") from exc

                # write spatial to obsm
                self._adata.obsm["spatial"] = (pd.DataFrame(index=self._adata.obs_names)
                                               .join(ti_po_df[["array_col","array_row"]])
                                               .to_numpy()
                                               .astype(np.float32))

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

            # verify array_row,array_col,pxl_row,pxl_col
            if array_row is None:
                array_row = ["x"]
            else:
                array_row = [array_row]
                
            if array_col is None:
                array_col = ["y"]
            else:
                array_col = [array_col]
                
            if pxl_row is None:
                pxl_row = ["x"]
            else:
                pxl_row = [pxl_row] 
                
            if pxl_col is None:
                pxl_col = ["y"]
            else:
                pxl_col = [pxl_col]
                
            if barcode is None:
                barcode = ["barcode"]
            else:
                barcode = [barcode]
                
            # load the tissue position information
            try:
                ti_po_df = self._adata.obs.copy()
                ti_po_df["barcode"] = self._adata.obs_names.to_numpy()
                ti_po_df[["array_row","array_col"]] = ti_po_df[array_row+array_col].copy()
                ti_po_df[["pxl_row","pxl_col"]] = ti_po_df[pxl_row+pxl_col].copy()
                ti_po_df.index = (ti_po_df["barcode"]
                                  .to_numpy()
                                  .astype(str))
            except Exception as exc:
                raise ValueError("Columns not found in adata.obs for stereo platform.") from exc

            # write spatial to obsm
            self._adata.obsm["spatial"] = (pd.DataFrame(index=self._adata.obs_names)
                                           .join(ti_po_df[["array_col","array_row"]])
                                           .to_numpy()
                                           .astype(np.float32))
            
            # create the Image_Pango class
            self._adata.uns["source_image"] = Image_Pango(image_path=image_path,
                                                          microns_per_pxl=microns_per_pxl)

            # add the registration information
            self._adata.uns["source_image"].add_coordinate_mapping(coordinate_mapping=ti_po_df[["barcode","array_row","array_col","pxl_row","pxl_col"]],
                                                                   scalefactor=scalefactor,
                                                                   mapping_key="barcode_registration")

        return None

    def get_stardist_labels(self,
                            model_path:str|None = None,
                            microns_per_pixel:float=0.3,
                            platform:Literal["10x","stereo"]="10x",
                            model_name:str|None = None) -> None:
        
        # resize the source image to adapt the stardist model
        if platform == "10x":
            self._adata.uns["source_image"].resize_source_image(microns_per_pixel=microns_per_pixel)
            if model_name is None:
                model_name = "2D_versatile_he"
        elif platform == "stereo":
            if model_name is None:
                model_name = "2D_versatile_fluo"

        # load pixel matrix
        self._adata.uns["source_image"].load_pixel_matrix(normalize=True,
                                                          platform=platform)

        # fit the model
        self._adata.uns["source_image"].fit_stardist_models(model_path=model_path,
                                                            platform=platform,
                                                            model_name=model_name)

        # expand the label to 3 pixel surrounding
        self._adata.uns["source_image"].expand_mask_labels(mask_label="stardist_label",
                                                           platform=platform)

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
                         platform:str|None = None,
                         chip_series:str|None = None) -> ad.AnnData:
        
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
        
        # delete the X
        del self._adata.X

        print("start to aggregate the adata")
        self._adata = sc.get.aggregate(self._adata,
                                       by=label,
                                       func="sum",
                                       layer="counts")
        print("finished aggregating the adata")
        
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
        gc.collect()
        
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
        
        # sort the adata by obs_names
        cell_names = pd.Index(self._adata.obs_names).astype(str)
        order_nums = cell_names.str.rsplit("_", n=1).str[-1].astype(int)
        self._adata = self._adata[np.argsort(order_nums)].copy()

        return self._adata

    def add_obs_to_coordinate_mapping(self,
                                      mapping_key:str,
                                      obs_cols:list) -> None:
        
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
                        categorical_covariate_keys:list | None=None,
                        continuous_covariate_keys:list | None=None,
                        highly_variable_genes:bool=False):
        
        # set the random seed for reproducibility
        scvi.settings.seed = 2026

        # select the HCGs
        if highly_variable_genes:
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
                        continuous_covariate_keys:list=None,
                        highly_variable_genes:bool=False):

        # set the random seed for reproducibility
        scvi.settings.seed = 2026

        # confirm the log1p
        self._adata.X = self._adata.layers["log1p"].copy()

        # select the HCGs
        if highly_variable_genes:
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
                                  n_pcs:int=None):

        # build the neighbor graph
        sc.pp.neighbors(self._adata,
                        use_rep=use_rep,
                        n_neighbors=n_neighbors,
                        n_pcs=n_pcs)

        # build the UMAP and t-SNE space
        sc.tl.umap(self._adata, 
                   min_dist=0.8)
        sc.tl.tsne(self._adata, 
                   use_rep=use_rep)

        # build the leiden clustering
        sc.tl.leiden(self._adata, 
                     resolution=0.5, 
                     key_added="leiden", 
                     flavor="igraph",
                     n_iterations=-1)

        return None

    def conduct_spatial_cluster(self,
                                rank_method:Literal["wilcoxon","logreg"]="wilcoxon",
                                max_iter:int|None = None) -> None:

        if rank_method=="logreg" and max_iter is None:
            max_iter = 100

        self._adata.X = self._adata.layers["counts"].copy()
        sc.pp.filter_cells(self._adata,
                           min_counts=1,
                           inplace=True)
        sc.pp.filter_genes(self._adata,
                           min_cells=1,
                           inplace=True)
        sc.experimental.pp.highly_variable_genes(
            self._adata,
            flavor='pearson_residuals',
            n_top_genes=2000
            )

        adata_hvg = self._adata[:,self._adata.var["highly_variable"]].copy()

        sc.pp.filter_cells(adata_hvg,
                           min_counts=1,
                           inplace=True)
        sc.pp.filter_genes(adata_hvg,
                           min_cells=1,
                           inplace=True)

        sc.experimental.pp.normalize_pearson_residuals(adata_hvg,
                                                       inplace=True)

        sc.pp.pca(adata_hvg, 
                  n_comps=50,
                  random_state=42)

        self._adata._inplace_subset_obs(self._adata.obs_names.isin(adata_hvg.obs_names))
        self._adata.obsm["X_pca"] = (pd.DataFrame(adata_hvg.obsm["X_pca"],
                                                  index=adata_hvg.obs_names)
                                     .reindex(self._adata.obs_names)
                                     .values)
        
        sc.pp.neighbors(self._adata,
                        random_state=42,
                        n_neighbors=20,
                        use_rep="X_pca",
                        n_pcs=50) 
        
        sq.gr.spatial_neighbors_knn(self._adata,
                                    n_neighs = 20)

        sl.spatialleiden(self._adata, 
                         layer_ratio=0.5,
                         n_iterations=-1,
                         random_state=42,
                         latent_neighbors_key="connectivities",
                         spatial_neighbors_key="spatial_connectivities",
                         key_added="spatialleiden")

        self._adata.X = self._adata.layers["counts"].copy()
        sc.pp.normalize_total(self._adata, 
                              target_sum=1e4)
        sc.pp.log1p(self._adata)
        self._adata.layers["log1p"] = self._adata.X.copy()

        sc.tl.rank_genes_groups(self._adata, 
                                groupby='spatialleiden', 
                                method=rank_method,
                                max_iter=max_iter)
        return None

    def map_cell_type(self,
                      cache_dir:str | None=None,
                      query_markers_path:str | None=None,
                      precomputed_stats_path:str | None=None,
                      output_dir:str | None=None,
                      makers_name:Literal["mouse_markers_230821"]="mouse_markers_230821",
                      stats_name:Literal["precomputed_stats_ABC_revision_230821"]="precomputed_stats_ABC_revision_230821") -> None:

        map_cell_type_pango(adata=self._adata,
                            cache_dir=cache_dir,
                            query_markers_path=query_markers_path,
                            precomputed_stats_path=precomputed_stats_path,
                            output_dir=output_dir,
                            makers_name=makers_name,
                            stats_name=stats_name,
                            inplace=True)

        return None
    
    def add_ensembl_ids(self,
                        gtf_path:str|None = None) -> None:
        
        map_df = (convert_gene_symbols_to_ensembl_ids_pango(gene_symbols=self._adata.var_names.tolist(),
                                                            gtf_path=gtf_path)
                  .set_index("gene_symbol"))
        
        self._adata.var = map_df.join(self._adata.var,
                                      how="right")
        
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

def concat_pango(adata_list:list[ad.AnnData],
                 batch_key:list,
                 save_var:list | None=None,
                 normalize:bool=False,
                 highly_variable_genes:bool=False) -> ad.AnnData:

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

    # select the HCGs
    if  highly_variable_genes:
        for ad_obj in adata_list:
            ad_obj.X = ad_obj.layers["log1p"]
            sc.pp.highly_variable_genes(ad_obj,
                                        n_top_genes=6000,
                                        subset=True)
    # concat adata list
    adata_comb = ad.concat(adata_list,
                           join="inner",
                           merge="first",
                           label="batch",
                           keys=batch_key,
                           index_unique="_")

    return adata_comb

def get_the_rank_genes_pango(adata:ad.AnnData,
                             rank_group:str="spatialleiden",
                             pval_cutoff:float=0.05,
                             log2fc_min:float=1,
                             gene_num:int=500):

    # get the total gene set of meaningful rank genes
    all_sig = set()
    for grp in adata.obs[rank_group].unique().astype(str):
        df = (sc.get.rank_genes_groups_df(adata,
                                         group=grp, 
                                         pval_cutoff=pval_cutoff, 
                                         log2fc_min=log2fc_min)
                                         .sort_values('scores', ascending=False))

        df = df[df["scores"]>0]

        if df.shape[0] < gene_num:
            all_sig.update(df['names'])
        else:
            all_sig.update(df.loc[:gene_num,"names"])

    gene_list = list(all_sig)

    return gene_list

