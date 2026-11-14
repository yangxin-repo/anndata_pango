
import shutil
import tempfile
from pathlib import Path
from typing import Literal

import anndata as ad

from .anndata_pango import load_10x_h5_pango


def stardist_pipeline_stage1_10x(*,
                                 h5_file_path:str,
                                 tissue_position_path:str,
                                 source_image_path:str,
                                 microns_per_pxl:float,
                                 scalefactor:float,
                                 chip_series:str,
                                 image_save_path:str|None = None,
                                 platform:str="10x",
                                 save_label_image:bool=False) -> str:
    
    # create a temporary directory to store intermediate files
    tmp_dir = tempfile.mkdtemp(prefix="stardist_segmentation_10x_")
    
    # load the 10x h5 file and process it with Stardist
    adata = load_10x_h5_pango(h5_file_path)
    
    # load the source image and tissue position data into the AnnData object
    adata.pango.load_source_image(tissue_position_path=tissue_position_path,
                                  image_path=source_image_path,
                                  microns_per_pxl=microns_per_pxl,
                                  scalefactor=scalefactor,
                                  platform=platform)
    
    # fit the Stardist model to the source image and generate label masks
    adata.pango.get_stardist_labels(platform=platform)

    # save the label image if requested
    if save_label_image:
        
        if  image_save_path is None:
            raise ValueError("There is no image save path:image_save_path")
        else:
            img = adata.uns["source_image"].get_mask_label_plot(mask_labels="stardist_label",
                                                                expanded_mask_labels="stardist_label_expanded")
            img.save(f"{image_save_path}/stardist_image_{chip_series}.png")
    
    # save the AnnData object with the Stardist labels to a temporary file
    del adata.uns["source_image"]
    adata.pango.save(save_path=f"{tmp_dir}/{chip_series}.h5ad")
    
    return f"{tmp_dir}/{chip_series}.h5ad"

def stardist_pipeline_stage1_stereo(*,
                                    h5_file_path:str,
                                    source_image_path:str,
                                    chip_series:str,
                                    platform:str = "stereo",
                                    image_save_path:str|None = None,
                                    save_label_image:bool=False) -> str:
    
    # create a temporary directory to store intermediate files
    tmp_dir = tempfile.mkdtemp(prefix="stardist_segmentation_stereo_")
    
    # read the raw h5ad file
    adata = ad.read_h5ad(h5_file_path)
    
    # load the source image and tissue position data into the AnnData object
    adata.pango.load_source_image(image_path=source_image_path,
                                  microns_per_pxl=0.5,
                                  platform=platform)
    
    # fit the Stardist model to the source image and generate label masks
    adata.pango.get_stardist_labels(platform=platform)
    
    # save the label image if requested
    if save_label_image:
            
        if  image_save_path is None:
            raise ValueError("There is no image save path:image_save_path")
        else:
            img = adata.uns["source_image"].get_mask_label_plot(mask_labels="stardist_label",
                                                                expanded_mask_labels="stardist_label_expanded")
            img.save(f"{image_save_path}/stardist_image_{chip_series}.png")

    # save the AnnData object with the Stardist labels to a temporary file
    del adata.uns["source_image"]
    adata.pango.save(save_path=f"{tmp_dir}/{chip_series}.h5ad")
    
    return f"{tmp_dir}/{chip_series}.h5ad"

def stardist_pipeline_stage2(*,
                             adata_path:str,
                             chip_series:str,
                             gtf_path:str,
                             output_dir:str,
                             platform:Literal["10x","stereo"],
                             map_cell_type_cache_dir:str|None=None,
                             map_cell_types:bool=True) -> str:
    
    # read the segemented adata
    adata = ad.read_h5ad(adata_path)
    
    # aggregate labels
    adata = adata.pango.aggregate_labels(label="stardist_label_expanded",
                                         except_labels=[0],
                                         platform=platform,
                                         chip_series=chip_series)

    # transfer gene symbol to ensembl ids
    adata.pango.add_ensembl_ids(gtf_path=gtf_path)
    adata.var["gene_symbol"] = adata.var_names.to_numpy()

    # confirm the var_names are unique and set them to ensembl ids
    adata.var_names = adata.var["ensembl_id"].to_numpy().astype(str)
    adata.var_names_make_unique()
    
    # allocate cell types
    if map_cell_types:
        if map_cell_type_cache_dir is None:
            raise ValueError("map_cell_type_cache_dir is None, please provide a valid cache directory for cell type mapping.")
        else:
            adata.pango.map_cell_type(cache_dir=map_cell_type_cache_dir)
    
    # perform spatial cluster
    adata.pango.conduct_spatial_cluster(rank_method="logreg", 
                                        max_iter=1000,
                                        layer_ratio=0.5)
    
    # save adata
    adata_save_path = f"{output_dir}/{chip_series}.h5ad"
    adata.pango.save(save_path=adata_save_path)
    
    # clean up the temporary directory
    shutil.rmtree(Path(adata_path).parent)  
    
    return adata_save_path