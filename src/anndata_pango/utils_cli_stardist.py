
import tempfile

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
                                    save_label_image:bool=False):
    
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