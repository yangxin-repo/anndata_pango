
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Literal

import scanpy as sc

from .anndata_pango import load_10x_h5_pango


def _worker_entry_cluster(kwargs:dict,
                          conn:mp.connection.Connection) -> None:
    
    # import the conduct function
    from .utils_cli_stardist import stardist_pipeline_stage2
    
    # run and return the results
    try:
        path = stardist_pipeline_stage2(**kwargs)
        conn.send(("ok", path))
    except Exception as exc:            # noqa: BLE001 — report, don't die silently
        conn.send(("error", repr(exc)))
    finally:
        conn.close()
        
    return None

def _worker_entry_10x(kwargs:dict,
                      conn:mp.connection.Connection) -> None:
    """
    Worker entry point for conducting the Stardist pipeline.

    This function is designed to be called by a worker process. It unpacks the provided 
    keyword arguments and invokes the main Stardist pipeline function.

    Returns:
        None
    """
    
    # import the conduct function
    from .utils_cli_stardist import stardist_pipeline_stage1_10x
    
    # run and return the results
    try:
        path = stardist_pipeline_stage1_10x(**kwargs)
        conn.send(("ok", path))
    except Exception as exc:            # noqa: BLE001 — report, don't die silently
        conn.send(("error", repr(exc)))
    finally:
        conn.close()
        
    return None

def _worker_entry_stereo(kwargs:dict,
                         conn:mp.connection.Connection) -> None:
    """
    Worker entry point for conducting the Stardist pipeline.

    This function is designed to be called by a worker process. It unpacks the provided 
    keyword arguments and invokes the main Stardist pipeline function.

    Returns:
        None
    """
    
    # import the conduct function
    from .utils_cli_stardist import stardist_pipeline_stage1_stereo
    
    # run and return the results
    try:
        path = stardist_pipeline_stage1_stereo(**kwargs)
        conn.send(("ok", path))
    except Exception as exc:            # noqa: BLE001 — report, don't die silently
        conn.send(("error", repr(exc)))
    finally:
        conn.close()
        
    return None

def run_stardist_pipeline_cell_segmentation(*,
                                            h5_file_path:str,
                                            source_image_path:str,
                                            chip_series:str,
                                            tissue_position_path:str|None = None,
                                            scalefactor:float|None = None,
                                            microns_per_pxl:float|None = None,
                                            image_save_path:str|None = None,
                                            platform:Literal["10x","stereo"]="10x",
                                            save_label_image:bool=False) -> str:
    
    # create the subprocess context
    ctx = mp.get_context("spawn") 
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    
    if platform == "10x":
        # pack the params
        kwargs = {"h5_file_path":h5_file_path,
                  "tissue_position_path":tissue_position_path,
                  "source_image_path":source_image_path,
                  "microns_per_pxl":microns_per_pxl,
                  "scalefactor":scalefactor,
                  "chip_series":chip_series,
                  "image_save_path":image_save_path,
                  "platform":platform,
                  "save_label_image":save_label_image}
    
        # create a subprocess
        p = ctx.Process(target=_worker_entry_10x, 
                        args=(kwargs, child_conn), 
                        daemon=False)
    
    if platform == "stereo":
        # pack the params
        kwargs = {"h5_file_path":h5_file_path,
                  "source_image_path":source_image_path,
                  "chip_series":chip_series,
                  "image_save_path":image_save_path,
                  "platform":platform,
                  "save_label_image":save_label_image}
    
        # create a subprocess
        p = ctx.Process(target=_worker_entry_stereo, 
                        args=(kwargs, child_conn), 
                        daemon=False)
        
    # start the subprocess
    p.start()
    child_conn.close() 
    
    try:
        status, payload = parent_conn.recv()   # blocks until worker reports
    except EOFError:                        # child died before sending anything
        status, payload = "error", f"worker exited abruptly (exitcode={p.exitcode})"
    finally:
        p.join()
        parent_conn.close()

    if p.exitcode != 0 or status != "ok":
        raise RuntimeError(
            f"Stardist stage-1 worker failed (exit {p.exitcode}):\n{payload}"
        )
    if not Path(payload).exists():
        raise RuntimeError(f"worker reported {payload!r} but file does not exist")
    
    return payload

def run_stardist_pipeline_cell_cluser_and_annotation(*,
                                                     adata_path:str,
                                                     chip_series:str,
                                                     gtf_path:str,
                                                     map_cell_type_cache_dir:str,
                                                     output_dir:str,
                                                     platform:Literal["10x","stereo"],
                                                     map_cell_types:bool) -> str:
    
    # create the subprocess context
    ctx = mp.get_context("spawn") 
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    
    # pack the params
    kwargs = {"adata_path":adata_path,
              "output_dir":output_dir,
              "chip_series":chip_series,
              "gtf_path":gtf_path,
              "platform":platform,
              "map_cell_type_cache_dir":map_cell_type_cache_dir,
              "map_cell_types":map_cell_types}
    
    # create a subprocess
    p = ctx.Process(target=_worker_entry_cluster, 
                    args=(kwargs, child_conn), 
                    daemon=False)
    
    # start the subprocess
    p.start()
    child_conn.close() 
    
    try:
        status, payload = parent_conn.recv()   # blocks until worker reports
    except EOFError:                        # child died before sending anything
        status, payload = "error", f"worker exited abruptly (exitcode={p.exitcode})"
    finally:
        p.join()
        parent_conn.close()

    if p.exitcode != 0 or status != "ok":
        raise RuntimeError(
            f"Stardist stage-2 worker failed (exit {p.exitcode}):\n{payload}"
        )
    if not Path(payload).exists():
        raise RuntimeError(f"worker reported {payload!r} but file does not exist")
    
    return payload

def conduct_stardist_pipeline(*,
                              h5_file_path:str,
                              source_image_path:str,
                              chip_series:str,
                              gtf_path:str,
                              output_dir:str,
                              platform:Literal["10x","stereo"],
                              scalefactor:float|None = None,
                              microns_per_pxl:float|None = None,
                              tissue_position_path:str|None = None,
                              save_label_image:bool=False,
                              image_save_path:str|None = None,
                              map_cell_type_cache_dir:str|None = None,
                              map_cell_types:bool=True):  # noqa: F821
    """
    Conduct the Stardist pipeline for cell segmentation.

    This function orchestrates the entire Stardist pipeline, including image preprocessing, 
    model inference, and post-processing of segmentation results. It handles input and output 
    paths, manages temporary files, and ensures that the pipeline runs smoothly.

    Returns:
        adata path: str
    """
    # Implementation of the Stardist pipeline goes here.
    
    if platform == "10x":
        
        # verfiy the params
        if (scalefactor is None) | (microns_per_pxl is None):
            raise ValueError("must input scalefactor and microns_per_pxl")
        if (tissue_position_path is None):
            raise ValueError("must input tissue_position_path")
        
        # conduct the stardist cell segmentation in a subprocess    
        segmented_adata_path = run_stardist_pipeline_cell_segmentation(h5_file_path=h5_file_path,
                                                                       tissue_position_path=tissue_position_path,
                                                                       source_image_path=source_image_path,
                                                                       microns_per_pxl=microns_per_pxl,
                                                                       scalefactor=scalefactor,
                                                                       image_save_path=image_save_path,
                                                                       chip_series=chip_series,
                                                                       save_label_image=save_label_image,
                                                                       platform=platform)
        
        adata_path = run_stardist_pipeline_cell_cluser_and_annotation(adata_path=segmented_adata_path,
                                                                      chip_series=chip_series,
                                                                      gtf_path=gtf_path,
                                                                      map_cell_type_cache_dir=map_cell_type_cache_dir,
                                                                      output_dir=output_dir,
                                                                      platform=platform,
                                                                      map_cell_types=map_cell_types)

        return adata_path
    
    if platform == "stereo":
    
        # conduct the stardist cell segmentation in a subprocess    
        segmented_adata_path = run_stardist_pipeline_cell_segmentation(h5_file_path=h5_file_path,
                                                             source_image_path=source_image_path,
                                                             image_save_path=image_save_path,
                                                             chip_series=chip_series,
                                                             save_label_image=save_label_image,
                                                             platform=platform)
    
        adata_path = run_stardist_pipeline_cell_cluser_and_annotation(adata_path=segmented_adata_path,
                                                                      chip_series=chip_series,
                                                                      gtf_path=gtf_path,
                                                                      map_cell_type_cache_dir=map_cell_type_cache_dir,
                                                                      output_dir=output_dir,
                                                                      platform=platform,
                                                                      map_cell_types=map_cell_types)

        return adata_path