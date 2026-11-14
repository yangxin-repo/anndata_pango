
from anndata import AnnData

from .anndata_pango import Pango_Accessor, concat_pango, load_10x_h5_pango
from .cli_stardist import conduct_stardist_pipeline
from .image_pango.image_pango import Image_Pango

# add class annotation
class AnnData(AnnData):
    pango: Pango_Accessor

__all__ = ["Image_Pango", "Pango_Accessor", "concat_pango", "conduct_stardist_pipeline", "load_10x_h5_pango"]