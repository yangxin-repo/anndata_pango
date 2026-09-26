
from typing import TYPE_CHECKING

from .image_pango.image_pango import Image_Pango as Image_Pango
from .anndata_pango import load_10x_h5_pango as load_10x_h5_pango

from anndata import AnnData
from .anndata_pango import Pango_Accessor as Pango_Accessor

# add class annotation
if TYPE_CHECKING:
    class _AnnDataWithPango(AnnData):
        pango: Pango_Accessor

__all__ = ["Pango_Accessor","Image_Pango","load_10x_h5_pango"]