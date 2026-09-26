
from .image_pango.image_pango import Image_Pango
from .anndata_pango import load_10x_h5_pango

from anndata import AnnData
from .anndata_pango import Pango_Accessor

# add class annotation
class AnnData(AnnData):
    pango: Pango_Accessor