
import numpy as np
import pandas as pd
import stardist.models
import matplotlib.pyplot as plt
import csbdeep.utils 
from scipy.sparse import csr_matrix
from PIL import Image
from skimage import segmentation
from scipy import ndimage

from . import coordinate_system_pango as csp

from .utils_coordinate_system_pango import (estimate_affine_matrix_from_coordinate_mapping,
                                            build_affine_matrix,
                                            apply_affine_matrix_to_coordinate)
Image.MAX_IMAGE_PIXELS = None

class Image_Pango:
    def __init__(self,
                 image_path:str,
                 microns_per_pxl:float=1):
        # load the image
        self.source_image = Image.open(image_path)

        # build the coordinate
        self.coordinate = csp.Coordinate_Pango(source_microns_per_pixel=microns_per_pxl,
                                                                       source_pixel_coordinate=self.source_image.size)

        # build the slot for pixel matrix
        self.pixel_matrix = None

        # build the slot for the mask matrix
        self.mask_matrix = dict()

    def save_source_image(self,
                          path:str):

        self.source_image.save(path)

        return None

    def load_pixel_matrix(self,
                          normalize=True):

        # load the pixel matrix as array
        pxl_mat = np.asarray(self.source_image)

        # normalize the pixel matrix
        if normalize:
            pxl_mat = csbdeep.utils.normalize(pxl_mat, 1, 99.8, axis=(0, 1))

        self.pixel_matrix = pxl_mat

        return None

    def fit_stardist_models(self,
                            model_path:str,
                            model_name:str = "2D_versatile_he"):

        # load the model
        model = stardist.models.StarDist2D(None,
                                           name=model_name,
                                           basedir=model_path)

        # fit the model
        labs, details = model.predict_instances(self.pixel_matrix,
                                                n_tiles=(8, 8, 1))
        
        self.mask_matrix["stardist_label"] = csr_matrix(labs)

        return None

    def add_coordinate_mapping(self,
                               coordinate_mapping:pd.DataFrame,
                               mapping_key:str=None,
                               scalefactor:float=None):

        # verify the mapping_symbol
        if mapping_key is None:
            mapping_key = "coordinate_mapping"

        # write the coordinate mapping
        self.coordinate.coordinate_mapping[mapping_key] = coordinate_mapping.copy()

        # estimate the affine matrix
        self.coordinate.affine_matrix[mapping_key] = estimate_affine_matrix_from_coordinate_mapping(coordinate_mapping=self.coordinate.coordinate_mapping[mapping_key])

        # transform the affine matrix with scalefactor
        if scalefactor is not None:
            self.coordinate.affine_matrix[mapping_key] = ((np.vstack([build_affine_matrix(am_x=scalefactor,
                                                                                          am_y=scalefactor),
                                                                      [0,0,1]])) @
                                                          (np.vstack([self.coordinate.affine_matrix[mapping_key],
                                                                      [0,0,1]])))[:2,:]

            # calculate the coordinate mapping with scalefactor
            coordinate_new = apply_affine_matrix_to_coordinate(coordinate=self.coordinate.coordinate_mapping[mapping_key],
                                                               affine_matrix=self.coordinate.affine_matrix[mapping_key])
            self.coordinate.coordinate_mapping[mapping_key]["pxl_row"] = coordinate_new[:,0]
            self.coordinate.coordinate_mapping[mapping_key]["pxl_col"] = coordinate_new[:,1]

        return None

    def extract_mapping_area(self,
                             mapping_key:str=None):

        # verify mapping_key
        if mapping_key is None:
            mapping_key = "coordinate_mapping"

        # get the mapping area
        xmin_crop = int(np.floor(np.min(self.coordinate.coordinate_mapping[mapping_key]["pxl_col"])))
        ymin_crop = int(np.floor(np.min(self.coordinate.coordinate_mapping[mapping_key]["pxl_row"])))
        xmax_crop = int(np.ceil(np.max(self.coordinate.coordinate_mapping[mapping_key]["pxl_col"])))
        ymax_crop = int(np.ceil(np.max(self.coordinate.coordinate_mapping[mapping_key]["pxl_row"])))

        # verify the crop boundary
        if  xmin_crop < 0:
            xmin_crop = 0
        if  ymin_crop < 0:
            ymin_crop = 0
        if  xmax_crop > self.coordinate.source_coordinate["pixel_coordinate"][0]:
            xmax_crop = self.coordinate.source_coordinate["pixel_coordinate"][0]
        if  ymax_crop > self.coordinate.source_coordinate["pixel_coordinate"][1]:
            ymax_crop = self.coordinate.source_coordinate["pixel_coordinate"][1]

        self.source_image = self.source_image.crop((xmin_crop, ymin_crop, xmax_crop, ymax_crop))
        self.coordinate.crop_source_image((xmin_crop, ymin_crop, xmax_crop, ymax_crop))

        return None

    def extract_image_area(self,
                           mapping_key:str=None):

        # verify mapping_key
        if mapping_key is None:
            mapping_key = "coordinate_mapping"

        # extract the mapping coordinate
        mapping_coordinate = self.coordinate.coordinate_mapping[mapping_key].copy()

        # generate the filter mask
        mask = (((mapping_coordinate["pxl_row"] >= 0) &
                 (mapping_coordinate["pxl_row"] < np.ceil(self.coordinate.source_coordinate["pixel_coordinate"][1])) &
                 (mapping_coordinate["pxl_col"] >= 0) &
                 (mapping_coordinate["pxl_col"] < np.ceil(self.coordinate.source_coordinate["pixel_coordinate"][0]))))

        # apply filter mask

        mapping_coordinate = mapping_coordinate.loc[mask,:]

        return mapping_coordinate

    def crop_source_image(self,
                          xmin_crop:int,
                          ymin_crop:int,
                          xmax_crop:int,
                          ymax_crop:int):

        # verify the crop boundary
        if  xmin_crop < 0:
            xmin_crop = 0
        if  ymin_crop < 0:
            ymin_crop = 0
        if  xmax_crop > self.coordinate.source_coordinate["pixel_coordinate"][0]:
            xmax_crop = self.coordinate.source_coordinate["pixel_coordinate"][0]
        if  ymax_crop > self.coordinate.source_coordinate["pixel_coordinate"][1]:
            ymax_crop = self.coordinate.source_coordinate["pixel_coordinate"][1]

        # crop the source image
        self.source_image = self.source_image.crop((xmin_crop, ymin_crop, xmax_crop, ymax_crop))

        # crop the coordinate
        self.coordinate.crop_source_image((xmin_crop, ymin_crop, xmax_crop, ymax_crop))

        return None

    def resize_source_image(self,
                            scalefactor:float=None,
                            microns_per_pixel:float=None):

        # verify scalefactor and microns_per_pixel
        if (microns_per_pixel is None) == (scalefactor is None):
            raise ValueError("must pass microns_per_pixel or scalefactor")
        elif scalefactor is None:
            scalefactor = self.coordinate.source_coordinate["microns_per_pixel"]/microns_per_pixel

        # calculate the new affine matrix
        self.coordinate.resize_source_image(scalefactor)

        # resize the source image
        self.source_image = self.source_image.resize(size=self.coordinate.source_coordinate["pixel_coordinate"], 
                                                     resample=Image.Resampling.LANCZOS)

        return None

    def expand_mask_labels(self,
                           mask_label):

        self.mask_matrix[f"{mask_label}_expanded"] = csr_matrix(segmentation.expand_labels(label_image=(self.mask_matrix[mask_label]
                                                                                                        .toarray()),
                                                                                           distance=3))

        return None

    def resample_mask_labels(self,
                             mask_label:str,
                             mapping_key:str):

        # calculate the resample labels using nearest
        self.coordinate.coordinate_mapping[mapping_key][mask_label] = ndimage.map_coordinates(input = self.mask_matrix[mask_label].toarray(),
                                                                                              coordinates = (self.coordinate.coordinate_mapping[mapping_key][["pxl_row","pxl_col"]]
                                                                                                             .to_numpy()
                                                                                                             .T),
                                                                                              order=0,
                                                                                              cval=0)

        return None

    def resample_source_image(self,
                              mapping_key:str):

        # build the inverse affine matrix
        affine_matrix = np.vstack([self.coordinate.affine_matrix[mapping_key],
                                  [0, 0, 1]])

        # get the pixel matrix without normalization
        pixel_mat = np.asarray(self.source_image)

        # resample and stack the pixel matrix
        warped = np.stack(
            [ndimage.affine_transform(
                input = pixel_mat[:,:,i],
                matrix=affine_matrix,
                output_shape=((np.max(self.coordinate.coordinate_mapping[mapping_key]["array_row"]
                                      .to_numpy())+1),
                              (np.max(self.coordinate.coordinate_mapping[mapping_key]["array_col"]
                                      .to_numpy())+1)),
                order=1,
                cval=255) for i in range(3)],
            axis=-1)

        return warped

    def rebuild_label_matrix(self,
                             mapping_key:str,
                             label:str):

        label_matrix = (self.coordinate.coordinate_mapping[mapping_key]
                        .pivot(index="array_row",
                               columns="array_col",
                               values=label)
                        .reindex(index = range((np.max(self.coordinate.coordinate_mapping[mapping_key]["array_row"]
                                                      .to_numpy())+1)),
                                 columns = range((np.max(self.coordinate.coordinate_mapping[mapping_key]["array_col"]
                                                        .to_numpy())+1)))
                        .to_numpy())

        label_matrix = np.nan_to_num(label_matrix, nan=0).astype(int)

        return label_matrix

    def add_labels_to_coordinate_mapping(self,
                                         mapping_key:str,
                                         label_df:pd.DataFrame,
                                         by:str):

        self.coordinate.coordinate_mapping[mapping_key] = self.coordinate.coordinate_mapping[mapping_key].merge(label_df,
                                                                                                                on=by,
                                                                                                                how="left")

        return None


    def show_labels(self,
                    mapping_key:str,
                    mask_label:str):

        # shut off show automatically
        plt.ioff()

        # build the plot panel
        fig, ax = plt.subplots()
        ax.axis("off")

        # rebuild the label matrix
        label_mat = self.rebuild_label_matrix(mapping_key=mapping_key, label=mask_label)

        # show image
        ax.imshow(self.resample_source_image(mapping_key=mapping_key), interpolation="nearest")
        ax.imshow(np.ma.masked_where(label_mat == 0, label_mat), interpolation="nearest", cmap="tab20", alpha=0.5)

        plt.show()

        return None

    def get_mapping_coordinate(self,
                               mapping_key:str):

        return self.coordinate.coordinate_mapping[mapping_key].copy()

    def write_h5(self):

        return None


