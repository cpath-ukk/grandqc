# MAIN LOOP TO PROCESS WSI
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm
import cv2
import json

import rasterio.features
import shapely
import geopandas as gpd
from collections import defaultdict

#Helper functions
def to_tensor_x(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(image, preprocessing_fn, model_size):
    if image.size != model_size:
        image = image.resize(model_size)
        print('resized')
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x


def make_1class_map_thr(mask, class_colors):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(1, len(class_colors)+1):
        idx = mask == l
        r[idx] = class_colors [l-1][0]
        g[idx] = class_colors [l-1][1]
        b[idx] = class_colors [l-1][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def slide_process_single(model, tis_det_map_mpp, slide, patch_n_w_l0, patch_n_h_l0, p_s, m_p_s, colors,
                         ENCODER_MODEL_1,ENCODER_WEIGHTS, DEVICE, BACK_CLASS, MPP_MODEL_1, mpp, w_l0, h_l0):
    '''
    Tissue detection map is generated under MPP = 4, therefore model patch size of (512,512) corresponds to tis_det_map patch
    size of (128,128).
    '''

    model_size = (m_p_s, m_p_s)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_MODEL_1, ENCODER_WEIGHTS)

    # Start loop
    for he in tqdm(range(patch_n_h_l0), total=patch_n_h_l0):
        h = he * p_s + 1
        if (he == 0):
            h = 0
        # print("Current cycle ", he + 1, " of ", patch_n_h_l0)
        for wi in range(patch_n_w_l0):
            w = wi * p_s + 1
            if (wi == 0):
                w = 0
            #he = 12
            #wi = 15
            td_patch = tis_det_map_mpp [he*m_p_s:(he+1)*m_p_s,wi*m_p_s:(wi+1)*m_p_s]
            if td_patch.shape != (512,512):
                # td_patch padding (incase td_patch does not equal (512,512))
                original_shape = td_patch.shape

                # Desired shape
                desired_shape = (512, 512)

                # Calculate padding needed
                padding = [(0, desired_shape[i] - original_shape[i]) for i in range(2)]

                # Apply padding
                td_patch_ = np.pad(td_patch, padding, mode='constant')
            else:
                td_patch_ = td_patch

            if np.count_nonzero(td_patch == 0) > 50: #here change to check of segmentation map
                # Generate patch
                work_patch = slide.read_region((w, h), 0, (p_s, p_s))
                work_patch = work_patch.convert('RGB')

                # Resize to model patch size
                work_patch = work_patch.resize((m_p_s, m_p_s), Image.Resampling.LANCZOS)

                image_pre = get_preprocessing(work_patch, preprocessing_fn, model_size)
                x_tensor = torch.from_numpy(image_pre).to(DEVICE).unsqueeze(0)
                predictions = model.predict(x_tensor)
                predictions = (predictions.squeeze().cpu().numpy())

                mask_raw = np.argmax(predictions, axis=0).astype('int8')
                mask = np.where(td_patch_ == 1, BACK_CLASS, mask_raw)


            else:
                mask = np.full((512,512), BACK_CLASS)



            if (wi == 0):
                temp_image = mask

            else:
                temp_image = np.concatenate((temp_image, mask), axis=1)

        if (he == 0):
            end_image = temp_image

        else:
            end_image = np.concatenate((end_image, temp_image), axis=0)

    # now get size of padded region (buffer) at Model MPP
    buffer_right_l = int((w_l0 - (patch_n_w_l0 * p_s)) * mpp / MPP_MODEL_1)
    buffer_bottom_l = int((h_l0 - (patch_n_h_l0 * p_s)) * mpp / MPP_MODEL_1)
    # firstly bottom
    buffer_bottom = np.full((buffer_bottom_l, end_image.shape[1]), 0)
    temp_image = np.concatenate((end_image, buffer_bottom), axis=0)
    # now right side
    temp_image_he, temp_image_wi = temp_image.shape  # width and height
    buffer_right = np.full((temp_image_he, buffer_right_l), 0)
    end_image = np.concatenate((temp_image, buffer_right), axis=1).astype(np.uint8)

    end_image_1class = make_1class_map_thr(end_image, colors)
    end_image_1class = Image.fromarray(end_image_1class)
    end_image_1class = end_image_1class.resize((patch_n_w_l0*50, patch_n_h_l0*50), Image.Resampling.LANCZOS)


    return end_image_1class, end_image


def diff_polys(positive_polygon, intersecting_neg_polys):
    """Subtract negative polygons from positive polygon."""
    if intersecting_neg_polys:                    
        # Unify negative polygons first for better performance
        big_neg_poly = shapely.ops.unary_union(intersecting_neg_polys)    
        positive_polygon = positive_polygon.difference(big_neg_poly)
    return positive_polygon

def mask_to_polygons(mask):
    """
    Convert binary mask to polygons
    
    Args:
        mask: Binary numpy array with foreground as True
        
    Returns:
        List of Shapely polygons
    """
    
    # Extract shapes
    shapes_gen = rasterio.features.shapes(np.int16(mask))
    
    # Separate positive and negative polygons
    positive_polys = []
    negative_polys = []    
    for shape_data, value in shapes_gen:
        poly = shapely.geometry.shape(shape_data)
        if value:  # Foreground
            positive_polys.append(poly)
        elif not value:  # Background holes
            negative_polys.append(poly)
    
    # Early exit if no positive polygons
    if not positive_polys:
        return []
    
    # Hole-polygon mapping using spatial indexing
    if negative_polys:
        # Group negative polygons by which positive polygons they intersect
        poly_holes_map = defaultdict(list)
        
        for neg_poly in negative_polys:
            for i, pos_poly in enumerate(positive_polys):
                if pos_poly.intersects(neg_poly):
                    poly_holes_map[i].append(neg_poly)
        
        # Apply differences
        result_polys = []
        for i, pos_poly in enumerate(positive_polys):
            if i in poly_holes_map:
                result_poly = diff_polys(pos_poly, poly_holes_map[i])
            else:
                result_poly = pos_poly
            result_polys.append(result_poly)
    else:
        result_polys = positive_polys
    
    return result_polys


def mask_to_geojson(mask_path, output_path, scale_factor=1.0):
    """
    Convert a semantic segmentation mask to GeoJSON with coordinate scaling

    Parameters:
    -----------
    mask_path : str
        Path to the input PNG mask file
    output_path : str
        Path to save the output GeoJSON file
    scale_factor : float, optional, should be: model_mpp / slide_mpp
        Factor to scale coordinates by (default: 1.0)

    Returns:
    --------
    None
    """
    # Define class mapping
    CLASS_MAPPING = {
        1: "Normal Tissue",
        2: "Fold",
        3: "Darkspot & Foreign Object",
        4: "PenMarking",
        5: "Edge & Air Bubble",
        6: "OOF",  # Out of Focus
        7: "Background"
    }

    # Read the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    # Dictionary to store features for each class
    features = []

    # Iterate through unique class values (1 to 7)
    for class_value in range(2, 7):
        # Create a binary mask for the current class and extract polygons
        class_mask = mask == class_value
        polygons = mask_to_polygons(class_mask)                
        features.extend([{'class_id': class_value,'classification':CLASS_MAPPING.get(class_value, "Unknown"),'area':poly.area*(scale_factor ** 2),'geometry': poly} for poly in polygons])        
    if len(features)>0:
        gdf = gpd.GeoDataFrame(features).set_geometry('geometry')        
        gdf.geometry = gdf.scale(xfact=scale_factor, yfact=scale_factor, origin=(0,0))
        geojson = json.loads(gdf.to_json()) # this is a bit silly, but not sure how else to add metadata
        geojson['metadata'] = dict(class_mapping=CLASS_MAPPING, scale_factor=scale_factor)         

    else:
        geojson = {
            "type": "FeatureCollection",
            "features": [],
            "metadata": {"class_mapping": CLASS_MAPPING, "scale_factor": scale_factor}
        }    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)