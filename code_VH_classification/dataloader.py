
import os
import re
import pandas as pd
from nilearn import image
from preprocessing import mask_image

def get_data(input_dir: str, meta_data: pd.DataFrame, label_col_name: str, activity_map_type):

    ''' Extract data from input directory. Extract labels, activity map type, image, and subject IDs.''' 

    if '.DS_Store' in os.listdir(input_dir):
        # only applicable for MacOS users. Deletes hidden files tso that os.listdir works properly
        os.remove(input_dir + ".DS_Store") 
    
    participants_paths = os.listdir(input_dir)

    # use list comprehension to filter the filenames, should contain "sub-" according to BIDS format, the correct activity map, and should match one participant in the meta data file
    filtered_filenames = [filename for filename in participants_paths \
                        if "sub-" in filename and \
                        any(act_map in filename for act_map in activity_map_type) and \
                        any(participant in filename for participant in meta_data.index)]

    sub_ids = []
    activity_map_list = []
    img_list = []
    for filename in filtered_filenames:
        sub_id = filename.split("_")[0]
        sub_ids.append(sub_id)

        act_map = filename.split("_")[-1]
        act_map = act_map.split(".")[0]
        activity_map_list.append(act_map)
        img_path = os.path.join(input_dir, filename)
        img = image.load_img(img_path)
        img_list.append(img)

    label_list = [meta_data[label_col_name][sub_id] for sub_id in sub_ids]

    img_dict = pd.DataFrame(columns=('sub_ids', 'labels', 'activity_map_type', "feature_vector_size", 'images'))
    
    img_dict.labels = label_list
    img_dict.activity_map_type = activity_map_list
    img_dict.images = img_list
    img_dict.sub_ids = sub_ids

    # no feature selection applied yet so "all" is inserted as feature vector size
    img_dict.feature_vector_size = "all"
    
    return img_dict


