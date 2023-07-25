import pandas as pd
import numpy as np
import nibabel as nib
import time


def pearson_correlation(labels, activity_maps_flat):

    ''' 
    Calculate voxel-wise Pearson correlation for the activity maps. 
    '''
    
    # labels from meta_data
    labels = pd.DataFrame(labels, columns=['labels'])
    labels['labels'] = labels['labels'].astype('category')
    labels['labels'] = labels['labels'].cat.codes

    '''
    Convert activity map to a dataframe

    Participant IDs are the rows and the columns are all voxels (with coordinates [i,j,k], flattened array). 
    The values are the intensity value at a specific voxel location for one subject. 

    It looks like this (example values):

                   0       1       2       3  ... 
    sub-1234     0.2     0.0     0.0     0.0  ...  
    sub-2345     0.1     0.0     0.2     0.0  ...   
    sub-3456     0.0     0.5     0.0     0.0  ...  
    ...          ...     ...     ...     ...


    '''
    
    activity_df = pd.DataFrame.from_dict(activity_maps_flat, orient='index') 
    
    corr_arr = []

    # for each voxel
    for i in range(activity_df.shape[1]):

        # calculate Pearson correlation between intensity values and corresponding subject phenotype labels (f.e. HC)
        corr = activity_df[i].corr(labels['labels'], method='pearson')
        corr_arr.append(corr)

    corr_arr = np.array(corr_arr)

    print("\n Number of finite correlation values inside pearson", len(corr_arr[np.isfinite(corr_arr) == True]))

    return corr_arr   

def run_feature_saliency_pearson(tr_data, tst_data, act_map):

    # select images for this activity map    
    tr_images = tr_data.loc[tr_data['activity_map_type'] == act_map]

    # get training flat feature vectors
    tr_maps_flat = {}
    tr_labels = {}
    for _, row in tr_images.iterrows():
        # read image
        activity_map_flat = row['images']
        sub = row['sub_ids']
        tr_maps_flat[sub] = activity_map_flat
        tr_labels[sub] = row['labels']
    tr_labels = pd.Series(tr_labels)

    # get testing flat feature vectors
    tst_images = tst_data.loc[tst_data['activity_map_type'] == act_map]
    tst_maps_flat = {}
    tst_labels = {}
    for _, row in tst_images.iterrows():
        # read image
        activity_map_flat = row['images']
        sub = row['sub_ids']
        tst_maps_flat[sub] = activity_map_flat
        tst_labels[sub] = row['labels']
    tst_labels = pd.Series(tst_labels)

    # determine voxel saliency and save in correlation matrix (using only training samples)
    strt_pears = time.time()
    tr_corr_mat = pearson_correlation(tr_labels, tr_maps_flat)
    print("pearson took %s seconds :"%(time.time() - strt_pears))

    print('tr maps flat', tr_maps_flat)

    return tr_corr_mat, tr_maps_flat, tst_maps_flat, tr_labels, tst_labels


def get_salient_voxel_locations(corr_matrix: np.array, feature_vector_size: int):

    ''' 
    Determine n salient voxel locations with n being the feature vector size specified by the user. 
    When the Pearson correlation of a voxel is in the top n correlation values, the mask value will be set to 1. 
    All other values are set to 0.  
    '''

    # flatten matrix and set NaN values to 0
    corr_matrix_flat = np.nan_to_num(corr_matrix)

    if len(abs(np.unique(corr_matrix_flat))) < feature_vector_size:

        # partial sort of the matrix to find the n highest value and set to threshold

        # get unique absolute values in the flattened matrix
        unique_abs_values = abs(np.unique(corr_matrix_flat))
        # get number of unique absolute values                       
        length_abs_values = len(unique_abs_values)        
        # sort the array until nth highest value is found                          
        partial_sorted_arr = np.partition(unique_abs_values, -length_abs_values)   
        # get the nth highest value 
        threshold_val = partial_sorted_arr[-length_abs_values]                     

        print("Attention: The salient voxel number %f is smaller than the specified feature vector size %f", unique_abs_values, feature_vector_size)
    
    else: 
        # partial sort of the matrix to find the n highest value and set to threshold

        # get unique absolute values in the flattened matrix
        unique_abs_values = abs(np.unique(corr_matrix_flat))                        
        # sort the array until nth highest value is found
        partial_sorted_arr = np.partition(unique_abs_values, -feature_vector_size)  
        # get the nth highest value, n = feature_vector_size
        threshold_val = partial_sorted_arr[-feature_vector_size]                    
    

    # set all (absolute) values below the threshold to 0 and all others to the original value (corr_matrix_flat value)
    corr_matr_thres_flat = np.where((list(map(abs, corr_matrix_flat)) <= threshold_val), 0, 1.0) 

    # print shape
    print("\n \n \n First 50 values in thresholded correlation matrix (only salient voxels)", corr_matr_thres_flat[corr_matr_thres_flat != 0][:50])
    
    return corr_matr_thres_flat


def get_salient_map(activity_maps_flat, corr_matr_thres_flat: np.array): 

    ''' 
    Mask images with matrix obtained in get_salient_voxel_locations (1s for salient voxels and 0s for all others). 
    Upon multiplication with the activity map (intensity values), only the mask values will be kept. 
    All intensity values that are multiplied with 1 stay in the matrix while all others are multiplied with 0 and therefore discarded.
    Outputs the masked matrix.
    '''

    # initializing
    salient_map_list = {}
    corr_matr_thres_flat = np.array(corr_matr_thres_flat)

    for participant in activity_maps_flat.keys():

        if np.isnan(activity_maps_flat[participant]).any():
            print("There are NaN values in the data!")


        fl_arr = np.array(activity_maps_flat[participant])
        indices_to_keep = np.where(corr_matr_thres_flat != 0.0)[0]
        print("First 15 indices to keep: ", indices_to_keep[:15])

        sel_vox_activity_map = np.delete(fl_arr, np.where(corr_matr_thres_flat == 0.0))
        print("First 15 values in selected voxel map: ",sel_vox_activity_map[:15])

        # append to dictionary
        salient_map_list[participant] = sel_vox_activity_map

    print("\n\n salient image list", salient_map_list.keys())
    return salient_map_list, indices_to_keep
    


def get_salient_nii_image(activity_maps_flat, affine_map, corr_matr_thres: np.array, init_shape): 

    ''' 
    Mask images with matrix obtained in get_salient_voxel_locations (1s for salient voxels and 0s for all others). 
    Upon multiplication with the activity map (intensity values), only the mask values will be kept. 
    All intensity values that are multiplied with 1 stay in the matrix while all others are multiplied with 0 and therefore discarded.
    Outputs the masked nifti image.
    '''

    # initializing
    salient_image_list = {}
    map_affine = affine_map

    for participant in activity_maps_flat.keys():

        if np.isnan(activity_maps_flat[participant]).any():
            print("There are NaN values in the data!")

        activity_map_ind = activity_maps_flat[participant].reshape(init_shape)
        
        sel_vox_activity_map = np.multiply(activity_map_ind, corr_matr_thres)
        sel_vox_image = nib.Nifti1Image(sel_vox_activity_map, affine=map_affine)

        # append to dictionary
        salient_image_list[participant] = sel_vox_image

    print("\n\n salient image list", salient_image_list.keys())
    return salient_image_list
