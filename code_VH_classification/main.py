## 
# Code by Helena Hazeu (2023) 
# Master's thesis: "Towards Automatic Classification of Lifetime Visual Hallucinations in Psychosis using Resting-State fMRI" at the Rijksuniversiteit Groningen and the UMCG (LEO lab). 
# If you have any questions you can contact me at helena.hazeu@gmail.com
# Copyright (c) Helena Hazeu 2023
##


# import libraries
import re
import os
import warnings
import time
import sys
import argparse
import configparser
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from sklearn.model_selection import StratifiedKFold

# import functions from files
from feature_selection import run_feature_saliency_pearson, get_salient_map, get_salient_voxel_locations
from plotting import z_plot, plot_histogram, save_nii_image, plot_reduced_matrix
from dataloader import get_data
from preprocessing import mask_image, mask_all_images_MNI
from classifiers import run_linear_svm_classification, run_nonlinear_svm_classification, run_rf_classification
from report_results import report_average_over_folds

# set a random seed for reproducibility
random_seed = 2023
np.random.seed(random_seed)

# silent certain warnings
warnings.filterwarnings('ignore')


def get_input_arguments():


    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type = str, default = "Config",
                        help = "Path of config file (.yaml)")

    args = parser.parse_args()

    return args.config_file

def check_affine(affine_list):
    # check if all affine maps in a list are equal     
    check = all(np.array_equal(affine_map, affine_list[0]) for affine_map in affine_list)

    if(check):
        # all affine maps are equal to each other
        pass
    else:
        print("Error: Not all affine maps in the image array are the same. Please provide a list of images with equal affine maps.")



def main():

    ###### Setup 

    # get parameters from command line
    config_file_path = get_input_arguments()

    config = configparser.ConfigParser()
    config.read(config_file_path)

    print("Classification %s program started ... " %(config["Type"]["model_type"]))
    print("For further terminal output please check the corresponding log files (experiment_log.txt in log folder).")


    # create CSV to save all results
    exp_dir = config["Paths"]["experiment_dir_path"]
    classes_for_classification = [config["Experiment"]["negative_label"], config["Experiment"]["positive_label"]]
    classes_str = "_" + str(classes_for_classification[0]) + "vs" + str(classes_for_classification[1])
    time_date = time.strftime("%d-%m-%y_%Hh%Mmin", time.localtime())
    csv_path = os.path.join(exp_dir,  config["Id"]["experiment_id"] + time_date + classes_str + "_experiment_results.csv")
    df = pd.DataFrame(columns=['id', 
    'type', 
    'activity_map', 
    'feat_vec_size', 
    'classes', 
    'folds', 
    'this_fold',
    'test_acc', 
    'test_sens', 
    'test_spec', 
    'confusion_matrix',
    'list_train', 
    'list_tst',
    'list_predictions', 
    'list_true_test_labels'])
    
    df.to_csv(csv_path, index = False)

    ###### 1. Initializing 

    # get start of execution time
    start_time = time.time()

    ### set config information
    model_type =  config["Type"]["model_type"]
    input_dir_path = config["Paths"]["input_dir_path"]
    experiment_dir_path = config["Paths"]["experiment_dir_path"]

    mask_2mm = config["Paths"]["mask_path_2mm"]
    mask_3mm = config["Paths"]["mask_path_3mm"]

    verbose = str(config["Experiment"]["verbose"]).lower().strip()  

    # if more than one activity map type, split them
    activity_map_type = config["Experiment"]["activity_map_type"].split(",")
    activity_map_type = [str(i).strip() for i in activity_map_type] 
    activity_map_type = [i.lower() if i != 'fastECM' else i for i in activity_map_type] 

    # if more than one feature vector size, split
    feature_vector_size = config["Experiment"]["feature_vector_size"].split(",")
    feature_vector_size = [j.strip() for j in feature_vector_size]
    # convert numbers to ints and preserve "all" as string
    feature_vector_size = [int(j) if j != 'all' else j for j in feature_vector_size] 
    
    label_col_name = config["Experiment"]["label_col_name"]
    index_col_name = config["Experiment"]["index_col_name"]
    scoring_measure = config["Experiment"]["scoring_measure"]
    positive_label = config["Experiment"]["positive_label"]
    
    experiment_id = str(config["Id"]["experiment_id"]) + "_" + \
        model_type + "_" + \
        "".join(activity_map_type) + "_" + \
        "".join(str(i) for i in feature_vector_size) + "_" + \
        classes_for_classification[0] + "vs" + \
        classes_for_classification[1]

    ### specify file paths

    meta_data_file_path = os.path.join(input_dir_path, "participants.tsv")

    if not meta_data_file_path:
        FileNotFoundError
        print('Error, the meta data file cannot be found or is not named "participants.tsv"')
        exit(1)

    # make experiment folder and log file
    os.makedirs(experiment_dir_path, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir_path, experiment_id), exist_ok=True)
    experiment_subdir = os.path.join(experiment_dir_path, experiment_id)
    log_file = os.path.join(experiment_dir_path, os.path.join(experiment_id, "experiment_log.txt"))

    # make folder for figures
    os.makedirs(os.path.join(experiment_dir_path, experiment_id, "Figures"), exist_ok=True)
    figures_subdir = os.path.join(experiment_dir_path, experiment_id, "Figures")

    # make folder for nifti images
    os.makedirs(os.path.join(experiment_dir_path, experiment_id, "Nifti_images"), exist_ok=True)
    nii_imgs_subdir = os.path.join(experiment_dir_path, experiment_id, "Nifti_images")

    # redirect output to log file
    sys.stdout = open(log_file, 'w')

    # print out all information to the log file
    print("input directory path:", input_dir_path)
    print("experiment directory path:", experiment_dir_path)
    print("experiment csv path:", csv_path)
    print("experiment subdirectory path:", experiment_subdir)
    print("meta data file path:", meta_data_file_path)
    print("log file path:", log_file)
    print("experiment id:", experiment_id)
    print("mask path 2mm:", mask_2mm)
    print("mask path 3mm:", mask_3mm)
    print("verbose (true or false):", verbose)
    print("activity map type:", activity_map_type) 
    print("feature vector size:", feature_vector_size)
    print("name of class-label column in meta data file:", label_col_name)
    print("name of subject identifier column in meta data file:", index_col_name)
    print("scoring measure:", scoring_measure)
    print("positive label:", positive_label)
    print("classes chosen for classification:", classes_for_classification)
    print("random seed:", random_seed)

    ### check input parameters
    
    if len(classes_for_classification) != 2:
        KeyError
        print("Error: Wrong input length. The number of selected phenotypes for classification needs to be equal to 2. Please change the input to two classes (positive_label and negative_label have to be set in the config .yaml file).")

    # find example images for all activity map types
    example_img_paths = {}
    for act_map in activity_map_type:
        found = False
        for file in os.listdir(input_dir_path):
            if re.findall(str("_" + act_map), file):
                if re.findall('sub-', file):
                    example_img_paths[act_map] = os.path.join(input_dir_path, file)
                    found = True
                    break

    if found == False:
        FileExistsError 
        print("Error: There is no file in the specified file path, that contains the activity map type that was requested. Please check input_file_path and activity_map_type in the config .yaml file.")


    # extract labels
    meta_data = pd.read_csv(meta_data_file_path, index_col=index_col_name, sep='\t')
    meta_data[label_col_name].astype("category") # use category type for storage optimization
    meta_data['Gender'].astype("category")


    # choose only selected classes
    meta_data_for_classes = meta_data.loc[meta_data[label_col_name].isin(classes_for_classification)]  
    print(meta_data_for_classes)

    # save all data in a dictionary
    data_dict = get_data(input_dir_path, meta_data_for_classes, label_col_name, activity_map_type)

    # mask images to MNI template before feature selection using provided masks
    data_dict = mask_all_images_MNI(data_dict, mask_2mm, mask_3mm)

    init_shape = {}
    affine_map = {}
    example_img_dict = {}

    ### check input data and plot an example image for each activity map
    for act_map in activity_map_type:
        # load example nifti image 
        example_img = image.load_img(example_img_paths[act_map])

        if act_map == 'vmhc':
            example_img = mask_image(example_img, mask_2mm)
        else:
            example_img = mask_image(example_img, mask_3mm)

        # get shape and affine of images (differs between activity map types)
        affine_map[act_map] = example_img.affine
        init_shape[act_map] = example_img.shape
        print("Shape of example image for {} activity map from the dataset is: {}".format(act_map, init_shape[act_map]))

        # check if all affine maps for one activity map type are the same
        affine_list = [img.affine for img in data_dict.loc[data_dict['activity_map_type'] == act_map]['images']]
        check_affine(affine_list)

        if verbose == 'true':
            # plot example data
            z_plot(example_img, figures_subdir, title=("example_img_" + act_map))
            plot_histogram(example_img, figures_subdir, title=("example_img_" + act_map))

        example_img_dict[act_map] = example_img


    ### 2. Initialize cross validation strategy

    global n_folds 
    n_folds = 5
    cv = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)

    fold = 0
    for train, test in cv.split(meta_data_for_classes.index, meta_data_for_classes[label_col_name]): 
        list_train_ids = list(meta_data_for_classes.iloc[train].index)
        list_test_ids = list(meta_data_for_classes.iloc[test].index)
        
        fold += 1
        print("This is fold {}".format(fold))
        
        # get training data and flatten matrix
        tr_data = data_dict.loc[data_dict['sub_ids'].isin(list_train_ids)]
        tr_data['images'] = tr_data['images'].apply(lambda x: np.nan_to_num(x.get_fdata().flatten()))

        tst_data = data_dict.loc[data_dict['sub_ids'].isin(list_test_ids)]
        tst_data['images'] = tst_data['images'].apply(lambda x: np.nan_to_num(x.get_fdata().flatten()))

        indices_to_keep_df = pd.DataFrame(columns=['indices', 'activity_map_type', 'feature_vector_size'])


        #### 3. Determine feature saliency

        for act_map in activity_map_type:
            print("\n\nDo feature saliency calculation with Pearson for activity map {}".format(act_map))

            tr_corr_mat, tr_maps_flat, tst_maps_flat, tr_labels, tst_labels = run_feature_saliency_pearson(tr_data, tst_data, act_map)

            # for each feature vector size, mask image so that only salient voxels remain
            for feat_vec_size in feature_vector_size:
                if feat_vec_size != "all":
                    print("\n\nStart feature selection process for activity map {} and feature vector size = {}".format(act_map, feat_vec_size))

                    # determine most salient voxel locations
                    tr_corr_matr_thres = get_salient_voxel_locations(tr_corr_mat, feat_vec_size)
                    print("Shape of selection mask ", tr_corr_matr_thres.shape)

                    if fold == 3 and verbose == 'true':
                        
                        # plot selection mask
                        tr_corr_matr_thres_img = tr_corr_matr_thres.reshape(init_shape[act_map])
                        tr_corr_matr_thres_img = nib.Nifti1Image(tr_corr_matr_thres_img, affine=affine_map[act_map])

                        # plot image
                        z_plot(tr_corr_matr_thres_img, figures_subdir, \
                               title=("selection_mask_" + \
                                        act_map + "_" + str(feat_vec_size) + "_" + str(fold)))
                        
                        # save image as 3D nii file
                        save_nii_image(tr_corr_matr_thres_img, nii_imgs_subdir, \
                                       title=("selection_mask_" + \
                                                act_map + "_" + str(feat_vec_size) + "_" + str(fold)))


                    ### 4. Feature selection

                    # select salient voxels from activity map
                    tr_map_list, indices_to_keep = get_salient_map(tr_maps_flat, tr_corr_matr_thres) 
                    tr_map_list = pd.Series(tr_map_list)

                    indices_to_keep_df = pd.concat([indices_to_keep_df, 
                                        pd.DataFrame({'indices' : [indices_to_keep], 
                                                    'activity_map_type' : act_map, 
                                                    'feature_vector_size': feat_vec_size})]) 

                    print("Dataframe of indices to keep: ")
                    print(indices_to_keep_df)

                    # select salient voxels in test images based on training map selection (generalizability)   
                    tst_map_list, _ = get_salient_map(tst_maps_flat, tr_corr_matr_thres)
                    tst_map_list = pd.Series(tst_map_list)

                    # plot most salient voxel locations
                    if fold == 3 and verbose == 'true':
                        for map in tr_map_list:
                            print("length map ", len(map))
                            print("type map", type(map))
                            new_map = np.zeros_like(tr_corr_matr_thres)
                            new_map[indices_to_keep] = map
                            print("new map:", new_map[:15])
                            new_map = new_map.reshape(init_shape[act_map])

                            new_map_img = nib.Nifti1Image(new_map, affine=affine_map[act_map])

                            z_plot(new_map_img, figures_subdir, 
                                   title=("selected_voxels_" + 
                                          act_map + "_" + str(feat_vec_size) + "_" + str(fold)))

                            break


                    # append data after feature selection to dataframe 
                    # train set 
                    tr_new_data = pd.DataFrame(columns=tr_data.columns)
                    tr_new_data.labels = tr_labels
                    tr_new_data.activity_map_type = act_map
                    tr_new_data.images = tr_map_list
                    tr_new_data.sub_ids = tr_map_list.index
                    tr_new_data.feature_vector_size = feat_vec_size

                    tr_data = pd.concat([tr_data, tr_new_data], ignore_index = True)

                    # test set
                    tst_new_data = pd.DataFrame(columns=tst_data.columns)
                    tst_new_data.labels = tst_labels
                    tst_new_data.activity_map_type = act_map
                    tst_new_data.images = tst_map_list
                    tst_new_data.sub_ids = tst_map_list.index
                    tst_new_data.feature_vector_size = feat_vec_size

                    tst_data = pd.concat([tst_data, tst_new_data], ignore_index = True)

            else:
                '''
                When feature vector size = all: feature selection is not needed!
                ''' 

                # plot salient voxels
                if fold == 3 and verbose == 'true':
                    # plot most salient voxel locations

                    tr_corr_matr_thres_img = tr_corr_mat.reshape(init_shape[act_map])
                    tr_corr_matr_thres_img = nib.Nifti1Image(tr_corr_matr_thres_img, affine=affine_map[act_map])

                    # plot image
                    z_plot(tr_corr_matr_thres_img, figures_subdir, title=("pearson_correlation_map_" + act_map + "_" + str(feat_vec_size) + "_" + str(fold)))
                    # save image as 3D nii file
                    save_nii_image(tr_corr_matr_thres_img, nii_imgs_subdir, title=("pearson_correlation_map_" + act_map + "_" + str(feat_vec_size) + "_" + str(fold)))

                pass

        ## print data used in this fold
        print("Train data:")
        print(tr_data[['sub_ids', 'labels']]) 
        print("Test data:")
        print(tst_data[['sub_ids', 'labels']])  


        #### 5. Classification

        for act_map in activity_map_type:
            for feat_vec_size in feature_vector_size:
                print("\n\nStart classification data selection procedure for {} activity map and feature vector size = {}...".format(act_map, feat_vec_size))

                # create train and test set lists
                condition_tr = ((tr_data['activity_map_type'] == act_map) & (tr_data['feature_vector_size'] == feat_vec_size))
                tr_image_list = pd.Series(tr_data.loc[condition_tr]['images']) 
                tr_labels = pd.Series(tr_data.loc[condition_tr]['labels'])
                tr_ids = pd.Series(tr_data.loc[condition_tr]['sub_ids'])
                print('tr_image_list ')
                print(tr_image_list)
                          

                condition_tst = ((tst_data['activity_map_type'] == act_map) & (tst_data['feature_vector_size'] == feat_vec_size))
                tst_image_list = pd.Series(tst_data.loc[condition_tst]['images']) 
                tst_labels = pd.Series(tst_data.loc[condition_tst]['labels'])
                tst_ids = pd.Series(tst_data.loc[condition_tst]['sub_ids'])

                print("Train subjects: ", tr_ids)
                print("Labels: ", tr_labels)
                print("Corresponding image list: ", tr_image_list.keys())

                print("Test subjects: ", tst_ids)
                print("Labels: ", tst_labels)
                print("Corresponding image list: ", tst_image_list.keys())

                # classification depending on selected model type
                if model_type == "SVM":
                    linear_svm_prediction, accuracy, sensitivity, specificity, conf_matrix, coefs= run_linear_svm_classification(tr_image_list, tr_labels, tst_image_list, tst_labels, scoring_measure, random_seed)

                    print("Test accuracy: {}, sens {}, spec {}".format(accuracy, sensitivity, specificity))
                    print("SVM coefs: ")
                    print(coefs)

                    # plot weights
                    if fold == 3:
                        if feat_vec_size != 'all':
                            # convert sparse weight matrix to 3D image
                            weight_map_img = plot_reduced_matrix(example_img_dict[act_map], 
                                                                 indices_to_keep_df, 
                                                                 act_map, 
                                                                 feat_vec_size, 
                                                                 coefs, 
                                                                 init_shape[act_map],
                                                                 affine_map[act_map])

                        else:
                            # convert non-sparse weight matrix (no feature selection) to 3D image
                            coef_map = coefs.reshape(init_shape[act_map])
                            weight_map_img = nib.Nifti1Image(coef_map, affine=affine_map[act_map])

                        # plot image
                        z_plot(weight_map_img, figures_subdir, \
                                           title=("SVM_weights_" + act_map + "_" + str(feat_vec_size) + "_" + str(fold)))
                        
                        # save image as 3D file
                        if verbose == 'true':
                            save_nii_image(weight_map_img, nii_imgs_subdir, \
                                                title=("SVM_weights_" + act_map + "_" + str(feat_vec_size) + "_" + str(fold)))


                    #### 6. Save results 

                    # append data to CSV file where results are stored
                    data = [{'id' : experiment_id,
                                'type': model_type, 
                                'activity_map' : act_map, 
                                'feat_vec_size' : feat_vec_size, 
                                'classes' : str(classes_for_classification[0] + "vs" + classes_for_classification[1]), 
                                'folds' : n_folds, 
                                'this_fold' : fold,
                                'test_acc' : accuracy, 
                                'test_sens' : sensitivity, 
                                'test_spec' : specificity,
                                'confusion_matrix' : conf_matrix.flatten(),
                                'list_train': list(tr_ids.values), 
                                'list_tst': list(tst_ids.values), 
                                'list_predictions': list(linear_svm_prediction), 
                                'list_true_test_labels': list(tst_labels.astype('category').cat.codes)}]


                    data = pd.DataFrame.from_dict(data)

                    print(data)

                    data.to_csv(csv_path, index=False, header=False, mode="a")
                
                elif model_type == "nonlinear_SVM":

                    nonlinear_svm_prediction, accuracy, sensitivity, specificity, conf_matrix, coefs= run_nonlinear_svm_classification(tr_image_list, tr_labels, tst_image_list, tst_labels, scoring_measure, random_seed)

                    print("Test accuracy: {}, sens {}, spec {}".format(accuracy, sensitivity, specificity))
                    print("SVM coefs: ")
                    print(coefs)

                    # plot weights
                    if fold == 3:
                        if feat_vec_size != 'all':
                            # convert sparse weight matrix to 3D image
                            weight_map_img = plot_reduced_matrix(example_img_dict[act_map], 
                                                                 indices_to_keep_df, 
                                                                 act_map, 
                                                                 feat_vec_size, 
                                                                 coefs, 
                                                                 init_shape[act_map],
                                                                 affine_map[act_map])

                        else:
                            # convert non-sparse weight matrix (no feature selection) to 3D image
                            coef_map = coefs.reshape(init_shape[act_map])
                            weight_map_img = nib.Nifti1Image(coef_map, affine=affine_map[act_map])

                        # plot image
                        z_plot(weight_map_img, figures_subdir, \
                                           title=("SVM_weights_" + act_map + "_" + str(feat_vec_size) + "_" + str(fold)))
                        
                        # save image as 3D file
                        if verbose == 'true':
                            save_nii_image(weight_map_img, nii_imgs_subdir, \
                                                title=("SVM_weights_" + act_map + "_" + str(feat_vec_size) + "_" + str(fold)))


                    #### 6. Save results 

                    # append data to CSV file where results are stored
                    data = [{'id' : experiment_id,
                                'type': model_type, 
                                'activity_map' : act_map, 
                                'feat_vec_size' : feat_vec_size, 
                                'classes' : str(classes_for_classification[0] + "vs" + classes_for_classification[1]), 
                                'folds' : n_folds, 
                                'this_fold' : fold,
                                'test_acc' : accuracy, 
                                'test_sens' : sensitivity, 
                                'test_spec' : specificity,
                                'confusion_matrix' : conf_matrix.flatten(),
                                'list_train': list(tr_ids.values), 
                                'list_tst': list(tst_ids.values), 
                                'list_predictions': list(nonlinear_svm_prediction), 
                                'list_true_test_labels': list(tst_labels.astype('category').cat.codes)}]


                    data = pd.DataFrame.from_dict(data)

                    print(data)

                    data.to_csv(csv_path, index=False, header=False, mode="a")
                

                elif model_type == "RF":
                    rf_decoder, rf_prediction, accuracy, sensitivity, specificity, conf_matrix, coefs = run_rf_classification(tr_image_list, tr_labels, tst_image_list, tst_labels, scoring_measure, random_seed)

                    print("Test accuracy: {}, sens {}, spec {}".format(accuracy, sensitivity, specificity))
                    print(coefs)

                    # plot weights
                    if fold == 3:
                        if feat_vec_size != 'all':
                            # convert sparse weight matrix to 3D image
                            weight_map_img = plot_reduced_matrix(example_img_dict[act_map], 
                                                                 indices_to_keep_df, 
                                                                 act_map, 
                                                                 feat_vec_size, 
                                                                 coefs, 
                                                                 init_shape[act_map],
                                                                 affine_map[act_map])

                        else:
                            # convert non-sparse weight matrix (no feature selection) to 3D image
                            coef_map = coefs.reshape(init_shape[act_map])
                            weight_map_img = nib.Nifti1Image(coef_map, affine=affine_map[act_map])

                        # plot image
                        z_plot(weight_map_img, figures_subdir, \
                                           title=("RF_weights_" + act_map + "_" + str(feat_vec_size) + "_" + str(fold)))
                        
                        # save image as 3D file
                        if verbose == 'true':
                            save_nii_image(weight_map_img, nii_imgs_subdir, \
                                                title=("RF_weights_" + act_map + "_" + str(feat_vec_size) + "_" + str(fold)))


                    #### 6. Save results 

                    # append data to CSV file where results are stored
                    data = [{'id' : experiment_id,
                                'type': model_type, 
                                'activity_map' : act_map, 
                                'feat_vec_size' : feat_vec_size, 
                                'classes' : str(classes_for_classification[0] + "vs" + classes_for_classification[1]), 
                                'folds' : n_folds, 
                                'this_fold' : fold,
                                'test_acc' : accuracy, 
                                'test_sens' : sensitivity, 
                                'test_spec' : specificity,
                                'confusion_matrix' : conf_matrix.flatten(),
                                'list_train': list(tr_ids.values), 
                                'list_tst': list(tst_ids.values), 
                                'list_predictions': list(rf_prediction), 
                                'list_true_test_labels': list(tst_labels.astype('category').cat.codes)}]


                    data = pd.DataFrame.from_dict(data)

                    print(data)

                    data.to_csv(csv_path, index=False, header=False, mode="a")
                




    # stop time
    print("--- %s seconds ---" % (time.time() - start_time))


 
    #### 7. Final logging of average accuracy over folds
    report_average_over_folds(csv_path, 
                              config, 
                              n_folds, 
                              classes_for_classification, 
                              activity_map_type, 
                              feature_vector_size, 
                              verbose)



if __name__ == "__main__":

    main() 
