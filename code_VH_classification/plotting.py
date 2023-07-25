import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting
import nibabel as nib 


def z_plot(img, experiment_subdir, title):
    plotting.plot_stat_map(img, cut_coords = (-27, -9, 3, 12, 30, 48, 57), display_mode='z', output_file = os.path.join(experiment_subdir, (title + ".png")))
    return


def save_nii_image(img, experiment_subdir, title):
    plotting.plot_stat_map(img, cut_coords = (-27, -9, 3, 12, 30, 48, 57), display_mode='z', output_file = os.path.join(experiment_subdir, (title + '.nii')))
    return


def plot_histogram(img, experiment_subdir, title):
    intensities_array = img.get_fdata()

    # exclude inf values and 0s for plotting
    intensities_array_flat = pd.DataFrame(intensities_array.flatten())
    intensities_array_flat = np.ma.masked_invalid(intensities_array_flat)
    intensities_array_flat = intensities_array_flat[intensities_array_flat != 0.0]
    
    # plot distribution of example data
    plt.hist(intensities_array_flat, bins=100)
    plt.savefig(os.path.join(experiment_subdir, (title + "_img_distribution.png")))


def conf_matrix_plot(tp, fn, fp, tn, path):
    # Plot confusion matrix
    
    conf_matrix = np.array([[tp, fn], [fp, tn]]) 

    # Visualize the confusion matrix using a heatmap
    plt.figure(figsize=(4, 4))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
                xticklabels=['Actual Positive', 'Actual Negative'],
                yticklabels=['Predicted Positive', 'Predicted Negative'])
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(path)


def plot_reduced_matrix(example_img, indices_to_keep_df, act_map, feat_vec_size, coefs, init_shape, affine_map):
    coef_map = np.zeros_like(example_img.get_fdata().flatten())
    ind = indices_to_keep_df.loc[(indices_to_keep_df['activity_map_type'] == act_map) & (indices_to_keep_df['feature_vector_size'] == feat_vec_size), 'indices'].values[0]
    ind = np.array(ind).astype(int)
    coef_map[ind] = coefs
    print("new map: ", coef_map[:15])
    coef_map = coef_map.reshape(init_shape)
    weight_map_img = nib.Nifti1Image(coef_map, affine=affine_map)

    return weight_map_img