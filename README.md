# Classification of visual hallucinations in rs-fMRI using machine learning

## 1. Project description


While VH in physiological disorders were investigated more thoroughly, research on the classification of VH in people with psychosis is rather sparse~\cite{waters_visual_2014,schutte_functional_2021,alderson-day_auditory_2016}.
Investigations into neuroimaging biomarkers of VH can inform further research on this topic and possibly complement clinical diagnosis procedures for VH in the future.

The aim of this project is to develop a reusable machine learning pipeline for the purpose of classifying lifetime visual hallucinations in rs-fMRI images of individuals with psychosis. 
Based on previous research on other hallucination modalities several feature extraction methods and classifiers were investigated~\cite{chyzhyk_discrimination_2015, schutte_functional_2021, van_ommen_impaired_2022, chyzhyk_computer_2015}. 
Specifically, I implemented regional homogeneity (ReHo), amplitude of low frequency fluctuations (ALFF), fractional amplitude of low frequency fluctuations (fALFF), voxel-mirrored homotopic connectivity (VMHC), and eigenvector centrality mapping (ECM) features~\cite{liu_decreased_2006, turner_multi-site_2013, zhu_improved_2022, hoptman_decreased_2012, van_ommen_impaired_2022}. 
Pearson correlation was used for feature selection to reduce the computational complexity during classification, extract the most salient features, and increase the signal-to-noise ratio of the data~\cite{chyzhyk_discrimination_2015, grana_resting_2017}.
A linear SVM was utilized for classification in line with previous research on auditory hallucinations and schizophrenia~\cite{fovet_decoding_2022, de_pierrefeu_prediction_2018, chyzhyk_discrimination_2015}.
To increase the algorithm's explainability and the clinical applicability of the obtained results, I compared the features with respect to the classification accuracy, as well as the visual interpretability of the feature weights.

The main contributions of this project are: 

	- Creating an automated and reusable pipeline for preprocessing, feature extraction, feature selection, classification, and analysis of rs-fMRI images.
	- Testing the pipeline on a dataset comprised of 45 individuals with schizophrenia and healthy controls.
    - Assessing five feature extraction methods for detection of lifetime VH, considering classification accuracy as well as interpretability of the feature weights.
    
The results of this research were condensed into the master's thesis titled "Towards Automatic Classification of Lifetime Visual Hallucinations in Psychosis using Resting-State fMRI" (2023).

This project was supervised by Ashkan Nejad, Andreea I. Sburlea, and Frans W. Cornelissen at the Rijksuniversiteit Groningen and the University Medical Center Groningen (UMCG).

Author information:

Helena Luise Hazeu

Rijksuniversiteit Groningen (Masters Computational Cognitive Science)

July 2023

For questions, you can contact me at helena.hazeu@gmail.com



# 2. Table of contents


The following graph shows an overview of the created pipeline:
![Pipeline](/repository/assets/pipeline.pdf?raw=true "Schematic graph of analysis pipeline")
This project was created on MacOS 11.5.2 in Python (Version Python 3.10.10) and tested on a MacBook Pro 2020, 16GB RAM, M1 Chip. Please note that paths in MacOS use backslashes (/) while Windows uses forward slashes (\). Please make sure to use a forward slash when inserting file paths on Windows. 

## 3. Data

The data should be present in the BIDS format (https://bids-specification.readthedocs.io/). Moreover, all MRI images should be in the Nifti format (https://nifti.nimh.nih.gov). This can be achieved using software such as dcm2nii (https://www.nitrc.org/projects/dcm2nii/).

## 4. Preprocessing and feature extraction

Once all data is present in the BIDS format, it can be preprocessed and the feature maps can be extracted. These steps were conducted using the C-PAC pipeline (https://fcp-indi.github.io/docs/latest/user/quick).
The configuration file used can be found under the name *config_cpac_preprocessing.yml*. 
For more information on the preprocessing and feature extractionprocedure, please refer to the Methods and Fundamentals sections in my master's thesis.

To run C-PAC, I would advise using the docker or singularity implementation of C-PAC. Details can be found at https://fcp-indi.github.io/docs/latest/user/quick.

After installing docker or singularity, you have to pull the C-PAC container from the respective platform:

singularity pull FCP-INDI/C-PAC
 
or

docker pull fcpindi/c-pac:latest

To run the C-PAC container, you have to specify several file paths:

cpac -B /path/to/data/configs:/configs \
             --image fcpindi/c-pac --tag latest \
             run /path/to/data /path/for/outputs \
             --save_working_dir

## VMHC, ALFF, fALFF, ReHo

After preprocessing, you should have an output folder that contains all files necessary for the subsequent analysis and classification steps. 

For simplicity, I gathered all necessary images in one folder. 
This data folder should contain files from the C-PAC output with the following names:

Data/
├── participants.tsv
├── sub-0001_ses-1_task-rest_desc-1_vmhc.nii.gz
├── sub-0001_ses-1_task-rest_space-template_desc-1_alff.nii.gz
├── sub-0001_ses-1_task-rest_space-template_desc-1_falff.nii.gz
└── sub-0001_ses-1_task-rest_space-template_desc-2_reho.nii.gz

The files starting with *sub* should be present for each participant (0001, 0002, ... or other numbers) if you plan to use the pipeline on the respective feature type. You can also only include files of one feature type in the folder. 

The participants.tsv file should be present in your data repository. It could look like this:

![Pipeline](/repository/assets/participants_file.pdf?raw=true "First few rows of an example participants.tsv file")

## fastECM

For extracting the fastECM features, I used the fastECM method by Wink et al. (https://github.com/amwink/bias/tree/master/matlab). The Matlab implementation is well documented. By using the GUI provided when downloading the repository, I converted all images to fastECM feature maps. I used MATLAB R2023a (https://en.mathworks.com/products/matlab.html).

I first copied the images with the following name from the C-PAC output for each participant into a folder:

sub-0001_ses-1_task-rest_space-template_desc-cleaned-2_bold.nii.gz

This is the preprocessed fMRI scan (BOLD image) that can be used to compute fastECM.

After cloning the fastECM repository:

git clone git@github.com:amwink/bias.git

I opened MATLAB and entered:

>> addpath(genpath('bias'))
>> fegui

which opens the GUI.

By clicking on "add files" I imported the preprocessed BOLD images of all subjects into the fastECM GUI. Then I clicked on "use mask" to import the 3mm MNI mask I downloaded from (https://git.bcbl.eu/brainhackdonostia/material-2019/tree/2fd574bb4340651ed6e853e88d551db9d6b1ee44/fMRI/). This makes sure that no voxels outside of the MNI template remain in the BOLD scan and to ensure the proper working of fastECM. On the "Settings" tab I changed the connectivity calculation setting to "ReLU". Finally, I clicked on "estimate fECM" on the "Files" tab to compute fastECM. 

I advise not to import too many images at a time (less than 10).

This results in images of the form:

sub-0001_ses-1_task-rest_space-template_desc-cleaned-2_bold_fastECM.nii.gz

which I also inserted into the data folder. 
If you want to run all feature types, your folder should look like this:

Data/
├── participants.tsv
├── sub-0001_ses-1_task-rest_desc-1_vmhc.nii.gz
├── sub-0001_ses-1_task-rest_space-template_desc-1_alff.nii.gz
├── sub-0001_ses-1_task-rest_space-template_desc-1_falff.nii.gz
├── sub-0001_ses-1_task-rest_space-template_desc-2_reho.nii.gz
└── sub-0001_ses-1_task-rest_space-template_desc-cleaned-2_bold_fastECM.nii.gz

## 5. Visual Hallucinations Classification pipeline

All subsequent steps can be performed using the pipeline provided in this repository. An overview can be seen in the figure below:

![Pipeline](/repository/assets/flowchart.png?raw=true "Schematic graph of the current pipeline")

To this end, you have to clone this repository like so:

git clone git@github.com:LEO-UMCG/Visual_Hallucinations_Classification.git

Once you have the repository, you can create and activate a virtual environment with conda (not necessary but strongly recommended):

conda create —-prefix /Users/me/VHclassification 
conda activate VHclassification

Then you install the dependencies. Move to the folder containing the repository (the one you cloned into), and enter this in the terminal:

conda install --yes --file requirements.txt

Now you are all set to run the Visual Hallucinations Classification pipeline. 

To configure the pipeline, you have to change the *config-VH_classification.yaml* file. All necessary information to change the file are in the file itself.

Once you are satisfied with your configuration, you can run:

python3 code_VH_classification/main.py -c ./config-VH_classification.yaml

Run this in the terminal while being in the main folder for this project: files_VH/
The flag "-c" is used to enter the configuration file. Change this accordingly if your configuration file is located somewhere else. 

The program should run fully automated from here and produces the following output file tree:

experiment_directory/ 
    ├── experimentID/
    |   ├── experiment_log.txt
    |   └── Figures/
    |   |   ├── brain maps and weight images in png format
    |   └── Nifti_images/
    |   |   ├── brain images in 3D nifti format
    └── experimentID_experiment_results.csv


where the path to the experiment_directory and experimentID can be set in the configuration file.
The CSV file contains the classification results. 
The following subsections will give an overview of the files created and the functions in each file.


## main.py

The **main** file orchestrates all other files and is used to run the full pipeline. The data is first loaded using dataloader.py and masked to MNI using preprocessing.py. Then five-fold cross-validation is started by 


## dataloader.py
Loads the necessary data from other locations. 

The **get_data** function extracts all necessary meta data from the meta data file (labels, activity map type, and subject IDs) and loads the nifti images for the correct subject IDs from the folder provided as input data directory in the configuration file. The function saves all of this in a dictionary for convenient use in the subsequent processing steps. 

## preprocessing.py

Is used to mask the images to the MNI152 brain template.

**mask_image** masks the provided image to the MNI mask.

**mask_all_images** masks all images in the data dictionary to either a 2mm or a 3mm MNI mask depending on the feature type considered. VMHC needs a 2mm mask while all other feature types need a mask with a 3mm resolution.

## feature_selection.py

Contains function for performing feature selection by Pearson correlation and get the corresponding matrices and images in a convenient format.

**pearson_correlation** calculates the Pearson correlation between the intensity values and the corresponding subject phenotype labels (f.e. HC or PSVH). 
Firstly converts the activity map to a dataframe. The participant IDs are the rows and the columns are all voxels (with coordinates [i,j,k], flattened array). The values are the intensity value at a specific voxel location for one subject. 

It looks like this (example values):

                0       1       2       3  ... 
sub-1234     0.2     0.0     0.0     0.0  ...  
sub-2345     0.1     0.0     0.2     0.0  ...   
sub-3456     0.0     0.5     0.0     0.0  ...  
...          ...     ...     ...     ...

Then the Pearson correlation is calculated for each column of this dataframe (intensity values for one voxel over all training subjects) and the subject labels.



**run_feature_saliency_pearson** is used to arrange the data in a convenient format before performing feature selection by using the **pearson_correlation** function.

**get_salient_voxel_locations**
Determines the n most salient voxel locations with n being the feature vector size specified by the user. 
When the Pearson correlation of a voxel is in the top n strongest absolute correlation values, the mask value will be set to 1. All other values are set to 0. 

**get_salient_map** masks images with matrix obtained in get_salient_voxel_locations.
Upon multiplication with the activity map (intensity values), only the most salient voxel values will be kept in the activity map. 
All intensity values that are multiplied with 1 stay in the matrix, all others are deleted. Outputs the masked matrix (flattened array where unsalient voxels are deleted).


## classifiers.py

Contains the code to run all classifiers. 

**run_linear_svm_classification** runs classification with a linear SVM using a parameter grid for hyperparameter tuning. Prediction on the test set is performed with the best performing classifier from the grid search. Outputs performance values.

**run_nonlinear_svm_classification** runs classification with a non-linear SVM using a parameter grid for hyperparameter tuning. Prediction on the test set is performed with the best performing classifier from the grid search. Outputs performance values.

**run_rf_classification** runs classification with a random forest using a parameter grid for hyperparameter tuning. Prediction on the test set is performed with the best performing classifier from the grid search. Outputs performance values.

## plotting.py

Contains all functions for plotting of brain maps, histograms, and confusion matrices. 


## report_results.py

**report_average_over_folds** extracts all performance values logged in the results CSV and determines the average performance over the five folds for all conducted experiments in this run. 
