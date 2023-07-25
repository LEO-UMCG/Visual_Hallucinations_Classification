from nilearn.image import math_img

def mask_image(nii_image, mask):
    # mask image to MNI space (or any other mask depending on mask file)
    masked_image = math_img('img*mask', img=nii_image, mask=mask)
    return masked_image

def mask_all_images_MNI(data_dict, mask_2mm, mask_3mm):

    ''' Masks all images provided in the data_dict to the MNI152 template. '''
    
    for index, row in data_dict.iterrows():
        # read image
        img = row['images']

        # mask images to MNI according to the activity map type, VMHC needs 2mm while the rest is in 3mm resolution
        if row['activity_map_type'] == 'vmhc':
            img = mask_image(img, mask_2mm)
        else: 
            img = mask_image(img, mask_3mm)
        
        data_dict.loc[index, 'images'] = img 

    return data_dict