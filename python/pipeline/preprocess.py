from __future__ import generators, division, absolute_import, with_statement, print_function, unicode_literals

import SimpleITK as sitk
# from downloaddata import fetch_data as fdata
from extensies import preprocessing as my
from extensies import normalization
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import sys


# Define parameters of preprocessing
root_path = '../../data/PROSTATEx'
path_to_data = '../../data/'
modality = 'ktrans' 
new_spacing = (0.5,0.5,0.5)
orientation = 't'
patch_size = (40,40,1)

target_path = os.path.join(path_to_data,modality,orientation, 
                            str(patch_size[0]) + 'x' + str(patch_size[1]) + 'x' + str(patch_size[2]))

# x = 1.5
# offsets = [[0,0,0],[-x,x,x],[x,x,x],[-x,-x,x],[x,-x,x],
#             [-x,x,0],[x,x,0],[-x,-x,0],[x,-x,0],[-x,x,-x],[x,x,-x],[-x,-x,-x],[x,-x,-x]]
offsets = [[0,0,0]]

if __name__ == "__main__":
    
    # read files with informations about findings and MRI images
    df_images = pd.read_csv(path_to_data + 'info/ProstateX-Images-Train.csv')
    df_findings = pd.read_csv(path_to_data + 'info/ProstateX-Findings-Train.csv')
    
    
    new_df = pd.DataFrame(columns = ['ProxID','fid','zone','ClinSig','name'])
    roi_volumes = []
    # findings = df_findings.drop(df_findings.index[[33,34,35,36,37,154]])
    findings = df_findings.drop(df_findings.index[[33,34,35,36,37,154,5,44,45,46,64,
                                                   81,84,87,110,114,131,145,162,176,179,190,193,215,230,246,264,265,268,275,292,304,325]])
    findings = findings[findings['zone'] != 'SV']
    
    # iterate throught all findings
    index = 0
    for idx,(_,row) in enumerate(findings.iterrows()):
        print(row['ProxID'])

        # Load image by modality
        if modality == 'ktrans':
            path_to_image = my.get_ktrans_path(row['ProxID'],path_to_data)
            image = sitk.ReadImage(path_to_image)
        else:
            #
            path_to_image = my.get_path(row['ProxID'], modality, root_path)
            fixed_series_filenames = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path_to_image)
            image = sitk.ReadImage(fixed_series_filenames)
        
        # fit normalizer
        normalizer = normalization.ScaleNormalization()
        normalizer.fit(sitk.GetArrayFromImage(image))
        # resample image to same spacing in all directions
        image = my.resample_image_to_spacing(image, new_spacing, sitk.sitkLinear)
        
        # get center of lesion from dataset       
        center = [float(x) for x in row['pos'].split()] 
        # extract region of interest
        for i,offset in enumerate(offsets):
            volume  = my.get_patch_from_image(image, patch_size, center, orientation)
            # normalise data
            print(volume.dtype)
            volume = normalizer.normalise(volume)
            
            # define informations about image
            file_name = row['ProxID']+'_'+str(index)+'_'+row['zone']+'.nii'
            new_df.loc[index] = row[['ProxID','fid','zone','ClinSig']]
            new_df.loc[index,'name'] = file_name
            new_df.loc[index,'ID'] = idx
            new_df.loc[index,'normalization'] = normalizer.name
            # save image to file
            my.save_image(volume, os.path.join(target_path,file_name))

            index += 1
    #save informations about images to the directory
    new_df.to_csv(os.path.join(target_path,'info.csv'))