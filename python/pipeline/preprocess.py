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


root_path = '../../data/PROSTATEx'
path_to_data = '../../data/'
modality = 't2tsetra'
new_spacing = (0.5,0.5,3)
orientation = 't'
patch_size = (40,40,1)

target_path = os.path.join(path_to_data,modality,orientation, 
                            str(patch_size[0]) + 'x' + str(patch_size[1]) + 'x' + str(patch_size[2]))

if __name__ == "__main__":
    
    
    df_images = pd.read_csv(path_to_data + 'info/ProstateX-Images-Train.csv')
    df_findings = pd.read_csv(path_to_data + 'info/ProstateX-Findings-Train.csv')
    
    
    new_df = pd.DataFrame(columns = ['ProxID','fid','zone','ClinSig','name'])
    roi_volumes = []
    findings = df_findings.drop(df_findings.index[[33,34,35,36,37,154]])
    
    # iterate throught all findings
    for idx,row in findings.iterrows():
        print(idx)

        if modality == 'ktrans':
            path_to_image = my.get_ktrans_path(row['ProxID'],path_to_data)
            image = sitk.ReadImage(path_to_image)
        else:
            path_to_image = my.get_path(row['ProxID'], modality, root_path)
            fixed_series_filenames = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path_to_image)
            image = sitk.ReadImage(fixed_series_filenames)
        
        # fit normalizer
        normalizer = normalization.ZScoreNormalization()
        normalizer.fit(sitk.GetArrayFromImage(image))
        # resample image to same spacing in all directions
        image = my.resample_image_to_spacing(image, new_spacing, sitk.sitkLinear)
               
        # extract region of interest
        center = [float(x) for x in row['pos'].split()] 
        volume  = my.get_patch_from_image(image, patch_size, center, orientation)
        # normalise data
        volume = normalizer.normalise(volume)
        
        #save image
        file_name = row['ProxID']+'_'+str(idx)+'_'+row['zone']+'.nii'
        new_df.loc[idx] = row[['ProxID','fid','zone','ClinSig']]
        new_df.loc[idx,'name'] = file_name
    
        my.save_image(volume, os.path.join(target_path,file_name))
    
    new_df.to_csv(os.path.join(target_path,'info.csv'))