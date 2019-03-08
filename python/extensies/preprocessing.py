import SimpleITK as sitk
import numpy as np 
import matplotlib.pyplot as plt
import os 
import re
import pandas as pd
from extensies import gui

class DicomReader():
    
    def __init__(self,data_directory):
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)

        if not series_IDs:
            print("ERROR: given directory \""+data_directory+"\" does not contain a DICOM series.")
            return

        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])

        self.series_reader = sitk.ImageSeriesReader()
        self.series_reader.SetFileNames(series_file_names)

        # Configure the reader to load all of the DICOM tags (public+private):
        # By default tags are not loaded (saves time).
        # By default if tags are loaded, the private tags are not loaded.
        # We explicitly configure the reader to load tags, including the
        # private ones.
        self.series_reader.MetaDataDictionaryArrayUpdateOn()
        self.series_reader.LoadPrivateTagsOn()
        
    def GetImage(self):
        
        image3D = series_reader.Execute()
        return image3D
        
    
def read_dicom_series(data_directory):
    
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
    
    if not series_IDs:
        print("ERROR: given directory \""+data_directory+"\" does not contain a DICOM series.")
        return
    
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    # Configure the reader to load all of the DICOM tags (public+private):
    # By default tags are not loaded (saves time).
    # By default if tags are loaded, the private tags are not loaded.
    # We explicitly configure the reader to load tags, including the
    # private ones.
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()

    return image3D


def myshow(img, title=None, margin=0.05):
    
    if (img.GetDimension() == 3):
        img = sitk.Tile( (img[img.GetSize()[0]//2,:,:],
                          img[:,img.GetSize()[1]//2,:],
                          img[:,:,img.GetSize()[2]//2]), [2,2])
            
    
    aimg = sitk.GetArrayViewFromImage(img)
    
    xsize,ysize = aimg.shape

    dpi=80
    
    # Make a figure big enough to accommodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1+margin)*ysize / dpi, (1+margin)*xsize / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    t = ax.imshow(aimg)
    if len(aimg.shape) == 2:
        t.set_cmap("gray")
    if(title):
        plt.title(title)
    plt.show()
    
    
def image_information(image):
    print("Image size:      ",image.GetSize())
    print("Image spacing:   ",image.GetSpacing())
    print("Pixel depth :    ",image.GetPixelIDTypeAsString())
    print("Image direction: ",image.GetDirection())
    print("Image direction: ",image.GetNumberOfComponentsPerPixel())
    print("Image origin:    ",image.GetOrigin())

def in_boundaries(shape,ijk,size):
    if ijk[0] + size[0] > shape[0] or ijk[0] - size[0] < 0:
        return False
    elif ijk[1] + size[1] > shape[1] or ijk[1] - size[1] < 0:
        return False
    elif ijk[2] + size[2] > shape[2] or ijk[2] - size[2] < 0:
        return False
    
    return True
    
def extract_roi(image, ijk, size):
    
    if not in_boundaries(image.shape,ijk,size):
        return image, False
           
    offset_x = size[0] // 2
    offset_y = size[1] // 2
    
    return image[ijk[0]-offset_x:ijk[0]+offset_x, ijk[1]-offset_y:ijk[1]+offset_y, ijk[2]], True

def save_image(image,filename):
    
    directory = os.path.dirname(filename)
    
    if not os.path.exists(directory):
        os.makedirs(directory) 
    
    im = sitk.GetImageFromArray(image, isVector=False)
    sitk.WriteImage(im, filename, True)
    
    
    
    
def get_path(patient,modality = 't2w', root_path = ''):
    modality = modality.lower()
    # get path to infividual patient 
    path = os.path.join(root_path, patient)
    # list all directories in patient directory
    patient_dirs = os.listdir(path)
    # join first dir with prefix
    study_paths = [os.path.join(path,directory) for directory in patient_dirs]
    # list all directories that contain dicom series

    final_dirs = list(filter(lambda x: re.search(modality, x.lower() ),sorted(os.listdir(study_paths[0]))))

    return os.path.join(study_paths[0],final_dirs[0])

def get_ktrans_path(patient, root_path = ''):
    path_to_image = os.path.join(root_path,'Ktrans',patient)
    image_name  = patient + '-Ktrans.mhd'
    return os.path.join(path_to_image,image_name)  


def get_patch_from_image(image = None, size = None, center = None, orientation = 't'):
        
    if orientation == 's':
        size = (size[0],size[2],size[1])
    elif orientation == 'a':
        size = (size[2],size[1],size[0])
    elif orientation == 't':
        pass
    else:
        raise Exception('Wrong orientation type! Chose from this tree: t,s,a')
       
    center_idxs = np.array(image.TransformPhysicalPointToContinuousIndex(center)) + 0.5
    vol = sitk.GetArrayFromImage(image)
    image_size = image.GetSize()
    
    max_axis = (int(round(idx + lenght / 2)) for idx,lenght in zip(center_idxs,size))
    min_axis = (int(round(idx - lenght / 2)) for idx,lenght in zip(center_idxs,size))
    
    boundaries = list(zip(max_axis,min_axis))
    
    tresholds = list(treshold + 1 >= ma and 0 <= mi for treshold,(ma,mi) in zip(image_size, boundaries))
    
    if False in tresholds:  
        raise Exception('Boundaries out of image!')
    
    #reverse boundaries because vol has reversed axes
    boundaries = list(reversed(boundaries))
    
    patch = vol[boundaries[0][1]:boundaries[0][0] ,boundaries[1][1]:boundaries[1][0],boundaries[2][1]:boundaries[2][0]]
    #transform arrays for different orientations to shape for image with shape X/Y/Z    
    if orientation == 's':
        patch = np.flip(patch,1)
        patch = np.swapaxes(patch,0,1)
    elif orientation == 'a':
        patch = np.swapaxes(patch,0,1)
        patch = np.swapaxes(patch,0,2)
        patch = np.flip(np.flip(patch,2),0)
    
    # when we took 2d image then transform dimension
    if 1 in size:
        patch = patch[0]
        
    
    return patch
    
    
def resample_image_to_spacing(image,spacing,interpolator,report = False):
    #compute new size of image
    
    phys_size  =   [ org_spacing*(org_size - 1) for org_spacing,org_size in zip(image.GetSpacing(),image.GetSize())]
    new_size = [ int(round(phys / space)) + 1 for phys,space in zip(phys_size, spacing)]
    
    
    #Set up resample
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(spacing)
    resample.SetSize(new_size)
    
    newimage = resample.Execute(image)
    if report:
        print('Size:{} to {}, Spacing: {} to {}'.format(image.GetSize(),
                                                    newimage.GetSize(),image.GetSpacing(),newimage.GetSpacing()))
    
    return newimage