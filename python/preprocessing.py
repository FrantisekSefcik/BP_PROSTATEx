import SimpleITK as sitk
import numpy as np 
import matplotlib.pyplot as plt

import pandas as pd
import gui

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
    im = sitk.GetImageFromArray(image, isVector=False)
    sitk.WriteImage(im, filename, True)