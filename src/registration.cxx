/*=========================================================================
*
*  Copyright Insight Software Consortium
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*         http://www.apache.org/licenses/LICENSE-2.0.txt
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
*=========================================================================*/

//  Software Guide : BeginLatex
//
//  Probably the most common representation of datasets in clinical
//  applications is the one that uses sets of DICOM slices in order to compose
//  3-dimensional images. This is the case for CT, MRI and PET scanners. It is
//  very common therefore for image analysts to have to process volumetric
//  images stored in a set of DICOM files belonging to a
//  common DICOM series.
//
//  The following example illustrates how to use ITK functionalities in order
//  to read a DICOM series into a volume and then save this volume in another
//  file format.
//
//  The example begins by including the appropriate headers. In particular we
//  will need the \doxygen{GDCMImageIO} object in order to have access to the
//  capabilities of the GDCM library for reading DICOM files, and the
//  \doxygen{GDCMSeriesFileNames} object for generating the lists of filenames
//  identifying the slices of a common volumetric dataset.
//
//  \index{itk::ImageSeriesReader!header}
//  \index{itk::GDCMImageIO!header}
//  \index{itk::GDCMSeriesFileNames!header}
//  \index{itk::ImageFileWriter!header}
//
//  Software Guide : EndLatex

// Software Guide : BeginCodeSnippet
#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
// Software Guide : EndCodeSnippet

typedef signed short    PixelType;
typedef unsigned short    PixelTypePNG;
const unsigned int      Dimension3D = 3;
const unsigned int      Dimension2D = 2;
typedef itk::Image< PixelType, Dimension3D >         ImageType3D;
typedef itk::Image< PixelType, Dimension2D >         ImageType2D;
typedef itk::Image< PixelTypePNG, Dimension2D >         ImageTypePNG;
typedef itk::ImageSeriesReader< ImageType3D >        ReaderType;


// Software Guide : EndCodeSnippet

ReaderType::Pointer readDicomSeri(char *fileName) {

	ReaderType::Pointer reader = ReaderType::New();

	typedef itk::GDCMImageIO       ImageIOType;
	ImageIOType::Pointer dicomIO = ImageIOType::New();

	reader->SetImageIO(dicomIO);

	typedef itk::GDCMSeriesFileNames NamesGeneratorType;
	NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();

	nameGenerator->SetUseSeriesDetails(true);
	nameGenerator->AddSeriesRestriction("0008|0021");

	nameGenerator->SetDirectory(fileName);
	// Software Guide : EndCodeSnippet


	try
	{
		std::cout << std::endl << "The directory: " << std::endl;
		std::cout << std::endl << fileName << std::endl << std::endl;
		std::cout << "Contains the following DICOM Series: ";
		std::cout << std::endl << std::endl;


		typedef std::vector< std::string >    SeriesIdContainer;

		const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();

		SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
		SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
		while (seriesItr != seriesEnd)
		{
			std::cout << seriesItr->c_str() << std::endl;
			++seriesItr;
		}

		std::string seriesIdentifier;

		seriesIdentifier = seriesUID.begin()->c_str();



		std::cout << std::endl << std::endl;
		std::cout << "Now reading series: " << std::endl << std::endl;
		std::cout << seriesIdentifier << std::endl;
		std::cout << std::endl << std::endl;


		typedef std::vector< std::string >   FileNamesContainer;
		FileNamesContainer fileNames;

		fileNames = nameGenerator->GetFileNames(seriesIdentifier);

		reader->SetFileNames(fileNames);

		try
		{
			reader->Update();
			return reader;
		}
		catch (itk::ExceptionObject &ex)
		{
			std::cout << ex << std::endl;
		}

	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
	}
}

int main(int argc, char* argv[])
{

	if (argc < 3)
	{
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0] << " DicomDirectory  outputFileName  xCoordinate yCoordinate sliceNumber size"
			<< std::endl;
		return EXIT_FAILURE;
	}
	try{

		// Software Guide : BeginCodeSnippet
		typedef itk::ImageFileWriter< ImageType2D > WriterType;
		WriterType::Pointer writer = WriterType::New();

		writer->SetFileName(argv[2]);
		
		// nacitanie dicom serie
		ReaderType::Pointer reader = ReaderType::New();
		reader = readDicomSeri(argv[1]);


		//FILTER/////////////////////////////////////////
		typedef itk::ExtractImageFilter< ImageType3D,
			ImageType2D > FilterType;
		FilterType::Pointer filter = FilterType::New();
		filter->InPlaceOn();
		filter->SetDirectionCollapseToSubmatrix();
		// Software Guide : EndCodeSnippet


		//  Software Guide : BeginLatex
		//
		//  The ExtractImageFilter requires a region to be defined by the
		//  user. The region is specified by an \doxygen{Index} indicating the
		//  pixel where the region starts and an \doxygen{Size} indicating how many
		//  pixels the region has along each dimension. In order to extract a $2D$
		//  image from a $3D$ data set, it is enough to set the size of the region
		//  to $0$ in one dimension.  This will indicate to
		//  ExtractImageFilter that a dimensional reduction has been
		//  specified. Here we take the region from the largest possible region of
		//  the input image. Note that \code{UpdateOutputInformation()} is being
		//  called first on the reader. This method updates the metadata in
		//  the output image without actually reading in the bulk-data.
		//
		//  Software Guide : EndLatex


		// Software Guide : BeginCodeSnippet
		reader->UpdateOutputInformation();
		ImageType3D::RegionType inputRegion =
			reader->GetOutput()->GetLargestPossibleRegion();
		// Software Guide : EndCodeSnippet


		//  Software Guide : BeginLatex
		//
		//  We take the size from the region and collapse the size in the $Z$
		//  component by setting its value to $0$. This will indicate to the
		//  ExtractImageFilter that the output image should have a
		//  dimension less than the input image.
		//
		//  Software Guide : EndLatex

		// Software Guide : BeginCodeSnippet
		const unsigned int regionSize = atoi(argv[6]);
		ImageType3D::SizeType size = inputRegion.GetSize();
		size[0] = regionSize;
		size[1] = regionSize;
		size[2] = 0;
		// Software Guide : EndCodeSnippet

		//  Software Guide : BeginLatex
		//
		//  Note that in this case we are extracting a $Z$ slice, and for that
		//  reason, the dimension to be collapsed is the one with index $2$. You
		//  may keep in mind the association of index components
		//  $\{X=0,Y=1,Z=2\}$. If we were interested in extracting a slice
		//  perpendicular to the $Y$ axis we would have set \code{size[1]=0;}.
		//
		//  Software Guide : EndLatex


		//  Software Guide : BeginLatex
		//
		//  Then, we take the index from the region and set its $Z$ value to the
		//  slice number we want to extract. In this example we obtain the slice
		//  number from the command line arguments.
		//
		//  Software Guide : EndLatex

		// Software Guide : BeginCodeSnippet
		ImageType3D::IndexType start = inputRegion.GetIndex();
		const unsigned int i = atoi(argv[3]);
		const unsigned int j = atoi(argv[4]);
		const unsigned int sliceNumber = atoi(argv[5]);
		start[0] = i - regionSize / 2;
		start[1] = j - regionSize / 2;
		start[2] = sliceNumber;
		// Software Guide : EndCodeSnippet


		



		//  Software Guide : BeginLatex
		//
		//  Finally, an \doxygen{ImageRegion} object is created and initialized with
		//  the start and size we just prepared using the slice information.
		//
		//  Software Guide : EndLatex

		// Software Guide : BeginCodeSnippet
		ImageType3D::RegionType desiredRegion;
		desiredRegion.SetSize(size);
		desiredRegion.SetIndex(start);
		// Software Guide : EndCodeSnippet


		//  Software Guide : BeginLatex
		//
		//  Then the region is passed to the filter using the
		//  \code{SetExtractionRegion()} method.
		//
		//  \index{itk::ExtractImageFilter!SetExtractionRegion()}
		//
		//  Software Guide : EndLatex


		// Software Guide : BeginCodeSnippet
		filter->SetExtractionRegion(desiredRegion);
		// Software Guide : EndCodeSnippet


		//  Software Guide : BeginLatex
		//
		//  Below we connect the reader, filter and writer to form the data
		//  processing pipeline.
		//
		//  Software Guide : EndLatex

		// Software Guide : BeginCodeSnippet
		ImageType3D::Pointer image;
		image = reader->GetOutput();

		const ImageType3D::IndexType LeftEyeIndex = { { i,j,sliceNumber } };
		ImageType3D::PointType LeftEyePoint;
		image->TransformIndexToPhysicalPoint(LeftEyeIndex, LeftEyePoint);
		// Software Guide : EndCodeSnippet

		std::cout << "===========================================" << std::endl;
		std::cout << "The Left Eye Location is " << LeftEyePoint << std::endl;
		for (int x = i - 3; x <= i + 3; x++) {
			for (int y = j - 3; y <= j + 3; y++) {
				const ImageType3D::IndexType LeftEyeIndex = { { x,j,sliceNumber } };
				image->SetPixel(LeftEyeIndex, 0);
				const ImageType3D::IndexType RightEyeIndex = { { i,y,sliceNumber } };
				image->SetPixel(RightEyeIndex, 0);
			}	
		}
		

		const ImageType3D::DirectionType & ImageDirectionCosines = image->GetDirection();

		std::cout << "===========================================" << std::endl;
		std::cout << "Direction " << ImageDirectionCosines[0] <<','<< ImageDirectionCosines[1]<< ',' << ImageDirectionCosines[2] << std::endl;

		filter->SetInput(image);
		writer->SetInput(filter->GetOutput());


		// Software Guide : EndCodeSnippet

		std::cout << "Writing the image as " << std::endl << std::endl;
		std::cout << argv[2] << std::endl << std::endl;

		try
		{
			// Software Guide : BeginCodeSnippet
			writer->Update();
			// Software Guide : EndCodeSnippet
		}
		catch (itk::ExceptionObject &ex)
		{
			std::cout << ex << std::endl;
			return EXIT_FAILURE;
		}
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}


	return EXIT_SUCCESS;
}