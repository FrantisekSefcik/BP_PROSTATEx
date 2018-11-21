

#include <stdio.h>

#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"

#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"

#include "itkMetaDataObjectBase.h"
// Software Guide : EndCodeSnippet

typedef unsigned short    PixelType;
typedef unsigned short    PixelTypePNG;
const unsigned int      Dimension3D = 3;
const unsigned int      Dimension2D = 2;
typedef itk::Image< PixelType, Dimension3D >         ImageType3D;
typedef itk::Image< PixelType, Dimension2D >         ImageType2D;
typedef itk::Image< PixelTypePNG, Dimension2D >         ImageTypePNG;
typedef itk::ImageSeriesReader< ImageType3D >        ReaderType;


// Software Guide : EndCodeSnippet
std::string FindDicomTag(const std::string & entryId, const itk::GDCMImageIO::Pointer dicomIO)
{
	std::string tagvalue;
	bool found = dicomIO->GetValueFromTag(entryId, tagvalue);
	if (!found)
	{
		tagvalue = "NOT FOUND";
	}
	return tagvalue;
}


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
		std::cerr << argv[0] << " DicomDirectory  outputFileName  xCoordinate yCoordinate zCoordinate degrees scale_x scale_y"
			<< std::endl;
		return EXIT_FAILURE;
	}
	try {

		// Software Guide : BeginCodeSnippet
		typedef itk::ImageFileWriter< ImageType2D > WriterType;
		WriterType::Pointer writer = WriterType::New();

		writer->SetFileName(argv[2]);

		// nacitanie dicom serie
		ReaderType::Pointer reader = ReaderType::New();
		reader = readDicomSeri(argv[1]);

		typedef itk::Image< signed short, 3 >         ImageTypeMY;

		ImageType3D::Pointer imageMy = reader->GetOutput();
		ImageType3D::IndexType pixelIndex;
		ImageType3D::PointType pixelPoint;
		float x = atof(argv[3]);
		float y = atof(argv[4]);
		float z = atof(argv[5]);
		float region = atof(argv[6]);
		float scale_x = atof(argv[7]);
		float scale_y = atof(argv[8]);

		int region_sizeX = region / scale_x;
		int region_sizeY = region / scale_y;

		pixelPoint[0] = x;
		pixelPoint[1] = y;
		pixelPoint[2] = z;

		imageMy->TransformPhysicalPointToIndex(pixelPoint, pixelIndex);

		std::cout << "xindex = " << pixelIndex[0] << std::endl;
		std::cout << "Yindex = " << pixelIndex[1] << std::endl;
		std::cout << "zIndex = " << pixelIndex[2] << std::endl;

		std::cout << "region x = " << region_sizeX << std::endl;
		std::cout << "region z = " << region_sizeY << std::endl;
		
		

		itk::ImageIOBase::IOPixelType pixelType
			= reader->GetImageIO()->GetPixelType();
		itk::ImageIOBase::IOComponentType componentType
			= reader->GetImageIO()->GetComponentType();
		std::cout << "PixelType: " << reader->GetImageIO()
			->GetPixelTypeAsString(pixelType) << std::endl;
		std::cout << "Component Type: " << reader->GetImageIO()
			->GetComponentTypeAsString(componentType) << std::endl;

		//FILTER//////Extract one slice from DICOM series///////////////////////////////////
		typedef itk::ExtractImageFilter< ImageType3D,
			ImageType2D > FilterType;
		FilterType::Pointer filter = FilterType::New();
		filter->InPlaceOn();
		filter->SetDirectionCollapseToSubmatrix();


		reader->UpdateOutputInformation();
		ImageType3D::RegionType inputRegion =
			reader->GetOutput()->GetLargestPossibleRegion();

		ImageType3D::SizeType size = inputRegion.GetSize();
		size[0] = region_sizeX;
		size[1] = region_sizeY;
		size[2] = 0;

		ImageType3D::IndexType start = inputRegion.GetIndex();
		const unsigned int sliceNumber = atoi(argv[5]);
		start[0] = pixelIndex[0] - (region_sizeX / 2);
		start[1] = pixelIndex[1] - (region_sizeY / 2);
		start[2] = pixelIndex[2];

		ImageType3D::RegionType desiredRegion;
		desiredRegion.SetSize(size);
		desiredRegion.SetIndex(start);

		filter->SetExtractionRegion(desiredRegion);

		ImageType2D::Pointer image;

		filter->SetInput(reader->GetOutput());
		////////////////////////////

		//////rotate image



		//typedef itk::ResampleImageFilter<
		//	ImageType2D, ImageType2D >  FilterRotateType;

		//FilterRotateType::Pointer rotateFilter = FilterRotateType::New();



		//typedef itk::AffineTransform< double, Dimension2D >  TransformType;
		//TransformType::Pointer transform = TransformType::New();
		//// Software Guide : EndCodeSnippet


		//typedef itk::LinearInterpolateImageFunction<
		//	ImageType2D, double >  InterpolatorType;
		//InterpolatorType::Pointer interpolator = InterpolatorType::New();

		//rotateFilter->SetInterpolator(interpolator);

		//rotateFilter->SetDefaultPixelValue(100);

		//filter->Update();
		//const ImageType2D * inputImage = filter->GetOutput();

		//const ImageType2D::SpacingType & spacing = inputImage->GetSpacing();
		//const ImageType2D::PointType & origin = inputImage->GetOrigin();
		//ImageType2D::SizeType size2 =
		//	inputImage->GetLargestPossibleRegion().GetSize();

		//std::cout << "imageCenterX = " << size2[0] << std::endl;
		//std::cout << "imageCenterY = " << size2[1] << std::endl;

		//rotateFilter->SetOutputOrigin(origin);
		//rotateFilter->SetOutputSpacing(spacing);
		//rotateFilter->SetOutputDirection(inputImage->GetDirection());
		//rotateFilter->SetSize(size2);
		//// Software Guide : EndCodeSnippet


		//rotateFilter->SetInput(filter->GetOutput());



		////  Software Guide : BeginLatex
		////
		////  Rotations are performed around the origin of physical coordinates---not
		////  the image origin nor the image center. Hence, the process of
		////  positioning the output image frame as it is shown in Figure
		////  \ref{fig:ResampleImageFilterOutput10} requires three steps.  First, the
		////  image origin must be moved to the origin of the coordinate system. This
		////  is done by applying a translation equal to the negative values of the
		////  image origin.
		////
		//// \begin{figure}
		//// \center
		//// \includegraphics[width=0.44\textwidth]{BrainProtonDensitySliceBorder20}
		//// \includegraphics[width=0.44\textwidth]{ResampleImageFilterOutput10}
		//// \itkcaption[Effect of the Resample filter rotating an image]{Effect of the
		//// resample filter rotating an image.}
		//// \label{fig:ResampleImageFilterOutput10}
		//// \end{figure}
		////
		////
		////  \index{itk::AffineTransform!Translate()}
		////
		////  Software Guide : EndLatex

		//// Software Guide : BeginCodeSnippet




		////TransformType::OutputVectorType translation1;
		//float i = atof(argv[3]);
		//float j = atof(argv[4]);
		////const double imageCenterX = origin[0] + spacing[0] * (size2[0] - i);
		////const double imageCenterY = origin[1] + spacing[1] * (size2[1] - j);

		////translation1[0] = -imageCenterX;
		////translation1[1] = -imageCenterY;

		////transform->Translate(translation1);
		//// Software Guide : EndCodeSnippet

		//TransformType::InputPointType rotationCenter;
		//rotationCenter[0] = i;
		//rotationCenter[1] = j;
		//std::cerr << origin[0] << " origin " << origin[1] << ',' << std::endl;
		//std::cerr << spacing[0] << " origin " << spacing[1] << ',' << std::endl;
		//transform->SetCenter(rotationCenter);

		//std::cout << "imageCenterX = " << rotationCenter[0] << std::endl;
		//std::cout << "imageCenterY = " << rotationCenter[1] << std::endl;


		//const double angleInDegrees = atof(argv[6]);
		//// Software Guide : BeginCodeSnippet
		//const double degreesToRadians = std::atan(1.0) / 45.0;
		//
		//// Software Guide : EndCodeSnippet








		//typedef itk::RegionOfInterestImageFilter< ImageType2D,
		//	ImageType2D > FilterExtractType;

		//FilterExtractType::Pointer extractFilter = FilterExtractType::New();
		//// Software Guide : EndCodeSnippet


		////  Software Guide : BeginLatex
		////
		////  The RegionOfInterestImageFilter requires a region to be
		////  defined by the user. The region is specified by an \doxygen{Index}
		////  indicating the pixel where the region starts and an \doxygen{Size}
		////  indicating how many pixels the region has along each dimension. In this
		////  example, the specification of the region is taken from the command line
		////  arguments (this example assumes that a 2D image is being processed).
		////
		////  Software Guide : EndLatex

		//int a = (i - origin[0]) / spacing[0];
		//int b = (j - origin[1]) / spacing[1];


		//std::cout << "CenterX = " << a << std::endl;
		//std::cout << "CenterY = " << b << std::endl;
		//// Software Guide : BeginCodeSnippet
		//ImageType2D::IndexType start3;
		//start3[0] = a - 40;
		//start3[1] = b - 40;
		//// Software Guide : EndCodeSnippet


		//// Software Guide : BeginCodeSnippet
		//ImageType2D::SizeType size3;
		//size3[0] = 80;
		//size3[1] = 80;
		//// Software Guide : EndCodeSnippet




		//// Software Guide : BeginCodeSnippet
		//ImageType2D::RegionType desiredRegionB;
		//desiredRegionB.SetSize(size3);
		//desiredRegionB.SetIndex(start3);
		//// Software Guide : EndCodeSnippet




		//// Software Guide : BeginCodeSnippet
		//extractFilter->SetRegionOfInterest(desiredRegionB);
		//// Software Guide : EndCodeSnippet


		//for (int i = 1; i < 3; i++) {

		//	const double angle = angleInDegrees * degreesToRadians * i;
		//	transform->Rotate2D(angle);
		//	// Software Guide : EndCodeSnippet


		//	// Software Guide : BeginCodeSnippet
		//	//TransformType::OutputVectorType translation2;
		//	//translation2[0] = imageCenterX;
		//	//translation2[1] = imageCenterY;
		//	//transform->Translate(translation2, false);
		//	rotateFilter->SetTransform(transform);

		//	extractFilter->SetInput(rotateFilter->GetOutput());


		//	writer->SetInput(extractFilter->GetOutput());


		//	// Software Guide : EndCodeSnippet

		//	std::cout << "Writing the image as " << std::endl << std::endl;
		//	std::cout << argv[2] << std::endl << std::endl;
            
			writer->SetInput(filter->GetOutput());

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
		
		//}

		
	}
	catch (itk::ExceptionObject &ex)
	{
		std::cout << ex << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}