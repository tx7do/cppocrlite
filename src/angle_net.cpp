#include "angle_net.hpp"
#include "ocr_utils.hpp"
#include <numeric>

const float AngleNet::MEAN_VALUES[3] = { 127.5f, 127.5f, 127.5f };

const float AngleNet::NORM_VALUES[3] = { 1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f };

const int AngleNet::ANGLE_DST_WIDTH = 192;

const int AngleNet::ANGLE_DST_HEIGHT = 32;

Angle scoreToAngle(const std::vector<float>& outputData)
{
	int maxIndex = 0;
	float maxScore = -1000.0f;
	for (size_t i = 0; i < outputData.size(); ++i)
	{
		if (i == 0) maxScore = outputData[i];
		else if (outputData[i] > maxScore)
		{
			maxScore = outputData[i];
			maxIndex = (int)i;
		}
	}
	return { maxIndex, maxScore };
}

AngleNet::AngleNet(bool isOutputAngleImg)
	: Session("AngleNet", ORT_LOGGING_LEVEL_ERROR), _isOutputAngleImg(isOutputAngleImg)
{
}

AngleNet::~AngleNet() = default;

Angle AngleNet::getAngle(cv::Mat& src)
{
	std::vector<float> inputTensorValues = substractMeanNormalize(src, MEAN_VALUES, NORM_VALUES);

	std::array<int64_t, 4> inputShape{ 1, src.channels(), src.rows, src.cols };

	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo,
		inputTensorValues.data(), inputTensorValues.size(),
		inputShape.data(), inputShape.size());
	assert(inputTensor.IsTensor());

	auto& outputTensor = Session::run(&inputTensor, 1, 1);
	assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

	std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

	int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());

	const auto* floatArray = outputTensor.front().GetTensorMutableData<float>();
	std::vector<float> outputData(floatArray, floatArray + outputCount);
	return scoreToAngle(outputData);
}

std::vector<Angle> AngleNet::getAngles(std::vector<cv::Mat>& partImages,
	const char* path, const char* imgName,
	bool doAngle, bool mostAngle)
{
	size_t size = partImages.size();
	std::vector<Angle> angles(size);
	if (doAngle)
	{
		for (size_t i = 0; i < size; ++i)
		{
			double startAngle = getCurrentTime();
			auto angleImg = adjustTargetImg(partImages[i], ANGLE_DST_WIDTH, ANGLE_DST_HEIGHT);
			Angle angle = getAngle(angleImg);
			double endAngle = getCurrentTime();
			angle.time = endAngle - startAngle;

			angles[i] = angle;

			// OutPut AngleImg
			if (_isOutputAngleImg)
			{
				std::string angleImgFile = getDebugImgFilePath(path, imgName, (int)i, "-angle-");
				saveImg(angleImg, angleImgFile.c_str());
			}
		}
	}
	else
	{
		for (size_t i = 0; i < size; ++i)
		{
			angles[i] = Angle{ -1, 0.f };
		}
	}
	//Most Possible AngleIndex
	if (doAngle && mostAngle)
	{
		auto angleIndexes = getAngleIndexes(angles);
		double sum = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
		double halfPercent = (double)angles.size() / 2.0f;
		int mostAngleIndex;
		if (sum < halfPercent)
		{
			//all angle set to 0
			mostAngleIndex = 0;
		}
		else
		{
			//all angle set to 1
			mostAngleIndex = 1;
		}
		printf("Set All Angle to mostAngleIndex(%d)\n", mostAngleIndex);
		for (auto& i : angles)
		{
			Angle angle = i;
			angle.index = mostAngleIndex;
			i = angle;
		}
	}

	return angles;
}
