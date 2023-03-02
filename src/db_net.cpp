#include "db_net.hpp"
#include "ocr_utils.hpp"
#include <onnxruntime_cxx_api.h>

const float DbNet::MEAN_VALUES[3] = { 0.485f * 255, 0.456f * 255, 0.406f * 255 };

const float DbNet::NORM_VALUES[3] = { 1.0f / 0.229f / 255.0f, 1.0f / 0.224f / 255.0f, 1.0f / 0.225f / 255.0f };

DbNet::DbNet()
	: Session("DbNet", ORT_LOGGING_LEVEL_ERROR)
{
}

DbNet::~DbNet() = default;

std::vector<TextBox>
DbNet::getTextBoxes(cv::Mat& src, ScaleParam& s, float boxScoreThresh, float boxThresh, float unClipRatio)
{
	cv::Mat srcResize;
	resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));

	std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, MEAN_VALUES, NORM_VALUES);
	std::array<int64_t, 4> inputShape{ 1, srcResize.channels(), srcResize.rows, srcResize.cols };

	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
		inputTensorValues.size(), inputShape.data(),
		inputShape.size());
	assert(inputTensor.IsTensor());

	auto& outputTensor = Session::run(&inputTensor, 1, 1);
	assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

	std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
	int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1,
		std::multiplies<int64_t>());

	auto* floatArray = outputTensor.front().GetTensorMutableData<float>();

	//-----Data preparation-----
	cv::Mat fMapMat(srcResize.rows, srcResize.cols, CV_32FC1);
	memcpy(fMapMat.data, floatArray, outputCount * sizeof(float));

	//-----boxThresh-----
	cv::Mat norfMapMat;
	norfMapMat = fMapMat > boxThresh;

	return findRsBoxes(fMapMat, norfMapMat, s, boxScoreThresh, unClipRatio);
}

std::vector<TextBox> DbNet::findRsBoxes(const cv::Mat& fMapMat, const cv::Mat& norfMapMat,
	ScaleParam& s,
	const float boxScoreThresh, const float unClipRatio)
{
	float minArea = 3;
	std::vector<TextBox> rsBoxes;
	rsBoxes.clear();
	std::vector<std::vector<cv::Point>> contours;
	findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	for (const auto& contour : contours)
	{
		float minSideLen, perimeter;
		auto minBox = getMinBoxes(contour, minSideLen, perimeter);
		if (minSideLen < minArea)
		{
			continue;
		}
		float score = boxScoreFast(fMapMat, contour);
		if (score < boxScoreThresh)
		{
			continue;
		}
		//---use clipper start---
		auto clipBox = unClip(minBox, perimeter, unClipRatio);
		auto clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);
		//---use clipper end---

		if (minSideLen < minArea + 2)
		{
			continue;
		}

		for (auto& j : clipMinBox)
		{
			j.x = (j.x / s.ratioWidth);
			j.x = (std::min)((std::max)(j.x, 0), s.srcWidth);

			j.y = (j.y / s.ratioHeight);
			j.y = (std::min)((std::max)(j.y, 0), s.srcHeight);
		}

		rsBoxes.emplace_back(TextBox{ clipMinBox, score });
	}
	reverse(rsBoxes.begin(), rsBoxes.end());
	return rsBoxes;
}
