#include "crnn_net.hpp"
#include "ocr_utils.hpp"
#include <fstream>
#include <numeric>

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last)
{
	return std::distance(first, std::max_element(first, last));
}

const float CrnnNet::MEAN_VALUES[3] = { 127.5, 127.5, 127.5 };

const float CrnnNet::NORM_VALUES[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };

const int CrnnNet::DST_HEIGHT = 32;

CrnnNet::CrnnNet(bool isOutputDebugImg)
	: Session("CrnnNet", ORT_LOGGING_LEVEL_ERROR), _isOutputDebugImg(isOutputDebugImg)
{
}

CrnnNet::~CrnnNet() = default;

bool CrnnNet::loadKeys(const std::string& keysPath)
{
	_keys.clear();

	std::ifstream inFile;
	inFile.open(keysPath.c_str(), std::ios::in);
	if (!inFile.is_open())
	{
		printf("The keys.txt file was not found\n");
		return false;
	}

	std::string strLine;
	while (getline(inFile, strLine))
	{
		if (strLine.empty()) continue;
		// line中不包括每行的换行符
		_keys.push_back(strLine);
	}

	if (_keys.size() != 5531)
	{
		fprintf(stderr, "missing keys\n");
		return false;
	}
	printf("total keys size(%lu)\n", _keys.size());

	return true;
}

TextLine CrnnNet::scoreToTextLine(const std::vector<float>& outputData, int h, int w)
{
	size_t keySize = _keys.size();
	std::string strRes;
	std::vector<float> scores;
	size_t lastIndex = 0;
	size_t maxIndex;
	float maxValue;

	for (int i = 0; i < h; i++)
	{
		maxIndex = 0;
		maxValue = -1000.f;
		//do softmax
		std::vector<float> exps(w);
		for (int j = 0; j < w; j++)
		{
			float expSingle = exp(outputData[i * w + j]);
			exps.at(j) = expSingle;
		}
		float partition = accumulate(exps.begin(), exps.end(), 0.0);//row sum
		maxIndex = int(argmax(exps.begin(), exps.end()));
		maxValue = float(*std::max_element(exps.begin(), exps.end())) / partition;
		if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex)))
		{
			scores.emplace_back(maxValue);
			strRes.append(_keys[maxIndex - 1]);
		}
		lastIndex = maxIndex;
	}
	return { strRes, scores };
}

TextLine CrnnNet::getTextLine(const cv::Mat& src)
{
	float scale = (float)DST_HEIGHT / (float)src.rows;
	int dstWidth = int((float)src.cols * scale);

	cv::Mat srcResize;
	resize(src, srcResize, cv::Size(dstWidth, DST_HEIGHT));

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
	std::vector<float> outputData(floatArray, floatArray + outputCount);
	return scoreToTextLine(outputData, outputShape[0], outputShape[2]);
}

std::vector<TextLine> CrnnNet::getTextLines(std::vector<cv::Mat>& partImg, const char* path, const char* imgName)
{
	int size = partImg.size();
	std::vector<TextLine> textLines(size);
	for (int i = 0; i < size; ++i)
	{
		//OutPut DebugImg
		if (_isOutputDebugImg)
		{
			std::string debugImgFile = getDebugImgFilePath(path, imgName, i, "-debug-");
			saveImg(partImg[i], debugImgFile.c_str());
		}

		//getTextLine
		double startCrnnTime = getCurrentTime();
		TextLine textLine = getTextLine(partImg[i]);
		double endCrnnTime = getCurrentTime();
		textLine.time = endCrnnTime - startCrnnTime;
		textLines[i] = textLine;
	}
	return textLines;
}
