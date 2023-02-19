#include "ocrlite.hpp"
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>

int main()
{
	std::string modelPath("../../models/");

	std::string
		modelDetPath(modelPath + "dbnet.onnx"),
		modelClsPath(modelPath + "angle_net.onnx"),
		modelRecPath(modelPath + "crnn_lite_lstm.onnx"),
		keysPath(modelPath + "keys.txt");

	std::string imgDir("../_fixtures/2.jpg");

	int numThread = 4;

	int padding = 50;
	int maxSideLen = 1024;
	float boxScoreThresh = 0.6f;
	float boxThresh = 0.3f;
	float unClipRatio = 2.6f;
	bool doAngle = true;
	bool mostAngle = true;

	OcrLite ocrLite;
	ocrLite.setNumThread(numThread);
	ocrLite.initLogger(true, false, true);

	ocrLite.enableResultTxt("./", "memory.jpg");
	ocrLite.log("=====Input Params=====\n");
	ocrLite.log(
		"numThread(%d),padding(%d),maxSideLen(%d),boxScoreThresh(%f),boxThresh(%f),unClipRatio(%f),doAngle(%d),mostAngle(%d)\n",
		numThread,
		padding,
		maxSideLen,
		boxScoreThresh,
		boxThresh,
		unClipRatio,
		doAngle,
		mostAngle);

	ocrLite.initModels(modelDetPath, modelClsPath, modelRecPath, keysPath);

	std::ifstream ifs;
	ifs.open(imgDir, std::ios::in);

	if (!ifs.is_open())
	{
		ocrLite.log("open file failed\n");
		return 1;
	}

	ifs.seekg(0, std::ios::end);
	auto length = ifs.tellg();
	ifs.seekg(0, std::ios::beg);

	std::vector<char> buff;
	buff.resize(length);
	ifs.read(buff.data(), length);
	ifs.close();

	cv::Mat matImg;
	try
	{
		matImg = cv::imdecode(cv::Mat(buff), CV_LOAD_IMAGE_UNCHANGED);
	}
	catch (std::exception& e)
	{
		ocrLite.log("decode image error: %s\n", e.what());
		return 2;
	}

	OcrResult result = ocrLite.detect(matImg, padding, maxSideLen,
		boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
	ocrLite.log("%s\n", result.strRes.c_str());

	return 0;
}
