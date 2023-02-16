#include <cstdio>
#include "ocrlite.hpp"

int main(int argc, char** argv)
{
	std::string
		modelsDir("../models/"),
		modelDetPath("dbnet.onnx"),
		modelClsPath("angle_net.onnx"),
		modelRecPath("crnn_lite_lstm.onnx"),
		keysPath("keys.txt");

	std::string
		imgDir("_fixtures/"),
		imgName("1.jpg");

	int numThread = 4;

	int padding = 50;
	int maxSideLen = 1024;
	float boxScoreThresh = 0.6f;
	float boxThresh = 0.3f;
	float unClipRatio = 2.0f;
	bool doAngle = true;
	bool mostAngle = true;

	OcrLite ocrLite;
	ocrLite.setNumThread(numThread);
	ocrLite.initLogger(
		true,//isOutputConsole
		false,//isOutputPartImg
		true);//isOutputResultImg

	ocrLite.enableResultTxt(imgDir.c_str(), imgName.c_str());
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

	OcrResult result = ocrLite.detect(imgDir.c_str(), imgName.c_str(), padding, maxSideLen,
		boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
	ocrLite.log("%s\n", result.strRes.c_str());

	return 0;
}
