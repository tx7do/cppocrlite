#pragma once

#include "opencv2/core.hpp"
#include <onnxruntime_cxx_api.h>
#include "ocr_struct.hpp"
#include "db_net.hpp"
#include "angle_net.hpp"
#include "crnn_net.hpp"

class OcrLite
{
public:
	OcrLite();
	~OcrLite();

public:
	void setNumThread(int numOfThread);

	bool initModels(const std::string& detPath, const std::string& clsPath,
		const std::string& recPath, const std::string& keysPath);

public:
	void initLogger(bool isConsole, bool isPartImg, bool isResultImg);

	void enableResultTxt(const char* path, const char* imgName);

	void log(const char* format, ...);

public:
	OcrResult detect(const char* path, const char* imgName,
		int padding, int maxSideLen,
		float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle);

	OcrResult detect(const cv::Mat& mat,
		int padding, int maxSideLen,
		float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle);

private:
	std::vector<cv::Mat> getPartImages(cv::Mat& src, std::vector<TextBox>& textBoxes,
		const char* path, const char* imgName) const;

	OcrResult detect(const char* path, const char* imgName,
		cv::Mat& src, cv::Rect& originRect, ScaleParam& scale,
		float boxScoreThresh = 0.6f, float boxThresh = 0.3f,
		float unClipRatio = 2.0f, bool doAngle = true, bool mostAngle = true);

private:
	bool _isOutputConsole{ false };
	bool _isOutputPartImg{ false };
	bool _isOutputResultTxt{ false };
	bool _isOutputResultImg{ false };

	FILE* _resultTxt{ nullptr };

private:
	DbNet _dbNet;
	AngleNet _angleNet;
	CrnnNet _crnnNet;
};
