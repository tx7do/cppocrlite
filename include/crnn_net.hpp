#pragma once

#include "ocr_struct.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "ocr_session.hpp"

class CrnnNet : public Session
{
public:
	CrnnNet(bool isOutputDebugImg = false);
	~CrnnNet();

public:
	bool loadKeys(const std::string& keysPath);

public:
	std::vector<TextLine> getTextLines(std::vector<cv::Mat>& partImg, const char* path, const char* imgName);

private:
	TextLine scoreToTextLine(const std::vector<float>& outputData, int h, int w);

	TextLine getTextLine(const cv::Mat& src);

private:
	bool _isOutputDebugImg = false;
	std::vector<std::string> _keys;

private:
	static const float MEAN_VALUES[3];
	static const float NORM_VALUES[3];

	static const int DST_HEIGHT;
};
