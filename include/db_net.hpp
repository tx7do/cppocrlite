#pragma once

#include "ocr_struct.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "ocr_session.hpp"

class DbNet : public Session
{
public:
	DbNet();
	~DbNet();

public:
	std::vector<TextBox> getTextBoxes(cv::Mat& src, ScaleParam& s, float boxScoreThresh,
		float boxThresh, float unClipRatio);

private:
	static std::vector<TextBox> findRsBoxes(const cv::Mat& fMapMat, const cv::Mat& norfMapMat, ScaleParam& s,
		float boxScoreThresh, float unClipRatio);

private:
	static const float MEAN_VALUES[3];
	static const float NORM_VALUES[3];
};
