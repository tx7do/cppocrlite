#pragma once

#include <opencv2/opencv.hpp>
#include "ocr_session.hpp"

class AngleNet : public Session
{
public:
	AngleNet(bool isOutputAngleImg = false);
	~AngleNet();

public:
	std::vector<Angle> getAngles(std::vector<cv::Mat>& partImages,
		const char* path, const char* imgName,
		bool doAngle, bool mostAngle);

private:
	Angle getAngle(cv::Mat& src);

private:
	bool _isOutputAngleImg = false;

private:
	static const float MEAN_VALUES[3];
	static const float NORM_VALUES[3];

	static const int ANGLE_DST_WIDTH;
	static const int ANGLE_DST_HEIGHT;
};
