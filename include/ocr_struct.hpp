#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include "ocrlite_port.hpp"

struct OCRLITE_PORT ScaleParam
{
	int srcWidth;
	int srcHeight;

	int dstWidth;
	int dstHeight;

	float ratioWidth;
	float ratioHeight;
};

struct OCRLITE_PORT TextBox
{
	std::vector<cv::Point> boxPoint;
	float score;
};

struct OCRLITE_PORT Angle
{
	int index;
	float score;
	double time;
};

struct OCRLITE_PORT TextLine
{
	std::string text;
	std::vector<float> charScores;
	double time;
};

struct OCRLITE_PORT TextBlock
{
	double angleTime;
	double crnnTime;
	double blockTime;

	std::vector<cv::Point> boxPoint;
	float boxScore;

	int angleIndex;
	float angleScore;

	std::vector<float> charScores;

	std::string text;
};

struct OCRLITE_PORT OcrResult
{
	double dbNetTime;
	double detectTime;

	std::vector<TextBlock> textBlocks;
	cv::Mat boxImg;

	std::string strRes;
};
