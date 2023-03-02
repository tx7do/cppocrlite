#include "ocrlite.hpp"


#include <onnxruntime_cxx_api.h>

#include "ocr_utils.hpp"
#include "db_net.hpp"
#include "angle_net.hpp"
#include "crnn_net.hpp"
#include "ocr_logger.hpp"

OcrLite::OcrLite()
	: _logger(new OcrLogger), _dbNet(new DbNet), _angleNet(new AngleNet), _crnnNet(new CrnnNet)
{
}

OcrLite::~OcrLite()
{
	if (_dbNet != nullptr)
	{
		delete _dbNet;
		_dbNet = nullptr;
	}
	if (_angleNet != nullptr)
	{
		delete _angleNet;
		_angleNet = nullptr;
	}
	if (_crnnNet != nullptr)
	{
		delete _crnnNet;
		_crnnNet = nullptr;
	}
	if (_logger != nullptr)
	{
		delete _logger;
		_logger = nullptr;
	}
}

void OcrLite::setNumThread(int numOfThread)
{
	if (_dbNet) _dbNet->setNumThread(numOfThread);
	if (_angleNet) _angleNet->setNumThread(numOfThread);
	if (_crnnNet) _crnnNet->setNumThread(numOfThread);
}

void OcrLite::initLogger(bool isOutputConsole, bool isOutputPartImg, bool isOutputResultImg)
{
	enableConsoleLogger(isOutputConsole);
	enableOutputPartImage(isOutputPartImg);
	enableOutputResultImage(isOutputResultImg);
}

void OcrLite::enableOutputPartImage(bool enable)
{
	_isOutputPartImg = enable;
}

void OcrLite::enableOutputResultImage(bool enable)
{
	_isOutputResultImg = enable;
}

void OcrLite::enableConsoleLogger(bool enable)
{
	if (_logger)
	{
		_logger->enableOutputConsole(enable);
	}
}

void OcrLite::enableFileLogger(const char* path, const char* imgName)
{
	if (_logger)
	{
		auto resultTxtPath = getResultTxtFilePath(path, imgName);
		_logger->startLog(resultTxtPath);
		_logger->log("resultTxtPath(%s)", resultTxtPath.c_str());
	}
}

void OcrLite::log(const char* format, ...)
{
	if (!_logger->isEnable())
	{
		return;
	}

	char* buffer = (char*)malloc(8192);
	va_list args;
	va_start(args, format);
	vsprintf(buffer, format, args);
	va_end(args);

	_logger->logString(buffer);

	free(buffer);
}

bool OcrLite::initModels(const std::string& detPath, const std::string& clsPath,
	const std::string& recPath, const std::string& keysPath)
{
	if (_dbNet == nullptr || _angleNet == nullptr || _crnnNet == nullptr)
	{
		return false;
	}

	_logger->log("=====Init Models=====");

	_logger->log("--- Init DbNet ---");
	if (!_dbNet->initModel(detPath))
	{
		return false;
	}

	_logger->log("--- Init AngleNet ---");
	if (!_angleNet->initModel(clsPath))
	{
		return false;
	}

	_logger->log("--- Init CrnnNet ---");
	if (!_crnnNet->initModel(recPath))
	{
		return false;
	}
	if (!_crnnNet->loadKeys(keysPath))
	{
		return false;
	}

	_logger->log("Init Models Success!");

	return true;
}

cv::Mat makePadding(cv::Mat& src, const int padding)
{
	if (padding <= 0) return src;
	cv::Scalar paddingScalar = { 255, 255, 255 };
	cv::Mat paddingSrc;
	cv::copyMakeBorder(src, paddingSrc, padding, padding, padding, padding, cv::BORDER_ISOLATED, paddingScalar);
	return paddingSrc;
}

OcrResult OcrLite::detect(const char* path, const char* imgName,
	const int padding, const int maxSideLen,
	float boxScoreThresh, float boxThresh, float unClipRatio,
	bool doAngle, bool mostAngle)
{
	std::string imgFile = getSrcImgFilePath(path, imgName);

	cv::Mat bgrSrc = imread(imgFile, cv::IMREAD_COLOR);//default : BGR
	cv::Mat originSrc;
	cvtColor(bgrSrc, originSrc, cv::COLOR_BGR2RGB);// convert to RGB
	int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
	int resize;
	if (maxSideLen <= 0 || maxSideLen > originMaxSide)
	{
		resize = originMaxSide;
	}
	else
	{
		resize = maxSideLen;
	}
	resize += 2 * padding;
	cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
	cv::Mat paddingSrc = makePadding(originSrc, padding);
	ScaleParam scale = getScaleParam(paddingSrc, resize);
	OcrResult result;
	result = detect(path, imgName, paddingSrc, paddingRect, scale,
		boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
	return result;
}

std::vector<cv::Mat> OcrLite::getPartImages(cv::Mat& src, std::vector<TextBox>& textBoxes,
	const char* path, const char* imgName) const
{
	std::vector<cv::Mat> partImages;
	for (size_t i = 0; i < textBoxes.size(); ++i)
	{
		cv::Mat partImg = getRotateCropImage(src, textBoxes[i].boxPoint);
		partImages.emplace_back(partImg);
		//OutPut DebugImg
		if (_isOutputPartImg)
		{
			std::string debugImgFile = getDebugImgFilePath(path, imgName, (int)i, "-part-");
			saveImg(partImg, debugImgFile.c_str());
		}
	}
	return partImages;
}

OcrResult OcrLite::detect(const char* path, const char* imgName,
	cv::Mat& src, cv::Rect& originRect, ScaleParam& scale,
	float boxScoreThresh, float boxThresh, float unClipRatio,
	bool doAngle, bool mostAngle)
{
	cv::Mat textBoxPaddingImg = src.clone();
	int thickness = getThickness(src);

	_logger->log("=====Start detect=====");
	_logger->log("ScaleParam(sw:%d,sh:%d,dw:%d,dh:%d,%f,%f)",
		scale.srcWidth, scale.srcHeight,
		scale.dstWidth, scale.dstHeight,
		scale.ratioWidth, scale.ratioHeight);

	_logger->log("---------- step: dbNet getTextBoxes ----------");
	double startTime = getCurrentTime();
	std::vector<TextBox> textBoxes = _dbNet->getTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
	double endDbNetTime = getCurrentTime();
	double dbNetTime = endDbNetTime - startTime;
	_logger->log("dbNetTime(%fms)", dbNetTime);

	for (size_t i = 0; i < textBoxes.size(); ++i)
	{
		_logger
			->log("TextBox[%d](+padding)[score(%f),[x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d]]",
				i,
				textBoxes[i].score,
				textBoxes[i].boxPoint[0].x,
				textBoxes[i].boxPoint[0].y,
				textBoxes[i].boxPoint[1].x,
				textBoxes[i].boxPoint[1].y,
				textBoxes[i].boxPoint[2].x,
				textBoxes[i].boxPoint[2].y,
				textBoxes[i].boxPoint[3].x,
				textBoxes[i].boxPoint[3].y);
	}

	_logger->log("---------- step: drawTextBoxes ----------");
	drawTextBoxes(textBoxPaddingImg, textBoxes, thickness);

	//---------- getPartImages ----------
	std::vector<cv::Mat> partImages = getPartImages(src, textBoxes, path, imgName);

	_logger->log("---------- step: angleNet getAngles ----------");
	std::vector<Angle> angles;
	angles = _angleNet->getAngles(partImages, path, imgName, doAngle, mostAngle);

	//Log Angles
	for (size_t i = 0; i < angles.size(); ++i)
	{
		_logger
			->log("angle[%d][index(%d), score(%f), time(%fms)]", i, angles[i].index, angles[i].score, angles[i].time);
	}

	//Rotate partImgs
	for (size_t i = 0; i < partImages.size(); ++i)
	{
		if (angles[i].index == 0)
		{
			partImages.at(i) = matRotateClockWise180(partImages[i]);
		}
	}

	_logger->log("---------- step: crnnNet getTextLine ----------");
	std::vector<TextLine> textLines = _crnnNet->getTextLines(partImages, path, imgName);
	//Log TextLines
	for (size_t i = 0; i < textLines.size(); ++i)
	{
		_logger->log("textLine[%d](%s)", i, textLines[i].text.c_str());
		std::ostringstream txtScores;
		for (size_t s = 0; s < textLines[i].charScores.size(); ++s)
		{
			if (s == 0)
			{
				txtScores << textLines[i].charScores[s];
			}
			else
			{
				txtScores << " ," << textLines[i].charScores[s];
			}
		}
		_logger->log("textScores[%d]{%s}", i, std::string(txtScores.str()).c_str());
		_logger->log("crnnTime[%d](%fms)", i, textLines[i].time);
	}

	std::vector<TextBlock> textBlocks;
	for (size_t i = 0; i < textLines.size(); ++i)
	{
		std::vector<cv::Point> boxPoint = std::vector<cv::Point>(4);
		int padding = originRect.x;//padding conversion
		boxPoint[0] = cv::Point(textBoxes[i].boxPoint[0].x - padding, textBoxes[i].boxPoint[0].y - padding);
		boxPoint[1] = cv::Point(textBoxes[i].boxPoint[1].x - padding, textBoxes[i].boxPoint[1].y - padding);
		boxPoint[2] = cv::Point(textBoxes[i].boxPoint[2].x - padding, textBoxes[i].boxPoint[2].y - padding);
		boxPoint[3] = cv::Point(textBoxes[i].boxPoint[3].x - padding, textBoxes[i].boxPoint[3].y - padding);
		TextBlock textBlock{
			angles[i].time, textLines[i].time, angles[i].time + textLines[i].time,
			boxPoint, textBoxes[i].score,
			angles[i].index, angles[i].score,
			textLines[i].charScores, textLines[i].text,
		};
		textBlocks.emplace_back(textBlock);
	}

	double endTime = getCurrentTime();
	double fullTime = endTime - startTime;
	_logger->log("=====End detect=====");
	_logger->log("FullDetectTime(%fms)", fullTime);

	//cropped to original size
	cv::Mat rgbBoxImg, textBoxImg;

	if (originRect.x > 0 && originRect.y > 0)
	{
		textBoxPaddingImg(originRect).copyTo(rgbBoxImg);
	}
	else
	{
		rgbBoxImg = textBoxPaddingImg;
	}
	cvtColor(rgbBoxImg, textBoxImg, cv::COLOR_RGB2BGR);//convert to BGR for Output Result Img

	//Save result.jpg
	if (_isOutputResultImg)
	{
		auto resultImgFile = getResultImgFilePath(path, imgName);
		saveImg(textBoxImg, resultImgFile.c_str());
	}

	std::string strRes;
	for (auto& textBlock : textBlocks)
	{
		strRes.append(textBlock.text);
		strRes.append("\n");
	}

	return OcrResult{ dbNetTime, fullTime, textBlocks, textBoxImg, strRes };
}

OcrResult OcrLite::detect(const cv::Mat& mat,
	int padding, int maxSideLen,
	float boxScoreThresh, float boxThresh, float unClipRatio,
	bool doAngle, bool mostAngle)
{
	cv::Mat originSrc;
	cvtColor(mat, originSrc, cv::COLOR_BGR2RGB);// convert to RGB
	int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
	int resize;
	if (maxSideLen <= 0 || maxSideLen > originMaxSide)
	{
		resize = originMaxSide;
	}
	else
	{
		resize = maxSideLen;
	}
	resize += 2 * padding;
	cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
	cv::Mat paddingSrc = makePadding(originSrc, padding);
	ScaleParam scale = getScaleParam(paddingSrc, resize);
	OcrResult result;
	result = detect(nullptr, nullptr, paddingSrc, paddingRect, scale,
		boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
	return result;
}
