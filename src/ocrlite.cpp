#include "ocrlite.hpp"
#include "ocr_utils.hpp"
#include <cstdarg> // windows & linux

OcrLite::OcrLite() = default;

OcrLite::~OcrLite()
{
	if (_resultTxt != nullptr)
	{
		fclose(_resultTxt);
		_resultTxt = nullptr;
	}
}

void OcrLite::setNumThread(int numOfThread)
{
	_dbNet.setNumThread(numOfThread);
	_angleNet.setNumThread(numOfThread);
	_crnnNet.setNumThread(numOfThread);
}

void OcrLite::initLogger(bool isOutputConsole, bool isOutputPartImg, bool isOutputResultImg)
{
	_isOutputConsole = isOutputConsole;
	_isOutputPartImg = isOutputPartImg;
	_isOutputResultImg = isOutputResultImg;
}

void OcrLite::enableResultTxt(const char* path, const char* imgName)
{
	_isOutputResultTxt = true;
	std::string resultTxtPath = getResultTxtFilePath(path, imgName);
	log("resultTxtPath(%s)\n", resultTxtPath.c_str());
	_resultTxt = fopen(resultTxtPath.c_str(), "w");
}

bool OcrLite::initModels(const std::string& detPath, const std::string& clsPath,
	const std::string& recPath, const std::string& keysPath)
{
	log("=====Init Models=====\n");

	log("--- Init DbNet ---\n");
	if (!_dbNet.initModel(detPath))
	{
		return false;
	}

	log("--- Init AngleNet ---\n");
	if (!_angleNet.initModel(clsPath))
	{
		return false;
	}

	log("--- Init CrnnNet ---\n");
	if (!_crnnNet.initModel(recPath))
	{
		return false;
	}
	if (!_crnnNet.loadKeys(keysPath))
	{
		return false;
	}

	log("Init Models Success!\n");

	return true;
}

void OcrLite::log(const char* format, ...)
{
	if (!(_isOutputConsole || _isOutputResultTxt)) return;
	char* buffer = (char*)malloc(8192);
	va_list args;
	va_start(args, format);
	vsprintf(buffer, format, args);
	va_end(args);
	if (_isOutputConsole) printf("%s", buffer);
	if (_isOutputResultTxt && _resultTxt != nullptr) fprintf(_resultTxt, "%s", buffer);
	free(buffer);
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

	log("=====Start detect=====\n");
	log("ScaleParam(sw:%d,sh:%d,dw:%d,dh:%d,%f,%f)\n",
		scale.srcWidth, scale.srcHeight,
		scale.dstWidth, scale.dstHeight,
		scale.ratioWidth, scale.ratioHeight);

	log("---------- step: dbNet getTextBoxes ----------\n");
	double startTime = getCurrentTime();
	std::vector<TextBox> textBoxes = _dbNet.getTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
	double endDbNetTime = getCurrentTime();
	double dbNetTime = endDbNetTime - startTime;
	log("dbNetTime(%fms)\n", dbNetTime);

	for (size_t i = 0; i < textBoxes.size(); ++i)
	{
		log("TextBox[%d](+padding)[score(%f),[x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d]]\n", i,
			textBoxes[i].score,
			textBoxes[i].boxPoint[0].x, textBoxes[i].boxPoint[0].y,
			textBoxes[i].boxPoint[1].x, textBoxes[i].boxPoint[1].y,
			textBoxes[i].boxPoint[2].x, textBoxes[i].boxPoint[2].y,
			textBoxes[i].boxPoint[3].x, textBoxes[i].boxPoint[3].y);
	}

	log("---------- step: drawTextBoxes ----------\n");
	drawTextBoxes(textBoxPaddingImg, textBoxes, thickness);

	//---------- getPartImages ----------
	std::vector<cv::Mat> partImages = getPartImages(src, textBoxes, path, imgName);

	log("---------- step: angleNet getAngles ----------\n");
	std::vector<Angle> angles;
	angles = _angleNet.getAngles(partImages, path, imgName, doAngle, mostAngle);

	//Log Angles
	for (size_t i = 0; i < angles.size(); ++i)
	{
		log("angle[%d][index(%d), score(%f), time(%fms)]\n", i, angles[i].index, angles[i].score, angles[i].time);
	}

	//Rotate partImgs
	for (size_t i = 0; i < partImages.size(); ++i)
	{
		if (angles[i].index == 0)
		{
			partImages.at(i) = matRotateClockWise180(partImages[i]);
		}
	}

	log("---------- step: crnnNet getTextLine ----------\n");
	std::vector<TextLine> textLines = _crnnNet.getTextLines(partImages, path, imgName);
	//Log TextLines
	for (size_t i = 0; i < textLines.size(); ++i)
	{
		log("textLine[%d](%s)\n", i, textLines[i].text.c_str());
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
		log("textScores[%d]{%s}\n", i, std::string(txtScores.str()).c_str());
		log("crnnTime[%d](%fms)\n", i, textLines[i].time);
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
		TextBlock textBlock{ boxPoint, textBoxes[i].score, angles[i].index, angles[i].score,
							 angles[i].time, textLines[i].text, textLines[i].charScores, textLines[i].time,
							 angles[i].time + textLines[i].time };
		textBlocks.emplace_back(textBlock);
	}

	double endTime = getCurrentTime();
	double fullTime = endTime - startTime;
	log("=====End detect=====\n");
	log("FullDetectTime(%fms)\n", fullTime);

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
		std::string resultImgFile = getResultImgFilePath(path, imgName);
		imwrite(resultImgFile, textBoxImg);
	}

	std::string strRes;
	for (auto& textBlock : textBlocks)
	{
		strRes.append(textBlock.text);
		strRes.append("\n");
	}

	return OcrResult{ dbNetTime, textBlocks, textBoxImg, fullTime, strRes };
}
