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
	/// 设置工作线程数
	/// @param [in] numOfThread 线程数
	void setNumThread(int numOfThread);

	/// 初始化模型
	/// @param [in] detPath dbNet模型文件名(含扩展名)
	/// @param [in] clsPath angleNet模型文件名(含扩展名)
	/// @param [in] recPath crnnNet模型文件名(含扩展名)
	/// @param [in] keysPath keys.txt文件名(含扩展名)
	bool initModels(const std::string& detPath, const std::string& clsPath,
		const std::string& recPath, const std::string& keysPath);

public:
	void initLogger(bool isConsole, bool isPartImg, bool isResultImg);

	void enableResultTxt(const char* path, const char* imgName);

	void log(const char* format, ...);

public:
	/// 识别图片（文件）
	/// @param [in] imgPath 目标图片路径，可以相对路径也可以绝对路径。
	/// @param [in] imgName 图片的文件名
	/// @param [in] padding 图像预处理，在图片外周添加白边，用于提升识别率，文字框没有正确框住所有文字时，增加此值。
	/// @param [in] maxSideLen 按图片最长边的长度，此值为0代表不缩放，例：1024，如果图片长边大于1024则把图像整体缩小到1024再进行图像分割计算，如果图片长边小于1024则不缩放，如果图片长边小于32，则缩放到32。
	/// @param [in] boxScoreThresh 文字框置信度阈值，文字框没有正确框住所有文字时，减小此值。
	/// @param [in] boxThresh
	/// @param [in] unClipRatio 单个文字框大小倍率，越大时单个文字框越大。此项与图片的大小相关，越大的图片此值应该越大。
	/// @param [in] doAngle 启用文字方向检测，只有图片倒置的情况下(旋转90~270度的图片)，才需要启用文字方向检测。
	/// @param [in] mostAngle
	OcrResult detect(const char* imgPath, const char* imgName,
		int padding, int maxSideLen,
		float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle);

	/// 识别图片（矩阵）
	/// @param [in] mat 图形矩阵，可以从文件，也可以从内存中读取。
	/// @param [in] padding 图像预处理，在图片外周添加白边，用于提升识别率，文字框没有正确框住所有文字时，增加此值。
	/// @param [in] maxSideLen 按图片最长边的长度，此值为0代表不缩放，例：1024，如果图片长边大于1024则把图像整体缩小到1024再进行图像分割计算，如果图片长边小于1024则不缩放，如果图片长边小于32，则缩放到32。
	/// @param [in] boxScoreThresh 文字框置信度阈值，文字框没有正确框住所有文字时，减小此值。
	/// @param [in] boxThresh
	/// @param [in] unClipRatio 单个文字框大小倍率，越大时单个文字框越大。此项与图片的大小相关，越大的图片此值应该越大。
	/// @param [in] doAngle 启用文字方向检测，只有图片倒置的情况下(旋转90~270度的图片)，才需要启用文字方向检测。
	/// @param [in] mostAngle
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
