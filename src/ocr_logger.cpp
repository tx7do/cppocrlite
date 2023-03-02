#include "ocr_logger.hpp"
#include <cstdarg> // windows & linux
#include <iostream>

OcrLogger::OcrLogger() = default;

OcrLogger::~OcrLogger()
{
	if (_log_file.is_open())
	{
		_log_file.close();
	}
}

bool OcrLogger::startLog(const std::string& strLoggerFile)
{
	_log_file.open(strLoggerFile, std::ios::app);
	bool isOpen = _log_file.is_open();
	if (isOpen)
	{
		_enableOutputFile = true;
	}
	return isOpen;
}

void OcrLogger::log(const char* format, ...)
{
	if (!(_enableOutputConsole || _enableOutputFile)) return;

	char* buffer = (char*)malloc(8192);
	va_list args;
	va_start(args, format);
	vsprintf(buffer, format, args);
	va_end(args);

	logString(buffer);

	free(buffer);
}

void OcrLogger::logString(const char* str)
{
	if (!(_enableOutputConsole || _enableOutputFile)) return;

	if (_enableOutputConsole)
	{
		std::unique_lock<std::mutex> lck (_mutex);
		std::cout << str << std::endl;
	}
	if (_enableOutputFile && _log_file.is_open())
	{
		std::unique_lock<std::mutex> lck (_mutex);
		_log_file << str << std::endl << std::flush;
	}
}

void OcrLogger::enableOutputConsole(bool enable)
{
	_enableOutputConsole = enable;
}

void OcrLogger::enableOutputFile(bool enable)
{
	_enableOutputFile = enable;
}

bool OcrLogger::isEnable() const
{
	return _enableOutputConsole || _enableOutputFile;
}
