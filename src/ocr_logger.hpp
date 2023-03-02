#pragma once

#include <fstream>
#include <mutex>

class OcrLogger
{
public:
	OcrLogger();
	~OcrLogger();

public:
	bool startLog(const std::string& strLoggerFile);

public:
	void enableOutputConsole(bool enable);
	void enableOutputFile(bool enable);

	bool isEnable() const;

public:
	void log(const char* format, ...);
	void logString(const char* str);

private:
	std::ofstream _log_file;

	bool _enableOutputConsole{ false };
	bool _enableOutputFile{ false };

	std::mutex _mutex;
};
