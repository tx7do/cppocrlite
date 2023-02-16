#pragma once

#include "ocr_struct.hpp"
#include <onnxruntime_cxx_api.h>


class Session
{
public:
	explicit Session(const std::string& sessionName, OrtLoggingLevel loggingLevel);
	~Session();

public:
	void setNumThread(int numOfThread);

	bool initModel(const std::string& modelFilePath);

protected:
	std::vector<Ort::Value>& run(const Ort::Value* inputValues, size_t inputCount, size_t outputCount);

protected:
	void readInputNames();
	void readOutputNames();

	void clearSession();
	Ort::Session* createSession(const std::string& modelFilePath);

protected:
	Ort::Env _env;

	Ort::Session* _session{ nullptr };
	Ort::SessionOptions _sessionOptions;

protected:
	Ort::AllocatorWithDefaultOptions _allocator;

	std::vector<Ort::AllocatedStringPtr> _inputNamePtrs;
	std::vector<const char*> _inputNodeNames;

	std::vector<Ort::AllocatedStringPtr> _outputNamePtrs;
	std::vector<const char*> _outputNodeNames;

	std::vector<Ort::Value> _outputTensor;
};
