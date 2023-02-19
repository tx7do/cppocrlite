#include "ocr_session.hpp"
#include <iostream>

Session::Session(const std::string& sessionName, OrtLoggingLevel loggingLevel)
	: _env(loggingLevel, sessionName.c_str())
{
}

Session::~Session()
{
	clearSession();
}

void Session::setNumThread(int numOfThread)
{
	//===session options===
	// Sets the number of threads used to parallelize the execution within nodes
	// A value of 0 means ORT will pick a default
	//sessionOptions.SetIntraOpNumThreads(numThread);
	//set OMP_NUM_THREADS=16

	// Sets the number of threads used to parallelize the execution of the graph (across nodes)
	// If sequential execution is enabled this value is ignored
	// A value of 0 means ORT will pick a default
	_sessionOptions.SetInterOpNumThreads(numOfThread);

	// Sets graph optimization level
	// ORT_DISABLE_ALL -> To disable all optimizations
	// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
	// ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
	// ORT_ENABLE_ALL -> To Enable All possible optimizations
	_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

bool Session::initModel(const std::string& modelFilePath)
{
	clearSession();

	_session = createSession(modelFilePath);
	assert(_session);
	if (_session == nullptr)
	{
		return false;
	}

	readInputNames();
	readOutputNames();

	return true;
}

std::vector<Ort::Value>& Session::run(const Ort::Value* inputValues, size_t inputCount, size_t outputCount)
{
	_outputTensor.clear();
	_outputTensor.reserve(outputCount);
	for (size_t i = 0; i < outputCount; ++i)
	{
		_outputTensor.emplace_back(nullptr);
	}

	_session->Run(Ort::RunOptions{ nullptr },
		_inputNodeNames.data(),
		inputValues,
		inputCount,
		_outputNodeNames.data(),
		_outputTensor.data(),
		outputCount);

	return _outputTensor;
}

void Session::clearSession()
{
	if (_session != nullptr)
	{
		delete _session;
		_session = nullptr;
	}
}

Ort::Session* Session::createSession(const std::string& modelFilePath)
{
	try
	{
#ifdef _WIN32
		std::wstring wstrPath = strToWstr(modelFilePath);
		return new Ort::Session(_env, wstrPath.c_str(), _sessionOptions);
#else
		return new Ort::Session(_env, modelFilePath.c_str(), _sessionOptions);
#endif
	}
	catch (std::exception& ex)
	{
		return nullptr;
	}
}

void Session::readInputNames()
{
	assert(_session);
	if (_session == nullptr)
	{
		return;
	}

	_inputNamePtrs.clear();
	_inputNodeNames.clear();

	size_t numInputNodes = _session->GetInputCount();
	if (numInputNodes < 1)
	{
		return;
	}

	_inputNamePtrs.reserve(numInputNodes);
	_inputNodeNames.reserve(numInputNodes);
	for (size_t i = 0; i < numInputNodes; ++i)
	{
		auto inputName = _session->GetInputNameAllocated(i, _allocator);
		// std::cout << "Input " << i << " : name =" << inputName.get() << std::endl;
		_inputNodeNames.push_back(inputName.get());
		_inputNamePtrs.push_back(std::move(inputName));
	}
}

void Session::readOutputNames()
{
	assert(_session);
	if (_session == nullptr)
	{
		return;
	}

	_outputNamePtrs.clear();
	_outputNodeNames.clear();

	size_t numOutputNodes = _session->GetOutputCount();
	if (numOutputNodes < 1)
	{
		return;
	}

	_outputNamePtrs.reserve(numOutputNodes);
	_outputNodeNames.reserve(numOutputNodes);
	for (size_t i = 0; i < numOutputNodes; ++i)
	{
		auto outputName = _session->GetOutputNameAllocated(i, _allocator);
		// std::cout << "Output " << i << " : name =" << outputName.get() << std::endl;
		_outputNodeNames.push_back(outputName.get());
		_outputNamePtrs.push_back(std::move(outputName));
	}
}
