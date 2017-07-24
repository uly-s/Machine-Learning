#pragma once
#include "Neural Net - primitive array.h"



class Trainer
{

protected: // DATA

	// pointer to the neural net for training
	NeuralNet* network;

	// learning rate
	double learningRate;

	// momentum
	double momentum;

	// epochs
	int epoch;
	int maxEpochs;

	// size of patterns and target
	int patternSize;
	int targetSize;

	// accuracy
	double desiredAccuracy;

	// weight changes
	double** deltaInput;
	double** deltaOutput;

	// error gradients
	double* hiddenGradient;
	double* outputGradient;

	// accuracy stats
	double trainingSetAccuracy;
	double validationSetAccuracy;
	double generalizationSetAccuracy;
	double trainingSetMSE;
	double validationSetMSE;
	double generalizationSetMSE;

	// use batch learning
	bool useBatch;
	
	// PROTECTED METHODS

	// backpropagate errors through network
	void backpropagate(double* desiredValues)
	{
		// get output error gradients and set delta output
		for (int i = 0; i < network->NumOutput; i++)
		{
			// output gradoemt
			outputGradient[i] = getOutputErrorGradient(desiredValues[i], network->OutputNodes[i]);

			// for each weight update, get change based on gradient
			for (int j = 0; j <= network->NumHidden; j++)
			{
				if (!useBatch) deltaOutput[j][i] = changeOutput(i, j);
				
				else deltaOutput[j][i] += changeOutput(i, j);
			}
		}

		// get hidden error gradients and set delta ouput
		for (int i = 0; i <= network->NumHidden; i++)
		{
			// get hidden gradient
			hiddenGradient[i] = getHiddenErrorGradient(i);

			// set delta output
			for (int j = 0; j <= network->NumInput; j++)
			{
				if (!useBatch) deltaInput[j][i] = changeHidden(i, j);

				else deltaInput[j][i] += changeHidden(i, j);
			}
		}

		// if using batch, do not immediatel update weights
		if (!useBatch) updateWeights();
	}

	// update weights according to delta
	void updateWeights()
	{
		// update input weights
		for (int i = 0; i <= network->NumInput; i++)
		{
			for (int j = 0; j <= network->NumHidden; j++)
			{
				network->InputWeights[i][j] += deltaInput[i][j];

				// if using batch set delta to 0
				if (useBatch) deltaInput[i][j] = 0;
			}
		}

		// update outputweights
		for (int i = 0; i <= network->NumHidden; i++)
		{
			for (int j = 0; j <= network->NumOutput; j++)
			{
				network->OutputWeights[i][j] += deltaOutput[i][j];

				if (useBatch) deltaOutput[i][j] = 0;
			}
		}
	}

	// run a training epoch 
	void runEpoch(int epochs, double** patterns, double** targets)
	{
		for (int i = 0; i < epochs; i++)
		{
			Epoch(patterns[i], targets[i]);
		}
	}

	// run a single training example
	void Epoch(double* pattern, double* target)
	{
		// feed pattern foward
		network->FeedFoward(pattern);

		// backpropagate network
		backpropagate(target);
	}

public: // PUBLIC METHODS

	// set learning rate, momentum, and batch learning
	void setTrainingParameters(double learningRate, double momentum, bool batch)
	{
		this->learningRate = learningRate;
		this->momentum = momentum;
		this->useBatch = batch;
	}

	// set max epochs and desired accuracy
	void setStoppingConditions(int maxEpochs, double desiredAccuracy)
	{
		this->maxEpochs = maxEpochs;
		this->desiredAccuracy = desiredAccuracy;
	}

	// use batch learning
	void useBatchLearning(bool batch)
	{
		this->useBatch = batch;
	}

	// set pattern size
	void PatternSize(int pattern)
	{
		patternSize = pattern;
	}

	// set target size
	void TargetSize(int target)
	{
		targetSize = target;
	}

	// TRAIN NETWORK, takes number of epochs and data for processing
	void trainNetwork(int epochs, double** patterns, double** targets)
	{
		runEpoch(epochs, patterns, targets);
	}

	// OPERATORS

	// CONSTRUCTORS

	// default constructor
	Trainer()
	{
		network = NULL;

		learningRate = 0;
		momentum = 0;

		epoch = 0;
		maxEpochs = 0;

		patternSize = 0;
		targetSize = 0;

		desiredAccuracy = 0;

		deltaInput = NULL;
		deltaOutput = NULL;

		hiddenGradient = NULL;
		outputGradient = NULL;

		trainingSetAccuracy = 0;
		validationSetAccuracy = 0;
		generalizationSetAccuracy = 0;
		trainingSetMSE = 0;
		validationSetMSE = 0;
		generalizationSetMSE = 0;

		useBatch = false;
	}

	// initializer
	Trainer(NeuralNet* net)
	{
		network = net;
		
		learningRate = 0.001;
		momentum = 0.9;
		
		epoch = 0;
		maxEpochs = 1500;

		patternSize = 0;
		targetSize = 0;
		
		desiredAccuracy = 0;

		deltaInput = new double*[network->NumInput + 1];
		deltaOutput = new double*[network->NumHidden + 1];

		network->zero(deltaInput, network->NumInput + 1, network->NumHidden + 1);
		network->zero(deltaOutput, network->NumHidden + 1, network->NumOutput);

		hiddenGradient = new double[network->NumHidden];
		outputGradient = new double[network->NumOutput];

		network->zero(hiddenGradient, network->NumHidden + 1);
		network->zero(outputGradient, network->NumOutput);

		trainingSetAccuracy = 0;
		validationSetAccuracy = 0;
		generalizationSetAccuracy = 0;
		trainingSetMSE = 0;
		validationSetMSE = 0;
		generalizationSetMSE = 0;

		useBatch = false;
	}

	// destructor
	~Trainer()
	{

	}

protected:

	// get output error gradient
	double getOutputErrorGradient(double desiredValue, double actualValue)
	{
		return desiredValue * (1 - desiredValue) * (desiredValue - actualValue);
	}

	// git hidden error gradient
	double getHiddenErrorGradient(int index)
	{
		double weightedSum = 0;

		for (int i = 0; i < network->NumOutput; i++)
		{
			weightedSum += network->OutputWeights[index][i] * outputGradient[i];
		}

		// return gradient
		return network->HiddenNodes[index] * (1 - network->HiddenNodes[index]) * weightedSum;
	}

	// get change in output 
	double changeOutput(int output, int hidden)
	{
		if (!useBatch) return learningRate * network->HiddenNodes[hidden] * outputGradient[output] + momentum
			* deltaOutput[output][hidden];

		else return learningRate * network->HiddenNodes[hidden] * outputGradient[output];
	}

	// get change in hidden
	double changeHidden(int hidden, int input)
	{
		if (!useBatch) return learningRate * network->InputNodes[input] * hiddenGradient[hidden] + momentum
			* deltaInput[input][hidden];

		else return learningRate * network->InputNodes[input] * hiddenGradient[hidden];
	}

};