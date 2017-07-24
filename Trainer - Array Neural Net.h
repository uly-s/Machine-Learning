#pragma once
#include "Neural Net - primitive array.h"



class Trainer
{

protected:

	// pointer to the neural net for training
	NeuralNet* network;

	// learning rate
	double learningRate;

	// momentum
	double momentum;

	// epochs
	int epoch;
	int maxEpochs;

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
	

public:

	// default constructor
	Trainer()
	{
		network = NULL;
	}

	// initializer
	Trainer(NeuralNet* net)
	{
		network = net;
	}

	// 
	~Trainer()
	{

	}

};