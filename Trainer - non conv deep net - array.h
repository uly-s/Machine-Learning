#pragma once
#include "Deep Neural Network - non convolutional - array implementation.h"


class Trainer
{

protected:
	
	// FIELDS

	// pointer to network
	DeepNet* net;

	// learning rate
	double LR;

	// desired accuracy
	double accuracy;

	// number of epochs
	int epochs;

	// max epochs
	int maxEpochs;

	// batch size
	int batchSize;

	// input error
	double* inputError;

	// hidden error
	double** hiddenError;

	// output error
	double* outputError;

	// changed to input weights
	double** deltaInput;

	// changed to hidden
	double*** deltaHidden;

	// changed to output weights
	double** deltaOutput;

	// PROTECTED METHODS

	// backpropagate 
	void backpropagate(double* targets)
	{
		// set output errors
		for (int i = 0; i < net->numOutput; i++)
		{
			outputError[i] = targets[i] - net->outputNodes[i];
		}

		// call backpropagate output
		backpropagateOutput();

		// call backpropagate hidden
		backpropagateHidden();

		// call backpropagate input
		backpropagateInput();

		// get weight changes


	}

	// backpropagate output nodes and output weights
	void backpropagateOutput()
	{
		// for each node in the last hidden layer
		for (int i = 0; i <= net->hiddenWidths[net->hiddenIndex]; i++)
		{
			// multiply the error at output by the weight of the hidden node leading to that output
			// to get the weighted rror at that node
			for (int j = 0; j < net->numOutput; j++)
			{
				hiddenError[net->hiddenIndex][i] += outputError[j] * net->outputWeights[i][j];
			}
		}


	}

	// backpropagate from the bottom hidden layer to the top
	void backpropagateHidden()
	{
		// for each hidden layer, get the weighted error
		// from the previous layer
		for (int i = net->hiddenIndex - 1; i > 0; i--)
		{
			// for each node in the layer
			for (int j = 0; j <= net->hiddenWidths[i]; j++)
			{
				// for each hidden weight
				for (int k = 0; k <= net->hiddenWidths[i + 1]; k++)
				{
					hiddenError[i][j] += hiddenError[i + 1][j] * net->hiddenWeights[i][j][k];
				}
			}
		}
	}

	// backpropagate hidden errors to input layer
	void backpropagateInput()
	{
		// for each input node
		for (int i = 0; i <= net->numInput; i++)
		{
			// for each node in the top hidden layer
			for (int j = 0; j <= net->hiddenWidths[0]; j++)
			{
				inputError[i] += hiddenError[0][j] * net->inputWeights[i][j];
			}
		}
	}

	// run individual training epoch
	void Epoch(double* input, double* targets)
	{
		net->FeedFoward(input);

		backpropagate(targets);
	}

	// run batch
	void Batch(double** inputs, double** targets)
	{
		for (int i = 0; i < batchSize; i++)
		{
			Epoch(inputs[i], targets[i]);
		}
	}




public:

	// PUBLIC METHODS
	
	
	// train on data for some number of epochs
	void Train(double*** data, double*** targets, int epochs)
	{
		double trainingAccuracy = 0;

		int epoch = 0;

		int index = 0;

		while (epoch < epochs && epoch < maxEpochs && trainingAccuracy < accuracy)
		{
			Batch(data[index], targets[index]);

			epoch += batchSize;
		}
	}

	// set training parameters, batch size, learning rate, desired accuracy, max epochs
	void Parameters(int batchSize, double learningRate, double targetAccuracy, int maxEpochs)
	{
		this->batchSize = batchSize;
		
		LR = learningRate;

		accuracy = targetAccuracy;

		this->maxEpochs = maxEpochs;
	}




	// OPERATORS

	// CONSTRUCTORS

	// default constructor
	Trainer()
	{
		net = NULL;

		accuracy, epochs, batchSize, maxEpochs, LR = 0;

		inputError, hiddenError, outputError = NULL;

		deltaInput, deltaHidden, deltaOutput = NULL;
	}

	// initializer
	Trainer(DeepNet* network)
	{
		// I've been working for a while and I'm feeling no motivation to document this

		net = network;

		accuracy, epochs, batchSize, maxEpochs, LR = 0;

		inputError = new double[net->numInput];

		for (int i = 0; i <= net->numInput; i++)
		{
			inputError[i] = 0;

		}


		hiddenError = new double*[net->hiddenLayers];

		for (int i = 0; i < net->hiddenLayers; i++)
		{
			hiddenError[i] = new double[net->hiddenWidths[i]];

			for (int j = 0; j <= net->hiddenWidths[i]; j++)
			{
				hiddenError[i][j] = 0;

			}

		}

		outputError = new double[net->numOutput];

		for (int i = 0; i < net->numOutput; i++)
		{
			outputError[i] = 0;

		}


		deltaInput = new double*[net->numInput + 1];

		for (int i = 0; i <= net->numInput; i++)
		{
			deltaInput[i] = new double[net->hiddenWidths[0]];

			for (int j = 0; j <= net->hiddenWidths[0]; j++)
			{
				deltaInput[i][j] = 0;

			}

		}


		deltaHidden = new double**[net->hiddenLayers];

		for (int i = 0; i < net->hiddenIndex; i++)
		{
			deltaHidden[i] = new double*[net->hiddenWidths[i] + 1];

			for (int j = 0; j <= net->hiddenWidths[i]; j++)
			{
				deltaHidden[i][j] = new double[net->hiddenWidths[i + 1] + 1];

				for (int k = 0; k <= net->hiddenWidths[i + 1]; k++)
				{
					deltaHidden[i][j][k] = 0;

				}

			}

		}


		deltaOutput = new double*[net->hiddenWidths[net->hiddenIndex] + 1];

		for (int i = 0; i <= net->hiddenWidths[net->hiddenIndex]; i++)
		{
			deltaOutput[i] = new double[net->numOutput];

			for (int j = 0; j < net->numOutput; j++)
			{
				deltaOutput[i][j] = 0;

			}

		}

	}

	// copy constructor
	Trainer(Trainer& trainer)
	{

	}

	// destructor
	~Trainer()
	{

	}


protected:

	// set input weight changes

	// set hidden weight changes

	// set output weight changes

	// delta or weight prime, the formula for how we change the weights
	double weightPrime(double weight, double error)
	{
		return weight;
	}






};