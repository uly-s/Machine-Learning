#pragma once
#include "Abstract Neural Net.h"

class NeuralNet : public AbstractNeuralNet
{

protected:

	// number of input, hidden, and output neurons
	int numInput, numHidden, numOutput;

	// input neurons
	double* inputNodes;

	// hidden neurons
	double* hiddenNodes;

	// output neurons
	double* outputNodes;

	// input to hidden weights
	double** inputWeights;

	// hidden to output weights;
	double** outputWeights;

	// initialize neurons
	void zeroInitialize(double* nodes, int num)
	{
		nodes = new double[num];

		for (int i = 0; i < num; i++)
		{
			nodes[i] = 0;
		}
	}

	// zero weights
	void zeroInitialize(double** weights, int height, int width)
	{
		for (int i = 0; i < height; i++)
		{
			weights[i] = new double[width];

			for (int j = 0; j < width; j++)
			{
				weights[i][j] = 0;
			}
		}
	}


public:

	// default constructor
	NeuralNet()
	{
		numInput, numHidden, numOutput = 0;

		inputNodes = NULL;
		hiddenNodes = NULL;
		outputNodes = NULL;

		inputWeights = NULL;
		outputWeights = NULL;
	}

	// initializer
	NeuralNet(int inputNum, int hiddenNum, int outputNum)
	{
		numInput = inputNum;
		numHidden = hiddenNum;
		numOutput = outputNum;

		// initialize input neurons
		zeroInitialize(inputNodes, numInput);

		// set bias neuron

		// initialize hidden neurons
		zeroInitialize(hiddenNodes, numHidden);

		// set bias neuron

		// initialize output neurons
		zeroInitialize(outputNodes, numOutput);

		// initialize input weights
		zeroInitialize(inputWeights, numInput, numHidden);

		// initialize output weights
		zeroInitialize(outputWeights, numHidden, numOutput);

	}


};