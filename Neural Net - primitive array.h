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
		// initialize hidden neurons
		// initialize output neurons

		inputNodes = new double[numInput + 1];
		hiddenNodes = new double[numHidden + 1];
		outputNodes = new double[numOutput];

		zeroInitialize(inputNodes, numInput);
		zeroInitialize(hiddenNodes, numHidden);
		zeroInitialize(outputNodes, numOutput);

		// set bias neurons
		inputNodes[numInput] = -1;
		hiddenNodes[numHidden] = -1;
		
		
		// initialize input weights
		// initialize output weights

		inputWeights = new double*[numInput];
		outputWeights = new double*[numHidden];

		zeroInitialize(inputWeights, numInput, numHidden);
		zeroInitialize(outputWeights, numHidden, numOutput);

	}


};