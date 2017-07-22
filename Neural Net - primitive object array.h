#pragma once
#include "Abstract Neural Net.h"
#include "Edge - Object Array Neural Net.h"
#include "Neuron - Object Array Neural Net.h"

class NeuralNet : public AbstractNeuralNet
{

protected:

	// number of input, hidden, and output neurons
	int numInput, numHidden, numOutput;

	// input neurons
	Neuron* input;

	// hidden neurons
	Neuron* hidden;

	// output neurons
	Neuron* output;

	// input to hidden weights
	Edge** inputWeights;

	// hidden to output weights;
	Edge** outputWeights;

	// initialize neurons
	void zeroInitialize(Neuron* nodes, int num)
	{
		nodes = new Neuron[num];

		for (int i = 0; i < num; i++)
		{
			nodes[i] = Neuron();
		}
	}

	// zero weights
	void zeroInitialize(Edge** weights, int height, int width)
	{
		for (int i = 0; i < height; i++)
		{
			weights[i] = new Edge[width];

			for (int j = 0; j < width; j++)
			{
				weights[i][j] = Edge();
			}
		}
	}


public:

	// default constructor
	NeuralNet()
	{
		numInput, numHidden, numOutput = 0;

		input = NULL;
		hidden = NULL;
		output = NULL;
		
		inputWeights = NULL;
		outputWeights = NULL;
	}

	// initializer
	NeuralNet(int inputNodes, int hiddenNodes, int outputNodes)
	{
		numInput = inputNodes;
		numHidden = hiddenNodes;
		numOutput = outputNodes;

		// initialize input neurons
		zeroInitialize(input, numInput);

		// initialize hidden neurons
		zeroInitialize(hidden, numHidden);

		// initialize output neurons
		zeroInitialize(output, numOutput);

		// initialize input weights
		zeroInitialize(inputWeights, numInput, numHidden);

		// initialize output weights
		zeroInitialize(outputWeights, numHidden, numOutput);

	}


};
