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
		//nodes = new Neuron[num];

		for (int i = 0; i < num; i++)
		{
			nodes[i] = Neuron();
		}
	}

	// zero weights
	void zeroInitialize(Edge** weights, int height, int width)
	{
		//weights = new Edge*[height];

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
		// initialize hidden neurons
		// initialize output neurons

		input = new Neuron[numInput + 1];
		hidden = new Neuron[numHidden + 1];
		output = new Neuron[numOutput];
		
		zeroInitialize(input, numInput + 1);
		zeroInitialize(hidden, numHidden + 1);	
		zeroInitialize(output, numOutput);
		
		// set bias neurons
		input[numInput] = -1;
		hidden[numHidden] = -1;


		// initialize weights, input to hidden and hidden to output
		inputWeights = new Edge*[numInput];
		outputWeights = new Edge*[numHidden];

		zeroInitialize(inputWeights, numInput, numHidden);
		zeroInitialize(outputWeights, numHidden, numOutput);

	}


};
