#pragma once
#include <math.h>
#include "Abstract Neural Net.h"
#include "Edge - primitive object array Neural Net.h"
#include "Neuron - primitive object array Neural Net.h"

// for hidden neurons the index of the output edges starts at numInput + 1 
// to account for the bias neuron, so indexing goes to num input + 1 + num output

class NeuralNet : public AbstractNeuralNet
{

protected:

	// number of input, hidden, and output neurons
	int numInput, numHidden, numOutput;

	// input neurons
	Neuron** input;

	// hidden neurons
	Neuron** hidden;

	// output neurons
	Neuron** output;

	// initialize neurons
	void zero(Neuron** nodes, int num, int edges)
	{
		//nodes = new Neuron[num];

		for (int i = 0; i < num; i++)
		{
			nodes[i] = new Neuron(edges);
		}
	}

	// zero weights
	void zero(Edge** weights, int height, int width)
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

	// initialize weights to random values
	void InitializeWeights()
	{
		// get range for input nodes and hidden nodes
		double rangeHidden = 1 / sqrt((double) numInput);
		double rangeOutput = 1 / sqrt((double) numHidden);

		// initialize input weights
		for (int i = 0; i <= numInput; i++)
		{
			cout << "Input node: " << i << " ";

			for (int j = 0; j <= numHidden; j++)
			{
				input[i][j] = random(rangeHidden);

				cout << input[i][j] << " ";
			}
		}

		// initialize output weights
		for (int i = 0; i <= numHidden; i++)
		{
			cout << "Hidden node: " << i << " ";

			for (int j = 0; j < numOutput; j++)
			{
				hidden[i][j] = random(rangeOutput);

				cout << hidden[i][j] << " ";
			}

		}
	}

	// save weights

	// load weights


	// default constructor
	NeuralNet()
	{
		numInput, numHidden, numOutput = 0;

		input = NULL;
		hidden = NULL;
		output = NULL;
		
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

		input = new Neuron*[numInput + 1];
		hidden = new Neuron*[numHidden + 1];
		output = new Neuron*[numOutput];
		
		zero(input, numInput + 1, numHidden + 1);
		zero(hidden, numHidden + 1, numOutput);	
		zero(output, numOutput, 0);

		hidden[0][0] = 0;
		
		// set bias neurons
		*input[numInput] = -1;
		*hidden[numHidden] = -1;



	}

protected:

	// random value for initializing weights
	double random(double range)
	{
		return (((double) (rand() % 100) + 1) / 100 * 2 * range) - range;
	}



};
