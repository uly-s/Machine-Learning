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

	// activation function
	double ActivationFunction(double x)
	{
		
		// return sigmoid value of x, 1 over 1 + e ^ -x
		return 1 / (1 + exp(-x));
	}

	// overload for neuron
	double ActivationFunction(Neuron* neuron)
	{
		
		double weight = **neuron;
		
		return 1 / (1 + exp(-**neuron));
	}

	// feed pattern into input
	void FeedInput(double* data)
	{
		// set input nodes to data
		for (int i = 0; i < numInput; i++)
		{
			*input[i] = data[i];
		}
	}

	// feed input into hidden 
	void FeedHidden()
	{
		// set each hidden node to be the weighted sum of the input nodes
		for (int i = 0; i < numHidden; i++)
		{
			// clear value
			*hidden[i] = 0;
			
			// add input node j multiplied by weight j i for the weighted sum
			*hidden[i] = weightedSum(input, numInput + 1, i);
			
			double activated = ActivationFunction(hidden[i]);
			
			// set the result to the activation function
			*hidden[i] = activated;
			
		}
	}

	// feed hidden to output
	void FeedOutput()
	{
		// set each output node to be the weighted sum of the hidden nodes
		for (int i = 0; i < numOutput; i++)
		{
			// clear value
			*output[i] = 0;

			// multiply the input by its weight and add it
			*output[i] = weightedSum(hidden, numHidden + 1, i);

			// set to activation function
			*output[i] = ActivationFunction(**output[i]);
		}
	}

	// Feed pattern foward
	void FeedFoward(double* data)
	{
		// feed into input
		FeedInput(data);
	
		// feed input to hidden
		FeedHidden();

		// feed hidden to output
		FeedOutput();

	}

	// feed data foward through network
	int* FeedPatternFoward(double* data)
	{
		FeedFoward(data);

		// get copy of results
		int* result = new int[numOutput];

		// copy over results
		for (int i = 0; i < numOutput; i++)
		{
			result[i] = clampOutput(**output[i]);
		}

		// return results
		return result;

	}

	// error gradient

	// mean squared error




public:

	int* test(double* data)
	{
		return FeedPatternFoward(data);
	}

	// get set accuracy

	// get set mean squared accuracy

	// initialize weights to random values
	void InitializeWeights()
	{
		// get range for input nodes and hidden nodes
		double rangeHidden = 1 / sqrt((double) numInput);
		double rangeOutput = 1 / sqrt((double) numHidden);

		// initialize input weights
		for (int i = 0; i <= numInput; i++)
		{
			for (int j = 0; j <= numHidden; j++)
			{
				input[i][j] = random(rangeHidden);
			}
		}

		// initialize output weights
		for (int i = 0; i <= numHidden; i++)
		{
			for (int j = 0; j < numOutput; j++)
			{
				hidden[i][j] = random(rangeOutput);
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
		
		// set bias neurons
		*input[numInput] = -1;
		*hidden[numHidden] = -1;



	}

	// copy constructor
	NeuralNet(NeuralNet& net)
	{

	}

	// destructor
	~NeuralNet()
	{
		delete[] input;
		delete[] hidden;
		delete[] output;
	}

protected:

	// random value for initializing weights
	double random(double range)
	{
		return (((double) (rand() % 100) + 1) / 100 * 2 * range) - range;
	}

	// initialize neurons
	void zero(Neuron** nodes, int num, int edges)
	{
		//nodes = new Neuron[num];

		for (int i = 0; i < num; i++)
		{
			nodes[i] = new Neuron(edges);
		}
	}

	// get weighted sum of inputs or hidden layer, edge is the weight we multiply by corresponding
	// to that connection with the node
	double weightedSum(Neuron** neurons, int nodes,  int edge)
	{
		double sum = 0;

		for (int i = 0; i < nodes; i++)
		{
			sum += *neurons[i] * (*neurons[i])[edge];
		}

		return sum;
	}

	// rounds off number
	int clampOutput(double x)
	{
		if (x > 0.9) return 1;
		else if (x < 0.1) return 0;
		else return -1;
	}

};
