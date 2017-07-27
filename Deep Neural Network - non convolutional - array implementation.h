#pragma once
#include <iostream>
#include <math.h>

using namespace std;

// fully connected simple deep neural network

class DeepNet
{

protected:

	// FIELDS
	
	// number of input and output "neurons"
	int numInput, numOutput;

	// number of hidden layers and the size of those layers
	int hiddenLayers;
	int* hiddenWidths;

	// input "neurons"
	double* inputNodes;

	// hidden "neurons"
	double** hiddenNodes;

	// output "neurons"
	double* outputNodes;

	// weights

	// input to hidden weights
	double** inputWeights;

	// hidden to hidden weights
	double*** hiddenWeights;

	// hidden to output weights
	double** outputWeights;

	// PROTECTED METHODS

	// softmax function

	// sigmoid function

	// activation function, uses softmax or sigmoid

	// feed input

	// feed input to hidden, feeds hidden to hidden

	// feed hidden to output

	// feed input foward

	// backpropagate errors through network, recursive

	// backpropagation algorithm


public:

	// PUBLIC METHODS

	// feed pattern foward, return results

	// initialize weights to random values
	void InitializeWeights()
	{

	}

	// save weights

	// load weights

	// OPERATORS

	// assignment operator

	// ostream operator
	friend ostream& operator<< (ostream& os, DeepNet& net)
	{
		return net.print(os);
	}

	// CONSTRUCTORS

	// constructor - default
	DeepNet()
	{
		numInput, numOutput = 0;

		inputNodes, hiddenNodes, outputNodes,
			hiddenWidths = NULL;

		inputWeights, hiddenWeights = NULL;

	}

	// initializer, takes in number of input nodes, number of hidden layers, and an array 
	// of integers specifying the size of the hidden layers, and then the number of output 
	// nodes
	DeepNet(int numInput, int numHiddenLayers, int* hiddenDimensions, int numOutput)
	{


	}

	// simpler initializer, hard coded as a two hidden layer
	DeepNet(int numInput, int numHidden1, int numHidden2, int numOutput)
	{
		this->numInput = numInput;

		hiddenLayers = 2;

		hiddenWidths = new int[2];
		hiddenWidths[0] = numHidden1;
		hiddenWidths[1] = numHidden2;

		this->numOutput = numOutput;

		// initialize nodes, add 1 for bias neurons
		inputNodes = new double[numInput + 1];
		hiddenNodes = new double*[hiddenLayers];
		outputNodes = new double[numOutput];

		// zero initialize nodes
		zero(inputNodes, numInput);

		// initialize each hidden layer and zero them out
		for (int i = 0; i < hiddenLayers; i++)
		{
			hiddenNodes[i] = new double[hiddenWidths[i] + 1];
			
			zero(hiddenNodes[i], hiddenWidths[i] + 1);

			// set bias neuron
			hiddenNodes[i][hiddenWidths[i]] = -1;
		}

		// zero output
		zero(outputNodes, numOutput);

		// initialize weights

		// initialize input to hidden weights
		inputWeights = new double*[numInput];

		// initialize hidden to hidden weights
		hiddenWeights = new double**[hiddenLayers - 1];

		// initialize output weights (to the width of the last hidden layer at index
		// 1 less than the number of layers)
		outputWeights = new double*[hiddenWidths[hiddenLayers - 1] + 1];
		
		// zero input weights
		for (int i = 0; i <= numInput; i++)
		{
			inputWeights[i] = new double[hiddenWidths[0] + 1];

			// for each input node, set the weights for its connections to the first
			// layer of hidden nodes
			zero(inputWeights[i], hiddenWidths[0] + 1);
		};

		// zero hidden weights
		for (int i = 0; i < hiddenLayers; i++)
		{
			hiddenWeights[i] = new double*[hiddenWidths[i] + 1];

			for (int j = 0; j <= hiddenWidths[i]; i++)
			{
				if (i < hiddenLayers - 1)hiddenWeights[i][j] = new double[hiddenWidths[i + 1] + 1];
				else hiddenWeights[i][j] = new double[numOutput];
				
			}

		}		

	}

	// deeper initializer, hard coded as three hidden layers
	DeepNet(int numInput, int numHidden1, int numHidden2, int numHidden3, int numOutput)
	{

	}

	// copy constructor
	DeepNet(DeepNet& net)
	{

	}

	// destructor
	~DeepNet()
	{

	}


protected:

	// PROTECTED HELPER METHODS

	// set values to 0
	void zero(double* row, int size)
	{
		for (int i = 0; i < size; i++)
		{
			row[i] = 0;
		}
	};

	// random value for initialization of weights

	// get weighted sum of a layer for a node in the next layer

	// clamp output to a round number

	// round number, for formatting output
	double Round(double x)
	{
		return round(x * 1000000000) / 1000000000;
	}

	// print for ostream operator
	ostream& print(ostream& os)
	{
		// print input
		printLayer(os, inputNodes, numInput);

		os << endl;

		// print input weights
		for (int i = 0; i < hiddenWidths[0] + 1; i++)
		{
			printLayer(os, inputWeights[i], hiddenWidths[0] + 1);
		}

		os << endl;

		// print hidden nodes
		for (int i = 0; i < hiddenLayers; i++)
		{
			printLayer(os, hiddenNodes[i], hiddenWidths[i]);
		}

		os << endl;

		// print hidden weights
		for (int i = 0; i < hiddenLayers; i++)
		{
			for (int j = 0; j < hiddenWidths[i]; j++)
			{
				printLayer(os, hiddenWeights[i][j], hiddenWidths[i + 1] + 1);
			}
		}

		os << endl;

		// prin output nodes
		printLayer(os, outputNodes, numOutput);

		os << endl;


		return os;
	}

	// print layer helper function
	void printLayer(ostream& os, double* row, int size)
	{
		for (int i = 0; i < size; i++)
		{
			os << row[i] << " ";
		}

		os << endl;
	}

	// print weights helper function

};