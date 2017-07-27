#pragma once
#include <iostream>
#include <math.h>

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
	double** hiddenWeights;

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

	// backpropagate errors through network


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

	// CONSTRUCTORS

	// constructor - default
	DeepNet()
	{
		numInput, numOutput = 0;

		inputNodes, hiddenNodes, outputNodes,
			hiddenWidths = NULL;

	}

	// initializer, takes in number of input nodes, number of hidden layers, and an array 
	// of integers specifying the size of the hidden layers, and then the number of output 
	// nodes
	DeepNet(int numInput, int numHiddenLayers, int* hiddenDimensions, int numOutput)
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

	// random value for initialization of weights

	// get weighted sum of a layer for a node in the next layer

	// clamp output to a round number

	// print for ostream operator

	// print layer helper function

	// print weights helper function

};