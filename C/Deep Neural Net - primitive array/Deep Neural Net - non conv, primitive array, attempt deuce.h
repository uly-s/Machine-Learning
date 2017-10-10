#pragma once



class DeepNet
{

protected:

	// FIELDS

	// number of input nodes, number of hidden nodes in the first layer
	// number in the second layer, number in the third layer, number of output nodes
	int input, hidden, hidden1, hidden2, hidden3, output;

	// the neurons in a 2d format
	double** nodes;

	// widths of each row of nodes
	int* widths;

	// biases of each node
	double** bias;

	// weights, one 2d layer of weights per layer of nodes -1
	double*** weights;

	// FRIENDS

	// PROTECTED METHODS

	// ACTIVATION FUNCTIONS


	// FEED FORWARD FUNCTIONS



	// BACKPROPAGATION ALGORITHM



public:

	// PUBLIC METHODS

	// OPERATORS

	// CONSTRUCTORS

	// default constructor

	// two layer initializer

	// three layer initializer

	// copy constructor

	// destructor


protected:





};