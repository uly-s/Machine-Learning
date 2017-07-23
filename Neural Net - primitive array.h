#pragma once
#include "Abstract Neural Net.h"




class NeuralNet : public AbstractNeuralNet
{

protected:

	// number of input, hidden, and output neurons
	int NumInput, NumHidden, NumOutput;

	// input neurons
	double* InputNodes;

	// hidden neurons
	double* HiddenNodes;

	// output neurons
	double* OutputNodes;

	// input to hidden weights
	double** InputWeights;

	// hidden to output weights;
	double** OutputWeights;

	// get weighted sum  of node

	// sigmoid function
	
	// activation function

	// error gradient

	// mean squared error

	// initialize weights 



	// initialize neurons
	void zero(double* nodes, int num)
	{
		for (int i = 0; i < num; i++)
		{
			nodes[i] = 0;
		}
	}

	// zero weights
	void zero(double** weights, int height, int width)
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



	// save weights

	// load weights


	// default constructor
	NeuralNet()
	{
		NumInput, NumHidden, NumOutput = 0;

		InputNodes = NULL;
		HiddenNodes = NULL;
		OutputNodes = NULL;

		InputWeights = NULL;
		OutputWeights = NULL;

	}

	// initializer
	NeuralNet(int inputNum, int hiddenNum, int outputNum)
	{
		NumInput = inputNum;
		NumHidden = hiddenNum;
		NumOutput = outputNum;

		// initialize input neurons
		// initialize hidden neurons
		// initialize output neurons

		InputNodes = new double[NumInput + 1];
		HiddenNodes = new double[NumHidden + 1];
		OutputNodes = new double[NumOutput];

		zero(InputNodes, NumInput);
		zero(HiddenNodes, NumHidden);
		zero(OutputNodes, NumOutput);

		// set bias neurons
		InputNodes[NumInput] = -1;
		HiddenNodes[NumHidden] = -1;
		
		// initialize input weights
		// initialize output weights

		InputWeights = new double*[NumInput + 1];
		OutputWeights = new double*[NumHidden + 1];

		zero(InputWeights, NumInput, NumHidden);
		zero(OutputWeights, NumHidden, NumOutput);

	}

	// destructor
	~NeuralNet()
	{
		delete[] InputNodes;
		delete[] HiddenNodes;
		delete[] OutputNodes;
		
		delete[] InputWeights;
		delete[] OutputWeights;

	}


protected:



};


