#pragma once
#include <iostream>
#include <math.h>
#include "Abstract Neural Net.h"

using namespace std;

class Trainer;


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

	// declare Trainer a friend
	friend Trainer;

	// activation function
	double ActivationFunction(double x)
	{
		// return sigmoid value of x, 1 over 1 + e ^ -x
		return 1 / (1 + exp(-x));
	}

	// feed pattern into input
	void FeedInput(double* data)
	{
		// set input nodes to data
		for (int i = 0; i < NumInput; i++)
		{
			InputNodes[i] = data[i];
		}
	}

	// feed input into hidden 
	void FeedHidden()
	{
		// set each hidden node to be the weighted sum of the input nodes
		for (int i = 0; i < NumHidden; i++)
		{
			// clear value
			HiddenNodes[i] = 0;

			// add input node j multiplied by weight j i for the weighted sum
			for (int j = 0; j <= NumInput; j++) HiddenNodes[i] += InputNodes[j] * InputWeights[j][i];

			// set the result to the activation function
			HiddenNodes[i] = ActivationFunction(HiddenNodes[i]);
		}
	}

	// feed hidden to output
	void FeedOutput()
	{
		// set each output node to be the weighted sum of the hidden nodes
		for (int i = 0; i < NumOutput; i++)
		{
			// clear value
			OutputNodes[i] = 0;

			// multiply the input by its weight and add it
			for (int j = 0; j <= NumHidden; j++) OutputNodes[i] += HiddenNodes[j] * OutputWeights[j][i];

			// set to activation function
			OutputNodes[i] = ActivationFunction(OutputNodes[i]);
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
		int* output = new int[NumOutput];

		// copy over results
		for (int i = 0; i < NumOutput; i++)
		{
			output[i] = clampOutput(OutputNodes[i]);
		}

		// return results
		return output;

	}
	
	// error gradient

	// mean squared error

public:

	// save weights

	// load weights

	// initialize weights 
	void InitializeWeights()
	{
		// get range for input nodes and hidden nodes
		double rangeHidden = 1 / sqrt((double) NumInput);
		double rangeOutput = 1 / sqrt((double) NumHidden);

		// initialize input weights
		for (int i = 0; i <= NumInput; i++)
		{
			for (int j = 0; j <= NumHidden; j++)
			{
				InputWeights[i][j] = random(rangeHidden);
			}
		}

		// initialize output weights
		for (int i = 0; i <= NumHidden; i++)
		{
			for (int j = 0; j < NumOutput; j++)
			{
				OutputWeights[i][j] = random(rangeOutput);
			}
		}
	}

	// get set accuracy
	double getSetAccuracy(int size, double** set, double** targets)
	{
		// is correct flag
		bool correct = true;

		// correct counter
		int numIncorrect = 0;

		// local accuracy
		double accuracy = 0;



		// feed data foward, test if pattern is correct
		for (int i = 0; i < size; i++)
		{
			FeedFoward(set[i]);

			for (int j = 0; j < NumOutput; j++)
			{
				if (clampOutput(OutputNodes[i]) != targets[i][j]) correct = false;
			}

			if (!correct) numIncorrect++;
		}

		accuracy = 100 - (numIncorrect / size * 100);

		return accuracy;
	}

	friend ostream& operator<< (ostream& os, NeuralNet& net)
	{
		return net.print(os);
	}

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

		zero(InputWeights, NumInput + 1, NumHidden + 1);
		zero(OutputWeights, NumHidden + 1, NumOutput);

		// initialize weights to random values
		InitializeWeights();

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

	// random initial value for weights
	double random(double range)
	{
		return (((double) (rand() % 100) + 1) / 100 * 2 * range) - range;
	}

	// rounds off number
	int clampOutput(double x)
	{
		if (x > 0.9) return 1;
		else if (x < 0.1) return 0;
		else return -1;
	}

	// print for ostream
	ostream& print(ostream& os)
	{
		// print input nodes
		os << "Input Nodes: ";

		for (int i = 0; i <= NumInput; i++)
		{
			os << InputNodes[i] << " ";
		}

		os << endl << endl;

		// print input weights
		os << "Input to hidden weights:\n\n";

		for (int i = 0; i <= NumInput; i++)
		{
			if (i < 10) os << " ";

			os << i << ". ";

			for (int j = 0; j <= NumHidden; j++)
			{
				os << InputWeights[i][j] << " ";
			}

			os << endl;
		}

		// print hidden nodes
		os << endl << "Hidden Nodes: ";

		for (int i = 0; i <= NumHidden; i++)
		{
			os << HiddenNodes[i] << " ";
		}

		os << endl << endl;

		// print output weights
		os << "Hidden to output weights: \n\n";

		for (int i = 0; i <= NumHidden; i++)
		{
			if (i < 10) os << " ";

			os << i << ". ";

			for (int j = 0; j < NumOutput; j++)
			{
				os << OutputWeights[i][j] << " ";
			}

			os << endl;
		}

		// print output nodes
		os << endl << "Output Nodes: ";


		for (int i = 0; i < NumOutput; i++)
		{
			os << OutputNodes[i] << " ";
		}

		os << endl;

		return os;
	}


};


