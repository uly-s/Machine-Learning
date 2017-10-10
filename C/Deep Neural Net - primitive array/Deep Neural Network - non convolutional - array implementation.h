#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>

using namespace std;

// fully connected simple deep neural network

class Trainer;

class DeepNet
{

protected:

	// FIELDS
	
	// number of input, hidden, and output "neurons"
	int numInput, numHidden, numOutput;

	// number of hidden layers and the size of those layers
	int hiddenLayers;
	int* hiddenWidths;

	// index for last hidden layer
	int hiddenIndex;

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

	// FRIENDS
	friend Trainer;

	// PROTECTED METHODS

	// softmax function

	// sigmoid function
	double sigmoid(double x)
	{
		return 1 / (1 + exp(-x));
	}

	// derivative of sigmoid
	double sigmoidPrime(double x)
	{
		return sigmoid(x) * (1 - sigmoid(x));
	}

	// hyperbolic tangent function
	double hyperbolicTangent(double x)
	{
		return tanh(x);
	}

	// rectified linear activation function
	double rectifiedLinear(double x)
	{
		return max(0.0, x);
	}

	// activation function, uses 1. sigmoid, 2. softmax, or 3. tanh
	double ActivationFunction(double x)
	{
		return sigmoid(x);
	}

	// feed input
	void FeedInput(double* input)
	{
		// set each neuron to input
		for (int i = 0; i < numInput; i++)
		{
			inputNodes[i] = input[i];
		}

	};

	// feed feeds hidden to hidden
	void FeedHidden()
	{
		// feed input to hidden
		for (int i = 0; i <= hiddenWidths[0]; i++)
		{
			hiddenNodes[0][i] = ActivationFunction(weightedInput(i));
		}


		// feed one hidden layer to the next
		for (int i = 1; i < hiddenLayers; i++)
		{
			for (int j = 0; j <= hiddenWidths[i]; j++)
			{
				hiddenNodes[i][j] = ActivationFunction(weightedHidden(i - 1, j));
			}
		}

	};

	// feed hidden to output
	void FeedOutput()
	{
		for (int i = 0; i < numOutput; i++)
		{
			outputNodes[i] = ActivationFunction(weightedOutput(i));
		}
	};

	// feed input foward
	void FeedFoward(double* input)
	{
		FeedInput(input);

		FeedHidden();

		FeedOutput();
	}


public:

	// PUBLIC METHODS

	// feed pattern foward, return results

	// get set accuracy
	double getSetAccuracy(double** inputs, double** targets, int size, int index)
	{
		double accuracy = 0;

		int wrong = 0;

		bool right = true;

		for (int i = index; i < size + index; i++)
		{
			FeedFoward(inputs[i]);

			right = true;

			for (int j = 0; j < numOutput; j++)
			{
				if (clampOutput(outputNodes[j]) != targets[i][j])
				{
					right = false;
				}
			}

			if (!right) wrong++;
		}

		accuracy = 100 - ((double) wrong / (double) size * 100);

		return accuracy;
	}
	
	// initialize weights to random values
	void InitializeWeights()
	{
		// get range  of hidden nodes and output nodes
		double rangeHidden = 1 / sqrt((double) numHidden);
		double rangeOutput = 1 / sqrt((double) numOutput);

		// initialize input weights
		for (int i = 0; i <= numInput; i++)
		{
			for (int j = 0; j <= hiddenWidths[0]; j++)
			{
				inputWeights[i][j] = random(rangeHidden);
			}
		}

		// initialize hidden weights
		for (int i = 0; i < hiddenIndex; i++)
		{
			for (int j = 0; j <= hiddenWidths[i]; j++)
			{
				for (int k = 0; k <= hiddenWidths[i + 1]; k++)
				{
					hiddenWeights[i][j][k] = random(rangeHidden);
				}
			}
		}

		// initialize output weights
		for (int i = 0; i <= hiddenWidths[hiddenIndex]; i++)
		{
			for (int j = 0; j < numOutput; j++)
			{
				outputWeights[i][j] = random(rangeOutput);
			}
		}

	}

	// save weights
	void SaveWeights(ofstream& file)
	{
		// output input weights
	
		// for each input node
		for (int i = 0; i <= numInput; i++)
		{
			// for each node in the top hidden layer
			for (int j = 0; j <= hiddenWidths[0]; j++)
			{
				// send weight to ostream, as well as a space
				file << inputWeights[i][j] << " ";
			}

			// send a new line
			file << endl;
		}

		// new line
		file << endl;

		// save hidden weights

		// for each hidden layer
		for (int i = 0; i < hiddenIndex; i++)
		{
			// for each hidden node in the first layer
			for (int j = 0; j <= hiddenWidths[i]; j++)
			{
				// for each node in the next layer
				for (int k = 0; k <= hiddenWidths[i + 1]; k++)
				{
					// send weight to file plus a space
					file << hiddenWeights[i][j][k] << " ";
				}

				// send an endl
				file << endl;
			}

			// send an endl
			file << endl;
		}

		// save output weights

		// for each node in the last hidden layer
		for (int i = 0; i <= hiddenWidths[hiddenIndex]; i++)
		{
			// for each output node
			for (int j = 0; j < numOutput; j++)
			{
				file << outputWeights[i][j] << " ";
			}

			file << endl;

		}

		file << endl;

		file.close();


	}

	// load weights
	void LoadWeights(ifstream& file)
	{
		// read in input weights

		// for each input node
		for (int i = 0; i <= numInput; i++)
		{
			// for each node in the top hidden layer
			for (int j = 0; j <= hiddenWidths[0]; j++)
			{
				file >> inputWeights[i][j];
			}
		}

		// read in hidden weights

		// for each hidden layer
		for (int i = 0; i < hiddenIndex; i++)
		{
			// for each hidden node in this layer
			for (int j = 0; j <= hiddenWidths[i]; j++)
			{
				// for each node in the next layer
				for (int k = 0; k <= hiddenWidths[i + 1]; k++)
				{
					file >> hiddenWeights[i][j][k];
				}
			}
		}

		// read in output weights

		// for each node in the last hidden layer
		for (int i = 0; i <= hiddenWidths[hiddenIndex]; i++)
		{
			// for each output node
			for (int j = 0; j < numOutput; j++)
			{
				file >> outputWeights[i][j];
			}
		}

		// close stream
		file.close();
		
	}

	// OPERATORS

	// assignment operator
	DeepNet& operator= (DeepNet& net)
	{


		return *this;
	}

	// ostream operator
	friend ostream& operator<< (ostream& os, DeepNet& net)
	{
		return net.print(os);
	}

	// CONSTRUCTORS

	// constructor - default
	DeepNet()
	{
		numInput, numHidden, numOutput = 0;

		inputNodes, hiddenNodes, outputNodes,
			hiddenWidths = NULL;

		inputWeights, hiddenWeights = NULL;

	}

	// initializer, takes in number of input nodes, number of hidden layers, and an array 
	// of integers specifying the size of the hidden layers, and then the number of output 
	// nodes
	DeepNet(int numInput, int numHiddenLayers, int* hiddenDimensions, int numOutput)
	{
		this->numInput = numInput;

		hiddenLayers = numHiddenLayers;

		numHidden = 0;

		hiddenIndex = hiddenLayers - 1;

		hiddenWidths = new int[hiddenLayers];

		for (int i = 0; i < hiddenLayers; i++)
		{
			hiddenWidths[i] = hiddenDimensions[i];
			numHidden += hiddenDimensions[i];
		}


		this->numOutput = numOutput;

		//cout << "test" << endl;

		// initialize nodes, add 1 for bias neurons
		inputNodes = new double[numInput + 1];
		hiddenNodes = new double*[hiddenLayers];
		outputNodes = new double[numOutput];

		// zero initialize nodes
		zero(inputNodes, numInput + 1);

		// set bias neuron
		inputNodes[numInput] = -1;

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

		// initialize hidden to hidden weights, one less than the number of hidden layers
		// because we are using output weights for the last set of weights
		// to simplify things
		hiddenWeights = new double**[hiddenIndex];

		// initialize output weights (to the width of the last hidden layer at index
		// 1 less than the number of layers)
		outputWeights = new double*[hiddenWidths[hiddenIndex] + 1];

		// zero input weights
		for (int i = 0; i <= numInput; i++)
		{
			inputWeights[i] = new double[hiddenWidths[0] + 1];

			// for each input node, set the weights for its connections to the first
			// layer of hidden nodes
			zero(inputWeights[i], hiddenWidths[0] + 1);
		};

		// zero hidden weights
		for (int i = 0; i < hiddenIndex; i++)
		{
			hiddenWeights[i] = new double*[hiddenWidths[i] + 1];

			for (int j = 0; j <= hiddenWidths[i]; j++)
			{
				hiddenWeights[i][j] = new double[hiddenWidths[i + 1] + 1];

				for (int k = 0; k <= hiddenWidths[i + 1]; k++)
				{
					hiddenWeights[i][j][k] = 0;
				}
			}
		}

		// zero output weights, one weight from each of the last hidden nodes to each output nodes
		for (int i = 0; i <= hiddenWidths[hiddenIndex]; i++)
		{
			outputWeights[i] = new double[numOutput];

			for (int j = 0; j < numOutput; j++)
			{
				outputWeights[i][j] = 0;
			}
		}

	}

	// simpler initializer, hard coded as a two hidden layer
	DeepNet(int numInput, int numHidden1, int numHidden2, int numOutput)
	{
		this->numInput = numInput;

		hiddenLayers = 2;

		numHidden = numHidden1 + numHidden2;

		hiddenIndex = hiddenLayers - 1;

		hiddenWidths = new int[2];
		hiddenWidths[0] = numHidden1;
		hiddenWidths[1] = numHidden2;

		this->numOutput = numOutput;

		// initialize nodes, add 1 for bias neurons
		inputNodes = new double[numInput + 1];
		hiddenNodes = new double*[hiddenLayers];
		outputNodes = new double[numOutput];

		// zero initialize nodes
		zero(inputNodes, numInput + 1);

		// set bias neuron
		inputNodes[numInput] = -1;

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

		// initialize hidden to hidden weights, one less than the number of hidden layers
		// because we are using output weights for the last set of weights
		// to simplify things
		hiddenWeights = new double**[hiddenIndex];

		// initialize output weights (to the width of the last hidden layer at index
		// 1 less than the number of layers)
		outputWeights = new double*[hiddenWidths[hiddenIndex] + 1];
		
		// zero input weights
		for (int i = 0; i <= numInput; i++)
		{
			inputWeights[i] = new double[hiddenWidths[0] + 1];

			// for each input node, set the weights for its connections to the first
			// layer of hidden nodes
			zero(inputWeights[i], hiddenWidths[0] + 1);
		};

		// zero hidden weights
		for (int i = 0; i < hiddenIndex; i++)
		{
			hiddenWeights[i] = new double*[hiddenWidths[i] + 1];

			for (int j = 0; j <= hiddenWidths[i]; j++)
			{
				hiddenWeights[i][j] = new double[hiddenWidths[i + 1] + 1];

				for (int k = 0; k <= hiddenWidths[i + 1]; k++)
				{
					hiddenWeights[i][j][k] = 0;
				}	
			}
		}		

		// zero output weights, one weight from each of the last hidden nodes to each output nodes
		for (int i = 0; i <= hiddenWidths[hiddenIndex]; i++)
		{
			outputWeights[i] = new double[numOutput];

			for (int j = 0; j < numOutput; j++)
			{
				outputWeights[i][j] = 0;
			}
		}
	}

	// deeper initializer, hard coded as three hidden layers
	DeepNet(int numInput, int numHidden1, int numHidden2, int numHidden3, int numOutput)
	{
		this->numInput = numInput;

		hiddenLayers = 3;

		numHidden = numHidden1 + numHidden2 + numHidden3;

		hiddenIndex = hiddenLayers - 1;

		hiddenWidths = new int[3];
		hiddenWidths[0] = numHidden1;
		hiddenWidths[1] = numHidden2;
		hiddenWidths[2] = numHidden3;

		this->numOutput = numOutput;

		//cout << "test" << endl;

		// initialize nodes, add 1 for bias neurons
		inputNodes = new double[numInput + 1];
		hiddenNodes = new double*[hiddenLayers];
		outputNodes = new double[numOutput];

		// zero initialize nodes
		zero(inputNodes, numInput + 1);

		// set bias neuron
		inputNodes[numInput] = -1;

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

		// initialize hidden to hidden weights, one less than the number of hidden layers
		// because we are using output weights for the last set of weights
		// to simplify things
		hiddenWeights = new double**[hiddenIndex];

		// initialize output weights (to the width of the last hidden layer at index
		// 1 less than the number of layers)
		outputWeights = new double*[hiddenWidths[hiddenIndex] + 1];

		// zero input weights
		for (int i = 0; i <= numInput; i++)
		{
			inputWeights[i] = new double[hiddenWidths[0] + 1];

			// for each input node, set the weights for its connections to the first
			// layer of hidden nodes
			zero(inputWeights[i], hiddenWidths[0] + 1);
		};

		// zero hidden weights
		for (int i = 0; i < hiddenIndex; i++)
		{
			hiddenWeights[i] = new double*[hiddenWidths[i] + 1];

			for (int j = 0; j <= hiddenWidths[i]; j++)
			{
				hiddenWeights[i][j] = new double[hiddenWidths[i + 1] + 1];

				for (int k = 0; k <= hiddenWidths[i + 1]; k++)
				{
					hiddenWeights[i][j][k] = 0;
				}
			}
		}

		// zero output weights, one weight from each of the last hidden nodes to each output nodes
		for (int i = 0; i <= hiddenWidths[hiddenIndex]; i++)
		{
			outputWeights[i] = new double[numOutput];

			for (int j = 0; j < numOutput; j++)
			{
				outputWeights[i][j] = 0;
			}
		}
	}

	// copy constructor
	DeepNet(DeepNet& net)
	{

	}

	// destructor
	~DeepNet()
	{
		if (inputNodes != NULL) delete[] inputNodes;

		if (hiddenNodes != NULL) delete[] hiddenNodes;

		if (outputNodes != NULL) delete[] outputNodes;

		if (hiddenWidths != NULL) delete[] hiddenWidths;

		if (inputWeights != NULL) delete[] inputWeights;

		if (hiddenWeights != NULL) delete[] hiddenWeights;

		if (outputWeights != NULL) delete[] outputWeights;

		inputNodes, hiddenNodes, outputNodes, hiddenWidths,
			inputWeights, hiddenWeights, outputWeights = NULL;


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

	// random initial value for weights
	double random(double range)
	{
		return (((double) (rand() % 100) + 1) / 100 * 2 * range) - range;
	}

	// get weighted sum of input for a node in the next layer
	double weightedInput(int node)
	{
		double sum = 0;

		for (int i = 0; i <= numInput; i++)
		{
			sum += inputNodes[i] * inputWeights[i][node];
		}

		return sum;
	}

	// get weighted sum of a hidden layer for a node in the next layer
	double weightedHidden(int layer, int node)
	{
		double sum = 0;

		for (int i = 0; i <= hiddenWidths[layer]; i++)
		{
			sum += hiddenNodes[layer][i] * hiddenWeights[layer][i][node];
		}

		return sum;
	}

	// get weighted sum of output for a given output node
	double weightedOutput(int node)
	{
		double sum = 0;

		for (int i = 0; i <= hiddenWidths[hiddenIndex]; i++)
		{
			sum += hiddenNodes[hiddenIndex][i] * outputWeights[i][node];
		}

		return sum;
	}

	// clamp output to a round number
	int clampOutput(double x)
	{
		if (x > 0.9) return 1;
		else if (x < 0.1) return 0;
		else return -1;
	}

	// round number, for formatting output
	int Round(double x)
	{
		return (int) (round(x * 1) / 1);
	}

	// print for ostream operator
	ostream& print(ostream& os)
	{
		// print input
		printLayer(os, inputNodes, numInput + 1);

		os << endl;

		// print input weights
		for (int i = 0; i <= numInput; i++)
		{
			printLayer(os, inputWeights[i], hiddenWidths[0] + 1);
		}

		os << endl;

		// print hidden nodes
		for (int i = 0; i < hiddenLayers; i++)
		{
			printLayer(os, hiddenNodes[i], hiddenWidths[i] + 1);
		}

		os << endl;

		// print hidden weights
		for (int i = 0; i < hiddenIndex; i++)
		{
			for (int j = 0; j <= hiddenWidths[i]; j++)
			{
				printLayer(os, hiddenWeights[i][j], hiddenWidths[i + 1] + 1);
			}

		}

		os << endl;

		// print output weights
		for (int i = 0; i <= hiddenWidths[hiddenIndex]; i++)
		{
			printLayer(os, outputWeights[i], numOutput);
		}

		os << endl;

		// print output nodes
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