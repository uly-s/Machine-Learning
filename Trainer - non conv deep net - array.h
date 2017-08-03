#pragma once
#include "Deep Neural Network - non convolutional - array implementation.h"


class Trainer
{

protected:
	
	// FIELDS

	// pointer to network
	DeepNet* net;

	// learning rate
	double LR;

	// desired accuracy
	double accuracy;

	// epoch we are on
	int epoch;

	// number of epochs
	int epochs;

	// max epochs
	int maxEpochs;

	// batch size
	int batchSize;

	// size of training set
	int setSize;

	// number of wrong answers
	int wrong;

	// number of wrong ansers in a batch
	int wrongBatch;

	// input error
	double* inputError;

	// hidden error
	double** hiddenError;

	// output error
	double* outputError;

	// input error gradient
	double* inputGradient;

	// hidden error gradient
	double** hiddenGradient;

	// output gradient
	double* outputGradient;

	// changed to input weights
	double** deltaInput;

	// changed to hidden
	double*** deltaHidden;

	// changed to output weights
	double** deltaOutput;

	// PROTECTED METHODS

	// backpropagate 
	void backpropagate(double* targets)
	{
		// set output errors
		for (int i = 0; i < net->numOutput; i++)
		{
			outputError[i] = targets[i] - net->outputNodes[i];

			//cout << "[" << targets[i] << " " << net->outputNodes[i] << "] ";

		//	outputGradient[i] = targets[i] * (1 - targets[i]) * outputError[i];
		}

		//cout << endl;

		// call backpropagate output
		backpropagateOutput();

		// call backpropagate hidden
		backpropagateHidden();

		// call backpropagate input
		backpropagateInput();

		// get weight changes
		getDeltaInput();
		getDeltaHidden();
		getDeltaOutput();

	}

	// backpropagate output nodes and output weights
	void backpropagateOutput()
	{
		// for each node in the last hidden layer
		for (int i = 0; i <= net->hiddenWidths[net->hiddenIndex]; i++)
		{
			// clear hidden error
			hiddenError[net->hiddenIndex][i] = 0;

			// get weighted sum
			//double sum = 0;

			// multiply the error at output by the weight of the hidden node leading to that output
			// to get the weighted rror at that node
			for (int j = 0; j < net->numOutput; j++)
			{
				hiddenError[net->hiddenIndex][i] += outputError[j] * net->outputWeights[i][j];

			//	sum += net->outputWeights[i][j] * outputGradient[j];

				
			}

			//hiddenGradient[net->hiddenIndex][i] = net->hiddenNodes[net->hiddenIndex][i] * (1 - net->hiddenNodes[net->hiddenIndex][i]) * sum;
		}


	}

	// backpropagate from the bottom hidden layer to the top
	void backpropagateHidden()
	{
		// for each hidden layer, get the weighted error
		// from the previous layer
		for (int i = net->hiddenIndex - 1; i >= 0; i--)
		{
			// for each node in the layer
			for (int j = 0; j <= net->hiddenWidths[i]; j++)
			{
				// clear hidden error
				hiddenError[i][j] = 0;

				// get weighted sum
				//double sum = 0;

				// for each hidden weight
				for (int k = 0; k <= net->hiddenWidths[i + 1]; k++)
				{
					hiddenError[i][j] += hiddenError[i + 1][k] * net->hiddenWeights[i][j][k];

					//sum += hiddenGradient[i + 1][j] * net->hiddenWeights[i][j][k];
				}

				//hiddenGradient[i][j] = net->hiddenNodes[i][j] * (1 - net->hiddenNodes[i][j]) * sum;
			}
		}
	}

	// backpropagate hidden errors to input layer
	void backpropagateInput()
	{

		// for each input node
		for (int i = 0; i <= net->numInput; i++)
		{
			// clear input error
			inputError[i] = 0;

			double sum = 0;

			// for each node in the top hidden layer
			for (int j = 0; j <= net->hiddenWidths[0]; j++)
			{
				inputError[i] += hiddenError[0][j] * net->inputWeights[i][j];	
			}

			//cout << endl;

			//inputGradient[i] = net->inputNodes[i] * (1 - net->inputNodes[i]) * sum;
		}
	}

	// get delta input
	void getDeltaInput()
	{
		// for each input node
		for (int i = 0; i <= net->numInput; i++)
		{
			// for each node in the next row
			for (int j = 0; j <= net->hiddenWidths[0]; j++)
			{
				// change in input equals learning rate times error at node times sigmoid prime of input times input
				deltaInput[i][j] += LR * hiddenError[0][j] * net->sigmoidPrime(net->hiddenNodes[0][j]) * net->inputNodes[i];
			}
		}
	}

	// get delta hidden
	void getDeltaHidden()
	{
		// for each hidden layer
		for (int i = 0; i < net->hiddenIndex; i++)
		{
			// for each node in the first layer
			for (int j = 0; j <= net->hiddenWidths[i]; j++)
			{
				// update each hidden weight
				for (int k = 0; k <= net->hiddenWidths[i + 1]; k++)
				{
					// change in weight equals learning rate times error at node
					// times sigmoid prime of node value times node value
					deltaHidden[i][j][k] += LR * hiddenError[i + 1][k] * net->sigmoidPrime(net->hiddenNodes[i + 1][k])
						* net->hiddenNodes[i][j];
				}
			}
		}
	}

	// get delta output
	void getDeltaOutput()
	{
		// for each node in the last hidden layer
		for (int i = 0; i <= net->hiddenWidths[net->hiddenIndex]; i++)
		{
			// for each output node
			for (int j = 0; j < net->numOutput; j++)
			{
				// change in output equals learning rate times error at node times
				// sigmoid prime of node value times node value
				deltaOutput[i][j] += LR * outputError[j] * net->sigmoidPrime(net->outputNodes[j]) 
					* net->hiddenNodes[net->hiddenIndex][i];
			}
		}
	}

	// update weights
	void UpdateWeights()
	{
		// update input weights
		
		// for each input node
		for (int i = 0; i <= net->numInput; i++)
		{
			// for each node in the first hidden layer
			for (int j = 0; j <= net->hiddenWidths[0]; j++)
			{
				net->inputWeights[i][j] += deltaInput[i][j];

				deltaInput[i][j] = 0;
			}
		}

		// update hidden weights
		
		// for each hidden layer
		for (int i = 0; i < net->hiddenIndex; i++)
		{
			// for each node in layer i
			for (int j = 0; j <= net->hiddenWidths[i]; j++)
			{
				// for each node in the next layer
				for (int k = 0; k <= net->hiddenWidths[i + 1]; k++)
				{
					net->hiddenWeights[i][j][k] += deltaHidden[i][j][k];

					deltaHidden[i][j][k] = 0;
				}
			}
		}

		// update output weights

		// for each node int the last hidden layer
		for (int i = 0; i < net->hiddenWidths[net->hiddenIndex]; i++)
		{
			// for each output nodes
			for (int j = 0; j < net->numOutput; j++)
			{
				net->outputWeights[i][j] += deltaOutput[i][j];

				deltaOutput[i][j] = 0;
			}
		}



	}

	// run individual training epoch
	void Epoch(double* input, double* targets)
	{
		net->FeedFoward(input);

		backpropagate(targets);

		bool correct = true;

		for (int i = 0; i < net->numOutput; i++)
		{
			if (net->clampOutput(net->outputNodes[i]) != targets[i])
			{
				correct = false;
			}
		}

		epoch++;
		epochs++;

		if (!correct)
		{
			wrong++;
			wrongBatch++;
		}
	}

	// run batch
	void Batch(double** inputs, double** targets, int index)
	{
		for (int i = 0; i < batchSize; i++)
		{
			Epoch(inputs[index + i], targets[index + i]);
		}

		UpdateWeights();
	}


public:

	// PUBLIC METHODS
	
	
	// train on data for some number of epochs
	void Train(double** data, double** targets, int epochs)
	{
		double trainingAccuracy = 0;

		int index = 0;

		epoch = 0;

		bool correct = true;

		while (epoch < epochs && epoch < maxEpochs)
		{
			for (int i = index; i < index + batchSize && epoch < epochs; i++)
			{
				correct = true;

				net->FeedFoward(data[i]);

				backpropagate(targets[i]);

				for (int j = 0; j < net->numOutput; j++)
				{
					if (net->clampOutput(net->outputNodes[j]) != targets[i][j])
					{
						correct = false;
					}
				}


				if (!correct)
				{
					wrong++;
					wrongBatch++;
				}

				epoch++;
				this->epochs++;
			}

			UpdateWeights();


			cout << "Average: " << 100 - ((double) wrong / (double) this->epochs * 100);
			
			cout << ",  batch: " << 100 - ((double) wrongBatch / (double) batchSize * 100)
				<< ",  epochs: " << this->epochs << endl;

			index += batchSize;

			wrongBatch = 0;

		}

		trainingAccuracy = 100 - ((double) wrong / (double) this->epochs * 100);

		cout << "Training set accuracy: " << trainingAccuracy << ", wrong: " << wrong << ", epochs: " << epochs << endl;

		wrong = 0;


	}

	// set training parameters, batch size, learning rate, desired accuracy, max epochs
	void Parameters(int batchSize, double learningRate, double targetAccuracy, int maxEpochs)
	{
		this->batchSize = batchSize;
		
		LR = learningRate;

		accuracy = targetAccuracy;

		this->maxEpochs = maxEpochs;
	}

	// OPERATORS

	// CONSTRUCTORS

	// default constructor
	Trainer()
	{
		net = NULL;

		accuracy, epochs, batchSize, maxEpochs, LR, wrong, epoch, wrongBatch = 0;

		inputError, hiddenError, outputError = NULL;

		inputGradient, hiddenGradient, outputGradient = NULL;

		deltaInput, deltaHidden, deltaOutput = NULL;
	}

	// initializer
	Trainer(DeepNet* network)
	{
		// I've been working for a while and I'm feeling no motivation to document this

		net = network;

		accuracy, epochs, batchSize, maxEpochs, LR, wrong, epoch = 0;

		wrong = 0;
		wrongBatch = 0;
		epochs = 0;
		setSize = 0;

		inputError = new double[net->numInput + 1];

		for (int i = 0; i <= net->numInput; i++)
		{
			inputError[i] = 0;

		}


		hiddenError = new double*[net->hiddenLayers];

		for (int i = 0; i < net->hiddenLayers; i++)
		{
			hiddenError[i] = new double[net->hiddenWidths[i] + 1];
			
			for (int j = 0; j <= net->hiddenWidths[i]; j++)
			{
				hiddenError[i][j] = 0;

			}

		}

		outputError = new double[net->numOutput];

		for (int i = 0; i < net->numOutput; i++)
		{
			outputError[i] = 0;

		}

		inputGradient = new double[net->numInput + 1];

		for (int i = 0; i <= net->numInput; i++)
		{
			inputGradient[i] = 0;
		}

		hiddenGradient = new double*[net->hiddenLayers];

		for (int i = 0; i < net->hiddenLayers; i++)
		{
			hiddenGradient[i] = new double[net->hiddenWidths[i] + 1];

			for (int j = 0; j <= net->hiddenWidths[i]; j++)
			{
				hiddenGradient[i][j] = 0;
			}
		}

		outputGradient = new double[net->numOutput];

		for (int i = 0; i < net->numOutput; i++)
		{
			outputGradient[i] = 0;
		}


		deltaInput = new double*[net->numInput + 1];

		for (int i = 0; i <= net->numInput; i++)
		{
			deltaInput[i] = new double[net->hiddenWidths[0]];

			for (int j = 0; j <= net->hiddenWidths[0]; j++)
			{
				deltaInput[i][j] = 0;

			}

		}


		deltaHidden = new double**[net->hiddenLayers];

		for (int i = 0; i < net->hiddenIndex; i++)
		{
			deltaHidden[i] = new double*[net->hiddenWidths[i] + 1];

			for (int j = 0; j <= net->hiddenWidths[i]; j++)
			{
				deltaHidden[i][j] = new double[net->hiddenWidths[i + 1] + 1];

				for (int k = 0; k <= net->hiddenWidths[i + 1]; k++)
				{
					deltaHidden[i][j][k] = 0;

				}

			}

		}


		deltaOutput = new double*[net->hiddenWidths[net->hiddenIndex] + 1];

		for (int i = 0; i <= net->hiddenWidths[net->hiddenIndex]; i++)
		{
			deltaOutput[i] = new double[net->numOutput];

			for (int j = 0; j < net->numOutput; j++)
			{
				deltaOutput[i][j] = 0;

			}

		}

	}

	// copy constructor
	Trainer(Trainer& trainer)
	{

	}

	// destructor
	~Trainer()
	{

	}


protected:










};