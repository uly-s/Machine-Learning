#pragma once
#include "Deep Neural Network - non convolutional - array implementation.h"


class Trainer
{

protected:
	
	// pointer to network
	DeepNet* net;

	// input error
	double* inputError;

	// hidden error
	double** hiddenError;

	// output error
	double* outputError;

	// changed to input weights
	double** deltaInput;

	// changed to hidden
	double*** deltaHidden;

	// changed to output weights
	double** deltaOutput;



public:

	// default constructor
	Trainer()
	{
		net = NULL;

		inputError, hiddenError, outputError = NULL;

		deltaInput, deltaHidden, deltaOutput = NULL;
	}

	// initializer
	Trainer(DeepNet* network)
	{
		net = network;

		inputError = new double[net->numInput];

		for (int i = 0; i <= net->numInput; i++)
		{
			inputError[i] = 0;

		}


		hiddenError = new double*[net->hiddenLayers];

		for (int i = 0; i < net->hiddenLayers; i++)
		{
			hiddenError[i] = new double[net->hiddenWidths[i]];

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