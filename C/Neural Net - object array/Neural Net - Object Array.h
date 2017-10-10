#pragma once
#include "Abstract Neural Net.h"
#include "Neuron - Array Neural Net.h"
#include "Edge - Object Array Neural Net.h"
#include "Data-Structures\Array.h"

// Array neural network

class NeuralNet : public AbstractNeuralNet
{

protected:

	// number of neurons
	int numInput, numHidden, numOutput;

	// input neurons
	Array<Neuron>* inputNeurons;

	// hidden neurons
	Array<Neuron>* hiddenNeurons;

	// output neurons
	Array<Neuron>* outputNeurons;

	// input to hidden weights
	Array<Array<Edge>*>* inputWeights;

	// hidden to output weights
	Array<Array<Edge>*>* outputWeights;

	// initialize neurons
	void initializeNeurons()
	{
		// initialize input neurons
		inputNeurons = new Array<Neuron>(numInput + 1);

		// initialize hidden neurons
		hiddenNeurons = new Array<Neuron>(numHidden + 1);

		// initialize output neurons
		outputNeurons = new Array<Neuron>(numOutput);
	}

	// neuron initializer
	void initialize(Array<Neuron>& nerve)
	{
		for (int i = 0; i < nerve.Size(); i++) nerve[i] = Neuron();
	}

	// weight initializer helper
	void initialize(Array<Edge>& nerve)
	{
		for (int i = 0; i < nerve.Size(); i++) nerve[i] = Edge();
	}

	// weight initializer (zero not random)
	void initialize(Array<Array<Edge>*>& weights, int height, int width)
	{
		for (int i = 0; i < height; i++)
		{
			weights[i] = new Array<Edge>(width);

			initialize(*(weights[i]));
		}
	}


public:

	// default constructor
	NeuralNet()
	{
		numInput, numHidden, numOutput = 0;

		inputNeurons, hiddenNeurons, outputNeurons,
			inputWeights, outputWeights = NULL;
	}

	// initializer
	NeuralNet(int in, int hidden, int out)
	{
		numInput = in;
		numHidden = hidden;
		numOutput = out;

		inputNeurons = new Array<Neuron>(in);
		hiddenNeurons = new Array<Neuron>(hidden);
		outputNeurons = new Array<Neuron>(out);

		initialize(*inputNeurons);
		initialize(*hiddenNeurons);
		initialize(*outputNeurons);

		inputWeights = new Array<Array<Edge>*>(in);
		outputWeights = new Array<Array<Edge>*>(hidden);

		initialize(*inputWeights, in, hidden);
		initialize(*outputWeights, hidden, out);

	}



};

