#pragma once
#include "Abstract Neural Net.h"
#include "Neuron - Array Neural Net.h"
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


public:



};