#pragma once
<<<<<<< HEAD
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
=======
#include "Data-Structures\Array.h"
>>>>>>> 85c06ffcd520efe296db98b13ada9299da324419
