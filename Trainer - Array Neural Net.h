#pragma once
#include "Neural Net - primitive array.h"

class Trainer
{

protected:

	// pointer to the neural net for training
	NeuralNet* network;

	

public:

	// default constructor
	Trainer()
	{
		network = NULL;
	}

	// initialier
	Trainer(NeuralNet* net)
	{
		network = net;
	}

	~Trainer()
	{

	}

};