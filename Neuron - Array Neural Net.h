#pragma once
#include <iostream>

using namespace std;

class Neuron
{

protected:

	double weight;

	ostream& print(ostream& os)
	{
		os << weight;
		return os;
	}

public:

	friend ostream& operator<< (ostream& os, Neuron& neuron)
	{
		return neuron.print(os);
	}


	Neuron()
	{
		weight = 0;
	}

	Neuron(double weight)
	{
		this->weight = weight;
	}

	virtual ~Neuron()
	{
		weight = 0;
	}

};