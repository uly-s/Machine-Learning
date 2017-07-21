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
	};

public:

	friend ostream& operator<< (ostream& os, Neuron& neuron)
	{
		return neuron.print(os);
	};

	Neuron& operator= (Neuron& neuron)
	{
		weight = neuron.weight;

		return *this;
	}

	Neuron& operator= (const double& weight)
	{
		this->weight = weight;

		return *this;
	}

	bool operator== (Neuron& neuron)
	{
		return weight == neuron.weight;
	}

	bool operator== (const double& weight)
	{
		return this->weight == weight;
	}

	Neuron()
	{
		weight = 0;
	};

	Neuron(double weight)
	{
		this->weight = weight;
	}

	Neuron(Neuron& neuron)
	{
		weight = neuron.weight;
	}

	virtual ~Neuron()
	{
		weight = 0;
	}

};