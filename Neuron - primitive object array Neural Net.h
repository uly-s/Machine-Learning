#pragma once
#include <iostream>
#include "Data-Structures\Array.h"
#include "Edge - Object Array Neural Net.h"

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

	Neuron& operator= (const Neuron& neuron)
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

	double operator+ (Neuron& neuron)
	{
		return weight + neuron.weight;
	}

	double operator+ (const double& weight)
	{
		return this->weight + weight;
	}

	double operator+= (Neuron& neuron)
	{
		this->weight = this->weight + neuron.weight;
		return this->weight;
	}

	double operator+= (const double& weight)
	{
		this->weight = this->weight + weight;
		return this->weight;
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
