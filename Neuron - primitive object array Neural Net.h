#pragma once
#include <iostream>
#include "Data-Structures\Array.h"
#include "Edge - Object Array Neural Net.h"

using namespace std;

class Neuron
{

protected:

	// field for input and passing on
	double weight;

	// edges of the node
	Edge* edges;

	// ostream assistant
	ostream& print(ostream& os)
	{
		os << weight;
		return os;
	};

public:

	// ostream operator
	friend ostream& operator<< (ostream& os, Neuron& neuron)
	{
		return neuron.print(os);
	};

	// equal to neuron operator
	bool operator== (Neuron& neuron)
	{
		return weight == neuron.weight;
	}

	// equal to double operator
	bool operator== (const double& weight)
	{
		return this->weight == weight;
	}

	// neuron assignment operator
	Neuron& operator= (const Neuron& neuron)
	{
		weight = neuron.weight;

		return *this;
	}

	// double assignment operator
	Neuron& operator= (const double& weight)
	{
		this->weight = weight;

		return *this;
	}

	// neuron addition operator
	double operator+ (Neuron& neuron)
	{
		return weight + neuron.weight;
	}

	// double addition operator
	double operator+ (const double& weight)
	{
		return this->weight + weight;
	}

	// Neuron plus equals operator
	double operator+= (Neuron& neuron)
	{
		this->weight = this->weight + neuron.weight;
		return this->weight;
	}

	// double plus equals operator
	double operator+= (const double& weight)
	{
		this->weight = this->weight + weight;
		return this->weight;
	}

	// bracket operator
	Edge& operator[] (int index)
	{
		return edges[index];
	}

	// default constructor
	Neuron()
	{
		edges = NULL;
		weight = 0;
	};

	// initializer
	Neuron(double weight)
	{
		edges = NULL;
		this->weight = weight;
	}

	// initializer 
	Neuron(int edges)
	{
		this->edges = new Edge[edges];
		
		weight = 0;
	}

	// copy constructor
	Neuron(Neuron& neuron)
	{
		weight = neuron.weight;
	}

	// destructor
	virtual ~Neuron()
	{
		weight = 0;
	}

};
