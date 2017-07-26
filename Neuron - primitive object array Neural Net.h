#pragma once
#include <iostream>
#include "Data-Structures\Array.h"
#include "Edge - primitive object array Neural Net.h"

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

	// multiplication operator
	double operator* (double weight)
	{
		return this->weight * weight;
	}

	// multiplication operator for edge
	double operator* (Edge& edge)
	{
		return edge * weight;
	}

	// negative operator
	double operator- ()
	{
		return -weight;
	}

	// dereference operator
	double& operator* ()
	{
		return weight;
	}

	/*
	// conversion operator
	operator double()
	{
		return weight;
	}*/

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
		
		for (int i = 0; i < edges; i++)
		{
			this->edges[i] = Edge();
		}

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
