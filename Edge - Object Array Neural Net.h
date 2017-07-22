#pragma once
#include <iostream>

using namespace std;

class Edge
{

protected:

	double weight;

	ostream& print(ostream& os)
	{
		os << weight;
		return os;
	}


public:

	Edge& operator= (const double& weight)
	{
		this->weight = weight;

		return *this;
	}

	Edge& operator= (Edge& edge)
	{
		this->weight = edge.weight;

		return *this;
	}

	bool operator== (const double& weight)
	{
		return this->weight == weight;
	}

	bool operator== (Edge& edge)
	{
		return this->weight = edge.weight;
	}

	friend ostream& operator<< (ostream& os, Edge& edge)
	{
		return edge.print(os);
	}

	Edge()
	{
		weight = 0;
	}

	Edge(double weight)
	{
		this->weight = weight;
	}

	~Edge()
	{

	}
};
