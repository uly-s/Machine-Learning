#pragma once
#include <iostream>

using namespace std;

class Weight
{

protected:

	double weight;

	ostream& print(ostream& os)
	{
		os << weight;
		return os;
	}


public:

	Weight& operator= (const double& weight)
	{
		this->weight = weight;

		return *this;
	}

	Weight& operator= (Weight& weight)
	{
		this->weight = weight.weight;

		return *this;
	}

	bool operator== (const double& weight)
	{
		return this->weight == weight;
	}

	bool operator== (Weight& weight)
	{
		return this->weight = weight.weight;
	}

	friend ostream& operator<< (ostream& os, Weight& weight)
	{
		return weight.print(os);
	}

	Weight()
	{
		weight = 0;
	}

	Weight(double weight)
	{
		this->weight = weight;
	}

	~Weight()
	{

	}
};
