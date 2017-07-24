#include <iostream>
#include <typeinfo>
#include "Neural Net - primitive array.h"
//#include "Trainer - Array Neural Net.h"

using namespace std;


int main()
{
	

	NeuralNet* ANN = new NeuralNet(10, 10, 10);

	ANN->InitializeWeights();

	cout << *ANN << endl;

	char dummy;

	double** data = new double*[20000];

	for (int i = 0; i < 20000; i++)
	{
		data[i] = new double[19];

		for (int j = 0; j < 19; j++)
		{
			data[i][j] = 0;

			cin >> data[i][j];

			if (j < 18)	cin >> dummy;
			
			//cout << data[i][j] << " ";
		}

		//cout << endl;
	}

	for (int i = 0; i < 20000; i++)
	{
		ANN->Test(data[i]);
	}
	
	cout << *ANN << endl;

	return 0;
}