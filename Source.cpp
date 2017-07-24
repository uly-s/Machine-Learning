#include <iostream>
#include <typeinfo>
#include "Neural Net - primitive array.h"
#include "Trainer - Array Neural Net.h"

using namespace std;


int main()
{
	

	NeuralNet* ANN = new NeuralNet(16, 10, 3);

	ANN->InitializeWeights();

	cout << *ANN << endl;

	char dummy;

	double** patterns = new double*[1000];
	double** targets = new double*[1000];



	for (int i = 0; i < 1000; i++)
	{
		patterns[i] = new double[16];
		targets[i] = new double[3];

		for (int j = 0; j < 16; j++)
		{
			patterns[i][j] = 0;

			cin >> patterns[i][j];

			if (j < 18)	cin >> dummy;
			
		}

		for (int k = 0; k < 3; k++)
		{
			targets[i][k] = 0;

			cin >> targets[i][k];

			if (k < 2) cin >> dummy;
		}
		
	}

	for (int i = 0; i < 1000; i++)
	{
		for (int j = 0; j < 16; j++)
		{
		//	cout << patterns[i][j] << " ";
		}

		for (int k = 0; k < 3; k++)
		{
		//	cout << targets[i][k] << " ";
		}

		//cout << endl;
	}


	Trainer* trainer = new Trainer(ANN);

	trainer->setTrainingParameters(0.01, 0.9, false);

	trainer->setStoppingConditions(2000, 90);

	trainer->PatternSize(16);
	trainer->TargetSize(3);

	trainer->trainNetwork(1000, patterns, targets);

	cout << *ANN << endl;





	return 0;
}