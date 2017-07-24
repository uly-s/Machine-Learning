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
	int k = 0;

	double** patterns = new double*[10000];
	double** targets = new double*[10000];



	for (int i = 0; i < 10000; i++)
	{
		patterns[i] = new double[16];
		targets[i] = new double[3];

		for (int j = 0; j < 19; j++)
		{
			//patterns[i][j] = 0;


			if (j <  16) cin >> patterns[i][j];
			
			else
			{
				cin >> targets[i][k];
				k++;
			}
			

			if (j < 18)	cin >> dummy;
			
			//cout << data[i][j] << " ";
		}

		k = 0;

		//cout << endl;
	}

	for (int i = 0; i < 2000; i++)
	{
		ANN->Test(patterns[i]);
	}

	Trainer* trainer = new Trainer(ANN);

	trainer->useBatchLearning(false);

	trainer->setStoppingConditions(1000, 90);

	trainer->PatternSize(16);
	trainer->TargetSize(3);

	trainer->trainNetwork(1000, patterns, targets);

	cout << *ANN << endl;

	return 0;
}