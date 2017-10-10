#include <iostream>
#include <typeinfo>
#include "Neural Net - primitive array.h"
#include "Trainer - primitive array Neural Net.h"

using namespace std;


int main()
{
	

	NeuralNet* ANN = new NeuralNet(18, 8, 1);

	ANN->InitializeWeights();

	//cout << *ANN << endl;

	char dummy;

	double** patterns = new double*[20000];
	double** targets = new double*[20000];

	double*** patternSets = new double**[20];
	double*** targetSets = new double**[20];

	
	for (int i = 0; i < 20; i++)
	{
		patternSets[i] = new double*[1000];
		targetSets[i] = new double*[1000];

		for (int j = 0; j < 1000; j++)
		{
			patternSets[i][j] = new double[18];
			targetSets[i][j] = new double[1];

			for (int k = 0; k < 18; k++)
			{
				patternSets[i][j][k] = 0;

				cin >> patternSets[i][j][k];

				cin >> dummy;
			}

			for (int k = 0; k < 1; k++)
			{
				targetSets[i][j][k] = 0;

				cin >> targetSets[i][j][k];

				if (k < 1) cin >> dummy;
			}
		}
	}
	
	/*

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

		for (int j = 0; j < 3; j++)
		{
			targets[i][j] = 0;

			cin >> targets[i][j];

			if (j < 2) cin >> dummy;
		}



	}

	*/
	

	

	Trainer* trainer = new Trainer(ANN);

	trainer->setTrainingParameters(0.01, 0.9, false);

	trainer->setStoppingConditions(5000, 90);

	trainer->PatternSize(18);
	
	trainer->TargetSize(1);

	for (int i = 0; i < 10; i++)
	{
		trainer->trainNetwork(1000, patternSets[i], targetSets[i]);
	}

	cout << "Generalization set accuracy: " << ANN->getSetAccuracy(1000, patternSets[11], targetSets[11]) << "%\n\n";
	

	//cout << *ANN << endl;





	return 0;
}
