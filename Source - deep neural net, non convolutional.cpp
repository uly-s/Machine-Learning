#include <iostream>
#include <fstream>
#include <typeinfo>
#include <vector>
#include <math.h>
#include "Deep Neural Network - non convolutional - array implementation.h"
#include "Trainer - non conv deep net - array.h"

using namespace std;


void LoadTrainingData(int thousands, double*** training, double*** labels, double*** labelArrays, double** patterns, double** targets)
{
	char dummy = ' ';

	int index = 0;

	for (int i = 0; i < thousands; i++)
	{
		training[i] = new double*[1000];

		labels[i] = new double*[1000];

		labelArrays[i] = new double*[1000];

		for (int j = 0; j < 1000; j++)
		{
			training[i][j] = new double[784];
			labels[i][j] = new double[1];
			labelArrays[i][j] = new double[10];

			labels[i][j][0] = 0;

			cin >> labels[i][j][0];

			for (int k = 0; k < 10; k++)
			{
				labelArrays[i][j][k] = 0;

				if (k == labels[i][j][0])
				{
					labelArrays[i][j][k] = 1;
				}
			}

			cin >> dummy;

			for (int k = 0; k < 784; k++)
			{
				training[i][j][k] = 0;

				cin >> training[i][j][k];

				if (training[i][j][k] > 0) training[i][j][k] = 1;

				if (k < 783) cin >> dummy;

			}
		}
	}


}


int main()
{
	double*** training = new double**[50];

	double*** labels = new double**[50];

	double*** labelArrays = new double**[50];

	double** patterns = new double*[5000];

	double** targets = new double*[5000];

	/*
	for (int i = 0; i < 3000; i++)
	{
		patterns[i] = new double[784];
		targets[i] = new double[10];

		for (int j = 0; j < 784; j++)
		{
			patterns[i][j] = 0;
		}

		for (int j = 0; j < 10; j++)
		{
			targets[i][j] = 0;
		}
	}*/

	LoadTrainingData(2, training, labels, labelArrays, patterns, targets);

	DeepNet* deepnet = new DeepNet(784, 28, 112, 224, 10);
	
	Trainer* trainer = new Trainer(deepnet);

	deepnet->InitializeWeights();

	trainer->Parameters(25, 1.0, 90, 5000);
	
	trainer->Train(training[0], labelArrays[0], 1000);

	cout << deepnet->getSetAccuracy(training[1], labelArrays[1], 1000, 0) << endl;

	//cout << deepnet->getSetAccuracy(training[3], labelArrays[3], 1000) << endl;

	return 0;
}