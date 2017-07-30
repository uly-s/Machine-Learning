#include <iostream>
#include <typeinfo>
#include <vector>
#include "Deep Neural Network - non convolutional - array implementation.h"
#include "Trainer - non conv deep net - array.h"


int main()
{
	char dummy = ' ';

	int index = 0;

	double*** training = new double**[10];

	double*** labels = new double**[10];

	double*** labelArrays = new double**[10];

	for (int i = 0; i < 2; i++)
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

				if (k == labels[i][j][k])
				{
					labelArrays[i][j][k] = 1;
				}

			}

			cin >> dummy;

			for (int k = 0; k < 784; k++)
			{
				training[i][j][k] = 0;
				
				cin >> training[i][j][k];

				if (k < 783) cin >> dummy;
			}
		}
	}

	DeepNet* deepnet = new DeepNet(784, 800, 600, 10);

	Trainer* trainer = new Trainer(deepnet);

	deepnet->InitializeWeights();

	trainer->Parameters(50, 0.1, 90, 2000);
	
	trainer->Train(training, labelArrays, 2000);
	

	return 0;
}