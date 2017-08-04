#include <iostream>
#include <fstream>
#include <typeinfo>
#include <vector>
#include <math.h>
#include "Deep Neural Network - non convolutional - array implementation.h"
#include "Trainer - non conv deep net - array.h"

using namespace std;


// x labels, x * 784 patterns, x * 10 targets
void LoadTrainingData(int samples, double* labels, double** patterns, double** targets)
{
	char dummy = ' ';

	int index = 0;

	for (int i = 0; i < samples; i++)
	{
		patterns[i] = new double[784];
		targets[i] = new double[10];
		labels[i] = 0;

		// get label
		cin >> labels[i];

		// get comma
		cin >> dummy;

		for (int j = 0; j < 10; j++)
		{
			targets[i][j] = 0;

			// if index = label this is the target
			if (j == labels[i]) targets[i][j] = 1;
		}

		for (int j = 0; j < 784; j++)
		{
			patterns[i][j] = 0;

			// get pattern
			cin >> patterns[i][j];

			// get comma if we aren't at the end of the line
			if (j < 783) cin >> dummy;
		}
	}

}

// shuffle training data
void Shuffle(int samples, double* labels, double** patterns, double** targets)
{

}

// swap two array elements
void swap(double& A, double& B)
{
	// store the value of A
	double C = A;

	// set A to B
	A = B;

	// set B to C
	B = C;
}

// get a random index in a certain range
int randomIndex(int order)
{
	return (int) (rand() % order);
}




int main()
{

	// digits
	double* labels = new double[5000];

	// 28 * 28 patterns of images
	double** patterns = new double*[5000];

	// target arrays 0 - 9
	double** targets = new double*[5000];

	// read in x training samples 
	LoadTrainingData(2000, labels, patterns, targets);

	DeepNet* deepnet = new DeepNet(784, 112, 56, 10);

	deepnet->InitializeWeights();
	
	Trainer* trainer = new Trainer(deepnet);

	trainer->Parameters(25, 1.0, 90, 5000);
	
	//trainer->Train(patterns, targets, 1000);

	//cout << deepnet->getSetAccuracy(patterns, targets, 1000, 999) << endl;

	int order = 10;

	for (int i = 0; i < 10; i++)
	{
		cout << randomIndex(order) << endl;
	}


	return 0;
}
