#include <iostream>


using namespace std;


int main()
{


	char dummy;

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

				cout << patternSets[i][j][k] << " ";
			}

			for (int k = 0; k < 1; k++)
			{
				targetSets[i][j][k] = 0;

				cin >> targetSets[i][j][k];

				if (k < 1) cin >> dummy;

				cout << patternSets[i][j][k] << " ";
			}

			cout << endl;
		}
	}

	return 0;
}