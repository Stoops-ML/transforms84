#include "../include/distances.c"

#include <stdio.h>
#include <time.h>

#define N 10000000

double wgs84_mean_radius = (2.0 * 6378137.0 + 6356752.314245) / 3.0;

int main()
{
    int i;
    double *rrmStart = (double *)malloc(N * 3 * sizeof(double));
    double *rrmEnd = (double *)malloc(N * 3 * sizeof(double));
    double *mDistance = (double *)malloc(N * sizeof(double));
    clock_t start, end;
    for (i = 0; i < N; ++i) {
        rrmStart[i * 3] = 33.0 + (double)i / N;
        rrmStart[i * 3 + 1] = 34.0 + (double)i / N;
        rrmStart[i * 3 + 2] = 0.0;
        rrmEnd[i * 3] = 32.0 + (double)i / N;
        rrmEnd[i * 3 + 1] = 38.0 + (double)i / N;
        rrmEnd[i * 3 + 2] = 0.0;
    }

    start = clock();
    HaversineDouble(
        rrmStart,
        rrmEnd,
        N,
        1,
        wgs84_mean_radius,
        mDistance);
    end = clock();
    printf("Haversine distances (%lg): %f, %f, %f\n", 
        (end - start) / (long double)CLOCKS_PER_SEC,
        mDistance[0],
        mDistance[1],
        mDistance[2]);
}