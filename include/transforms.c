#include <Python.h>
#include <float.h>
#include <math.h>
#include <numpy/arrayobject.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_procs() 1
#endif

#include "definitions.h"

/*
UTM to geodetic transformation of double precision.
https://fypandroid.wordpress.com/2011/09/03/converting-utm-to-latitude-and-longitude-or-vice-versa/

@param double *mmUTM array of size nx1 easting, northing[m, m]
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
*/
void UTM2geodeticDoubleUnrolled(const double* mUTMX,
    const double* mUTMY,
    long ZoneNumber,
    char* ZoneLetter,
    long nPoints,
    double a,
    double b,
    double* radLat,
    double* radLong,
    double* mAlt)
{
    double k0 = 0.9996;
    double e2 = 1.0 - (b * b) / (a * a);
    double e = sqrt(e2);
    double ed2 = ((a * a) - (b * b)) / (b * b);
    double lon0 = (((double)ZoneNumber - 1.0) * 6.0 - 177.0) * PI / 180.0;
    double e1 = (1.0 - sqrt(1.0 - e2)) / (1.0 + sqrt(1.0 - e2));
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        double x = mUTMX[iPoint] - 500000.0;
        double y = mUTMY[iPoint];
        if (ZoneLetter[0] < 'N')
            y -= 10000000.0;
        double m = y / k0;
        double mu = m / (a * (1 - e2 / 4.0 - 3.0 * pow(e, 4) / 64.0 - 5.0 * pow(e, 6) / 256.0));
        double j1 = 3.0 * e1 / 2 - 27.0 * pow(e1, 3) / 32.0;
        double j2 = 21.0 * pow(e1, 2) / 16.0 - 55.0 * pow(e1, 4) / 32.0;
        double j3 = 151.0 * pow(e1, 3) / 96.0;
        double j4 = 1097.0 * pow(e1, 4) / 512.0;
        double fp_lat = mu + j1 * sin(2.0 * mu) + j2 * sin(4.0 * mu) + j3 * sin(6.0 * mu) + j4 * sin(8.0 * mu);
        double c1 = ed2 * pow(cos(fp_lat), 2);
        double t1 = pow(tan(fp_lat), 2);
        double r1 = a * (1 - e2) / pow((1 - e2 * pow(sin(fp_lat), 2)), 1.5);
        double n1 = a / sqrt(1 - e2 * pow(sin(fp_lat), 2));
        double d = x / (n1 * k0);
        double q1 = n1 * tan(fp_lat) / r1;
        double q2 = pow(d, 2) / 2.0;
        double q3 = (5 + 3 * t1 + 10 * c1 - 4 * pow(c1, 2) - 9.0 * ed2) * pow(d, 4) / 24.0;
        double q4 = (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * pow(t1, 2) - 252.0 * ed2 - 3 * pow(c1, 2)) * pow(d, 6) / 720.0;
        radLat[iPoint] = fp_lat - q1 * (q2 - q3 + q4);
        double q5 = d;
        double q6 = (1.0 + 2.0 * t1 + c1) * pow(d, 3) / 6.0;
        double q7 = (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * pow(c1, 2) + 8.0 * ed2 + 24.0 * pow(t1, 2)) * pow(d, 5) / 120.0;
        radLong[iPoint] = lon0 + (q5 - q6 + q7) / cos(fp_lat);
        mAlt[iPoint] = 0.0;
    }
}

/*
UTM to geodetic transformation of float precision.

@param double *mmUTM array of size nx1 easting, northing[m, m]
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
*/
void UTM2geodeticFloatUnrolled(const float* mUTMX,
    const float* mUTMY,
    long ZoneNumber,
    char* ZoneLetter,
    long nPoints,
    float a,
    float b,
    float* radLat,
    float* radLong,
    float* mAlt)
{
    float k0 = 0.9996f;
    float e2 = 1.0f - (b * b) / (a * a);
    float e = sqrtf(e2);
    float ed2 = ((a * a) - (b * b)) / (b * b);
    float lon0 = (((float)ZoneNumber - 1.0f) * 6.0f - 177.0f) * PIf / 180.0f;
    float e1 = (1.0f - sqrtf(1.0f - e2)) / (1.0f + sqrtf(1.0f - e2));
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        float x = mUTMX[iPoint] - 500000.0f;
        float y = mUTMY[iPoint];
        if (ZoneLetter[0] < 'N')
            y -= 10000000.0f;
        float m = y / k0;
        float mu = m / (a * (1 - e2 / 4.0f - 3.0f * powf(e, 4) / 64.0f - 5.0f * powf(e, 6) / 256.0f));
        float j1 = 3.0f * e1 / 2 - 27.0f * powf(e1, 3) / 32.0f;
        float j2 = 21.0f * powf(e1, 2) / 16.0f - 55.0f * powf(e1, 4) / 32.0f;
        float j3 = 151.0f * powf(e1, 3) / 96.0f;
        float j4 = 1097.0f * powf(e1, 4) / 512.0f;
        float fp_lat = mu + j1 * sinf(2.0f * mu) + j2 * sinf(4.0f * mu) + j3 * sinf(6.0f * mu) + j4 * sinf(8.0f * mu);
        float c1 = ed2 * powf(cosf(fp_lat), 2);
        float t1 = powf(tanf(fp_lat), 2);
        float r1 = a * (1 - e2) / powf((1 - e2 * powf(sinf(fp_lat), 2)), 1.5);
        float n1 = a / sqrtf(1 - e2 * powf(sinf(fp_lat), 2));
        float d = x / (n1 * k0);
        float q1 = n1 * tanf(fp_lat) / r1;
        float q2 = powf(d, 2) / 2.0f;
        float q3 = (5 + 3 * t1 + 10 * c1 - 4 * powf(c1, 2) - 9.0f * ed2) * powf(d, 4) / 24.0f;
        float q4 = (61.0f + 90.0f * t1 + 298.0f * c1 + 45.0f * powf(t1, 2) - 252.0f * ed2 - 3 * powf(c1, 2)) * powf(d, 6) / 720.0f;
        radLat[iPoint] = fp_lat - q1 * (q2 - q3 + q4);
        float q5 = d;
        float q6 = (1.0f + 2.0f * t1 + c1) * powf(d, 3) / 6.0f;
        float q7 = (5.0f - 2.0f * c1 + 28.0f * t1 - 3.0f * powf(c1, 2) + 8.0f * ed2 + 24.0f * powf(t1, 2)) * powf(d, 5) / 120.0f;
        radLong[iPoint] = lon0 + (q5 - q6 + q7) / cosf(fp_lat);
        mAlt[iPoint] = 0.0f;
    }
}

/*
Geodetic to UTM transformation of float precision.

@param float *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param float a semi-major axis
@param float b semi-minor axis
@param float *mmUTM array of size nx1 easting, northing[m, m]
*/
void geodetic2UTMFloatUnrolled(const float* radLat,
    const float* radLong,
    const float* mAlt,
    long nPoints,
    float a,
    float b,
    float* mUTMX,
    float* mUTMY)
{
    float k0 = 0.9996f;
    float e2 = 1.0f - (b * b) / (a * a);
    float e = sqrtf(e2);
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int zone = (radLong[iPoint] * 180.0 / PI + 180) / 6 + 1;
        float radCentralMeridian = ((float)(zone) * 6.0f - 183.0f) * PIf / 180.0f;
        float N = a / sqrtf(1 - e2 * powf(sinf(radLat[iPoint]), 2));
        float T = powf(tanf(radLat[iPoint]), 2);
        float C = (e2 * powf(cosf(radLat[iPoint]), 2)) / (1 - e2);
        float A = cosf(radLat[iPoint]) * (radLong[iPoint] - radCentralMeridian);
        float M = a * ((1.0f - powf(e, 2) / 4.0f - 3.0f * powf(e, 4) / 64.0f - 5.0f * powf(e, 6) / 256.0f) * radLat[iPoint] - (3.0f * powf(e, 2) / 8.0f + 3.0f * powf(e, 4) / 32.0f + 45.0f * powf(e, 6) / 1024.0f) * sinf(2.0f * radLat[iPoint]) + (15.0f * powf(e, 4) / 256.0f + 45.0f * powf(e, 6) / 1024.0f) * sinf(4.0f * radLat[iPoint]) - (35.0f * powf(e, 6) / 3072.0f) * sinf(6.0f * radLat[iPoint]));
        mUTMX[iPoint] = k0 * N * (A + (1.0f - T + C) * powf(A, 3) / 6.0f + (5.0f - 18.0f * T + powf(T, 2) + 72.0f * C - 58.0f * powf(e, 2)) * powf(A, 5) / 120.0f) + 500000.0f; // easting
        mUTMY[iPoint] = k0 * (M + N * tanf(radLat[iPoint]) * (powf(A, 2) / 2.0f + powf(A, 4) / 24.0f * (5.0f - T + 9.0f * C + 4.0f * powf(C, 2)) + powf(A, 6) / 720.0f * (61.0f - 58.0f * T + powf(T, 2) + 600.0f * C - 330.0f * powf(e, 2)))); // northing
        if (radLat[iPoint] < 0.0f)
            mUTMY[iPoint] += 10000000.0f;
    }
}

/*
Geodetic to UTM transformation of double precision.

@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmUTM array of size nx1 easting, northing[m, m]
*/
void geodetic2UTMDoubleUnrolled(const double* radLat,
    const double* radLong,
    const double* mAlt,
    long nPoints,
    double a,
    double b,
    double* mUTMX,
    double* mUTMY)
{
    double k0 = 0.9996;
    double e2 = 1.0 - (b * b) / (a * a);
    double e = sqrt(e2);
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int zone = (radLong[iPoint] * 180.0 / PI + 180) / 6 + 1;
        double radCentralMeridian = ((double)(zone) * 6.0 - 183.0) * PI / 180.0;
        double N = a / sqrt(1 - e2 * pow(sin(radLat[iPoint]), 2));
        double T = pow(tan(radLat[iPoint]), 2);
        double C = (e2 * pow(cos(radLat[iPoint]), 2)) / (1 - e2);
        double A = cos(radLat[iPoint]) * (radLong[iPoint] - radCentralMeridian);
        double M = a * ((1 - e2 / 4.0 - 3.0 * pow(e, 4) / 64.0 - 5.0 * pow(e, 6) / 256.0) * radLat[iPoint] - (3.0 * e2 / 8.0 + 3.0 * pow(e, 4) / 32.0 + 45.0 * pow(e, 6) / 1024.0) * sin(2.0 * radLat[iPoint]) + (15.0 * pow(e, 4) / 256.0 + 45 * pow(e, 6) / 1024.0) * sin(4.0 * radLat[iPoint]) - (35.0 * pow(e, 6) / 3072.0) * sin(6.0 * radLat[iPoint]));
        mUTMX[iPoint] = k0 * N * (A + (1.0 - T + C) * pow(A, 3) / 6.0 + (5.0 - 18.0 * T + pow(T, 2) + 72.0 * C - 58.0 * e2) * pow(A, 5) / 120.0) + 500000.0; // easting
        mUTMY[iPoint] = k0 * (M + N * tan(radLat[iPoint]) * (pow(A, 2) / 2.0 + pow(A, 4) / 24.0 * (5.0 - T + 9.0 * C + 4.0 * pow(C, 2)) + pow(A, 6) / 720.0 * (61.0 - 58.0 * T + pow(T, 2) + 600.0 * C - 330.0 * e2))); // northing
        if (radLat[iPoint] < 0.0)
            mUTMY[iPoint] += 10000000.0;
    }
}

/*
Geodetic to ECEF transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates

@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmXYZ array of size nx3 X, Y, Z [rad, rad, m]
*/
void geodetic2ECEFFloatUnrolled(const float* radLat,
    const float* radLong,
    const float* mAlt,
    long nPoints,
    float a,
    float b,
    float* mX,
    float* mY,
    float* mZ)
{
    float e2 = 1.0f - (b * b) / (a * a);
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        float N = a / sqrtf(1 - e2 * (sinf(radLat[iPoint]) * sinf(radLat[iPoint])));
        mX[iPoint] = (N + mAlt[iPoint]) * cosf(radLat[iPoint]) * cosf(radLong[iPoint]);
        mY[iPoint] = (N + mAlt[iPoint]) * cosf(radLat[iPoint]) * sinf(radLong[iPoint]);
        mZ[iPoint] = ((1 - e2) * N + mAlt[iPoint]) * sinf(radLat[iPoint]);
    }
}

/*
Geodetic to ECEF transformation of double precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates

@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmXYZ array of size nx3 X, Y, Z [m, m, m]
*/
void geodetic2ECEFDoubleUnrolled(const double* radLat,
    const double* radLong,
    const double* mAlt,
    long nPoints,
    double a,
    double b,
    double* mX,
    double* mY,
    double* mZ)
{
    double e2 = 1.0 - (b * b) / (a * a);
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        double N = a / sqrt(1 - e2 * sin(radLat[iPoint]) * sin(radLat[iPoint]));
        mX[iPoint] = (N + mAlt[iPoint]) * cos(radLat[iPoint]) * cos(radLong[iPoint]);
        mY[iPoint] = (N + mAlt[iPoint]) * cos(radLat[iPoint]) * sin(radLong[iPoint]);
        mZ[iPoint] = ((1 - e2) * N + mAlt[iPoint]) * sin(radLat[iPoint]);
    }
}

/*
ECEF to geodetic transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#The_application_of_Ferrari's_solution

@param double *mmmXYZ array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of ECEF points
@param double a semi-major axis
@param double b semi-minor axis
@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
*/
void ECEF2geodeticFloatUnrolled(const float* mX,
    const float* mY,
    const float* mZ,
    long nPoints,
    float a,
    float b,
    float* radLat,
    float* radLong,
    float* mAlt)
{
    long iPoint;
    float half = 0.5;
    float e2 = ((a * a) - (b * b)) / (a * a);
    float ed2 = ((a * a) - (b * b)) / (b * b);
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        float p = sqrtf(mX[iPoint] * mX[iPoint] + mY[iPoint] * mY[iPoint]);
        float F = 54 * b * b * mZ[iPoint] * mZ[iPoint];
        float G = p * p + (1 - e2) * mZ[iPoint] * mZ[iPoint] - e2 * (a * a - b * b);
        float c = e2 * e2 * F * p * p / (G * G * G);
        float s = cbrtf(1 + c + sqrtf(c * c + 2 * c));
        float k = s + 1 + 1 / s;
        float P = F / (3 * k * k * G * G);
        float Q = sqrtf(1 + 2 * e2 * e2 * P);
        float r0 = -P * e2 * p / (1 + Q) + sqrtf(half * a * a * (1 + 1 / Q) - P * (1 - e2) * mZ[iPoint] * mZ[iPoint] / (Q * (1 + Q)) - half * P * p * p);
        float U = sqrtf((p - e2 * r0) * (p - e2 * r0) + mZ[iPoint] * mZ[iPoint]);
        float V = sqrtf((p - e2 * r0) * (p - e2 * r0) + (1 - e2) * mZ[iPoint] * mZ[iPoint]);
        float z0 = b * b * mZ[iPoint] / (a * V);
        radLat[iPoint] = atanf((mZ[iPoint] + ed2 * z0) / p);
        radLong[iPoint] = atan2f(mY[iPoint], mX[iPoint]);
        mAlt[iPoint] = U * (1 - b * b / (a * V));
    }
}

/*
ECEF to geodetic transformation of double precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#The_application_of_Ferrari's_solution

@param double *mmmXYZ array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of ECEF points
@param double a semi-major axis
@param double b semi-minor axis
@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
*/
void ECEF2geodeticDoubleUnrolled(const double* mX,
    const double* mY,
    const double* mZ,
    long nPoints,
    double a,
    double b,
    double* radLat,
    double* radLong,
    double* mAlt)
{
    double e2 = ((a * a) - (b * b)) / (a * a);
    double ed2 = ((a * a) - (b * b)) / (b * b);
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        double p = sqrt(mX[iPoint] * mX[iPoint] + mY[iPoint] * mY[iPoint]);
        double F = 54 * b * b * mZ[iPoint] * mZ[iPoint];
        double G = p * p + (1 - e2) * mZ[iPoint] * mZ[iPoint] - e2 * (a * a - b * b);
        double c = e2 * e2 * F * p * p / (G * G * G);
        double s = cbrt(1 + c + sqrt(c * c + 2 * c));
        double k = s + 1 + 1 / s;
        double P = F / (3 * k * k * G * G);
        double Q = sqrt(1 + 2 * e2 * e2 * P);
        double r0 = -P * e2 * p / (1 + Q) + sqrt(0.5 * a * a * (1 + 1 / Q) - P * (1 - e2) * mZ[iPoint] * mZ[iPoint] / (Q * (1 + Q)) - 0.5 * P * p * p);
        double U = sqrt((p - e2 * r0) * (p - e2 * r0) + mZ[iPoint] * mZ[iPoint]);
        double V = sqrt((p - e2 * r0) * (p - e2 * r0) + (1 - e2) * mZ[iPoint] * mZ[iPoint]);
        double z0 = b * b * mZ[iPoint] / (a * V);
        radLat[iPoint] = atan((mZ[iPoint] + ed2 * z0) / p);
        radLong[iPoint] = atan2(mY[iPoint], mX[iPoint]);
        mAlt[iPoint] = U * (1 - b * b / (a * V));
    }
}

/*
ECEF to ENU transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

@param double *rrmLLALocalOrigin array of size nx3 of local reference point X,
Y, Z [m, m, m]
@param double *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void ECEF2ENUFloatUnrolled(float* radLatOrigin,
    const float* radLongOrigin,
    const float* mAltLocalOrigin,
    const float* mXTarget,
    const float* mYTarget,
    const float* mZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mXLocal,
    float* mYLocal,
    float* mZLocal)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mXLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    float* mYLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    float* mZLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    geodetic2ECEFFloatUnrolled(radLatOrigin, radLongOrigin, mAltLocalOrigin, nOriginPoints, (float)(a), (float)(b), mXLocalOrigin, mYLocalOrigin, mZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        float DeltaX = mXTarget[iPoint] - mXLocalOrigin[iOrigin];
        float DeltaY = mYTarget[iPoint] - mYLocalOrigin[iOrigin];
        float DeltaZ = mZTarget[iPoint] - mZLocalOrigin[iOrigin];
        mXLocal[iPoint] = -sinf(radLongOrigin[iOrigin]) * DeltaX + cosf(radLongOrigin[iOrigin]) * DeltaY;
        mYLocal[iPoint] = -sinf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * DeltaX + -sinf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * DeltaY + cosf(radLatOrigin[iOrigin]) * DeltaZ;
        mZLocal[iPoint] = cosf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * DeltaX + cosf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * DeltaY + sinf(radLatOrigin[iOrigin]) * DeltaZ;
    }
    free(mXLocalOrigin);
    free(mYLocalOrigin);
    free(mZLocalOrigin);
}

/*
ECEF to ENU transformation of double precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

@param double *rrmLLALocalOrigin array of size nx3 of local reference point X,
Y, Z [m, m, m]
@param double *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void ECEF2ENUDoubleUnrolled(const double* radLatOrigin,
    const double* radLongOrigin,
    const double* mAltLocalOrigin,
    const double* mXTarget,
    const double* mYTarget,
    const double* mZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mXLocal,
    double* mYLocal,
    double* mZLocal)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mXLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    double* mYLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    double* mZLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    geodetic2ECEFDoubleUnrolled(radLatOrigin, radLongOrigin, mAltLocalOrigin, nOriginPoints, a, b, mXLocalOrigin, mYLocalOrigin, mZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        double DeltaX = mXTarget[iPoint] - mXLocalOrigin[iOrigin];
        double DeltaY = mYTarget[iPoint] - mYLocalOrigin[iOrigin];
        double DeltaZ = mZTarget[iPoint] - mZLocalOrigin[iOrigin];
        mXLocal[iPoint] = -sin(radLongOrigin[iOrigin]) * DeltaX + cos(radLongOrigin[iOrigin]) * DeltaY;
        mYLocal[iPoint] = -sin(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * DeltaX + -sin(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * DeltaY + cos(radLatOrigin[iOrigin]) * DeltaZ;
        mZLocal[iPoint] = cos(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * DeltaX + cos(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * DeltaY + sin(radLatOrigin[iOrigin]) * DeltaZ;
    }
    free(mXLocalOrigin);
    free(mYLocalOrigin);
    free(mZLocalOrigin);
}

void ECEF2NEDFloatUnrolled(float* radLatOrigin,
    const float* radLongOrigin,
    const float* mAltLocalOrigin,
    const float* mXTarget,
    const float* mYTarget,
    const float* mZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mXLocal,
    float* mYLocal,
    float* mZLocal)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mXLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    float* mYLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    float* mZLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    geodetic2ECEFFloatUnrolled(radLatOrigin, radLongOrigin, mAltLocalOrigin, nOriginPoints, (float)(a), (float)(b), mXLocalOrigin, mYLocalOrigin, mZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        float DeltaX = mXTarget[iPoint] - mXLocalOrigin[iOrigin];
        float DeltaY = mYTarget[iPoint] - mYLocalOrigin[iOrigin];
        float DeltaZ = mZTarget[iPoint] - mZLocalOrigin[iOrigin];
        mXLocal[iPoint] = -sinf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * DeltaX + -sinf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * DeltaY + cosf(radLatOrigin[iOrigin]) * DeltaZ;
        mYLocal[iPoint] = -sinf(radLongOrigin[iOrigin]) * DeltaX + cosf(radLongOrigin[iOrigin]) * DeltaY;
        mZLocal[iPoint] = -cosf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * DeltaX + -cosf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * DeltaY + -sinf(radLatOrigin[iOrigin]) * DeltaZ;
    }
    free(mXLocalOrigin);
    free(mYLocalOrigin);
    free(mZLocalOrigin);
}

void ECEF2NEDDoubleUnrolled(const double* radLatOrigin,
    const double* radLongOrigin,
    const double* mAltLocalOrigin,
    const double* mXTarget,
    const double* mYTarget,
    const double* mZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mXLocal,
    double* mYLocal,
    double* mZLocal)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mXLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    double* mYLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    double* mZLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    geodetic2ECEFDoubleUnrolled(radLatOrigin, radLongOrigin, mAltLocalOrigin, nOriginPoints, a, b, mXLocalOrigin, mYLocalOrigin, mZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        double DeltaX = mXTarget[iPoint] - mXLocalOrigin[iOrigin];
        double DeltaY = mYTarget[iPoint] - mYLocalOrigin[iOrigin];
        double DeltaZ = mZTarget[iPoint] - mZLocalOrigin[iOrigin];
        mXLocal[iPoint] = -sin(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * DeltaX + -sin(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * DeltaY + cos(radLatOrigin[iOrigin]) * DeltaZ;
        mYLocal[iPoint] = -sin(radLongOrigin[iOrigin]) * DeltaX + cos(radLongOrigin[iOrigin]) * DeltaY;
        mZLocal[iPoint] = -cos(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * DeltaX + -cos(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * DeltaY + -sin(radLatOrigin[iOrigin]) * DeltaZ;
    }
    free(mXLocalOrigin);
    free(mYLocalOrigin);
    free(mZLocalOrigin);
}

void ECEF2NEDvFloatUnrolled(float* radLatOrigin,
    const float* radLongOrigin,
    const float* mAltLocalOrigin,
    const float* mXTarget,
    const float* mYTarget,
    const float* mZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    float* mXLocal,
    float* mYLocal,
    float* mZLocal)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXLocal[iPoint] = -sinf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mXTarget[iPoint] + -sinf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mYTarget[iPoint] + cosf(radLatOrigin[iOrigin]) * mZTarget[iPoint];
        mYLocal[iPoint] = -sinf(radLongOrigin[iOrigin]) * mXTarget[iPoint] + cosf(radLongOrigin[iOrigin]) * mYTarget[iPoint];
        mZLocal[iPoint] = -cosf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mXTarget[iPoint] + -cosf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mYTarget[iPoint] + -sinf(radLatOrigin[iOrigin]) * mZTarget[iPoint];
    }
}

void ECEF2NEDvDoubleUnrolled(const double* radLatOrigin,
    const double* radLongOrigin,
    const double* mAltLocalOrigin,
    const double* mXTarget,
    const double* mYTarget,
    const double* mZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    double* mXLocal,
    double* mYLocal,
    double* mZLocal)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXLocal[iPoint] = -sin(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mXTarget[iPoint] + -sin(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mYTarget[iPoint] + cos(radLatOrigin[iOrigin]) * mZTarget[iPoint];
        mYLocal[iPoint] = -sin(radLongOrigin[iOrigin]) * mXTarget[iPoint] + cos(radLongOrigin[iOrigin]) * mYTarget[iPoint];
        mZLocal[iPoint] = -cos(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mXTarget[iPoint] + -cos(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mYTarget[iPoint] + -sin(radLatOrigin[iOrigin]) * mZTarget[iPoint];
    }
}

void ECEF2ENUvFloatUnrolled(const float* radLatOrigin,
    const float* radLongOrigin,
    const float* mAltLocalOrigin,
    const float* mXTarget,
    const float* mYTarget,
    const float* mZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    float* mXLocal,
    float* mYLocal,
    float* mZLocal)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXLocal[iPoint] = -sinf(radLongOrigin[iOrigin]) * mXTarget[iPoint] + cosf(radLongOrigin[iOrigin]) * mYTarget[iPoint];
        mYLocal[iPoint] = -sinf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mXTarget[iPoint] + -sinf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mYTarget[iPoint] + cosf(radLatOrigin[iOrigin]) * mZTarget[iPoint];
        mZLocal[iPoint] = cosf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mXTarget[iPoint] + cosf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mYTarget[iPoint] + sinf(radLatOrigin[iOrigin]) * mZTarget[iPoint];
    }
}

void ECEF2ENUvDoubleUnrolled(const double* radLatOrigin,
    const double* radLongOrigin,
    const double* mAltLocalOrigin,
    const double* mXTarget,
    const double* mYTarget,
    const double* mZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    double* mXLocal,
    double* mYLocal,
    double* mZLocal)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXLocal[iPoint] = -sin(radLongOrigin[iOrigin]) * mXTarget[iPoint] + cos(radLongOrigin[iOrigin]) * mYTarget[iPoint];
        mYLocal[iPoint] = -sin(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mXTarget[iPoint] + -sin(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mYTarget[iPoint] + cos(radLatOrigin[iOrigin]) * mZTarget[iPoint];
        mZLocal[iPoint] = cos(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mXTarget[iPoint] + cos(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mYTarget[iPoint] + sin(radLatOrigin[iOrigin]) * mZTarget[iPoint];
    }
}

void ENU2ECEFvFloatUnrolled(const float* radLatOrigin,
    const float* radLongOrigin,
    const float* mAltLocalOrigin,
    const float* mXTargetLocal,
    const float* mYTargetLocal,
    const float* mZTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    float* mXTarget,
    float* mYTarget,
    float* mZTarget)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXTarget[iPoint] = -sinf(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sinf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + cosf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint];
        mYTarget[iPoint] = cosf(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sinf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + cosf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint];
        mZTarget[iPoint] = cosf(radLatOrigin[iOrigin]) * mYTargetLocal[iPoint] + sinf(radLatOrigin[iOrigin]) * mZTargetLocal[iPoint];
    }
}

void NED2ECEFvFloatUnrolled(const float* radLatOrigin,
    const float* radLongOrigin,
    const float* mAltLocalOrigin,
    const float* mXTargetLocal,
    const float* mYTargetLocal,
    const float* mZTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    float* mXTarget,
    float* mYTarget,
    float* mZTarget)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXTarget[iPoint] = -sinf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sinf(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + -cosf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint];
        mYTarget[iPoint] = -sinf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + cosf(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + -cosf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint];
        mZTarget[iPoint] = cosf(radLatOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sinf(radLatOrigin[iOrigin]) * mZTargetLocal[iPoint];
    }
}

void NED2ECEFvDoubleUnrolled(const double* radLatOrigin,
    const double* radLongOrigin,
    const double* mAltLocalOrigin,
    const double* mXTargetLocal,
    const double* mYTargetLocal,
    const double* mZTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    double* mXTarget,
    double* mYTarget,
    double* mZTarget)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXTarget[iPoint] = -sin(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sin(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + -cos(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint];
        mYTarget[iPoint] = -sin(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + cos(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + -cos(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint];
        mZTarget[iPoint] = cos(radLatOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sin(radLatOrigin[iOrigin]) * mZTargetLocal[iPoint];
    }
}

void ENU2ECEFvDoubleUnrolled(const double* radLatOrigin,
    const double* radLongOrigin,
    const double* mAltLocalOrigin,
    const double* mXTargetLocal,
    const double* mYTargetLocal,
    const double* mZTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    double* mXTarget,
    double* mYTarget,
    double* mZTarget)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXTarget[iPoint] = -sin(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sin(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + cos(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint];
        mYTarget[iPoint] = cos(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sin(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + cos(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint];
        mZTarget[iPoint] = cos(radLatOrigin[iOrigin]) * mYTargetLocal[iPoint] + sin(radLatOrigin[iOrigin]) * mZTargetLocal[iPoint];
    }
}

/*
ECEF to ENU transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *rrmLLALocalOrigin array of size nx3 of local reference point
latitude, longitude, height [rad, rad, m]
@param float *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void NED2ECEFFloatUnrolled(const float* radLatOrigin,
    const float* radLongOrigin,
    const float* mAltLocalOrigin,
    const float* mXTargetLocal,
    const float* mYTargetLocal,
    const float* mZTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mXTarget,
    float* mYTarget,
    float* mZTarget)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mXLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    float* mYLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    float* mZLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    geodetic2ECEFFloatUnrolled(radLatOrigin, radLongOrigin, mAltLocalOrigin, nOriginPoints, a, b, mXLocalOrigin, mYLocalOrigin, mZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXTarget[iPoint] = -sinf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sinf(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + -cosf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint] + mXLocalOrigin[iOrigin];
        mYTarget[iPoint] = -sinf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + cosf(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + -cosf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint] + mYLocalOrigin[iOrigin];
        mZTarget[iPoint] = cosf(radLatOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sinf(radLatOrigin[iOrigin]) * mZTargetLocal[iPoint] + mZLocalOrigin[iOrigin];
    }
    free(mXLocalOrigin);
    free(mYLocalOrigin);
    free(mZLocalOrigin);
}

/*
ECEF to ENU transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *rrmLLALocalOrigin array of size nx3 of local reference point
latitude, longitude, height [rad, rad, m]
@param float *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void NED2ECEFDoubleUnrolled(const double* radLatOrigin,
    const double* radLongOrigin,
    const double* mAltLocalOrigin,
    const double* mXTargetLocal,
    const double* mYTargetLocal,
    const double* mZTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mXTarget,
    double* mYTarget,
    double* mZTarget)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mXLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    double* mYLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    double* mZLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    geodetic2ECEFDoubleUnrolled(radLatOrigin, radLongOrigin, mAltLocalOrigin, nOriginPoints, a, b, mXLocalOrigin, mYLocalOrigin, mZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXTarget[iPoint] = -sin(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sin(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + -cos(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint] + mXLocalOrigin[iOrigin];
        mYTarget[iPoint] = -sin(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + cos(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + -cos(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint] + mYLocalOrigin[iOrigin];
        mZTarget[iPoint] = cos(radLatOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sin(radLatOrigin[iOrigin]) * mZTargetLocal[iPoint] + mZLocalOrigin[iOrigin];
    }
    free(mXLocalOrigin);
    free(mYLocalOrigin);
    free(mZLocalOrigin);
}

/*
ECEF to ENU transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *rrmLLALocalOrigin array of size nx3 of local reference point
latitude, longitude, height [rad, rad, m]
@param float *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void ENU2ECEFFloatUnrolled(const float* radLatOrigin,
    const float* radLongOrigin,
    const float* mAltLocalOrigin,
    const float* mXTargetLocal,
    const float* mYTargetLocal,
    const float* mZTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mXTarget,
    float* mYTarget,
    float* mZTarget)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mXLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    float* mYLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    float* mZLocalOrigin = (float*)malloc(nOriginPoints * sizeof(float));
    geodetic2ECEFFloatUnrolled(radLatOrigin, radLongOrigin, mAltLocalOrigin, nOriginPoints, a, b, mXLocalOrigin, mYLocalOrigin, mZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        mXTarget[iPoint] = -sinf(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sinf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + cosf(radLatOrigin[iOrigin]) * cosf(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint] + mXLocalOrigin[iOrigin];
        mYTarget[iPoint] = cosf(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sinf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + cosf(radLatOrigin[iOrigin]) * sinf(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint] + mYLocalOrigin[iOrigin];
        mZTarget[iPoint] = cosf(radLatOrigin[iOrigin]) * mYTargetLocal[iPoint] + sinf(radLatOrigin[iOrigin]) * mZTargetLocal[iPoint] + mZLocalOrigin[iOrigin];
    }
    free(mXLocalOrigin);
    free(mYLocalOrigin);
    free(mZLocalOrigin);
}

/*
ECEF to ENU transformation of double precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *rrmLLALocalOrigin array of size nx3 of local reference point
latitude, longitude, height [rad, rad, m]
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
*/
void ENU2ECEFDoubleUnrolled(const double* radLatOrigin,
    const double* radLongOrigin,
    const double* mAltLocalOrigin,
    const double* mXTargetLocal,
    const double* mYTargetLocal,
    const double* mZTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mXTarget,
    double* mYTarget,
    double* mZTarget)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mXLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    double* mYLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    double* mZLocalOrigin = (double*)malloc(nOriginPoints * sizeof(double));
    geodetic2ECEFDoubleUnrolled(radLatOrigin, radLongOrigin, mAltLocalOrigin, nOriginPoints, a, b, mXLocalOrigin, mYLocalOrigin, mZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iOrigin = iPoint * isOriginSizeOfTargets;
        // mXTarget[iPoint] = -sin(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + cos(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + mXLocalOrigin[iOrigin];
        // mYTarget[iPoint] = -sin(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sin(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + cos(radLatOrigin[iOrigin]) * mZTargetLocal[iPoint] + mYLocalOrigin[iOrigin];
        // mZTarget[iPoint] = cos(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + cos(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + sin(radLatOrigin[iOrigin]) * mZTargetLocal[iPoint] + mZLocalOrigin[iOrigin];
        mXTarget[iPoint] = -sin(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sin(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + cos(radLatOrigin[iOrigin]) * cos(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint] + mXLocalOrigin[iOrigin];
        mYTarget[iPoint] = cos(radLongOrigin[iOrigin]) * mXTargetLocal[iPoint] + -sin(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mYTargetLocal[iPoint] + cos(radLatOrigin[iOrigin]) * sin(radLongOrigin[iOrigin]) * mZTargetLocal[iPoint] + mYLocalOrigin[iOrigin];
        mZTarget[iPoint] = cos(radLatOrigin[iOrigin]) * mYTargetLocal[iPoint] + sin(radLatOrigin[iOrigin]) * mZTargetLocal[iPoint] + mZLocalOrigin[iOrigin];
    }
    free(mXLocalOrigin);
    free(mYLocalOrigin);
    free(mZLocalOrigin);
}

/*
ENU to AER transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf <-
includes additional errors and factors that could be implemented
https://www.lddgo.net/en/coordinate/ecef-enu

@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void NED2AERFloatUnrolled(const float* mN, const float* mE, const float* mD, long nPoints, float* radAz, float* radEl, float* mRange)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        radAz[iPoint] = atan2f(mE[iPoint], mN[iPoint]);
        if (radAz[iPoint] < 0)
            radAz[iPoint] = radAz[iPoint] + (2.0f * PIf);
        mRange[iPoint] = sqrtf(mN[iPoint] * mN[iPoint] + mE[iPoint] * mE[iPoint] + mD[iPoint] * mD[iPoint]);
        radEl[iPoint] = asinf(-mD[iPoint] / mRange[iPoint]);}
}

/*
ENU to AER transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf <-
includes additional errors and factors that could be implemented
https://www.lddgo.net/en/coordinate/ecef-enu

@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void NED2AERDoubleUnrolled(const double* mN, const double* mE, const double* mD, long nPoints, double* radAz, double* radEl, double* mRange)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        radAz[iPoint] = atan2(mE[iPoint], mN[iPoint]);
        if (radAz[iPoint] < 0)
            radAz[iPoint] = radAz[iPoint] + 2.0 * PI;
        mRange[iPoint] = sqrt(mN[iPoint] * mN[iPoint] + mE[iPoint] * mE[iPoint] + mD[iPoint] * mD[iPoint]);
        radEl[iPoint] = asin(-mD[iPoint] / mRange[iPoint]);
    }
}

/*
ENU to AER transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf <-
includes additional errors and factors that could be implemented
https://www.lddgo.net/en/coordinate/ecef-enu

@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void ENU2AERFloatUnrolled(const float* mE, const float* mN, const float* mU, long nPoints, float* radAz, float* radEl, float* mRange)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        radAz[iPoint] = atan2f(mE[iPoint], mN[iPoint]);
        if (radAz[iPoint] < 0)
            radAz[iPoint] = radAz[iPoint] + (2.0f * PIf);
        mRange[iPoint] = sqrtf(mE[iPoint] * mE[iPoint] + mN[iPoint] * mN[iPoint] + mU[iPoint] * mU[iPoint]);
        radEl[iPoint] = asinf(mU[iPoint] / mRange[iPoint]);
    }
}

/*
ENU to AER transformation of double precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void ENU2AERDoubleUnrolled(const double* mE, const double* mN, const double* mU, long nPoints, double* radAz, double* radEl, double* mRange)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        radAz[iPoint] = atan2(mE[iPoint], mN[iPoint]);
        if (radAz[iPoint] < 0)
            radAz[iPoint] = radAz[iPoint] + 2.0 * PI;
        mRange[iPoint] = sqrt(mE[iPoint] * mE[iPoint] + mN[iPoint] * mN[iPoint] + mU[iPoint] * mU[iPoint]);
        radEl[iPoint] = asin(mU[iPoint] / mRange[iPoint]);
    }
}

/*
AER to ENU transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf

@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
@param long nPoints Number of target points
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2NEDFloatUnrolled(const float* radAz, const float* radEl, const float* mRange, long nPoints, float* mN, float* mE, float* mD)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        mN[iPoint] = cosf(radEl[iPoint]) * cosf(radAz[iPoint]) * mRange[iPoint];
        mE[iPoint] = cosf(radEl[iPoint]) * sinf(radAz[iPoint]) * mRange[iPoint];
        mD[iPoint] = -sinf(radEl[iPoint]) * mRange[iPoint];
    }
}

/*
AER to ENU transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf

@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
@param long nPoints Number of target points
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2NEDDoubleUnrolled(const double* radAz, const double* radEl, const double* mRange, long nPoints, double* mN, double* mE, double* mD)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        mN[iPoint] = cos(radEl[iPoint]) * cos(radAz[iPoint]) * mRange[iPoint];
        mE[iPoint] = cos(radEl[iPoint]) * sin(radAz[iPoint]) * mRange[iPoint];
        mD[iPoint] = -sin(radEl[iPoint]) * mRange[iPoint];
    }
}

/*
AER to ENU transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf

@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
@param long nPoints Number of target points
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2ENUFloatUnrolled(const float* radAz, const float* radEl, const float* mRange, long nPoints, float* mE, float* mN, float* mU)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        mE[iPoint] = cosf(radEl[iPoint]) * sinf(radAz[iPoint]) * mRange[iPoint];
        mN[iPoint] = cosf(radEl[iPoint]) * cosf(radAz[iPoint]) * mRange[iPoint];
        mU[iPoint] = sinf(radEl[iPoint]) * mRange[iPoint];
    }
}

/*
AER to ENU transformation of double precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf

@param double *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
@param long nPoints Number of target points
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2ENUDoubleUnrolled(const double* radAz, const double* radEl, const double* mRange, long nPoints, double* mE, double* mN, double* mU)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        mE[iPoint] = cos(radEl[iPoint]) * sin(radAz[iPoint]) * mRange[iPoint];
        mN[iPoint] = cos(radEl[iPoint]) * cos(radAz[iPoint]) * mRange[iPoint];
        mU[iPoint] = sin(radEl[iPoint]) * mRange[iPoint];
    }
}

/*
UTM to geodetic transformation of double precision.
https://fypandroid.wordpress.com/2011/09/03/converting-utm-to-latitude-and-longitude-or-vice-versa/

@param double *mmUTM array of size nx1 easting, northing[m, m]
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
*/
void UTM2geodeticDoubleRolled(const double* mmUTM,
    long ZoneNumber,
    char* ZoneLetter,
    long nPoints,
    double a,
    double b,
    double* rrmLLA)
{
    double k0 = 0.9996;
    double e2 = 1.0 - (b * b) / (a * a);
    double e = sqrt(e2);
    double ed2 = ((a * a) - (b * b)) / (b * b);
    double lon0 = (((double)ZoneNumber - 1.0) * 6.0 - 177.0) * PI / 180.0;
    double e1 = (1.0 - sqrt(1.0 - e2)) / (1.0 + sqrt(1.0 - e2));
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        long iUTM = iPoint * NCOORDSIN2D;
        double x = mmUTM[iUTM + 0] - 500000.0;
        double y = mmUTM[iUTM + 1];
        if (ZoneLetter[0] < 'N')
            y -= 10000000.0;
        double m = y / k0;
        double mu = m / (a * (1 - e2 / 4.0 - 3.0 * pow(e, 4) / 64.0 - 5.0 * pow(e, 6) / 256.0));
        double j1 = 3.0 * e1 / 2 - 27.0 * pow(e1, 3) / 32.0;
        double j2 = 21.0 * pow(e1, 2) / 16.0 - 55.0 * pow(e1, 4) / 32.0;
        double j3 = 151.0 * pow(e1, 3) / 96.0;
        double j4 = 1097.0 * pow(e1, 4) / 512.0;
        double fp_lat = mu + j1 * sin(2.0 * mu) + j2 * sin(4.0 * mu) + j3 * sin(6.0 * mu) + j4 * sin(8.0 * mu);
        double c1 = ed2 * pow(cos(fp_lat), 2);
        double t1 = pow(tan(fp_lat), 2);
        double r1 = a * (1 - e2) / pow((1 - e2 * pow(sin(fp_lat), 2)), 1.5);
        double n1 = a / sqrt(1 - e2 * pow(sin(fp_lat), 2));
        double d = x / (n1 * k0);
        double q1 = n1 * tan(fp_lat) / r1;
        double q2 = pow(d, 2) / 2.0;
        double q3 = (5 + 3 * t1 + 10 * c1 - 4 * pow(c1, 2) - 9.0 * ed2) * pow(d, 4) / 24.0;
        double q4 = (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * pow(t1, 2) - 252.0 * ed2 - 3 * pow(c1, 2)) * pow(d, 6) / 720.0;
        rrmLLA[i + 0] = fp_lat - q1 * (q2 - q3 + q4);
        double q5 = d;
        double q6 = (1.0 + 2.0 * t1 + c1) * pow(d, 3) / 6.0;
        double q7 = (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * pow(c1, 2) + 8.0 * ed2 + 24.0 * pow(t1, 2)) * pow(d, 5) / 120.0;
        rrmLLA[i + 1] = lon0 + (q5 - q6 + q7) / cos(fp_lat);
        rrmLLA[i + 2] = 0.0;
    }
}

/*
UTM to geodetic transformation of float precision.

@param double *mmUTM array of size nx1 easting, northing[m, m]
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
*/
void UTM2geodeticFloatRolled(const float* mmUTM,
    long ZoneNumber,
    char* ZoneLetter,
    long nPoints,
    float a,
    float b,
    float* rrmLLA)
{
    float k0 = 0.9996f;
    float e2 = 1.0f - (b * b) / (a * a);
    float e = sqrtf(e2);
    float ed2 = ((a * a) - (b * b)) / (b * b);
    float lon0 = (((float)ZoneNumber - 1.0f) * 6.0f - 177.0f) * PIf / 180.0f;
    float e1 = (1.0f - sqrtf(1.0f - e2)) / (1.0f + sqrtf(1.0f - e2));
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        long iUTM = iPoint * NCOORDSIN2D;
        float x = mmUTM[iUTM + 0] - 500000.0f;
        float y = mmUTM[iUTM + 1];
        if (ZoneLetter[0] < 'N')
            y -= 10000000.0f;
        float m = y / k0;
        float mu = m / (a * (1 - e2 / 4.0f - 3.0f * powf(e, 4) / 64.0f - 5.0f * powf(e, 6) / 256.0f));
        float j1 = 3.0f * e1 / 2 - 27.0f * powf(e1, 3) / 32.0f;
        float j2 = 21.0f * powf(e1, 2) / 16.0f - 55.0f * powf(e1, 4) / 32.0f;
        float j3 = 151.0f * powf(e1, 3) / 96.0f;
        float j4 = 1097.0f * powf(e1, 4) / 512.0f;
        float fp_lat = mu + j1 * sinf(2.0f * mu) + j2 * sinf(4.0f * mu) + j3 * sinf(6.0f * mu) + j4 * sinf(8.0f * mu);
        float c1 = ed2 * powf(cosf(fp_lat), 2);
        float t1 = powf(tanf(fp_lat), 2);
        float r1 = a * (1 - e2) / powf((1 - e2 * powf(sinf(fp_lat), 2)), 1.5);
        float n1 = a / sqrtf(1 - e2 * powf(sinf(fp_lat), 2));
        float d = x / (n1 * k0);
        float q1 = n1 * tanf(fp_lat) / r1;
        float q2 = powf(d, 2) / 2.0f;
        float q3 = (5 + 3 * t1 + 10 * c1 - 4 * powf(c1, 2) - 9.0f * ed2) * powf(d, 4) / 24.0f;
        float q4 = (61.0f + 90.0f * t1 + 298.0f * c1 + 45.0f * powf(t1, 2) - 252.0f * ed2 - 3 * powf(c1, 2)) * powf(d, 6) / 720.0f;
        rrmLLA[i + 0] = fp_lat - q1 * (q2 - q3 + q4);
        float q5 = d;
        float q6 = (1.0f + 2.0f * t1 + c1) * powf(d, 3) / 6.0f;
        float q7 = (5.0f - 2.0f * c1 + 28.0f * t1 - 3.0f * powf(c1, 2) + 8.0f * ed2 + 24.0f * powf(t1, 2)) * powf(d, 5) / 120.0f;
        rrmLLA[i + 1] = lon0 + (q5 - q6 + q7) / cosf(fp_lat);
        rrmLLA[i + 2] = 0.0f;
    }
}

/*
Geodetic to UTM transformation of float precision.

@param float *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param float a semi-major axis
@param float b semi-minor axis
@param float *mmUTM array of size nx1 easting, northing[m, m]
*/
void geodetic2UTMFloatRolled(const float* rrmLLA,
    long nPoints,
    float a,
    float b,
    float* mmUTM)
{
    float k0 = 0.9996f;
    float e2 = 1.0f - (b * b) / (a * a);
    float e = sqrtf(e2);
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        long iUTM = iPoint * NCOORDSIN2D;
        int zone = (rrmLLA[i + 1] * 180.0 / PI + 180) / 6 + 1;
        float radCentralMeridian = ((float)(zone) * 6.0f - 183.0f) * PIf / 180.0f;
        float N = a / sqrtf(1 - e2 * powf(sinf(rrmLLA[i + 0]), 2));
        float T = powf(tanf(rrmLLA[i + 0]), 2);
        float C = (e2 * powf(cosf(rrmLLA[i + 0]), 2)) / (1 - e2);
        float A = cosf(rrmLLA[i + 0]) * (rrmLLA[i + 1] - radCentralMeridian);
        float M = a * ((1.0f - powf(e, 2) / 4.0f - 3.0f * powf(e, 4) / 64.0f - 5.0f * powf(e, 6) / 256.0f) * rrmLLA[i + 0] - (3.0f * powf(e, 2) / 8.0f + 3.0f * powf(e, 4) / 32.0f + 45.0f * powf(e, 6) / 1024.0f) * sinf(2.0f * rrmLLA[i + 0]) + (15.0f * powf(e, 4) / 256.0f + 45.0f * powf(e, 6) / 1024.0f) * sinf(4.0f * rrmLLA[i + 0]) - (35.0f * powf(e, 6) / 3072.0f) * sinf(6.0f * rrmLLA[i + 0]));
        mmUTM[iUTM + 0] = k0 * N * (A + (1.0f - T + C) * powf(A, 3) / 6.0f + (5.0f - 18.0f * T + powf(T, 2) + 72.0f * C - 58.0f * powf(e, 2)) * powf(A, 5) / 120.0f) + 500000.0f; // easting
        mmUTM[iUTM + 1] = k0 * (M + N * tanf(rrmLLA[i + 0]) * (powf(A, 2) / 2.0f + powf(A, 4) / 24.0f * (5.0f - T + 9.0f * C + 4.0f * powf(C, 2)) + powf(A, 6) / 720.0f * (61.0f - 58.0f * T + powf(T, 2) + 600.0f * C - 330.0f * powf(e, 2)))); // northing
        if (rrmLLA[i + 0] < 0.0f)
            mmUTM[iUTM + 1] += 10000000.0f;
    }
}

/*
Geodetic to UTM transformation of double precision.

@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmUTM array of size nx1 easting, northing[m, m]
*/
void geodetic2UTMDoubleRolled(const double* rrmLLA,
    long nPoints,
    double a,
    double b,
    double* mmUTM)
{
    double k0 = 0.9996;
    double e2 = 1.0 - (b * b) / (a * a);
    double e = sqrt(e2);
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        long iUTM = iPoint * NCOORDSIN2D;
        int zone = (rrmLLA[i + 1] * 180.0 / PI + 180) / 6 + 1;
        double radCentralMeridian = ((double)(zone) * 6.0 - 183.0) * PI / 180.0;
        double N = a / sqrt(1 - e2 * pow(sin(rrmLLA[i + 0]), 2));
        double T = pow(tan(rrmLLA[i + 0]), 2);
        double C = (e2 * pow(cos(rrmLLA[i + 0]), 2)) / (1 - e2);
        double A = cos(rrmLLA[i + 0]) * (rrmLLA[i + 1] - radCentralMeridian);
        double M = a * ((1 - e2 / 4.0 - 3.0 * pow(e, 4) / 64.0 - 5.0 * pow(e, 6) / 256.0) * rrmLLA[i + 0] - (3.0 * e2 / 8.0 + 3.0 * pow(e, 4) / 32.0 + 45.0 * pow(e, 6) / 1024.0) * sin(2.0 * rrmLLA[i + 0]) + (15.0 * pow(e, 4) / 256.0 + 45 * pow(e, 6) / 1024.0) * sin(4.0 * rrmLLA[i + 0]) - (35.0 * pow(e, 6) / 3072.0) * sin(6.0 * rrmLLA[i + 0]));
        mmUTM[iUTM + 0] = k0 * N * (A + (1.0 - T + C) * pow(A, 3) / 6.0 + (5.0 - 18.0 * T + pow(T, 2) + 72.0 * C - 58.0 * e2) * pow(A, 5) / 120.0) + 500000.0; // easting
        mmUTM[iUTM + 1] = k0 * (M + N * tan(rrmLLA[i + 0]) * (pow(A, 2) / 2.0 + pow(A, 4) / 24.0 * (5.0 - T + 9.0 * C + 4.0 * pow(C, 2)) + pow(A, 6) / 720.0 * (61.0 - 58.0 * T + pow(T, 2) + 600.0 * C - 330.0 * e2))); // northing
        if (rrmLLA[i + 0] < 0.0)
            mmUTM[iUTM + 1] += 10000000.0;
    }
}

/*
Geodetic to ECEF transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates

@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmXYZ array of size nx3 X, Y, Z [rad, rad, m]
*/
void geodetic2ECEFFloatRolled(const float* rrmLLA,
    long nPoints,
    float a,
    float b,
    float* mmmXYZ)
{
    float e2 = 1.0f - (b * b) / (a * a);
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        float N = a / sqrtf(1 - e2 * (sinf(rrmLLA[i + 0]) * sinf(rrmLLA[i + 0])));
        mmmXYZ[i + 0] = (N + rrmLLA[i + 2]) * cosf(rrmLLA[i + 0]) * cosf(rrmLLA[i + 1]);
        mmmXYZ[i + 1] = (N + rrmLLA[i + 2]) * cosf(rrmLLA[i + 0]) * sinf(rrmLLA[i + 1]);
        mmmXYZ[i + 2] = ((1 - e2) * N + rrmLLA[i + 2]) * sinf(rrmLLA[i + 0]);
    }
}

/*
Geodetic to ECEF transformation of double precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates

@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
@param long nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmXYZ array of size nx3 X, Y, Z [m, m, m]
*/
void geodetic2ECEFDoubleRolled(const double* rrmLLA,
    long nPoints,
    double a,
    double b,
    double* mmmXYZ)
{
    double e2 = 1.0 - (b * b) / (a * a);
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        double N = a / sqrt(1 - e2 * sin(rrmLLA[i + 0]) * sin(rrmLLA[i + 0]));
        mmmXYZ[i + 0] = (N + rrmLLA[i + 2]) * cos(rrmLLA[i + 0]) * cos(rrmLLA[i + 1]);
        mmmXYZ[i + 1] = (N + rrmLLA[i + 2]) * cos(rrmLLA[i + 0]) * sin(rrmLLA[i + 1]);
        mmmXYZ[i + 2] = ((1 - e2) * N + rrmLLA[i + 2]) * sin(rrmLLA[i + 0]);
    }
}

/*
ECEF to geodetic transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#The_application_of_Ferrari's_solution

@param double *mmmXYZ array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of ECEF points
@param double a semi-major axis
@param double b semi-minor axis
@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
*/
void ECEF2geodeticFloatRolled(const float* mmmXYZ,
    long nPoints,
    float a,
    float b,
    float* rrmLLA)
{
    long iPoint;
    float half = 0.5;
    float e2 = ((a * a) - (b * b)) / (a * a);
    float ed2 = ((a * a) - (b * b)) / (b * b);
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        float p = sqrtf(mmmXYZ[i + 0] * mmmXYZ[i + 0] + mmmXYZ[i + 1] * mmmXYZ[i + 1]);
        float F = 54 * b * b * mmmXYZ[i + 2] * mmmXYZ[i + 2];
        float G = p * p + (1 - e2) * mmmXYZ[i + 2] * mmmXYZ[i + 2] - e2 * (a * a - b * b);
        float c = e2 * e2 * F * p * p / (G * G * G);
        float s = cbrtf(1 + c + sqrtf(c * c + 2 * c));
        float k = s + 1 + 1 / s;
        float P = F / (3 * k * k * G * G);
        float Q = sqrtf(1 + 2 * e2 * e2 * P);
        float r0 = -P * e2 * p / (1 + Q) + sqrtf(half * a * a * (1 + 1 / Q) - P * (1 - e2) * mmmXYZ[i + 2] * mmmXYZ[i + 2] / (Q * (1 + Q)) - half * P * p * p);
        float U = sqrtf((p - e2 * r0) * (p - e2 * r0) + mmmXYZ[i + 2] * mmmXYZ[i + 2]);
        float V = sqrtf((p - e2 * r0) * (p - e2 * r0) + (1 - e2) * mmmXYZ[i + 2] * mmmXYZ[i + 2]);
        float z0 = b * b * mmmXYZ[i + 2] / (a * V);
        rrmLLA[i + 0] = atanf((mmmXYZ[i + 2] + ed2 * z0) / p);
        rrmLLA[i + 1] = atan2f(mmmXYZ[i + 1], mmmXYZ[i + 0]);
        rrmLLA[i + 2] = U * (1 - b * b / (a * V));
    }
}

/*
ECEF to geodetic transformation of double precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#The_application_of_Ferrari's_solution

@param double *mmmXYZ array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of ECEF points
@param double a semi-major axis
@param double b semi-minor axis
@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
*/
void ECEF2geodeticDoubleRolled(const double* mmmXYZ,
    long nPoints,
    double a,
    double b,
    double* rrmLLA)
{
    double e2 = ((a * a) - (b * b)) / (a * a);
    double ed2 = ((a * a) - (b * b)) / (b * b);
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        double p = sqrt(mmmXYZ[i + 0] * mmmXYZ[i + 0] + mmmXYZ[i + 1] * mmmXYZ[i + 1]);
        double F = 54 * b * b * mmmXYZ[i + 2] * mmmXYZ[i + 2];
        double G = p * p + (1 - e2) * mmmXYZ[i + 2] * mmmXYZ[i + 2] - e2 * (a * a - b * b);
        double c = e2 * e2 * F * p * p / (G * G * G);
        double s = cbrt(1 + c + sqrt(c * c + 2 * c));
        double k = s + 1 + 1 / s;
        double P = F / (3 * k * k * G * G);
        double Q = sqrt(1 + 2 * e2 * e2 * P);
        double r0 = -P * e2 * p / (1 + Q) + sqrt(0.5 * a * a * (1 + 1 / Q) - P * (1 - e2) * mmmXYZ[i + 2] * mmmXYZ[i + 2] / (Q * (1 + Q)) - 0.5 * P * p * p);
        double U = sqrt((p - e2 * r0) * (p - e2 * r0) + mmmXYZ[i + 2] * mmmXYZ[i + 2]);
        double V = sqrt((p - e2 * r0) * (p - e2 * r0) + (1 - e2) * mmmXYZ[i + 2] * mmmXYZ[i + 2]);
        double z0 = b * b * mmmXYZ[i + 2] / (a * V);
        rrmLLA[i + 0] = atan((mmmXYZ[i + 2] + ed2 * z0) / p);
        rrmLLA[i + 1] = atan2(mmmXYZ[i + 1], mmmXYZ[i + 0]);
        rrmLLA[i + 2] = U * (1 - b * b / (a * V));
    }
}

/*
ECEF to ENU transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

@param double *rrmLLALocalOrigin array of size nx3 of local reference point X,
Y, Z [m, m, m]
@param double *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void ECEF2ENUFloatRolled(const float* rrmLLALocalOrigin,
    const float* mmmXYZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mmmLocal)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mmmXYZLocalOrigin = (float*)malloc(nOriginPoints * NCOORDSIN3D * sizeof(float));
    geodetic2ECEFFloatRolled(rrmLLALocalOrigin, nOriginPoints, (float)(a), (float)(b), mmmXYZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        float DeltaX = mmmXYZTarget[iTarget + 0] - mmmXYZLocalOrigin[iOrigin + 0];
        float DeltaY = mmmXYZTarget[iTarget + 1] - mmmXYZLocalOrigin[iOrigin + 1];
        float DeltaZ = mmmXYZTarget[iTarget + 2] - mmmXYZLocalOrigin[iOrigin + 2];
        mmmLocal[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + cosf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY;
        mmmLocal[iTarget + 1] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + cosf(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
        mmmLocal[iTarget + 2] = cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + sinf(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
    }
    free(mmmXYZLocalOrigin);
}

/*
ECEF to ENU transformation of double precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ECEF_to_ENU

@param double *rrmLLALocalOrigin array of size nx3 of local reference point X,
Y, Z [m, m, m]
@param double *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void ECEF2ENUDoubleRolled(const double* rrmLLALocalOrigin,
    const double* mmmXYZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mmmLocal)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mmmXYZLocalOrigin = (double*)malloc(nOriginPoints * NCOORDSIN3D * sizeof(double));
    geodetic2ECEFDoubleRolled(
        rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        double DeltaX = mmmXYZTarget[iTarget + 0] - mmmXYZLocalOrigin[iOrigin + 0];
        double DeltaY = mmmXYZTarget[iTarget + 1] - mmmXYZLocalOrigin[iOrigin + 1];
        double DeltaZ = mmmXYZTarget[iTarget + 2] - mmmXYZLocalOrigin[iOrigin + 2];
        mmmLocal[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY;
        mmmLocal[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + cos(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
        mmmLocal[iTarget + 2] = cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + sin(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
    }
    free(mmmXYZLocalOrigin);
}

void ECEF2NEDFloatRolled(const float* rrmLLALocalOrigin,
    const float* mmmXYZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mmmLocal)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mmmXYZLocalOrigin = (float*)malloc(nOriginPoints * NCOORDSIN3D * sizeof(float));
    geodetic2ECEFFloatRolled(
        rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        float DeltaX = mmmXYZTarget[iTarget + 0] - mmmXYZLocalOrigin[iOrigin + 0];
        float DeltaY = mmmXYZTarget[iTarget + 1] - mmmXYZLocalOrigin[iOrigin + 1];
        float DeltaZ = mmmXYZTarget[iTarget + 2] - mmmXYZLocalOrigin[iOrigin + 2];
        mmmLocal[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + cosf(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
        mmmLocal[iTarget + 1] = -sinf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + cosf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY;
        mmmLocal[iTarget + 2] = -cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
    }
    free(mmmXYZLocalOrigin);
}

void ECEF2NEDDoubleRolled(const double* rrmLLALocalOrigin,
    const double* mmmXYZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mmmLocal)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mmmXYZLocalOrigin = (double*)malloc(nOriginPoints * NCOORDSIN3D * sizeof(double));
    geodetic2ECEFDoubleRolled(
        rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        double DeltaX = mmmXYZTarget[iTarget + 0] - mmmXYZLocalOrigin[iOrigin + 0];
        double DeltaY = mmmXYZTarget[iTarget + 1] - mmmXYZLocalOrigin[iOrigin + 1];
        double DeltaZ = mmmXYZTarget[iTarget + 2] - mmmXYZLocalOrigin[iOrigin + 2];
        mmmLocal[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + cos(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
        mmmLocal[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY;
        mmmLocal[iTarget + 2] = -cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + -sin(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
    }
    free(mmmXYZLocalOrigin);
}

void ECEF2NEDvFloatRolled(const float* rrmLLALocalOrigin,
    const float* mmmXYZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    float* mmmLocal)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmLocal[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
        mmmLocal[iTarget + 1] = -sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1];
        mmmLocal[iTarget + 2] = -cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
    }
}

void ECEF2NEDvDoubleRolled(const double* rrmLLALocalOrigin,
    const double* mmmXYZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    double* mmmLocal)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmLocal[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + cos(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
        mmmLocal[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1];
        mmmLocal[iTarget + 2] = -cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
    }
}

void ECEF2ENUvFloatRolled(const float* rrmLLALocalOrigin,
    const float* mmmXYZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    float* mmmLocal)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmLocal[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1];
        mmmLocal[iTarget + 1] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
        mmmLocal[iTarget + 2] = cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + sinf(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
    }
}

void ECEF2ENUvDoubleRolled(const double* rrmLLALocalOrigin,
    const double* mmmXYZTarget,
    long nTargets,
    int isOriginSizeOfTargets,
    double* mmmLocal)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmLocal[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1];
        mmmLocal[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + cos(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
        mmmLocal[iTarget + 2] = cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + sin(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
    }
}

void ENU2ECEFvFloatRolled(const float* rrmLLALocalOrigin,
    const float* mmmTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    float* mmmXYZTarget)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmXYZTarget[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 1] = cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 2] = cosf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 1] + sinf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2];
    }
}

void NED2ECEFvFloatRolled(const float* rrmLLALocalOrigin,
    const float* mmmTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    float* mmmXYZTarget)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmXYZTarget[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 1] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 2] = cosf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2];
    }
}

void NED2ECEFvDoubleRolled(const double* rrmLLALocalOrigin,
    const double* mmmTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    double* mmmXYZTarget)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmXYZTarget[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 2] = cos(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2];
    }
}

void ENU2ECEFvDoubleRolled(const double* rrmLLALocalOrigin,
    const double* mmmTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    double* mmmXYZTarget)
{
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmXYZTarget[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 1] = cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 2] = cos(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 1] + sin(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2];
    }
}

/*
ECEF to ENU transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *rrmLLALocalOrigin array of size nx3 of local reference point
latitude, longitude, height [rad, rad, m]
@param float *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void NED2ECEFFloatRolled(const float* rrmLLALocalOrigin,
    const float* mmmTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mmmXYZTarget)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mmmXYZLocalOrigin = (float*)malloc(nOriginPoints * NCOORDSIN3D * sizeof(float));
    geodetic2ECEFFloatRolled(rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmXYZTarget[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 0];
        mmmXYZTarget[iTarget + 1] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 1];
        mmmXYZTarget[iTarget + 2] = cosf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 2];
    }
    free(mmmXYZLocalOrigin);
}

/*
ECEF to ENU transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *rrmLLALocalOrigin array of size nx3 of local reference point
latitude, longitude, height [rad, rad, m]
@param float *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void NED2ECEFDoubleRolled(const double* rrmLLALocalOrigin,
    const double* mmmTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mmmXYZTarget)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mmmXYZLocalOrigin = (double*)malloc(nOriginPoints * NCOORDSIN3D * sizeof(double));
    geodetic2ECEFDoubleRolled(rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmXYZTarget[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 0];
        mmmXYZTarget[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 1];
        mmmXYZTarget[iTarget + 2] = cos(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 2];
    }
    free(mmmXYZLocalOrigin);
}

/*
ECEF to ENU transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *rrmLLALocalOrigin array of size nx3 of local reference point
latitude, longitude, height [rad, rad, m]
@param float *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void ENU2ECEFFloatRolled(const float* rrmLLALocalOrigin,
    const float* mmmTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mmmXYZTarget)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mmmXYZLocalOrigin = (float*)malloc(nOriginPoints * NCOORDSIN3D * sizeof(float));
    geodetic2ECEFFloatRolled(rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        mmmXYZTarget[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 0];
        mmmXYZTarget[iTarget + 1] = cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 1];
        mmmXYZTarget[iTarget + 2] = cosf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 1] + sinf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 2];
    }
    free(mmmXYZLocalOrigin);
}

/*
ECEF to ENU transformation of double precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_ENU_to_ECEF
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *rrmLLALocalOrigin array of size nx3 of local reference point
latitude, longitude, height [rad, rad, m]
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
*/
void ENU2ECEFDoubleRolled(const double* rrmLLALocalOrigin,
    const double* mmmTargetLocal,
    long nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mmmXYZTarget)
{
    long nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mmmXYZLocalOrigin = (double*)malloc(nOriginPoints * NCOORDSIN3D * sizeof(double));
    geodetic2ECEFDoubleRolled(
        rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    long iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        long iTarget = iPoint * NCOORDSIN3D;
        long iOrigin = iTarget * isOriginSizeOfTargets;
        // mmmXYZTarget[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + mmmXYZLocalOrigin[iOrigin + 0];
        // mmmXYZTarget[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cos(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 1];
        // mmmXYZTarget[iTarget + 2] = cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + sin(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 2];
        mmmXYZTarget[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 0];
        mmmXYZTarget[iTarget + 1] = cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 1];
        mmmXYZTarget[iTarget + 2] = cos(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 1] + sin(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2] + mmmXYZLocalOrigin[iOrigin + 2];
    }
    free(mmmXYZLocalOrigin);
}

/*
ENU to AER transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf <-
includes additional errors and factors that could be implemented
https://www.lddgo.net/en/coordinate/ecef-enu

@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void NED2AERFloatRolled(const float* mmmNED, long nPoints, float* rrmAER)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        rrmAER[i + 0] = atan2f(mmmNED[i + 1], mmmNED[i + 0]);
        if (rrmAER[i + 0] < 0)
            rrmAER[i + 0] = rrmAER[i + 0] + (2.0f * PIf);
        rrmAER[i + 2] = sqrtf(mmmNED[i + 0] * mmmNED[i + 0] + mmmNED[i + 1] * mmmNED[i + 1] + mmmNED[i + 2] * mmmNED[i + 2]);
        rrmAER[i + 1] = asinf(-mmmNED[i + 2] / rrmAER[i + 2]);
    }
}

/*
ENU to AER transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf <-
includes additional errors and factors that could be implemented
https://www.lddgo.net/en/coordinate/ecef-enu

@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void NED2AERDoubleRolled(const double* mmmNED, long nPoints, double* rrmAER)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        rrmAER[i + 0] = atan2(mmmNED[i + 1], mmmNED[i + 0]);
        if (rrmAER[i + 0] < 0)
            rrmAER[i + 0] = rrmAER[i + 0] + 2.0 * PI;
        rrmAER[i + 2] = sqrt(mmmNED[i + 0] * mmmNED[i + 0] + mmmNED[i + 1] * mmmNED[i + 1] + mmmNED[i + 2] * mmmNED[i + 2]);
        rrmAER[i + 1] = asin(-mmmNED[i + 2] / rrmAER[i + 2]);
    }
}

/*
ENU to AER transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf <-
includes additional errors and factors that could be implemented
https://www.lddgo.net/en/coordinate/ecef-enu

@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void ENU2AERFloatRolled(const float* mmmENU, long nPoints, float* rrmAER)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        rrmAER[i + 0] = atan2f(mmmENU[i + 0], mmmENU[i + 1]);
        if (rrmAER[i + 0] < 0)
            rrmAER[i + 0] = rrmAER[i + 0] + (2.0f * PIf);
        rrmAER[i + 2] = sqrtf(mmmENU[i + 0] * mmmENU[i + 0] + mmmENU[i + 1] * mmmENU[i + 1] + mmmENU[i + 2] * mmmENU[i + 2]);
        rrmAER[i + 1] = asinf(mmmENU[i + 2] / rrmAER[i + 2]);
    }
}

/*
ENU to AER transformation of double precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param long nPoints Number of target points
@param double *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void ENU2AERDoubleRolled(const double* mmmENU, long nPoints, double* rrmAER)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        rrmAER[i + 0] = atan2(mmmENU[i + 0], mmmENU[i + 1]);
        if (rrmAER[i + 0] < 0)
            rrmAER[i + 0] = rrmAER[i + 0] + 2.0 * PI;
        rrmAER[i + 2] = sqrt(mmmENU[i + 0] * mmmENU[i + 0] + mmmENU[i + 1] * mmmENU[i + 1] + mmmENU[i + 2] * mmmENU[i + 2]);
        rrmAER[i + 1] = asin(mmmENU[i + 2] / rrmAER[i + 2]);
    }
}

/*
AER to ENU transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf

@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
@param long nPoints Number of target points
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2NEDFloatRolled(const float* rrmAER, long nPoints, float* mmmNED)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        mmmNED[i + 0] = cosf(rrmAER[i + 1]) * cosf(rrmAER[i + 0]) * rrmAER[i + 2];
        mmmNED[i + 1] = cosf(rrmAER[i + 1]) * sinf(rrmAER[i + 0]) * rrmAER[i + 2];
        mmmNED[i + 2] = -sinf(rrmAER[i + 1]) * rrmAER[i + 2];
    }
}

/*
AER to ENU transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf

@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
@param long nPoints Number of target points
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2NEDDoubleRolled(const double* rrmAER, long nPoints, double* mmmNED)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        mmmNED[i + 0] = cos(rrmAER[i + 1]) * cos(rrmAER[i + 0]) * rrmAER[i + 2];
        mmmNED[i + 1] = cos(rrmAER[i + 1]) * sin(rrmAER[i + 0]) * rrmAER[i + 2];
        mmmNED[i + 2] = -sin(rrmAER[i + 1]) * rrmAER[i + 2];
    }
}

/*
AER to ENU transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf

@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
@param long nPoints Number of target points
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2ENUFloatRolled(const float* rrmAER, long nPoints, float* mmmENU)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        mmmENU[i + 0] = cosf(rrmAER[i + 1]) * sinf(rrmAER[i + 0]) * rrmAER[i + 2];
        mmmENU[i + 1] = cosf(rrmAER[i + 1]) * cosf(rrmAER[i + 0]) * rrmAER[i + 2];
        mmmENU[i + 2] = sinf(rrmAER[i + 1]) * rrmAER[i + 2];
    }
}

/*
AER to ENU transformation of double precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf

@param double *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
@param long nPoints Number of target points
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2ENUDoubleRolled(const double* rrmAER, long nPoints, double* mmmENU)
{
    long iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        long i = iPoint * NCOORDSIN3D;
        mmmENU[i + 0] = cos(rrmAER[i + 1]) * sin(rrmAER[i + 0]) * rrmAER[i + 2];
        mmmENU[i + 1] = cos(rrmAER[i + 1]) * cos(rrmAER[i + 0]) * rrmAER[i + 2];
        mmmENU[i + 2] = sin(rrmAER[i + 1]) * rrmAER[i + 2];
    }
}

static PyObject*
geodetic2UTMRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* rrmLLA;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args, "Odd", &rrmLLA, &a, &b))
        return NULL;
    rrmLLA = get_numpy_array(rrmLLA);
    if (PyErr_Occurred())
        return NULL;
    if (check_arrays_same_float_dtype(1, (PyArrayObject *[]){rrmLLA}) == 0) {
        rrmLLA = (PyArrayObject *)PyArray_CastToType(rrmLLA, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if ((PyArray_SIZE(rrmLLA) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    long nPoints = (int)PyArray_SIZE(rrmLLA) / NCOORDSIN3D;
    PyArrayObject* result_array;
    if ((nPoints == 1) && (PyArray_NDIM(rrmLLA) == 2)) {
        npy_intp dims[2] = { 2, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLA), dims, PyArray_TYPE(rrmLLA));
    } else if ((nPoints == 1) && (PyArray_NDIM(rrmLLA) == 3)) {
        npy_intp dims[3] = { 1, 2, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLA), dims, PyArray_TYPE(rrmLLA));
    } else if (nPoints > 1) {
        npy_intp dims[3] = { nPoints, 2, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLA), dims, PyArray_TYPE(rrmLLA));
    } else {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output array.");
        return NULL;
    }
    if (result_array == NULL)
        return NULL;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        geodetic2UTMDoubleRolled((double*)PyArray_DATA(rrmLLA), nPoints, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        geodetic2UTMFloatRolled((float*)PyArray_DATA(rrmLLA), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
geodetic2UTMUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLat, *radLon, *mAlt;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args, "OOOdd", &radLat, &radLon, &mAlt, &a, &b))
        return NULL;
    if (((radLat = get_numpy_array(radLat)) == NULL) || ((radLon = get_numpy_array(radLon)) == NULL) || ((mAlt = get_numpy_array(mAlt)) == NULL))
        return NULL;
    PyArrayObject *arrays[] = {radLat, radLon, mAlt};
    if (check_arrays_same_size(3, arrays) == 0)
        return NULL;
    if (check_arrays_same_float_dtype(3, arrays) == 0) {
        radLat = (PyArrayObject *)PyArray_CastToType(radLat, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLon = (PyArrayObject *)PyArray_CastToType(radLon, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAlt = (PyArrayObject *)PyArray_CastToType(mAlt, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }

    PyArrayObject *outX, *outY;
    outX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radLat), PyArray_SHAPE(radLat), PyArray_TYPE(radLat));
    outY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radLat), PyArray_SHAPE(radLat), PyArray_TYPE(radLat));
    if ((outX == NULL) || (outY == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(radLat);

    // run function
    switch (PyArray_TYPE(radLat)) {
    case NPY_DOUBLE:
        geodetic2UTMDoubleUnrolled((double*)PyArray_DATA(radLat), (double*)PyArray_DATA(radLon), (double*)PyArray_DATA(mAlt), nPoints, a, b, (double*)PyArray_DATA(outX), (double*)PyArray_DATA(outY));
        break;
    case NPY_FLOAT:
        geodetic2UTMFloatUnrolled((float*)PyArray_DATA(radLat), (float*)PyArray_DATA(radLon), (float*)PyArray_DATA(mAlt), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(outX), (float*)PyArray_DATA(outY));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(2);
    if (!tuple){
        Py_DECREF(outX);
        Py_DECREF(outY);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)outX);
    PyTuple_SetItem(tuple, 1, (PyObject*)outY);
    return tuple;
}

static PyObject*
geodetic2UTMWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 3)
        return geodetic2UTMRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 5)
        return geodetic2UTMUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either three or five inputs");
        return NULL;
    }
}

static PyObject*
UTM2geodeticRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* mmUTM;
    double a, b;
    PyObject* ZoneNumberPy;
    char* ZoneLetter;

    // checks
    if (!PyArg_ParseTuple(args, "OOsdd", &mmUTM, &ZoneNumberPy, &ZoneLetter, &a, &b))
        return NULL;
    if (!PyLong_Check(ZoneNumberPy)) {
        PyErr_SetString(PyExc_TypeError, "Zone number must be an integer");
        return NULL;
    }
    long ZoneNumber = PyLong_AsLong(ZoneNumberPy);
    mmUTM = get_numpy_array(mmUTM);
    if (PyErr_Occurred())
        return NULL;
    if (check_arrays_same_float_dtype(1, (PyArrayObject *[]){mmUTM}) == 0)
        mmUTM = (PyArrayObject *)PyArray_CastToType(mmUTM, PyArray_DescrFromType(NPY_FLOAT64), 0);
    if ((PyArray_SIZE(mmUTM) % NCOORDSIN2D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 2.");
        return NULL;
    }

    long nPoints = (int)PyArray_SIZE(mmUTM) / NCOORDSIN2D;
    PyArrayObject* result_array;
    if ((nPoints == 1) && (PyArray_NDIM(mmUTM) == 2)) {
        npy_intp dims[2] = { 3, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmUTM), dims, PyArray_TYPE(mmUTM));
    } else if ((nPoints == 1) && (PyArray_NDIM(mmUTM) == 3)) {
        npy_intp dims[3] = { 1, 3, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmUTM), dims, PyArray_TYPE(mmUTM));
    } else if (nPoints > 1) {
        npy_intp dims[3] = { nPoints, 3, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmUTM), dims, PyArray_TYPE(mmUTM));
    } else {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output array.");
        return NULL;
    }
    if (result_array == NULL)
        return NULL;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        UTM2geodeticDoubleRolled((double*)PyArray_DATA(mmUTM), ZoneNumber, ZoneLetter, nPoints, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        UTM2geodeticFloatRolled((float*)PyArray_DATA(mmUTM), ZoneNumber, ZoneLetter, nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
UTM2geodeticUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *mX, *mY;
    double a, b;
    PyObject* ZoneNumberPy;
    char* ZoneLetter;

    // checks
    if (!PyArg_ParseTuple(args, "OOOsdd", &mX, &mY, &ZoneNumberPy, &ZoneLetter, &a, &b))
        return NULL;
    if (((mX = get_numpy_array(mX)) == NULL) || ((mY = get_numpy_array(mY)) == NULL))
        return NULL;
    if (!PyLong_Check(ZoneNumberPy)) {
        PyErr_SetString(PyExc_TypeError, "Zone number must be an integer");
        return NULL;
    }
    long ZoneNumber = PyLong_AsLong(ZoneNumberPy);
    if (PyErr_Occurred())
        return NULL;
    PyArrayObject *arrays[] = {mX, mY};
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        mX = (PyArrayObject *)PyArray_CastToType(mX, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mY = (PyArrayObject *)PyArray_CastToType(mY, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (check_arrays_same_size(2, arrays) == 0)
        return NULL;

    PyArrayObject *radLat, *radLon, *mAlt;
    radLat = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mX), PyArray_SHAPE(mX), PyArray_TYPE(mX));
    radLon = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mX), PyArray_SHAPE(mX), PyArray_TYPE(mX));
    mAlt = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mX), PyArray_SHAPE(mX), PyArray_TYPE(mX));
    if ((radLat == NULL) || (radLon == NULL) || (mAlt == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }

    // run function
    long nPoints = (int)PyArray_SIZE(mX);
    switch (PyArray_TYPE(radLat)) {
    case NPY_DOUBLE:
        UTM2geodeticDoubleUnrolled((double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), ZoneNumber, ZoneLetter, nPoints, a, b, (double*)PyArray_DATA(radLat), (double*)PyArray_DATA(radLon), (double*)PyArray_DATA(mAlt));
        break;
    case NPY_FLOAT:
        UTM2geodeticFloatUnrolled((float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), ZoneNumber, ZoneLetter, nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(radLat), (float*)PyArray_DATA(radLon), (float*)PyArray_DATA(mAlt));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(radLat);
        Py_DECREF(radLon);
        Py_DECREF(mAlt);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)radLat);
    PyTuple_SetItem(tuple, 1, (PyObject*)radLon);
    PyTuple_SetItem(tuple, 2, (PyObject*)mAlt);
    return tuple;
}

static PyObject*
UTM2geodeticWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 5)
        return UTM2geodeticRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 6)
        return UTM2geodeticUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either five or six inputs");
        return NULL;
    }
}

static PyObject*
geodetic2ECEFRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* rrmLLA;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args, "Odd", &rrmLLA, &a, &b))
        return NULL;
    rrmLLA = get_numpy_array(rrmLLA);
    if (PyErr_Occurred())
        return NULL;
    if (check_arrays_same_float_dtype(1, (PyArrayObject *[]){rrmLLA}) == 0) {
        rrmLLA = (PyArrayObject *)PyArray_CastToType(rrmLLA, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if ((PyArray_SIZE(rrmLLA) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(rrmLLA), PyArray_SHAPE(rrmLLA), PyArray_TYPE(rrmLLA));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(rrmLLA) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        geodetic2ECEFDoubleRolled((double*)PyArray_DATA(rrmLLA), nPoints, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        geodetic2ECEFFloatRolled((float*)PyArray_DATA(rrmLLA), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
geodetic2ECEFUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLat, *radLon, *mAlt;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args, "OOOdd", &radLat, &radLon, &mAlt, &a, &b))
        return NULL;
    if (((radLat = get_numpy_array(radLat)) == NULL) || ((radLon = get_numpy_array(radLon)) == NULL) || ((mAlt = get_numpy_array(mAlt)) == NULL))
        return NULL;
    PyArrayObject *arrays[] = {radLat, radLon, mAlt};
    if (check_arrays_same_size(3, arrays) == 0)
        return NULL;
    if (check_arrays_same_float_dtype(3, arrays) == 0) {
        radLat = (PyArrayObject *)PyArray_CastToType(radLat, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLon = (PyArrayObject *)PyArray_CastToType(radLon, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAlt = (PyArrayObject *)PyArray_CastToType(mAlt, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }

    // prepare inputs
    PyArrayObject *outX, *outY, *outZ;
    outX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radLat), PyArray_SHAPE(radLat), PyArray_TYPE(radLat));
    outY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radLat), PyArray_SHAPE(radLat), PyArray_TYPE(radLat));
    outZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radLat), PyArray_SHAPE(radLat), PyArray_TYPE(radLat));
    if ((outX == NULL) || (outY == NULL) || (outZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(radLat);

    // run function
    switch (PyArray_TYPE(outX)) {
    case NPY_DOUBLE:
        geodetic2ECEFDoubleUnrolled((double*)PyArray_DATA(radLat), (double*)PyArray_DATA(radLon), (double*)PyArray_DATA(mAlt), nPoints, a, b, (double*)PyArray_DATA(outX), (double*)PyArray_DATA(outY), (double*)PyArray_DATA(outZ));
        break;
    case NPY_FLOAT:
        geodetic2ECEFFloatUnrolled((float*)PyArray_DATA(radLat), (float*)PyArray_DATA(radLon), (float*)PyArray_DATA(mAlt), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(outX), (float*)PyArray_DATA(outY), (float*)PyArray_DATA(outZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(outX);
        Py_DECREF(outY);
        Py_DECREF(outZ);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)outX);
    PyTuple_SetItem(tuple, 1, (PyObject*)outY);
    PyTuple_SetItem(tuple, 2, (PyObject*)outZ);
    return tuple;
}

static PyObject*
geodetic2ECEFWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 3)
        return geodetic2ECEFRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 5)
        return geodetic2ECEFUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either three or five inputs");
        return NULL;
    }
}

static PyObject*
ECEF2geodeticUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *mX, *mY, *mZ;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args, "OOOdd", &mX, &mY, &mZ, &a, &b))
        return NULL;
    if (((mX = get_numpy_array(mX)) == NULL) || ((mY = get_numpy_array(mY)) == NULL) || ((mZ = get_numpy_array(mZ)) == NULL))
        return NULL;
    PyArrayObject *arrays[] = {mX, mY, mZ};
    if (check_arrays_same_size(3, arrays) == 0)
        return NULL;
    if (check_arrays_same_float_dtype(3, arrays) == 0) {
        mX = (PyArrayObject *)PyArray_CastToType(mX, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mY = (PyArrayObject *)PyArray_CastToType(mY, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mZ = (PyArrayObject *)PyArray_CastToType(mZ, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    
     // prepare inputs
     PyArrayObject *radLat, *radLon, *mAlt;
     radLat = (PyArrayObject*)PyArray_SimpleNew(
         PyArray_NDIM(mX), PyArray_SHAPE(mX), PyArray_TYPE(mX));
     radLon = (PyArrayObject*)PyArray_SimpleNew(
         PyArray_NDIM(mX), PyArray_SHAPE(mX), PyArray_TYPE(mX));
     mAlt = (PyArrayObject*)PyArray_SimpleNew(
         PyArray_NDIM(mX), PyArray_SHAPE(mX), PyArray_TYPE(mX));
     if ((radLat == NULL) || (radLon == NULL) || (mAlt == NULL)) {
         PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
         return NULL;
     }
     long nPoints = (int)PyArray_SIZE(mX);
 
    // run function
    switch (PyArray_TYPE(radLat)) {
    case NPY_DOUBLE:
        ECEF2geodeticDoubleUnrolled((double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ), nPoints, a, b, (double*)PyArray_DATA(radLat), (double*)PyArray_DATA(radLon), (double*)PyArray_DATA(mAlt));
        break;
    case NPY_FLOAT:
        ECEF2geodeticFloatUnrolled((float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(radLat), (float*)PyArray_DATA(radLon), (float*)PyArray_DATA(mAlt));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(radLat);
        Py_DECREF(radLon);
        Py_DECREF(mAlt);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)radLat);
    PyTuple_SetItem(tuple, 1, (PyObject*)radLon);
    PyTuple_SetItem(tuple, 2, (PyObject*)mAlt);
    return tuple;
}

static PyObject*
ECEF2geodeticRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* mmmXYZ;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args, "Odd", &mmmXYZ, &a, &b))
        return NULL;
    mmmXYZ = get_numpy_array(mmmXYZ);
    if (check_arrays_same_float_dtype(1, (PyArrayObject *[]){mmmXYZ}) == 0)
        mmmXYZ = (PyArrayObject *)PyArray_CastToType(mmmXYZ, PyArray_DescrFromType(NPY_FLOAT64), 0);
    if (PyErr_Occurred())
        return NULL;
    if ((PyArray_SIZE(mmmXYZ) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mmmXYZ), PyArray_SHAPE(mmmXYZ), PyArray_TYPE(mmmXYZ));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmXYZ) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2geodeticDoubleRolled((double*)PyArray_DATA(mmmXYZ), nPoints, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2geodeticFloatRolled((float*)PyArray_DATA(mmmXYZ), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
ECEF2geodeticWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 3)
        return ECEF2geodeticRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 5)
        return ECEF2geodeticUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either three or five inputs");
        return NULL;
    }
}

static PyObject*
ECEF2ENUUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLatOrigin, *radLonOrigin, *mAltOrigin, *mXTarget, *mYTarget, *mZTarget;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args,
            "OOOOOOdd",
            &radLatOrigin,
            &radLonOrigin,
            &mAltOrigin,
            &mXTarget,
            &mYTarget,
            &mZTarget,
            &a,
            &b))
        return NULL;
    if (((radLatOrigin = get_numpy_array(radLatOrigin)) == NULL) || ((radLonOrigin = get_numpy_array(radLonOrigin)) == NULL) || ((mAltOrigin = get_numpy_array(mAltOrigin)) == NULL) || ((mXTarget = get_numpy_array(mXTarget)) == NULL) || ((mYTarget = get_numpy_array(mYTarget)) == NULL) || ((mZTarget = get_numpy_array(mZTarget)) == NULL))
        return NULL;
    if (check_arrays_same_float_dtype(6, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin, mXTarget, mYTarget, mZTarget}) == 0) {
        radLatOrigin = (PyArrayObject *)PyArray_CastToType(radLatOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLonOrigin = (PyArrayObject *)PyArray_CastToType(radLonOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAltOrigin = (PyArrayObject *)PyArray_CastToType(mAltOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mXTarget = (PyArrayObject *)PyArray_CastToType(mXTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mYTarget = (PyArrayObject *)PyArray_CastToType(mYTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mZTarget = (PyArrayObject *)PyArray_CastToType(mZTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (check_arrays_same_size(3, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin}) == 0)
        return NULL;
    if (check_arrays_same_size(3, (PyArrayObject *[]){mXTarget, mYTarget, mZTarget}) == 0)
        return NULL;

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(mXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)mXTarget) == PyArray_Size((PyObject*)radLatOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        ECEF2ENUDoubleUnrolled(
            (double*)PyArray_DATA(radLatOrigin), (double*)PyArray_DATA(radLonOrigin), (double*)PyArray_DATA(mAltOrigin), (double*)PyArray_DATA(mXTarget), (double*)PyArray_DATA(mYTarget), (double*)PyArray_DATA(mZTarget), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        ECEF2ENUFloatUnrolled(
            (float*)PyArray_DATA(radLatOrigin), (float*)PyArray_DATA(radLonOrigin), (float*)PyArray_DATA(mAltOrigin), (float*)PyArray_DATA(mXTarget), (float*)PyArray_DATA(mYTarget), (float*)PyArray_DATA(mZTarget), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(mX);
        Py_DECREF(mY);
        Py_DECREF(mZ);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)mX);
    PyTuple_SetItem(tuple, 1, (PyObject*)mY);
    PyTuple_SetItem(tuple, 2, (PyObject*)mZ);
    return tuple;
}

static PyObject*
ECEF2ENURolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *rrmLLALocalOrigin, *mmmXYZTarget;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args,
            "OOdd",
            &rrmLLALocalOrigin,
            &mmmXYZTarget,
            &a,
            &b))
        return NULL;
    mmmXYZTarget = get_numpy_array(mmmXYZTarget);
    rrmLLALocalOrigin = get_numpy_array(rrmLLALocalOrigin);
    PyArrayObject *arrays[] = {rrmLLALocalOrigin, mmmXYZTarget};
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        mmmXYZTarget = (PyArrayObject *)PyArray_CastToType(mmmXYZTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        rrmLLALocalOrigin = (PyArrayObject *)PyArray_CastToType(rrmLLALocalOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmXYZTarget)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmXYZTarget)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSIN3D) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmXYZTarget))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSIN3D) != 0 || (PyArray_SIZE(mmmXYZTarget) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(mmmXYZTarget),
        PyArray_SHAPE(mmmXYZTarget),
        PyArray_TYPE(mmmXYZTarget));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmXYZTarget) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)rrmLLALocalOrigin) == PyArray_Size((PyObject*)mmmXYZTarget));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2ENUDoubleRolled(
            (double*)PyArray_DATA(rrmLLALocalOrigin), (double*)PyArray_DATA(mmmXYZTarget), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2ENUFloatRolled(
            (float*)PyArray_DATA(rrmLLALocalOrigin), (float*)PyArray_DATA(mmmXYZTarget), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
ECEF2ENUWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 4)
        return ECEF2ENURolledWrapper(self, args);
    else if (PyTuple_Size(args) == 8)
        return ECEF2ENUUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either four or eight inputs");
        return NULL;
    }
}

static PyObject*
ECEF2NEDUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLatOrigin, *radLonOrigin, *mAltOrigin, *mXTarget, *mYTarget, *mZTarget;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args,
            "OOOOOOdd",
            &radLatOrigin,
            &radLonOrigin,
            &mAltOrigin,
            &mXTarget,
            &mYTarget,
            &mZTarget,
            &a,
            &b))
        return NULL;
    if (((radLatOrigin = get_numpy_array(radLatOrigin)) == NULL) || ((radLonOrigin = get_numpy_array(radLonOrigin)) == NULL) || ((mAltOrigin = get_numpy_array(mAltOrigin)) == NULL) || ((mXTarget = get_numpy_array(mXTarget)) == NULL) || ((mYTarget = get_numpy_array(mYTarget)) == NULL) || ((mZTarget = get_numpy_array(mZTarget)) == NULL))
        return NULL;
    PyArrayObject *arrays[] = {radLatOrigin, radLonOrigin, mAltOrigin, mXTarget, mYTarget, mZTarget};
    if (check_arrays_same_float_dtype(6, arrays) == 0) {
        radLatOrigin = (PyArrayObject *)PyArray_CastToType(radLatOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLonOrigin = (PyArrayObject *)PyArray_CastToType(radLonOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAltOrigin = (PyArrayObject *)PyArray_CastToType(mAltOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mXTarget = (PyArrayObject *)PyArray_CastToType(mXTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mYTarget = (PyArrayObject *)PyArray_CastToType(mYTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mZTarget = (PyArrayObject *)PyArray_CastToType(mZTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (check_arrays_same_size(3, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin}) == 0)
        return NULL;
    if (check_arrays_same_size(3, (PyArrayObject *[]){mXTarget, mYTarget, mZTarget}) == 0)
        return NULL;

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(mXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)mXTarget) == PyArray_Size((PyObject*)radLatOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        ECEF2NEDDoubleUnrolled(
            (double*)PyArray_DATA(radLatOrigin), (double*)PyArray_DATA(radLonOrigin), (double*)PyArray_DATA(mAltOrigin), (double*)PyArray_DATA(mXTarget), (double*)PyArray_DATA(mYTarget), (double*)PyArray_DATA(mZTarget), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        ECEF2NEDFloatUnrolled(
            (float*)PyArray_DATA(radLatOrigin), (float*)PyArray_DATA(radLonOrigin), (float*)PyArray_DATA(mAltOrigin), (float*)PyArray_DATA(mXTarget), (float*)PyArray_DATA(mYTarget), (float*)PyArray_DATA(mZTarget), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(mX);
        Py_DECREF(mY);
        Py_DECREF(mZ);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)mX);
    PyTuple_SetItem(tuple, 1, (PyObject*)mY);
    PyTuple_SetItem(tuple, 2, (PyObject*)mZ);
    return tuple;
}

static PyObject*
ECEF2NEDRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *rrmLLALocalOrigin, *mmmXYZTarget;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args,
            "OOdd",
            &rrmLLALocalOrigin,
            &mmmXYZTarget,
            &a,
            &b))
        return NULL;
    rrmLLALocalOrigin = get_numpy_array(rrmLLALocalOrigin);
    mmmXYZTarget = get_numpy_array(mmmXYZTarget);
    PyArrayObject *arrays[] = {rrmLLALocalOrigin, mmmXYZTarget};
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        mmmXYZTarget = (PyArrayObject *)PyArray_CastToType(mmmXYZTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        rrmLLALocalOrigin = (PyArrayObject *)PyArray_CastToType(rrmLLALocalOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmXYZTarget)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmXYZTarget)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSIN3D) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmXYZTarget))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSIN3D) != 0 || (PyArray_SIZE(mmmXYZTarget) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(mmmXYZTarget),
        PyArray_SHAPE(mmmXYZTarget),
        PyArray_TYPE(mmmXYZTarget));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmXYZTarget) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)rrmLLALocalOrigin) == PyArray_Size((PyObject*)mmmXYZTarget));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2NEDDoubleRolled(
            (double*)PyArray_DATA(rrmLLALocalOrigin), (double*)PyArray_DATA(mmmXYZTarget), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2NEDFloatRolled(
            (float*)PyArray_DATA(rrmLLALocalOrigin), (float*)PyArray_DATA(mmmXYZTarget), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
ECEF2NEDWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 4)
        return ECEF2NEDRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 8)
        return ECEF2NEDUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either four or eight inputs");
        return NULL;
    }
}

static PyObject*
ECEF2NEDvRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *rrmLLALocalOrigin, *mmmXYZTarget;

    // checks
    if (!PyArg_ParseTuple(args, "OO", &rrmLLALocalOrigin, &mmmXYZTarget))
        return NULL;
    rrmLLALocalOrigin = get_numpy_array(rrmLLALocalOrigin);
    mmmXYZTarget = get_numpy_array(mmmXYZTarget);
    PyArrayObject *arrays[] = {rrmLLALocalOrigin, mmmXYZTarget};
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        mmmXYZTarget = (PyArrayObject *)PyArray_CastToType(mmmXYZTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        rrmLLALocalOrigin = (PyArrayObject *)PyArray_CastToType(rrmLLALocalOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmXYZTarget)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmXYZTarget)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSIN3D) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmXYZTarget))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSIN3D) != 0 || (PyArray_SIZE(mmmXYZTarget) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(mmmXYZTarget),
        PyArray_SHAPE(mmmXYZTarget),
        PyArray_TYPE(mmmXYZTarget));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmXYZTarget) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)rrmLLALocalOrigin) == PyArray_Size((PyObject*)mmmXYZTarget));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2NEDvDoubleRolled(
            (double*)PyArray_DATA(rrmLLALocalOrigin), (double*)PyArray_DATA(mmmXYZTarget), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2NEDvFloatRolled(
            (float*)PyArray_DATA(rrmLLALocalOrigin), (float*)PyArray_DATA(mmmXYZTarget), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
ECEF2NEDvUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLatOrigin, *radLonOrigin, *mAltOrigin, *mXTarget, *mYTarget, *mZTarget;

    // checks
    if (!PyArg_ParseTuple(args,
        "OOOOOO",
        &radLatOrigin,
        &radLonOrigin,
        &mAltOrigin,
        &mXTarget,
        &mYTarget,
        &mZTarget))
        return NULL;
    if (((radLatOrigin = get_numpy_array(radLatOrigin)) == NULL) || ((radLonOrigin = get_numpy_array(radLonOrigin)) == NULL) || ((mAltOrigin = get_numpy_array(mAltOrigin)) == NULL) || ((mXTarget = get_numpy_array(mXTarget)) == NULL) || ((mYTarget = get_numpy_array(mYTarget)) == NULL) || ((mZTarget = get_numpy_array(mZTarget)) == NULL))
        return NULL;
    if (check_arrays_same_float_dtype(6, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin, mXTarget, mYTarget, mZTarget}) == 0) {
        radLatOrigin = (PyArrayObject *)PyArray_CastToType(radLatOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLonOrigin = (PyArrayObject *)PyArray_CastToType(radLonOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAltOrigin = (PyArrayObject *)PyArray_CastToType(mAltOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mXTarget = (PyArrayObject *)PyArray_CastToType(mXTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mYTarget = (PyArrayObject *)PyArray_CastToType(mYTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mZTarget = (PyArrayObject *)PyArray_CastToType(mZTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (check_arrays_same_size(3, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin}) == 0)
        return NULL;
    if (check_arrays_same_size(3, (PyArrayObject *[]){mXTarget, mYTarget, mZTarget}) == 0)
        return NULL;

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(mXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)mXTarget) == PyArray_Size((PyObject*)radLonOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        ECEF2NEDvDoubleUnrolled(
            (double*)PyArray_DATA(radLatOrigin), (double*)PyArray_DATA(radLonOrigin), (double*)PyArray_DATA(mAltOrigin), (double*)PyArray_DATA(mXTarget), (double*)PyArray_DATA(mYTarget), (double*)PyArray_DATA(mZTarget), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        ECEF2NEDvFloatUnrolled(
            (float*)PyArray_DATA(radLatOrigin), (float*)PyArray_DATA(radLonOrigin), (float*)PyArray_DATA(mAltOrigin), (float*)PyArray_DATA(mXTarget), (float*)PyArray_DATA(mYTarget), (float*)PyArray_DATA(mZTarget), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(mX);
        Py_DECREF(mY);
        Py_DECREF(mZ);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)mX);
    PyTuple_SetItem(tuple, 1, (PyObject*)mY);
    PyTuple_SetItem(tuple, 2, (PyObject*)mZ);
    return tuple;
}

static PyObject*
ECEF2NEDvWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 2)
        return ECEF2NEDvRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 6)
        return ECEF2NEDvUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either two or four inputs");
        return NULL;
    }
}

static PyObject*
ECEF2ENUvRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *rrmLLALocalOrigin, *mmmXYZTarget;

    // checks
    if (!PyArg_ParseTuple(args, "OO", &rrmLLALocalOrigin, &mmmXYZTarget))
        return NULL;
    rrmLLALocalOrigin = get_numpy_array(rrmLLALocalOrigin);
    mmmXYZTarget = get_numpy_array(mmmXYZTarget);
    PyArrayObject *arrays[] = {rrmLLALocalOrigin, mmmXYZTarget};
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        mmmXYZTarget = (PyArrayObject *)PyArray_CastToType(mmmXYZTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        rrmLLALocalOrigin = (PyArrayObject *)PyArray_CastToType(rrmLLALocalOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmXYZTarget)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmXYZTarget)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSIN3D) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmXYZTarget))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSIN3D) != 0 || (PyArray_SIZE(mmmXYZTarget) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(mmmXYZTarget),
        PyArray_SHAPE(mmmXYZTarget),
        PyArray_TYPE(mmmXYZTarget));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmXYZTarget) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)rrmLLALocalOrigin) == PyArray_Size((PyObject*)mmmXYZTarget));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2ENUvDoubleRolled(
            (double*)PyArray_DATA(rrmLLALocalOrigin), (double*)PyArray_DATA(mmmXYZTarget), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2ENUvFloatRolled(
            (float*)PyArray_DATA(rrmLLALocalOrigin), (float*)PyArray_DATA(mmmXYZTarget), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
ECEF2ENUvUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLatOrigin, *radLonOrigin, *mAltOrigin, *mXTarget, *mYTarget, *mZTarget;

    // checks
    if (!PyArg_ParseTuple(args,
        "OOOOOO",
        &radLatOrigin,
        &radLonOrigin,
        &mAltOrigin,
        &mXTarget,
        &mYTarget,
        &mZTarget))
        return NULL;
    if (((radLatOrigin = get_numpy_array(radLatOrigin)) == NULL) || ((radLonOrigin = get_numpy_array(radLonOrigin)) == NULL) || ((mAltOrigin = get_numpy_array(mAltOrigin)) == NULL) || ((mXTarget = get_numpy_array(mXTarget)) == NULL) || ((mYTarget = get_numpy_array(mYTarget)) == NULL) || ((mZTarget = get_numpy_array(mZTarget)) == NULL))
        return NULL;
    if (check_arrays_same_float_dtype(6, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin, mXTarget, mYTarget, mZTarget}) == 0) {
        radLatOrigin = (PyArrayObject *)PyArray_CastToType(radLatOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLonOrigin = (PyArrayObject *)PyArray_CastToType(radLonOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAltOrigin = (PyArrayObject *)PyArray_CastToType(mAltOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mXTarget = (PyArrayObject *)PyArray_CastToType(mXTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mYTarget = (PyArrayObject *)PyArray_CastToType(mYTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mZTarget = (PyArrayObject *)PyArray_CastToType(mZTarget, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (check_arrays_same_size(3, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin}) == 0)
        return NULL;
    if (check_arrays_same_size(3, (PyArrayObject *[]){mXTarget, mYTarget, mZTarget}) == 0)
        return NULL;

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), PyArray_TYPE(mXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(mXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)mXTarget) == PyArray_Size((PyObject*)radLatOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        ECEF2ENUvDoubleUnrolled(
            (double*)PyArray_DATA(radLatOrigin), (double*)PyArray_DATA(radLonOrigin), (double*)PyArray_DATA(mAltOrigin), (double*)PyArray_DATA(mXTarget), (double*)PyArray_DATA(mYTarget), (double*)PyArray_DATA(mZTarget), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        ECEF2ENUvFloatUnrolled(
            (float*)PyArray_DATA(radLatOrigin), (float*)PyArray_DATA(radLonOrigin), (float*)PyArray_DATA(mAltOrigin), (float*)PyArray_DATA(mXTarget), (float*)PyArray_DATA(mYTarget), (float*)PyArray_DATA(mZTarget), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(mX);
        Py_DECREF(mY);
        Py_DECREF(mZ);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)mX);
    PyTuple_SetItem(tuple, 1, (PyObject*)mY);
    PyTuple_SetItem(tuple, 2, (PyObject*)mZ);
    return tuple;
}

static PyObject*
ECEF2ENUvWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 2)
        return ECEF2ENUvRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 6)
        return ECEF2ENUvUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either two or four inputs");
        return NULL;
    }
}

static PyObject*
NED2ECEFUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLatOrigin, *radLonOrigin, *mAltOrigin, *mNLocal, *mELocal, *mDLocal;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args,
        "OOOOOOdd",
        &radLatOrigin,
        &radLonOrigin,
        &mAltOrigin,
        &mNLocal,
        &mELocal,
        &mDLocal, &a, &b))
        return NULL;
    if (((radLatOrigin = get_numpy_array(radLatOrigin)) == NULL) || ((radLonOrigin = get_numpy_array(radLonOrigin)) == NULL) || ((mAltOrigin = get_numpy_array(mAltOrigin)) == NULL) || ((mNLocal = get_numpy_array(mNLocal)) == NULL) || ((mELocal = get_numpy_array(mELocal)) == NULL) || ((mDLocal = get_numpy_array(mDLocal)) == NULL))
        return NULL;
    if (check_arrays_same_float_dtype(6, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin, mNLocal, mELocal, mDLocal}) == 0) {
        radLatOrigin = (PyArrayObject *)PyArray_CastToType(radLatOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLonOrigin = (PyArrayObject *)PyArray_CastToType(radLonOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAltOrigin = (PyArrayObject *)PyArray_CastToType(mAltOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mNLocal = (PyArrayObject *)PyArray_CastToType(mNLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mELocal = (PyArrayObject *)PyArray_CastToType(mELocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mDLocal = (PyArrayObject *)PyArray_CastToType(mDLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (check_arrays_same_size(3, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin}) == 0)
        return NULL;
    if (check_arrays_same_size(3, (PyArrayObject *[]){mNLocal, mELocal, mDLocal}) == 0)
        return NULL;

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(mNLocal);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)mNLocal) == PyArray_Size((PyObject*)radLatOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        NED2ECEFDoubleUnrolled(
            (double*)PyArray_DATA(radLatOrigin), (double*)PyArray_DATA(radLonOrigin), (double*)PyArray_DATA(mAltOrigin), (double*)PyArray_DATA(mNLocal), (double*)PyArray_DATA(mELocal), (double*)PyArray_DATA(mDLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        NED2ECEFFloatUnrolled(
            (float*)PyArray_DATA(radLatOrigin), (float*)PyArray_DATA(radLonOrigin), (float*)PyArray_DATA(mAltOrigin), (float*)PyArray_DATA(mNLocal), (float*)PyArray_DATA(mELocal), (float*)PyArray_DATA(mDLocal), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(mX);
        Py_DECREF(mY);
        Py_DECREF(mZ);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)mX);
    PyTuple_SetItem(tuple, 1, (PyObject*)mY);
    PyTuple_SetItem(tuple, 2, (PyObject*)mZ);
    return tuple;
}

static PyObject*
NED2ECEFRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *rrmLLALocalOrigin, *mmmLocal;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args,
            "OOdd",
            &rrmLLALocalOrigin,
            &mmmLocal,
            &a,
            &b))
        return NULL;
    rrmLLALocalOrigin = get_numpy_array(rrmLLALocalOrigin);
    mmmLocal = get_numpy_array(mmmLocal);
    PyArrayObject *arrays[] = {rrmLLALocalOrigin, mmmLocal};
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        mmmLocal = (PyArrayObject *)PyArray_CastToType(mmmLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        rrmLLALocalOrigin = (PyArrayObject *)PyArray_CastToType(rrmLLALocalOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmLocal)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmLocal)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSIN3D) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmLocal))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSIN3D) != 0 || (PyArray_SIZE(mmmLocal) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mmmLocal), PyArray_SHAPE(mmmLocal), PyArray_TYPE(mmmLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)rrmLLALocalOrigin) == PyArray_Size((PyObject*)mmmLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        NED2ECEFDoubleRolled(
            (double*)PyArray_DATA(rrmLLALocalOrigin), (double*)PyArray_DATA(mmmLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        NED2ECEFFloatRolled(
            (float*)PyArray_DATA(rrmLLALocalOrigin), (float*)PyArray_DATA(mmmLocal), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
NED2ECEFWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 4)
        return NED2ECEFRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 8)
        return NED2ECEFUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either four or eight inputs");
        return NULL;
    }
}

static PyObject*
ENU2ECEFUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLatOrigin, *radLonOrigin, *mAltOrigin, *mNLocal, *mELocal, *mDLocal;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args,
        "OOOOOOdd",
        &radLatOrigin,
        &radLonOrigin,
        &mAltOrigin,
        &mNLocal,
        &mELocal,
        &mDLocal,
            &a,
            &b))
        return NULL;
    if (((radLatOrigin = get_numpy_array(radLatOrigin)) == NULL) || ((radLonOrigin = get_numpy_array(radLonOrigin)) == NULL) || ((mAltOrigin = get_numpy_array(mAltOrigin)) == NULL) || ((mNLocal = get_numpy_array(mNLocal)) == NULL) || ((mELocal = get_numpy_array(mELocal)) == NULL) || ((mDLocal = get_numpy_array(mDLocal)) == NULL))
        return NULL;
    if (check_arrays_same_float_dtype(6, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin, mNLocal, mELocal, mDLocal}) == 0) {
        radLatOrigin = (PyArrayObject *)PyArray_CastToType(radLatOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLonOrigin = (PyArrayObject *)PyArray_CastToType(radLonOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAltOrigin = (PyArrayObject *)PyArray_CastToType(mAltOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mNLocal = (PyArrayObject *)PyArray_CastToType(mNLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mELocal = (PyArrayObject *)PyArray_CastToType(mELocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mDLocal = (PyArrayObject *)PyArray_CastToType(mDLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (check_arrays_same_size(3, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin}) == 0)
        return NULL;
    if (check_arrays_same_size(3, (PyArrayObject *[]){mNLocal, mELocal, mDLocal}) == 0)
        return NULL;

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(mNLocal);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)mNLocal) == PyArray_Size((PyObject*)radLatOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        ENU2ECEFDoubleUnrolled(
            (double*)PyArray_DATA(radLatOrigin), (double*)PyArray_DATA(radLonOrigin), (double*)PyArray_DATA(mAltOrigin), (double*)PyArray_DATA(mNLocal), (double*)PyArray_DATA(mELocal), (double*)PyArray_DATA(mDLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        ENU2ECEFFloatUnrolled(
            (float*)PyArray_DATA(radLatOrigin), (float*)PyArray_DATA(radLonOrigin), (float*)PyArray_DATA(mAltOrigin), (float*)PyArray_DATA(mNLocal), (float*)PyArray_DATA(mELocal), (float*)PyArray_DATA(mDLocal), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(mX);
        Py_DECREF(mY);
        Py_DECREF(mZ);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)mX);
    PyTuple_SetItem(tuple, 1, (PyObject*)mY);
    PyTuple_SetItem(tuple, 2, (PyObject*)mZ);
    return tuple;
}

static PyObject*
ENU2ECEFRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *rrmLLALocalOrigin, *mmmLocal;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args,
            "OOdd",
            &rrmLLALocalOrigin,
            &mmmLocal,
            &a,
            &b))
        return NULL;
    rrmLLALocalOrigin = get_numpy_array(rrmLLALocalOrigin);
    mmmLocal = get_numpy_array(mmmLocal);
    PyArrayObject *arrays[] = {rrmLLALocalOrigin, mmmLocal};
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        mmmLocal = (PyArrayObject *)PyArray_CastToType(mmmLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        rrmLLALocalOrigin = (PyArrayObject *)PyArray_CastToType(rrmLLALocalOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmLocal)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmLocal)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSIN3D) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmLocal))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSIN3D) != 0 || (PyArray_SIZE(mmmLocal) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mmmLocal), PyArray_SHAPE(mmmLocal), PyArray_TYPE(mmmLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)rrmLLALocalOrigin) == PyArray_Size((PyObject*)mmmLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ENU2ECEFDoubleRolled(
            (double*)PyArray_DATA(rrmLLALocalOrigin), (double*)PyArray_DATA(mmmLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ENU2ECEFFloatRolled(
            (float*)PyArray_DATA(rrmLLALocalOrigin), (float*)PyArray_DATA(mmmLocal), nPoints, isOriginSizeOfTargets, (float)a, (float)b, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
ENU2ECEFWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 4)
        return ENU2ECEFRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 8)
        return ENU2ECEFUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either four or eight inputs");
        return NULL;
    }
}

static PyObject*
ENU2ECEFvRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *rrmLLALocalOrigin, *mmmLocal;

    // checks
    if (!PyArg_ParseTuple(args,
            "OO",
            &rrmLLALocalOrigin,
            &mmmLocal))
        return NULL;
    rrmLLALocalOrigin = get_numpy_array(rrmLLALocalOrigin);
    mmmLocal = get_numpy_array(mmmLocal);
    PyArrayObject *arrays[] = {rrmLLALocalOrigin, mmmLocal};
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        mmmLocal = (PyArrayObject *)PyArray_CastToType(mmmLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        rrmLLALocalOrigin = (PyArrayObject *)PyArray_CastToType(rrmLLALocalOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmLocal)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmLocal)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSIN3D) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmLocal))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSIN3D) != 0 || (PyArray_SIZE(mmmLocal) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mmmLocal), PyArray_SHAPE(mmmLocal), PyArray_TYPE(mmmLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)rrmLLALocalOrigin) == PyArray_Size((PyObject*)mmmLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ENU2ECEFvDoubleRolled(
            (double*)PyArray_DATA(rrmLLALocalOrigin), (double*)PyArray_DATA(mmmLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ENU2ECEFvFloatRolled(
            (float*)PyArray_DATA(rrmLLALocalOrigin), (float*)PyArray_DATA(mmmLocal), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
ENU2ECEFvUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLatOrigin, *radLonOrigin, *mAltOrigin, *mELocal, *mNLocal, *mULocal;

    // checks
    if (!PyArg_ParseTuple(args,
        "OOOOOO",
        &radLatOrigin,
        &radLonOrigin,
        &mAltOrigin,
        &mELocal,
        &mNLocal,
        &mULocal))
        return NULL;
    if (((radLatOrigin = get_numpy_array(radLatOrigin)) == NULL) || ((radLonOrigin = get_numpy_array(radLonOrigin)) == NULL) || ((mAltOrigin = get_numpy_array(mAltOrigin)) == NULL) || ((mNLocal = get_numpy_array(mNLocal)) == NULL) || ((mELocal = get_numpy_array(mELocal)) == NULL) || ((mULocal = get_numpy_array(mULocal)) == NULL))
        return NULL;
    if (check_arrays_same_float_dtype(6, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin, mNLocal, mELocal, mULocal}) == 0) {
        radLatOrigin = (PyArrayObject *)PyArray_CastToType(radLatOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLonOrigin = (PyArrayObject *)PyArray_CastToType(radLonOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAltOrigin = (PyArrayObject *)PyArray_CastToType(mAltOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mNLocal = (PyArrayObject *)PyArray_CastToType(mNLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mELocal = (PyArrayObject *)PyArray_CastToType(mELocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mULocal = (PyArrayObject *)PyArray_CastToType(mULocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (check_arrays_same_size(3, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin}) == 0)
        return NULL;
    if (check_arrays_same_size(3, (PyArrayObject *[]){mNLocal, mELocal, mULocal}) == 0)
        return NULL;

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(mNLocal);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)mNLocal) == PyArray_Size((PyObject*)radLatOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
        case NPY_DOUBLE:
            ENU2ECEFvDoubleUnrolled(
                (double*)PyArray_DATA(radLatOrigin), (double*)PyArray_DATA(radLonOrigin), (double*)PyArray_DATA(mAltOrigin), (double*)PyArray_DATA(mELocal), (double*)PyArray_DATA(mNLocal), (double*)PyArray_DATA(mULocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
            break;
        case NPY_FLOAT:
            ENU2ECEFvFloatUnrolled(
                (float*)PyArray_DATA(radLatOrigin), (float*)PyArray_DATA(radLonOrigin), (float*)PyArray_DATA(mAltOrigin), (float*)PyArray_DATA(mELocal), (float*)PyArray_DATA(mNLocal), (float*)PyArray_DATA(mULocal), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                "Only 32 and 64 bit float types or all integer are accepted.");
            return NULL;
        }
    
        // output
        PyObject* tuple = PyTuple_New(3);
        if (!tuple){
            Py_DECREF(mX);
            Py_DECREF(mY);
            Py_DECREF(mZ);
            return NULL;
        }
        PyTuple_SetItem(tuple, 0, (PyObject*)mX);
        PyTuple_SetItem(tuple, 1, (PyObject*)mY);
        PyTuple_SetItem(tuple, 2, (PyObject*)mZ);
        return tuple;
    
}

static PyObject*
ENU2ECEFvWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 2)
        return ENU2ECEFvRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 6)
        return ENU2ECEFvUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either two or six inputs");
        return NULL;
    }
}

static PyObject*
NED2ECEFvUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radLatOrigin, *radLonOrigin, *mAltOrigin, *mNLocal, *mELocal, *mDLocal;

    // checks
    if (!PyArg_ParseTuple(args,
        "OOOOOO",
        &radLatOrigin,
        &radLonOrigin,
        &mAltOrigin,
        &mNLocal,
        &mELocal,
        &mDLocal))
        return NULL;
    if (((radLatOrigin = get_numpy_array(radLatOrigin)) == NULL) || ((radLonOrigin = get_numpy_array(radLonOrigin)) == NULL) || ((mAltOrigin = get_numpy_array(mAltOrigin)) == NULL) || ((mNLocal = get_numpy_array(mNLocal)) == NULL) || ((mELocal = get_numpy_array(mELocal)) == NULL) || ((mDLocal = get_numpy_array(mDLocal)) == NULL))
        return NULL;
    if (check_arrays_same_float_dtype(6, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin, mNLocal, mELocal, mDLocal}) == 0) {
        radLatOrigin = (PyArrayObject *)PyArray_CastToType(radLatOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radLonOrigin = (PyArrayObject *)PyArray_CastToType(radLonOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mAltOrigin = (PyArrayObject *)PyArray_CastToType(mAltOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mNLocal = (PyArrayObject *)PyArray_CastToType(mNLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mELocal = (PyArrayObject *)PyArray_CastToType(mELocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mDLocal = (PyArrayObject *)PyArray_CastToType(mDLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (check_arrays_same_size(3, (PyArrayObject *[]){radLatOrigin, radLonOrigin, mAltOrigin}) == 0)
        return NULL;
    if (check_arrays_same_size(3, (PyArrayObject *[]){mNLocal, mELocal, mDLocal}) == 0)
        return NULL;

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), PyArray_TYPE(mNLocal));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(mNLocal);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)mNLocal) == PyArray_Size((PyObject*)radLatOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
        case NPY_DOUBLE:
            NED2ECEFvDoubleUnrolled(
                (double*)PyArray_DATA(radLatOrigin), (double*)PyArray_DATA(radLonOrigin), (double*)PyArray_DATA(mAltOrigin), (double*)PyArray_DATA(mNLocal), (double*)PyArray_DATA(mELocal), (double*)PyArray_DATA(mDLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
            break;
        case NPY_FLOAT:
            NED2ECEFvFloatUnrolled(
                (float*)PyArray_DATA(radLatOrigin), (float*)PyArray_DATA(radLonOrigin), (float*)PyArray_DATA(mAltOrigin), (float*)PyArray_DATA(mNLocal), (float*)PyArray_DATA(mELocal), (float*)PyArray_DATA(mDLocal), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                "Only 32 and 64 bit float types or all integer are accepted.");
            return NULL;
        }
    
    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(mX);
        Py_DECREF(mY);
        Py_DECREF(mZ);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)mX);
    PyTuple_SetItem(tuple, 1, (PyObject*)mY);
    PyTuple_SetItem(tuple, 2, (PyObject*)mZ);
    return tuple;
}

static PyObject*
NED2ECEFvRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *rrmLLALocalOrigin, *mmmLocal;

    // checks
    if (!PyArg_ParseTuple(args,
            "OO",
            &rrmLLALocalOrigin,
            &mmmLocal))
        return NULL;
    rrmLLALocalOrigin = get_numpy_array(rrmLLALocalOrigin);
    mmmLocal = get_numpy_array(mmmLocal);
    PyArrayObject *arrays[] = {rrmLLALocalOrigin, mmmLocal};
    if (check_arrays_same_float_dtype(2, arrays) == 0) {
        mmmLocal = (PyArrayObject *)PyArray_CastToType(mmmLocal, PyArray_DescrFromType(NPY_FLOAT64), 0);
        rrmLLALocalOrigin = (PyArrayObject *)PyArray_CastToType(rrmLLALocalOrigin, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmLocal)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmLocal)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSIN3D) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmLocal))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSIN3D) != 0 || (PyArray_SIZE(mmmLocal) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mmmLocal), PyArray_SHAPE(mmmLocal), PyArray_TYPE(mmmLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)rrmLLALocalOrigin) == PyArray_Size((PyObject*)mmmLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        NED2ECEFvDoubleRolled(
            (double*)PyArray_DATA(rrmLLALocalOrigin), (double*)PyArray_DATA(mmmLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        NED2ECEFvFloatRolled(
            (float*)PyArray_DATA(rrmLLALocalOrigin), (float*)PyArray_DATA(mmmLocal), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
NED2ECEFvWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 2)
        return NED2ECEFvRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 6)
        return NED2ECEFvUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either two or six inputs");
        return NULL;
    }
}

static PyObject*
ENU2AERUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *mE, *mN, *mU;

    // checks
    if (!PyArg_ParseTuple(args, "OOO", &mE, &mN, &mU))
        return NULL;
    if (((mE = get_numpy_array(mE)) == NULL) || ((mN = get_numpy_array(mN)) == NULL) || ((mU = get_numpy_array(mU)) == NULL))
        return NULL;
    PyArrayObject *arrays[] = {mE, mN, mU};
    if (check_arrays_same_size(3, arrays) == 0)
        return NULL;
    if (check_arrays_same_float_dtype(3, arrays) == 0) {
        mE = (PyArrayObject *)PyArray_CastToType(mE, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mN = (PyArrayObject *)PyArray_CastToType(mN, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mU = (PyArrayObject *)PyArray_CastToType(mU, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }

    // prepare inputs
    PyArrayObject *radAz, *radEl, *mRange;
    radAz = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mE), PyArray_SHAPE(mE), PyArray_TYPE(mE));
    radEl = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mE), PyArray_SHAPE(mE), PyArray_TYPE(mE));
    mRange = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mE), PyArray_SHAPE(mE), PyArray_TYPE(mE));
    if ((radAz == NULL) || (radEl == NULL) || (mRange == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(mE);

    // run function
    switch (PyArray_TYPE(mE)) {
        case NPY_DOUBLE:
            ENU2AERDoubleUnrolled(
                (double*)PyArray_DATA(mE), (double*)PyArray_DATA(mN), (double*)PyArray_DATA(mU), nPoints, (double*)PyArray_DATA(radAz), (double*)PyArray_DATA(radEl), (double*)PyArray_DATA(mRange));
            break;
        case NPY_FLOAT:
            ENU2AERFloatUnrolled(
                (float*)PyArray_DATA(mE), (float*)PyArray_DATA(mN), (float*)PyArray_DATA(mU), nPoints, (float*)PyArray_DATA(radAz), (float*)PyArray_DATA(radEl), (float*)PyArray_DATA(mRange));
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                "Only 32 and 64 bit float types or all integer are accepted.");
            return NULL;
        }
    
    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(radAz);
        Py_DECREF(radEl);
        Py_DECREF(mRange);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)radAz);
    PyTuple_SetItem(tuple, 1, (PyObject*)radEl);
    PyTuple_SetItem(tuple, 2, (PyObject*)mRange);
    return tuple;
}

static PyObject*
ENU2AERRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* mmmENU;

    // checks
    if (!PyArg_ParseTuple(args, "O", &mmmENU))
        return NULL;
    mmmENU = get_numpy_array(mmmENU);
    if (mmmENU == NULL)
        return NULL;
    if (check_arrays_same_float_dtype(1, (PyArrayObject *[]){mmmENU}) == 0) {
        mmmENU = (PyArrayObject *)PyArray_CastToType(mmmENU, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if ((PyArray_SIZE(mmmENU) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mmmENU), PyArray_SHAPE(mmmENU), PyArray_TYPE(mmmENU));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmENU) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ENU2AERDoubleRolled((double*)PyArray_DATA(mmmENU), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ENU2AERFloatRolled((float*)PyArray_DATA(mmmENU), nPoints, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
ENU2AERWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 1)
        return ENU2AERRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 3)
        return ENU2AERUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either two or six inputs");
        return NULL;
    }
}

static PyObject*
NED2AERUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *mN, *mE, *mD;

    // checks
    if (!PyArg_ParseTuple(args, "OOO", &mN, &mE, &mD))
        return NULL;
    if (((mN = get_numpy_array(mN)) == NULL) || ((mE = get_numpy_array(mE)) == NULL) || ((mD = get_numpy_array(mD)) == NULL))
        return NULL;
    PyArrayObject *arrays[] = {mN, mE, mD};
    if (check_arrays_same_size(3, arrays) == 0)
        return NULL;
    if (check_arrays_same_float_dtype(3, arrays) == 0) {
        mN = (PyArrayObject *)PyArray_CastToType(mN, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mE = (PyArrayObject *)PyArray_CastToType(mE, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mD = (PyArrayObject *)PyArray_CastToType(mD, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }

    // prepare inputs
    PyArrayObject *radAz, *radEl, *mRange;
    radAz = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mN), PyArray_SHAPE(mN), PyArray_TYPE(mN));
    radEl = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mN), PyArray_SHAPE(mN), PyArray_TYPE(mN));
    mRange = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mN), PyArray_SHAPE(mN), PyArray_TYPE(mN));
    if ((radAz == NULL) || (radEl == NULL) || (mRange == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(mN);

    // run function
    switch (PyArray_TYPE(mN)) {
        case NPY_DOUBLE:
            NED2AERDoubleUnrolled(
                (double*)PyArray_DATA(mN), (double*)PyArray_DATA(mE), (double*)PyArray_DATA(mD), nPoints, (double*)PyArray_DATA(radAz), (double*)PyArray_DATA(radEl), (double*)PyArray_DATA(mRange));
            break;
        case NPY_FLOAT:
            NED2AERFloatUnrolled(
                (float*)PyArray_DATA(mN), (float*)PyArray_DATA(mE), (float*)PyArray_DATA(mD), nPoints, (float*)PyArray_DATA(radAz), (float*)PyArray_DATA(radEl), (float*)PyArray_DATA(mRange));
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                "Only 32 and 64 bit float types or all integer are accepted.");
            return NULL;
    }
    
    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(radAz);
        Py_DECREF(radEl);
        Py_DECREF(mRange);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)radAz);
    PyTuple_SetItem(tuple, 1, (PyObject*)radEl);
    PyTuple_SetItem(tuple, 2, (PyObject*)mRange);
    return tuple;
}

static PyObject*
NED2AERRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* mmmNED;

    // checks
    if (!PyArg_ParseTuple(args, "O", &mmmNED))
        return NULL;
    mmmNED = get_numpy_array(mmmNED);
    if (mmmNED == NULL)
        return NULL;
    if (check_arrays_same_float_dtype(1, (PyArrayObject *[]){mmmNED}) == 0) {
        mmmNED = (PyArrayObject *)PyArray_CastToType(mmmNED, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if ((PyArray_SIZE(mmmNED) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(mmmNED), PyArray_SHAPE(mmmNED), PyArray_TYPE(mmmNED));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(mmmNED) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        NED2AERDoubleRolled((double*)PyArray_DATA(mmmNED), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        NED2AERFloatRolled((float*)PyArray_DATA(mmmNED), nPoints, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
NED2AERWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 1)
        return NED2AERRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 3)
        return NED2AERUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either one or three inputs");
        return NULL;
    }
}

static PyObject*
AER2NEDRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* rrmAER;

    // checks
    if (!PyArg_ParseTuple(args, "O", &rrmAER))
        return NULL;
    rrmAER = get_numpy_array(rrmAER);
    if (rrmAER == NULL)
        return NULL;
    if ((PyArray_SIZE(rrmAER) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(rrmAER), PyArray_SHAPE(rrmAER), PyArray_TYPE(rrmAER));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(rrmAER) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        AER2NEDDoubleRolled((double*)PyArray_DATA(rrmAER), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        AER2NEDFloatRolled((float*)PyArray_DATA(rrmAER), nPoints, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
AER2NEDUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radAz, *radEl, *mRange;

    // checks
    if (!PyArg_ParseTuple(args, "OOO", &radAz, &radEl, &mRange))
        return NULL;
    if (((radAz = get_numpy_array(radAz)) == NULL) || ((radEl = get_numpy_array(radEl)) == NULL) || ((mRange = get_numpy_array(mRange)) == NULL)) {
        return NULL;
    }
    PyArrayObject *arrays[] = {radAz, radEl, mRange};
    if (check_arrays_same_size(3, arrays) == 0)
        return NULL;
    if (check_arrays_same_float_dtype(3, arrays) == 0) {
        radAz = (PyArrayObject *)PyArray_CastToType(radAz, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radEl = (PyArrayObject *)PyArray_CastToType(radEl, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mRange = (PyArrayObject *)PyArray_CastToType(mRange, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }

    // prepare inputs
    PyArrayObject *mN, *mE, *mD;
    mN = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radAz), PyArray_SHAPE(radAz), PyArray_TYPE(radAz));
    mE = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radAz), PyArray_SHAPE(radAz), PyArray_TYPE(radAz));
    mD = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radAz), PyArray_SHAPE(radAz), PyArray_TYPE(radAz));
    if ((mN == NULL) || (mE == NULL) || (mD == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(radAz);

    // run function
    switch (PyArray_TYPE(mN)) {
    case NPY_DOUBLE:
        AER2NEDDoubleUnrolled(
            (double*)PyArray_DATA(radAz), (double*)PyArray_DATA(radEl), (double*)PyArray_DATA(mRange), nPoints, (double*)PyArray_DATA(mN), (double*)PyArray_DATA(mE), (double*)PyArray_DATA(mD));
        break;
    case NPY_FLOAT:
        AER2NEDFloatUnrolled(
            (float*)PyArray_DATA(radAz), (float*)PyArray_DATA(radEl), (float*)PyArray_DATA(mRange), nPoints, (float*)PyArray_DATA(mN), (float*)PyArray_DATA(mE), (float*)PyArray_DATA(mD));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    
    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(mN);
        Py_DECREF(mE);
        Py_DECREF(mD);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)mN);
    PyTuple_SetItem(tuple, 1, (PyObject*)mE);
    PyTuple_SetItem(tuple, 2, (PyObject*)mD);
    return tuple;
}

static PyObject*
AER2NEDWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 1)
        return AER2NEDRolledWrapper(self, args);
    else if (PyTuple_Size(args) == 3)
        return AER2NEDUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either one or three inputs");
        return NULL;
    }
}

static PyObject*
AER2ENURolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* rrmAER;

    // checks
    if (!PyArg_ParseTuple(args, "O", &rrmAER))
        return NULL;
    rrmAER = get_numpy_array(rrmAER);
    if (rrmAER == NULL)
        return NULL;
    if (check_arrays_same_float_dtype(1, (PyArrayObject *[]){rrmAER}) == 0) {
        rrmAER = (PyArrayObject *)PyArray_CastToType(rrmAER, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }
    if ((PyArray_SIZE(rrmAER) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(rrmAER), PyArray_SHAPE(rrmAER), PyArray_TYPE(rrmAER));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(rrmAER) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        AER2ENUDoubleRolled((double*)PyArray_DATA(rrmAER), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        AER2ENUFloatRolled((float*)PyArray_DATA(rrmAER), nPoints, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
AER2ENUUnrolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject *radAz, *radEl, *mRange;

    // checks
    if (!PyArg_ParseTuple(args, "OOO", &radAz, &radEl, &mRange))
        return NULL;
    if (((radAz = get_numpy_array(radAz)) == NULL) || ((radEl = get_numpy_array(radEl)) == NULL) || ((mRange = get_numpy_array(mRange)) == NULL)) {
        return NULL;
    }
    PyArrayObject *arrays[] = {radAz, radEl, mRange};
    if (check_arrays_same_size(3, arrays) == 0)
        return NULL;
    if (check_arrays_same_float_dtype(3, arrays) == 0) {
        radAz = (PyArrayObject *)PyArray_CastToType(radAz, PyArray_DescrFromType(NPY_FLOAT64), 0);
        radEl = (PyArrayObject *)PyArray_CastToType(radEl, PyArray_DescrFromType(NPY_FLOAT64), 0);
        mRange = (PyArrayObject *)PyArray_CastToType(mRange, PyArray_DescrFromType(NPY_FLOAT64), 0);
    }

    // prepare inputs
    PyArrayObject *mE, *mN, *mU;
    mE = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radAz), PyArray_SHAPE(radAz), PyArray_TYPE(radAz));
    mN = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radAz), PyArray_SHAPE(radAz), PyArray_TYPE(radAz));
    mU = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(radAz), PyArray_SHAPE(radAz), PyArray_TYPE(radAz));
    if ((mE == NULL) || (mN == NULL) || (mU == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(radAz);

    // run function
    switch (PyArray_TYPE(mE)) {
    case NPY_DOUBLE:
        AER2ENUDoubleUnrolled(
            (double*)PyArray_DATA(radAz), (double*)PyArray_DATA(radEl), (double*)PyArray_DATA(mRange), nPoints, (double*)PyArray_DATA(mE), (double*)PyArray_DATA(mN), (double*)PyArray_DATA(mU));
        break;
    case NPY_FLOAT:
        AER2ENUFloatUnrolled(
            (float*)PyArray_DATA(radAz), (float*)PyArray_DATA(radEl), (float*)PyArray_DATA(mRange), nPoints, (float*)PyArray_DATA(mE), (float*)PyArray_DATA(mN), (float*)PyArray_DATA(mU));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    
    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(mE);
        Py_DECREF(mN);
        Py_DECREF(mU);
        return NULL;
    }
    PyTuple_SetItem(tuple, 0, (PyObject*)mE);
    PyTuple_SetItem(tuple, 1, (PyObject*)mN);
    PyTuple_SetItem(tuple, 2, (PyObject*)mU);
    return tuple;
}

static PyObject*
AER2ENUWrapper(PyObject* self, PyObject* args)
{
    if (PyTuple_Size(args) == 1)
        return AER2ENURolledWrapper(self, args);
    else if (PyTuple_Size(args) == 3)
        return AER2ENUUnrolledWrapper(self, args);
    else {
        PyErr_SetString(PyExc_TypeError, "Function accepts either one or three inputs");
        return NULL;
    }
}

// Method definition object for this extension, these argumens mean:
// ml_name: The name of the method
// ml_meth: Function pointer to the method implementation
// ml_flags: Flags indicating special features of this method, such as
//           accepting arguments, accepting keyword arguments, being a class
//           method, or being a static method of a class.
// ml_doc:  The docstring for the method
static PyMethodDef MyMethods[] = {
    { "geodetic2ECEF",
        geodetic2ECEFWrapper,
        METH_VARARGS,
        "Convert geodetic coordinate system to ECEF." },
    { "ECEF2geodetic",
        ECEF2geodeticWrapper,
        METH_VARARGS,
        "Convert ECEF to geodetic coordinate system." },
    { "ECEF2ENU", ECEF2ENUWrapper, METH_VARARGS, "Convert ECEF to ENU." },
    { "ENU2ECEF", ENU2ECEFWrapper, METH_VARARGS, "Convert ENU to ECEF." },
    { "ENU2AER", ENU2AERWrapper, METH_VARARGS, "Convert ENU to AER." },
    { "AER2ENU", AER2ENUWrapper, METH_VARARGS, "Convert AER to ENU." },
    { "ECEF2ENUv", ECEF2ENUvWrapper, METH_VARARGS, "Convert ECEF to ENU velocity." },
    { "ENU2ECEFv", ENU2ECEFvWrapper, METH_VARARGS, "Convert ENU to ECEF velocity." },
    { "ECEF2NED", ECEF2NEDWrapper, METH_VARARGS, "Convert ECEF to NED." },
    { "NED2ECEF", NED2ECEFWrapper, METH_VARARGS, "Convert NED to ECEF." },
    { "NED2ECEFv", NED2ECEFvWrapper, METH_VARARGS, "Convert NED to ECEF velocity." },
    { "ECEF2NEDv", ECEF2NEDvWrapper, METH_VARARGS, "Convert ECEF to NED velocity." },
    { "NED2AER", NED2AERWrapper, METH_VARARGS, "Convert NED to AER velocity." },
    { "AER2NED", AER2NEDWrapper, METH_VARARGS, "Convert AER to NED velocity." },
    { "geodetic2UTM",
        geodetic2UTMWrapper,
        METH_VARARGS,
        "Convert geodetic coordinate system to UTM." },
    { "UTM2geodetic",
        UTM2geodeticWrapper,
        METH_VARARGS,
        "Convert UTM to geodetic." },
    { NULL, NULL, 0, NULL }
};

// Module definition
static struct PyModuleDef transforms = {
    PyModuleDef_HEAD_INIT,
    "transforms",
    "Module that contains transform functions.",
    -1,
    MyMethods
};

// Module initialization function
PyMODINIT_FUNC
PyInit_transforms(void)
{
    import_array(); // Initialize the NumPy C API
    return PyModule_Create(&transforms);
}
