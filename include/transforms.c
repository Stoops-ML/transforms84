#include <Python.h>
#include <float.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <omp.h>

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
    if (!PyArg_ParseTuple(args, "O!dd", &PyArray_Type, &rrmLLA, &a, &b))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmLLA))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLA) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    PyArrayObject* inArray;
    if (PyArray_ISINTEGER(rrmLLA) == 0)
        inArray = rrmLLA;
    else {
        inArray = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLA), PyArray_SHAPE(rrmLLA), NPY_DOUBLE);
        if (inArray == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArray, rrmLLA) < 0) {
            Py_DECREF(inArray);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArray))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    long nPoints = (int)PyArray_SIZE(inArray) / NCOORDSIN3D;
    PyArrayObject* result_array;
    if ((nPoints == 1) && (PyArray_NDIM(inArray) == 2)) {
        npy_intp dims[2] = { 2, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(inArray), dims, PyArray_TYPE(inArray));
    } else if ((nPoints == 1) && (PyArray_NDIM(inArray) == 3)) {
        npy_intp dims[3] = { 1, 2, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(inArray), dims, PyArray_TYPE(inArray));
    } else if (nPoints > 1) {
        npy_intp dims[3] = { nPoints, 2, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(inArray), dims, PyArray_TYPE(inArray));
    } else {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output array.");
        return NULL;
    }
    if (result_array == NULL)
        return NULL;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        geodetic2UTMDoubleRolled((double*)PyArray_DATA(inArray), nPoints, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        geodetic2UTMFloatRolled((float*)PyArray_DATA(inArray), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
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
    if (!PyArg_ParseTuple(args, "O!O!O!dd", &PyArray_Type, &radLat, &PyArray_Type, &radLon, &PyArray_Type, &mAlt, &a, &b))
        return NULL;
    if (!((PyArray_ISCONTIGUOUS(radLat)) && (PyArray_ISCONTIGUOUS(radLon)) && (PyArray_ISCONTIGUOUS(mAlt)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLat) == PyArray_SIZE(radLon)) && (PyArray_SIZE(radLat) == PyArray_SIZE(mAlt)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be the same size.");
        return NULL;
    }

    PyArrayObject *inArrayLat, *inArrayLon, *inArrayAlt;
    if (PyArray_ISINTEGER(radLat) == 0)
        inArrayLat = radLat;
    else {
        inArrayLat = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLat), PyArray_SHAPE(radLat), NPY_DOUBLE);
        if (inArrayLat == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLat, radLat) < 0) {
            Py_DECREF(inArrayLat);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLat))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLon) == 0)
        inArrayLon = radLon;
    else {
        inArrayLon = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLon), PyArray_SHAPE(radLon), NPY_DOUBLE);
        if (inArrayLon == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLon, radLon) < 0) {
            Py_DECREF(inArrayLon);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLon))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAlt) == 0)
        inArrayAlt = mAlt;
    else {
        inArrayAlt = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAlt), PyArray_SHAPE(mAlt), NPY_DOUBLE);
        if (inArrayAlt == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAlt, mAlt) < 0) {
            Py_DECREF(inArrayAlt);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAlt))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    PyArrayObject *outX, *outY;
    outX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayLat), PyArray_SHAPE(inArrayLat), PyArray_TYPE(inArrayLat));
    outY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayLat), PyArray_SHAPE(inArrayLat), PyArray_TYPE(inArrayLat));
    if ((outX == NULL) || (outY == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayLat);

    // run function
    switch (PyArray_TYPE(inArrayLat)) {
    case NPY_DOUBLE:
        geodetic2UTMDoubleUnrolled((double*)PyArray_DATA(inArrayLat), (double*)PyArray_DATA(inArrayLon), (double*)PyArray_DATA(inArrayAlt), nPoints, a, b, (double*)PyArray_DATA(outX), (double*)PyArray_DATA(outY));
        break;
    case NPY_FLOAT:
        geodetic2UTMFloatUnrolled((float*)PyArray_DATA(inArrayLat), (float*)PyArray_DATA(inArrayLon), (float*)PyArray_DATA(inArrayAlt), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(outX), (float*)PyArray_DATA(outY));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(2);
    if (!tuple){
        Py_DECREF(inArrayLat);
        Py_DECREF(inArrayLon);
        Py_DECREF(inArrayAlt);
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
    if (!PyArg_ParseTuple(args, "O!Osdd", &PyArray_Type, &mmUTM, &ZoneNumberPy, &ZoneLetter, &a, &b))
        return NULL;
    if (!PyLong_Check(ZoneNumberPy)) {
        PyErr_SetString(PyExc_TypeError, "Zone number must be an integer");
        return NULL;
    }
    long ZoneNumber = PyLong_AsLong(ZoneNumberPy);
    if (PyErr_Occurred()) {
        return NULL; // Conversion failed
    }
    if (!(PyArray_ISCONTIGUOUS(mmUTM))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(mmUTM) % NCOORDSIN2D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 2.");
        return NULL;
    }

    PyArrayObject* inArray;
    if (PyArray_ISINTEGER(mmUTM) == 0)
        inArray = mmUTM;
    else {
        inArray = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmUTM), PyArray_SHAPE(mmUTM), NPY_DOUBLE);
        if (inArray == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArray, mmUTM) < 0) {
            Py_DECREF(inArray);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArray))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    long nPoints = (int)PyArray_SIZE(inArray) / NCOORDSIN2D;
    PyArrayObject* result_array;
    if ((nPoints == 1) && (PyArray_NDIM(inArray) == 2)) {
        npy_intp dims[2] = { 3, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(inArray), dims, PyArray_TYPE(inArray));
    } else if ((nPoints == 1) && (PyArray_NDIM(inArray) == 3)) {
        npy_intp dims[3] = { 1, 3, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(inArray), dims, PyArray_TYPE(inArray));
    } else if (nPoints > 1) {
        npy_intp dims[3] = { nPoints, 3, 1 };
        result_array = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(inArray), dims, PyArray_TYPE(inArray));
    } else {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output array.");
        return NULL;
    }
    if (result_array == NULL)
        return NULL;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        UTM2geodeticDoubleRolled((double*)PyArray_DATA(inArray), ZoneNumber, ZoneLetter, nPoints, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        UTM2geodeticFloatRolled((float*)PyArray_DATA(inArray), ZoneNumber, ZoneLetter, nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
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
    if ((mX = get_numpy_array(mX)) == NULL)  {
        PyErr_SetString(PyExc_ValueError, "mX must either be a numpy ndarray or a pandas Series.");
        Py_XDECREF(mX);
        Py_XDECREF(mY);
        return NULL;
    }
    if ((mY = get_numpy_array(mY)) == NULL)  {
        PyErr_SetString(PyExc_ValueError, "mY must either be a numpy ndarray or a pandas Series.");
        Py_XDECREF(mX);
        Py_XDECREF(mY);
        return NULL;
    }
    if (!PyLong_Check(ZoneNumberPy)) {
        PyErr_SetString(PyExc_TypeError, "Zone number must be an integer");
        return NULL;
    }
    long ZoneNumber = PyLong_AsLong(ZoneNumberPy);
    if (PyErr_Occurred()) {
        return NULL; // Conversion failed
    }
    if (!((PyArray_ISCONTIGUOUS(mX)) && (PyArray_ISCONTIGUOUS(mY)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }

    PyArrayObject *inArrayX, *inArrayY;
    if (PyArray_ISINTEGER(mX) == 0)
        inArrayX = mX;
    else {
        inArrayX = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mX), PyArray_SHAPE(mX), NPY_DOUBLE);
        if (inArrayX == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayX, mX) < 0) {
            Py_DECREF(inArrayX);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayX))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mY) == 0)
        inArrayY = mY;
    else {
        inArrayY = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mY), PyArray_SHAPE(mY), NPY_DOUBLE);
        if (inArrayY == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayY, mY) < 0) {
            Py_DECREF(inArrayY);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayY))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    PyArrayObject *radLat, *radLon, *mAlt;
    radLat = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayX), PyArray_SHAPE(inArrayX), PyArray_TYPE(inArrayX));
    radLon = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayX), PyArray_SHAPE(inArrayX), PyArray_TYPE(inArrayX));
    mAlt = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayX), PyArray_SHAPE(inArrayX), PyArray_TYPE(inArrayX));
    if ((radLat == NULL) || (radLon == NULL) || (mAlt == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }

    // run function
    long nPoints = (int)PyArray_SIZE(inArrayX);
    switch (PyArray_TYPE(radLat)) {
    case NPY_DOUBLE:
        UTM2geodeticDoubleUnrolled((double*)PyArray_DATA(inArrayX), (double*)PyArray_DATA(inArrayY), ZoneNumber, ZoneLetter, nPoints, a, b, (double*)PyArray_DATA(radLat), (double*)PyArray_DATA(radLon), (double*)PyArray_DATA(mAlt));
        break;
    case NPY_FLOAT:
        UTM2geodeticFloatUnrolled((float*)PyArray_DATA(inArrayX), (float*)PyArray_DATA(inArrayY), ZoneNumber, ZoneLetter, nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(radLat), (float*)PyArray_DATA(radLon), (float*)PyArray_DATA(mAlt));
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
        Py_DECREF(inArrayX);
        Py_DECREF(inArrayY);
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
    if (!PyArg_ParseTuple(args, "O!dd", &PyArray_Type, &rrmLLA, &a, &b))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmLLA))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLA) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    PyArrayObject* inArray;
    if (PyArray_ISINTEGER(rrmLLA) == 0)
        inArray = rrmLLA;
    else {
        inArray = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLA), PyArray_SHAPE(rrmLLA), NPY_DOUBLE);
        if (inArray == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArray, rrmLLA) < 0) {
            Py_DECREF(inArray);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArray))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArray), PyArray_SHAPE(inArray), PyArray_TYPE(inArray));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArray) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        geodetic2ECEFDoubleRolled((double*)PyArray_DATA(inArray), nPoints, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        geodetic2ECEFFloatRolled((float*)PyArray_DATA(inArray), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
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
    if (!PyArg_ParseTuple(args, "O!O!O!dd", &PyArray_Type, &radLat, &PyArray_Type, &radLon, &PyArray_Type, &mAlt, &a, &b))
        return NULL;
    if (!((PyArray_ISCONTIGUOUS(radLat)) && (PyArray_ISCONTIGUOUS(radLon)) && (PyArray_ISCONTIGUOUS(mAlt)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLat) == PyArray_SIZE(radLon)) && (PyArray_SIZE(radLat) == PyArray_SIZE(mAlt)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be the same size.");
        return NULL;
    }

    // ensure matching floating point types
    PyArrayObject *inArrayLat, *inArrayLon, *inArrayAlt;
    if (PyArray_ISINTEGER(radLat) == 0)
        inArrayLat = radLat;
    else {
        inArrayLat = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLat), PyArray_SHAPE(radLat), NPY_DOUBLE);
        if (inArrayLat == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLat, radLat) < 0) {
            Py_DECREF(inArrayLat);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLat))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLon) == 0)
        inArrayLon = radLon;
    else {
        inArrayLon = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLon), PyArray_SHAPE(radLon), NPY_DOUBLE);
        if (inArrayLon == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLon, radLon) < 0) {
            Py_DECREF(inArrayLon);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLon))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAlt) == 0)
        inArrayAlt = mAlt;
    else {
        inArrayAlt = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAlt), PyArray_SHAPE(mAlt), NPY_DOUBLE);
        if (inArrayAlt == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAlt, mAlt) < 0) {
            Py_DECREF(inArrayAlt);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAlt))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *outX, *outY, *outZ;
    outX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayLat), PyArray_SHAPE(inArrayLat), PyArray_TYPE(inArrayLat));
    outY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayLat), PyArray_SHAPE(inArrayLat), PyArray_TYPE(inArrayLat));
    outZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayLat), PyArray_SHAPE(inArrayLat), PyArray_TYPE(inArrayLat));
    if ((outX == NULL) || (outY == NULL) || (outZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayLat);

    // run function
    switch (PyArray_TYPE(outX)) {
    case NPY_DOUBLE:
        geodetic2ECEFDoubleUnrolled((double*)PyArray_DATA(inArrayLat), (double*)PyArray_DATA(inArrayLon), (double*)PyArray_DATA(inArrayAlt), nPoints, a, b, (double*)PyArray_DATA(outX), (double*)PyArray_DATA(outY), (double*)PyArray_DATA(outZ));
        break;
    case NPY_FLOAT:
        geodetic2ECEFFloatUnrolled((float*)PyArray_DATA(inArrayLat), (float*)PyArray_DATA(inArrayLon), (float*)PyArray_DATA(inArrayAlt), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(outX), (float*)PyArray_DATA(outY), (float*)PyArray_DATA(outZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(inArrayLat);
        Py_DECREF(inArrayLon);
        Py_DECREF(inArrayAlt);
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
    if (!PyArg_ParseTuple(args, "O!O!O!dd", &PyArray_Type, &mX, &PyArray_Type, &mY, &PyArray_Type, &mZ, &a, &b))
        return NULL;
    if (!((PyArray_ISCONTIGUOUS(mX)) && (PyArray_ISCONTIGUOUS(mY)) && (PyArray_ISCONTIGUOUS(mZ)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_SIZE(mX) == PyArray_SIZE(mY)) && (PyArray_SIZE(mX) == PyArray_SIZE(mZ)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be the same size.");
        return NULL;
    }

    PyArrayObject *inArrayX, *inArrayY, *inArrayZ;
    if (PyArray_ISINTEGER(mX) == 0)
        inArrayX = mX;
    else {
        inArrayX = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mX), PyArray_SHAPE(mX), NPY_DOUBLE);
        if (inArrayX == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayX, mX) < 0) {
            Py_DECREF(inArrayX);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayX))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mY) == 0)
        inArrayY = mY;
    else {
        inArrayY = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mY), PyArray_SHAPE(mY), NPY_DOUBLE);
        if (inArrayY == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayY, mY) < 0) {
            Py_DECREF(inArrayY);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayY))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mZ) == 0)
        inArrayZ = mZ;
    else {
        inArrayZ = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mZ), PyArray_SHAPE(mZ), NPY_DOUBLE);
        if (inArrayZ == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayZ, mZ) < 0) {
            Py_DECREF(inArrayZ);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayZ))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

     // prepare inputs
     PyArrayObject *radLat, *radLon, *mAlt;
     radLat = (PyArrayObject*)PyArray_SimpleNew(
         PyArray_NDIM(inArrayX), PyArray_SHAPE(inArrayX), PyArray_TYPE(inArrayX));
     radLon = (PyArrayObject*)PyArray_SimpleNew(
         PyArray_NDIM(inArrayX), PyArray_SHAPE(inArrayX), PyArray_TYPE(inArrayX));
     mAlt = (PyArrayObject*)PyArray_SimpleNew(
         PyArray_NDIM(inArrayX), PyArray_SHAPE(inArrayX), PyArray_TYPE(inArrayX));
     if ((radLat == NULL) || (radLon == NULL) || (mAlt == NULL)) {
         PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
         return NULL;
     }
     long nPoints = (int)PyArray_SIZE(inArrayX);
 
    // run function
    switch (PyArray_TYPE(radLat)) {
    case NPY_DOUBLE:
        ECEF2geodeticDoubleUnrolled((double*)PyArray_DATA(inArrayX), (double*)PyArray_DATA(inArrayY), (double*)PyArray_DATA(inArrayZ), nPoints, a, b, (double*)PyArray_DATA(radLat), (double*)PyArray_DATA(radLon), (double*)PyArray_DATA(mAlt));
        break;
    case NPY_FLOAT:
        ECEF2geodeticFloatUnrolled((float*)PyArray_DATA(inArrayX), (float*)PyArray_DATA(inArrayY), (float*)PyArray_DATA(inArrayZ), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(radLat), (float*)PyArray_DATA(radLon), (float*)PyArray_DATA(mAlt));
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
        Py_DECREF(inArrayX);
        Py_DECREF(inArrayY);
        Py_DECREF(inArrayZ);
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
    if (!PyArg_ParseTuple(args, "O!dd", &PyArray_Type, &mmmXYZ, &a, &b))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(mmmXYZ))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(mmmXYZ) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    PyArrayObject* inArray;
    if (PyArray_ISINTEGER(mmmXYZ) == 0)
        inArray = mmmXYZ;
    else {
        inArray = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmXYZ), PyArray_SHAPE(mmmXYZ), NPY_DOUBLE);
        if (inArray == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArray, mmmXYZ) < 0) {
            Py_DECREF(inArray);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArray))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArray), PyArray_SHAPE(inArray), PyArray_TYPE(inArray));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArray) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2geodeticDoubleRolled((double*)PyArray_DATA(inArray), nPoints, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2geodeticFloatRolled((float*)PyArray_DATA(inArray), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
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
            "O!O!O!O!O!O!dd",
            &PyArray_Type,
            &radLatOrigin,
            &PyArray_Type,
            &radLonOrigin,
            &PyArray_Type,
            &mAltOrigin,
            &PyArray_Type,
            &mXTarget,
            &PyArray_Type,
            &mYTarget,
            &PyArray_Type,
            &mZTarget,
            &a,
            &b))
        return NULL;
    if (!((PyArray_ISCONTIGUOUS(radLatOrigin)) && (PyArray_ISCONTIGUOUS(radLonOrigin)) && (PyArray_ISCONTIGUOUS(mAltOrigin)) && (PyArray_ISCONTIGUOUS(mXTarget)) && (PyArray_ISCONTIGUOUS(mYTarget)) && (PyArray_ISCONTIGUOUS(mZTarget)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(radLatOrigin) == PyArray_TYPE(radLonOrigin)) && (PyArray_TYPE(radLonOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mAltOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mXTarget) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mYTarget) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mZTarget) == PyArray_TYPE(radLatOrigin)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(mXTarget) == PyArray_SIZE(mYTarget)) && (PyArray_SIZE(mXTarget) == PyArray_SIZE(mZTarget)))) {
        PyErr_SetString(PyExc_ValueError, "Input target arrays must be the same size.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLatOrigin) == PyArray_SIZE(radLonOrigin)) && (PyArray_SIZE(radLatOrigin) == PyArray_SIZE(mAltOrigin)))) {
        PyErr_SetString(PyExc_ValueError, "Input origin arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayXTarget, *inArrayYTarget, *inArrayZTarget, *inArrayLatOrigin, *inArrayLonOrigin, *inArrayAltOrigin;
    if (PyArray_ISINTEGER(mXTarget) == 0)
        inArrayXTarget = mXTarget;
    else {
        inArrayXTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), NPY_DOUBLE);
        if (inArrayXTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayXTarget, mXTarget) < 0) {
            Py_DECREF(inArrayXTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayXTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mYTarget) == 0)
        inArrayYTarget = mYTarget;
    else {
        inArrayYTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mYTarget), PyArray_SHAPE(mYTarget), NPY_DOUBLE);
        if (inArrayYTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayYTarget, mYTarget) < 0) {
            Py_DECREF(inArrayYTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayYTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mZTarget) == 0)
        inArrayZTarget = mZTarget;
    else {
        inArrayZTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mZTarget), PyArray_SHAPE(mZTarget), NPY_DOUBLE);
        if (inArrayZTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayZTarget, mZTarget) < 0) {
            Py_DECREF(inArrayZTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayZTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLatOrigin) == 0)
        inArrayLatOrigin = radLatOrigin;
    else {
        inArrayLatOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLatOrigin), PyArray_SHAPE(radLatOrigin), NPY_DOUBLE);
        if (inArrayLatOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLatOrigin, radLatOrigin) < 0) {
            Py_DECREF(inArrayLatOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLatOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLonOrigin) == 0)
        inArrayLonOrigin = radLonOrigin;
    else {
        inArrayLonOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLonOrigin), PyArray_SHAPE(radLonOrigin), NPY_DOUBLE);
        if (inArrayLonOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLonOrigin, radLonOrigin) < 0) {
            Py_DECREF(inArrayLonOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLonOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAltOrigin) == 0)
        inArrayAltOrigin = mAltOrigin;
    else {
        inArrayAltOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAltOrigin), PyArray_SHAPE(mAltOrigin), NPY_DOUBLE);
        if (inArrayAltOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAltOrigin, mAltOrigin) < 0) {
            Py_DECREF(inArrayAltOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAltOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayXTarget) == PyArray_Size((PyObject*)inArrayAltOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        ECEF2ENUDoubleUnrolled(
            (double*)PyArray_DATA(inArrayLatOrigin), (double*)PyArray_DATA(inArrayLonOrigin), (double*)PyArray_DATA(inArrayAltOrigin), (double*)PyArray_DATA(inArrayXTarget), (double*)PyArray_DATA(inArrayYTarget), (double*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        ECEF2ENUFloatUnrolled(
            (float*)PyArray_DATA(inArrayLatOrigin), (float*)PyArray_DATA(inArrayLonOrigin), (float*)PyArray_DATA(inArrayAltOrigin), (float*)PyArray_DATA(inArrayXTarget), (float*)PyArray_DATA(inArrayYTarget), (float*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(inArrayLatOrigin);
        Py_DECREF(inArrayLonOrigin);
        Py_DECREF(inArrayAltOrigin);
        Py_DECREF(inArrayXTarget);
        Py_DECREF(inArrayYTarget);
        Py_DECREF(inArrayZTarget);
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
            "O!O!dd",
            &PyArray_Type,
            &rrmLLALocalOrigin,
            &PyArray_Type,
            &mmmXYZTarget,
            &a,
            &b))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmLLALocalOrigin)) || !(PyArray_ISCONTIGUOUS(mmmXYZTarget))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
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

    // ensure matching floating point types
    PyArrayObject *inArrayLocal, *inArrayOrigin;
    if (((PyArray_TYPE(rrmLLALocalOrigin) == NPY_FLOAT) && (PyArray_TYPE(mmmXYZTarget) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmLLALocalOrigin) == 0)) {
        inArrayOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLALocalOrigin), PyArray_SHAPE(rrmLLALocalOrigin), NPY_DOUBLE);
        if (inArrayOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayOrigin, rrmLLALocalOrigin) < 0) {
            Py_DECREF(inArrayOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayOrigin = rrmLLALocalOrigin;
    if (((PyArray_TYPE(mmmXYZTarget) == NPY_FLOAT) && (PyArray_TYPE(rrmLLALocalOrigin) == NPY_DOUBLE)) || (PyArray_ISFLOAT(mmmXYZTarget) == 0)) {
        inArrayLocal = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmXYZTarget), PyArray_SHAPE(mmmXYZTarget), NPY_DOUBLE);
        if (inArrayLocal == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLocal, mmmXYZTarget) < 0) {
            Py_DECREF(inArrayLocal);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLocal))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayLocal = mmmXYZTarget;

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(inArrayLocal),
        PyArray_SHAPE(inArrayLocal),
        PyArray_TYPE(inArrayLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2ENUDoubleRolled(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2ENUFloatRolled(
            (float*)PyArray_DATA(inArrayOrigin), (float*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
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
            "O!O!O!O!O!O!dd",
            &PyArray_Type,
            &radLatOrigin,
            &PyArray_Type,
            &radLonOrigin,
            &PyArray_Type,
            &mAltOrigin,
            &PyArray_Type,
            &mXTarget,
            &PyArray_Type,
            &mYTarget,
            &PyArray_Type,
            &mZTarget,
            &a,
            &b))
        return NULL;
    if (!((PyArray_ISCONTIGUOUS(radLatOrigin)) && (PyArray_ISCONTIGUOUS(radLonOrigin)) && (PyArray_ISCONTIGUOUS(mAltOrigin)) && (PyArray_ISCONTIGUOUS(mXTarget)) && (PyArray_ISCONTIGUOUS(mYTarget)) && (PyArray_ISCONTIGUOUS(mZTarget)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(radLatOrigin) == PyArray_TYPE(radLonOrigin)) && (PyArray_TYPE(radLonOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mAltOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mXTarget) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mYTarget) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mZTarget) == PyArray_TYPE(radLatOrigin)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(mXTarget) == PyArray_SIZE(mYTarget)) && (PyArray_SIZE(mXTarget) == PyArray_SIZE(mZTarget)))) {
        PyErr_SetString(PyExc_ValueError, "Input target arrays must be the same size.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLatOrigin) == PyArray_SIZE(radLonOrigin)) && (PyArray_SIZE(radLatOrigin) == PyArray_SIZE(mAltOrigin)))) {
        PyErr_SetString(PyExc_ValueError, "Input origin arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayXTarget, *inArrayYTarget, *inArrayZTarget, *inArrayLatOrigin, *inArrayLonOrigin, *inArrayAltOrigin;
    if (PyArray_ISINTEGER(mXTarget) == 0)
        inArrayXTarget = mXTarget;
    else {
        inArrayXTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), NPY_DOUBLE);
        if (inArrayXTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayXTarget, mXTarget) < 0) {
            Py_DECREF(inArrayXTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayXTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mYTarget) == 0)
        inArrayYTarget = mYTarget;
    else {
        inArrayYTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mYTarget), PyArray_SHAPE(mYTarget), NPY_DOUBLE);
        if (inArrayYTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayYTarget, mYTarget) < 0) {
            Py_DECREF(inArrayYTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayYTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mZTarget) == 0)
        inArrayZTarget = mZTarget;
    else {
        inArrayZTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mZTarget), PyArray_SHAPE(mZTarget), NPY_DOUBLE);
        if (inArrayZTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayZTarget, mZTarget) < 0) {
            Py_DECREF(inArrayZTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayZTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLatOrigin) == 0)
        inArrayLatOrigin = radLatOrigin;
    else {
        inArrayLatOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLatOrigin), PyArray_SHAPE(radLatOrigin), NPY_DOUBLE);
        if (inArrayLatOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLatOrigin, radLatOrigin) < 0) {
            Py_DECREF(inArrayLatOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLatOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLonOrigin) == 0)
        inArrayLonOrigin = radLonOrigin;
    else {
        inArrayLonOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLonOrigin), PyArray_SHAPE(radLonOrigin), NPY_DOUBLE);
        if (inArrayLonOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLonOrigin, radLonOrigin) < 0) {
            Py_DECREF(inArrayLonOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLonOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAltOrigin) == 0)
        inArrayAltOrigin = mAltOrigin;
    else {
        inArrayAltOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAltOrigin), PyArray_SHAPE(mAltOrigin), NPY_DOUBLE);
        if (inArrayAltOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAltOrigin, mAltOrigin) < 0) {
            Py_DECREF(inArrayAltOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAltOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayXTarget) == PyArray_Size((PyObject*)inArrayAltOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        ECEF2NEDDoubleUnrolled(
            (double*)PyArray_DATA(inArrayLatOrigin), (double*)PyArray_DATA(inArrayLonOrigin), (double*)PyArray_DATA(inArrayAltOrigin), (double*)PyArray_DATA(inArrayXTarget), (double*)PyArray_DATA(inArrayYTarget), (double*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        ECEF2NEDFloatUnrolled(
            (float*)PyArray_DATA(inArrayLatOrigin), (float*)PyArray_DATA(inArrayLonOrigin), (float*)PyArray_DATA(inArrayAltOrigin), (float*)PyArray_DATA(inArrayXTarget), (float*)PyArray_DATA(inArrayYTarget), (float*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(inArrayLatOrigin);
        Py_DECREF(inArrayLonOrigin);
        Py_DECREF(inArrayAltOrigin);
        Py_DECREF(inArrayXTarget);
        Py_DECREF(inArrayYTarget);
        Py_DECREF(inArrayZTarget);
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
            "O!O!dd",
            &PyArray_Type,
            &rrmLLALocalOrigin,
            &PyArray_Type,
            &mmmXYZTarget,
            &a,
            &b))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmLLALocalOrigin)) || !(PyArray_ISCONTIGUOUS(mmmXYZTarget))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
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

    // ensure matching floating point types
    PyArrayObject *inArrayLocal, *inArrayOrigin;
    if (((PyArray_TYPE(rrmLLALocalOrigin) == NPY_FLOAT) && (PyArray_TYPE(mmmXYZTarget) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmLLALocalOrigin) == 0)) {
        inArrayOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLALocalOrigin), PyArray_SHAPE(rrmLLALocalOrigin), NPY_DOUBLE);
        if (inArrayOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayOrigin, rrmLLALocalOrigin) < 0) {
            Py_DECREF(inArrayOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayOrigin = rrmLLALocalOrigin;
    if (((PyArray_TYPE(mmmXYZTarget) == NPY_FLOAT) && (PyArray_TYPE(rrmLLALocalOrigin) == NPY_DOUBLE)) || (PyArray_ISFLOAT(mmmXYZTarget) == 0)) {
        inArrayLocal = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmXYZTarget), PyArray_SHAPE(mmmXYZTarget), NPY_DOUBLE);
        if (inArrayLocal == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLocal, mmmXYZTarget) < 0) {
            Py_DECREF(inArrayLocal);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLocal))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayLocal = mmmXYZTarget;

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(inArrayLocal),
        PyArray_SHAPE(inArrayLocal),
        PyArray_TYPE(inArrayLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2NEDDoubleRolled(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2NEDFloatRolled(
            (float*)PyArray_DATA(inArrayOrigin), (float*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
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
    if (!PyArg_ParseTuple(args,
            "O!O!",
            &PyArray_Type,
            &rrmLLALocalOrigin,
            &PyArray_Type,
            &mmmXYZTarget))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmLLALocalOrigin)) || !(PyArray_ISCONTIGUOUS(mmmXYZTarget))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
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

    // ensure matching floating point types
    PyArrayObject *inArrayLocal, *inArrayOrigin;
    if (((PyArray_TYPE(rrmLLALocalOrigin) == NPY_FLOAT) && (PyArray_TYPE(mmmXYZTarget) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmLLALocalOrigin) == 0)) {
        inArrayOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLALocalOrigin), PyArray_SHAPE(rrmLLALocalOrigin), NPY_DOUBLE);
        if (inArrayOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayOrigin, rrmLLALocalOrigin) < 0) {
            Py_DECREF(inArrayOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayOrigin = rrmLLALocalOrigin;
    if (((PyArray_TYPE(mmmXYZTarget) == NPY_FLOAT) && (PyArray_TYPE(rrmLLALocalOrigin) == NPY_DOUBLE)) || (PyArray_ISFLOAT(mmmXYZTarget) == 0)) {
        inArrayLocal = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmXYZTarget), PyArray_SHAPE(mmmXYZTarget), NPY_DOUBLE);
        if (inArrayLocal == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLocal, mmmXYZTarget) < 0) {
            Py_DECREF(inArrayLocal);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLocal))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayLocal = mmmXYZTarget;

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(inArrayLocal),
        PyArray_SHAPE(inArrayLocal),
        PyArray_TYPE(inArrayLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2NEDvDoubleRolled(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2NEDvFloatRolled(
            (float*)PyArray_DATA(inArrayOrigin), (float*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(result_array));
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
            "O!O!O!O!O!O!",
            &PyArray_Type,
            &radLatOrigin,
            &PyArray_Type,
            &radLonOrigin,
            &PyArray_Type,
            &mAltOrigin,
            &PyArray_Type,
            &mXTarget,
            &PyArray_Type,
            &mYTarget,
            &PyArray_Type,
            &mZTarget))
        return NULL;
    if (!((PyArray_ISCONTIGUOUS(radLatOrigin)) && (PyArray_ISCONTIGUOUS(radLonOrigin)) && (PyArray_ISCONTIGUOUS(mAltOrigin)) && (PyArray_ISCONTIGUOUS(mXTarget)) && (PyArray_ISCONTIGUOUS(mYTarget)) && (PyArray_ISCONTIGUOUS(mZTarget)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(radLatOrigin) == PyArray_TYPE(radLonOrigin)) && (PyArray_TYPE(radLonOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mAltOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mXTarget) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mYTarget) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mZTarget) == PyArray_TYPE(radLatOrigin)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(mXTarget) == PyArray_SIZE(mYTarget)) && (PyArray_SIZE(mXTarget) == PyArray_SIZE(mZTarget)))) {
        PyErr_SetString(PyExc_ValueError, "Input target arrays must be the same size.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLatOrigin) == PyArray_SIZE(radLonOrigin)) && (PyArray_SIZE(radLatOrigin) == PyArray_SIZE(mAltOrigin)))) {
        PyErr_SetString(PyExc_ValueError, "Input origin arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayXTarget, *inArrayYTarget, *inArrayZTarget, *inArrayLatOrigin, *inArrayLonOrigin, *inArrayAltOrigin;
    if (PyArray_ISINTEGER(mXTarget) == 0)
        inArrayXTarget = mXTarget;
    else {
        inArrayXTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), NPY_DOUBLE);
        if (inArrayXTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayXTarget, mXTarget) < 0) {
            Py_DECREF(inArrayXTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayXTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mYTarget) == 0)
        inArrayYTarget = mYTarget;
    else {
        inArrayYTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mYTarget), PyArray_SHAPE(mYTarget), NPY_DOUBLE);
        if (inArrayYTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayYTarget, mYTarget) < 0) {
            Py_DECREF(inArrayYTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayYTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mZTarget) == 0)
        inArrayZTarget = mZTarget;
    else {
        inArrayZTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mZTarget), PyArray_SHAPE(mZTarget), NPY_DOUBLE);
        if (inArrayZTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayZTarget, mZTarget) < 0) {
            Py_DECREF(inArrayZTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayZTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLatOrigin) == 0)
        inArrayLatOrigin = radLatOrigin;
    else {
        inArrayLatOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLatOrigin), PyArray_SHAPE(radLatOrigin), NPY_DOUBLE);
        if (inArrayLatOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLatOrigin, radLatOrigin) < 0) {
            Py_DECREF(inArrayLatOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLatOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLonOrigin) == 0)
        inArrayLonOrigin = radLonOrigin;
    else {
        inArrayLonOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLonOrigin), PyArray_SHAPE(radLonOrigin), NPY_DOUBLE);
        if (inArrayLonOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLonOrigin, radLonOrigin) < 0) {
            Py_DECREF(inArrayLonOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLonOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAltOrigin) == 0)
        inArrayAltOrigin = mAltOrigin;
    else {
        inArrayAltOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAltOrigin), PyArray_SHAPE(mAltOrigin), NPY_DOUBLE);
        if (inArrayAltOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAltOrigin, mAltOrigin) < 0) {
            Py_DECREF(inArrayAltOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAltOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayXTarget) == PyArray_Size((PyObject*)inArrayAltOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        ECEF2NEDvDoubleUnrolled(
            (double*)PyArray_DATA(inArrayLatOrigin), (double*)PyArray_DATA(inArrayLonOrigin), (double*)PyArray_DATA(inArrayAltOrigin), (double*)PyArray_DATA(inArrayXTarget), (double*)PyArray_DATA(inArrayYTarget), (double*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        ECEF2NEDvFloatUnrolled(
            (float*)PyArray_DATA(inArrayLatOrigin), (float*)PyArray_DATA(inArrayLonOrigin), (float*)PyArray_DATA(inArrayAltOrigin), (float*)PyArray_DATA(inArrayXTarget), (float*)PyArray_DATA(inArrayYTarget), (float*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(inArrayLatOrigin);
        Py_DECREF(inArrayLonOrigin);
        Py_DECREF(inArrayAltOrigin);
        Py_DECREF(inArrayXTarget);
        Py_DECREF(inArrayYTarget);
        Py_DECREF(inArrayZTarget);
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
    if (!PyArg_ParseTuple(args,
            "O!O!",
            &PyArray_Type,
            &rrmLLALocalOrigin,
            &PyArray_Type,
            &mmmXYZTarget))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmLLALocalOrigin)) || !(PyArray_ISCONTIGUOUS(mmmXYZTarget))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
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

    // ensure matching floating point types
    PyArrayObject *inArrayLocal, *inArrayOrigin;
    if (((PyArray_TYPE(rrmLLALocalOrigin) == NPY_FLOAT) && (PyArray_TYPE(mmmXYZTarget) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmLLALocalOrigin) == 0)) {
        inArrayOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLALocalOrigin), PyArray_SHAPE(rrmLLALocalOrigin), NPY_DOUBLE);
        if (inArrayOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayOrigin, rrmLLALocalOrigin) < 0) {
            Py_DECREF(inArrayOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayOrigin = rrmLLALocalOrigin;
    if (((PyArray_TYPE(mmmXYZTarget) == NPY_FLOAT) && (PyArray_TYPE(rrmLLALocalOrigin) == NPY_DOUBLE)) || (PyArray_ISFLOAT(mmmXYZTarget) == 0)) {
        inArrayLocal = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmXYZTarget), PyArray_SHAPE(mmmXYZTarget), NPY_DOUBLE);
        if (inArrayLocal == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLocal, mmmXYZTarget) < 0) {
            Py_DECREF(inArrayLocal);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLocal))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayLocal = mmmXYZTarget;

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(inArrayLocal),
        PyArray_SHAPE(inArrayLocal),
        PyArray_TYPE(inArrayLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2ENUvDoubleRolled(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2ENUvFloatRolled(
            (float*)PyArray_DATA(inArrayOrigin), (float*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(result_array));
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
            "O!O!O!O!O!O!",
            &PyArray_Type,
            &radLatOrigin,
            &PyArray_Type,
            &radLonOrigin,
            &PyArray_Type,
            &mAltOrigin,
            &PyArray_Type,
            &mXTarget,
            &PyArray_Type,
            &mYTarget,
            &PyArray_Type,
            &mZTarget))
        return NULL;
    if (!((PyArray_ISCONTIGUOUS(radLatOrigin)) && (PyArray_ISCONTIGUOUS(radLonOrigin)) && (PyArray_ISCONTIGUOUS(mAltOrigin)) && (PyArray_ISCONTIGUOUS(mXTarget)) && (PyArray_ISCONTIGUOUS(mYTarget)) && (PyArray_ISCONTIGUOUS(mZTarget)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(radLatOrigin) == PyArray_TYPE(radLonOrigin)) && (PyArray_TYPE(radLonOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mAltOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mXTarget) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mYTarget) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mZTarget) == PyArray_TYPE(radLatOrigin)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(mXTarget) == PyArray_SIZE(mYTarget)) && (PyArray_SIZE(mXTarget) == PyArray_SIZE(mZTarget)))) {
        PyErr_SetString(PyExc_ValueError, "Input target arrays must be the same size.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLatOrigin) == PyArray_SIZE(radLonOrigin)) && (PyArray_SIZE(radLatOrigin) == PyArray_SIZE(mAltOrigin)))) {
        PyErr_SetString(PyExc_ValueError, "Input origin arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayXTarget, *inArrayYTarget, *inArrayZTarget, *inArrayLatOrigin, *inArrayLonOrigin, *inArrayAltOrigin;
    if (PyArray_ISINTEGER(mXTarget) == 0)
        inArrayXTarget = mXTarget;
    else {
        inArrayXTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mXTarget), PyArray_SHAPE(mXTarget), NPY_DOUBLE);
        if (inArrayXTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayXTarget, mXTarget) < 0) {
            Py_DECREF(inArrayXTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayXTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mYTarget) == 0)
        inArrayYTarget = mYTarget;
    else {
        inArrayYTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mYTarget), PyArray_SHAPE(mYTarget), NPY_DOUBLE);
        if (inArrayYTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayYTarget, mYTarget) < 0) {
            Py_DECREF(inArrayYTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayYTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mZTarget) == 0)
        inArrayZTarget = mZTarget;
    else {
        inArrayZTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mZTarget), PyArray_SHAPE(mZTarget), NPY_DOUBLE);
        if (inArrayZTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayZTarget, mZTarget) < 0) {
            Py_DECREF(inArrayZTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayZTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLatOrigin) == 0)
        inArrayLatOrigin = radLatOrigin;
    else {
        inArrayLatOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLatOrigin), PyArray_SHAPE(radLatOrigin), NPY_DOUBLE);
        if (inArrayLatOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLatOrigin, radLatOrigin) < 0) {
            Py_DECREF(inArrayLatOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLatOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLonOrigin) == 0)
        inArrayLonOrigin = radLonOrigin;
    else {
        inArrayLonOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLonOrigin), PyArray_SHAPE(radLonOrigin), NPY_DOUBLE);
        if (inArrayLonOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLonOrigin, radLonOrigin) < 0) {
            Py_DECREF(inArrayLonOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLonOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAltOrigin) == 0)
        inArrayAltOrigin = mAltOrigin;
    else {
        inArrayAltOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAltOrigin), PyArray_SHAPE(mAltOrigin), NPY_DOUBLE);
        if (inArrayAltOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAltOrigin, mAltOrigin) < 0) {
            Py_DECREF(inArrayAltOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAltOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayXTarget) == PyArray_Size((PyObject*)inArrayAltOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        ECEF2ENUvDoubleUnrolled(
            (double*)PyArray_DATA(inArrayLatOrigin), (double*)PyArray_DATA(inArrayLonOrigin), (double*)PyArray_DATA(inArrayAltOrigin), (double*)PyArray_DATA(inArrayXTarget), (double*)PyArray_DATA(inArrayYTarget), (double*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        ECEF2ENUvFloatUnrolled(
            (float*)PyArray_DATA(inArrayLatOrigin), (float*)PyArray_DATA(inArrayLonOrigin), (float*)PyArray_DATA(inArrayAltOrigin), (float*)PyArray_DATA(inArrayXTarget), (float*)PyArray_DATA(inArrayYTarget), (float*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(inArrayLatOrigin);
        Py_DECREF(inArrayLonOrigin);
        Py_DECREF(inArrayAltOrigin);
        Py_DECREF(inArrayXTarget);
        Py_DECREF(inArrayYTarget);
        Py_DECREF(inArrayZTarget);
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
            "O!O!O!O!O!O!dd",
            &PyArray_Type,
            &radLatOrigin,
            &PyArray_Type,
            &radLonOrigin,
            &PyArray_Type,
            &mAltOrigin,
            &PyArray_Type,
            &mNLocal,
            &PyArray_Type,
            &mELocal,
            &PyArray_Type,
            &mDLocal,
            &a,
            &b))
        return NULL;
    if (!((PyArray_ISCONTIGUOUS(radLatOrigin)) && (PyArray_ISCONTIGUOUS(radLonOrigin)) && (PyArray_ISCONTIGUOUS(mAltOrigin)) && (PyArray_ISCONTIGUOUS(mNLocal)) && (PyArray_ISCONTIGUOUS(mELocal)) && (PyArray_ISCONTIGUOUS(mDLocal)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(radLatOrigin) == PyArray_TYPE(radLonOrigin)) && (PyArray_TYPE(radLonOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mAltOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mELocal) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mNLocal) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mDLocal) == PyArray_TYPE(radLatOrigin)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(mELocal) == PyArray_SIZE(mNLocal)) && (PyArray_SIZE(mELocal) == PyArray_SIZE(mDLocal)))) {
        PyErr_SetString(PyExc_ValueError, "Input target arrays must be the same size.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLatOrigin) == PyArray_SIZE(radLonOrigin)) && (PyArray_SIZE(radLatOrigin) == PyArray_SIZE(mAltOrigin)))) {
        PyErr_SetString(PyExc_ValueError, "Input origin arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayXTarget, *inArrayYTarget, *inArrayZTarget, *inArrayLatOrigin, *inArrayLonOrigin, *inArrayAltOrigin;
    if (PyArray_ISINTEGER(mNLocal) == 0)
        inArrayXTarget = mNLocal;
    else {
        inArrayXTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), NPY_DOUBLE);
        if (inArrayXTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayXTarget, mNLocal) < 0) {
            Py_DECREF(inArrayXTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayXTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mELocal) == 0)
        inArrayYTarget = mELocal;
    else {
        inArrayYTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mELocal), PyArray_SHAPE(mELocal), NPY_DOUBLE);
        if (inArrayYTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayYTarget, mELocal) < 0) {
            Py_DECREF(inArrayYTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayYTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mDLocal) == 0)
        inArrayZTarget = mDLocal;
    else {
        inArrayZTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mDLocal), PyArray_SHAPE(mDLocal), NPY_DOUBLE);
        if (inArrayZTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayZTarget, mDLocal) < 0) {
            Py_DECREF(inArrayZTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayZTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLatOrigin) == 0)
        inArrayLatOrigin = radLatOrigin;
    else {
        inArrayLatOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLatOrigin), PyArray_SHAPE(radLatOrigin), NPY_DOUBLE);
        if (inArrayLatOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLatOrigin, radLatOrigin) < 0) {
            Py_DECREF(inArrayLatOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLatOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLonOrigin) == 0)
        inArrayLonOrigin = radLonOrigin;
    else {
        inArrayLonOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLonOrigin), PyArray_SHAPE(radLonOrigin), NPY_DOUBLE);
        if (inArrayLonOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLonOrigin, radLonOrigin) < 0) {
            Py_DECREF(inArrayLonOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLonOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAltOrigin) == 0)
        inArrayAltOrigin = mAltOrigin;
    else {
        inArrayAltOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAltOrigin), PyArray_SHAPE(mAltOrigin), NPY_DOUBLE);
        if (inArrayAltOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAltOrigin, mAltOrigin) < 0) {
            Py_DECREF(inArrayAltOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAltOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayXTarget) == PyArray_Size((PyObject*)inArrayAltOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        NED2ECEFDoubleUnrolled(
            (double*)PyArray_DATA(inArrayLatOrigin), (double*)PyArray_DATA(inArrayLonOrigin), (double*)PyArray_DATA(inArrayAltOrigin), (double*)PyArray_DATA(inArrayXTarget), (double*)PyArray_DATA(inArrayYTarget), (double*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        NED2ECEFFloatUnrolled(
            (float*)PyArray_DATA(inArrayLatOrigin), (float*)PyArray_DATA(inArrayLonOrigin), (float*)PyArray_DATA(inArrayAltOrigin), (float*)PyArray_DATA(inArrayXTarget), (float*)PyArray_DATA(inArrayYTarget), (float*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(inArrayLatOrigin);
        Py_DECREF(inArrayLonOrigin);
        Py_DECREF(inArrayAltOrigin);
        Py_DECREF(inArrayXTarget);
        Py_DECREF(inArrayYTarget);
        Py_DECREF(inArrayZTarget);
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
            "O!O!dd",
            &PyArray_Type,
            &rrmLLALocalOrigin,
            &PyArray_Type,
            &mmmLocal,
            &a,
            &b))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmLLALocalOrigin)) || !(PyArray_ISCONTIGUOUS(mmmLocal))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
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

    // ensure matching floating point types
    PyArrayObject *inArrayLocal, *inArrayOrigin;
    if (((PyArray_TYPE(rrmLLALocalOrigin) == NPY_FLOAT) && (PyArray_TYPE(mmmLocal) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmLLALocalOrigin) == 0)) {
        inArrayOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLALocalOrigin), PyArray_SHAPE(rrmLLALocalOrigin), NPY_DOUBLE);
        if (inArrayOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayOrigin, rrmLLALocalOrigin) < 0) {
            Py_DECREF(inArrayOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayOrigin = rrmLLALocalOrigin;
    if (((PyArray_TYPE(mmmLocal) == NPY_FLOAT) && (PyArray_TYPE(rrmLLALocalOrigin) == NPY_DOUBLE)) || (PyArray_ISFLOAT(mmmLocal) == 0)) {
        inArrayLocal = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmLocal), PyArray_SHAPE(mmmLocal), NPY_DOUBLE);
        if (inArrayLocal == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLocal, mmmLocal) < 0) {
            Py_DECREF(inArrayLocal);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLocal))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayLocal = mmmLocal;

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayLocal), PyArray_SHAPE(inArrayLocal), PyArray_TYPE(inArrayLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        NED2ECEFDoubleRolled(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        NED2ECEFFloatRolled(
            (float*)PyArray_DATA(inArrayOrigin), (float*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
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
            "O!O!O!O!O!O!dd",
            &PyArray_Type,
            &radLatOrigin,
            &PyArray_Type,
            &radLonOrigin,
            &PyArray_Type,
            &mAltOrigin,
            &PyArray_Type,
            &mNLocal,
            &PyArray_Type,
            &mELocal,
            &PyArray_Type,
            &mDLocal,
            &a,
            &b))
        return NULL;
    if (!((PyArray_ISCONTIGUOUS(radLatOrigin)) && (PyArray_ISCONTIGUOUS(radLonOrigin)) && (PyArray_ISCONTIGUOUS(mAltOrigin)) && (PyArray_ISCONTIGUOUS(mNLocal)) && (PyArray_ISCONTIGUOUS(mELocal)) && (PyArray_ISCONTIGUOUS(mDLocal)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(radLatOrigin) == PyArray_TYPE(radLonOrigin)) && (PyArray_TYPE(radLonOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mAltOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mELocal) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mNLocal) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mDLocal) == PyArray_TYPE(radLatOrigin)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(mELocal) == PyArray_SIZE(mNLocal)) && (PyArray_SIZE(mELocal) == PyArray_SIZE(mDLocal)))) {
        PyErr_SetString(PyExc_ValueError, "Input target arrays must be the same size.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLatOrigin) == PyArray_SIZE(radLonOrigin)) && (PyArray_SIZE(radLatOrigin) == PyArray_SIZE(mAltOrigin)))) {
        PyErr_SetString(PyExc_ValueError, "Input origin arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayXTarget, *inArrayYTarget, *inArrayZTarget, *inArrayLatOrigin, *inArrayLonOrigin, *inArrayAltOrigin;
    if (PyArray_ISINTEGER(mNLocal) == 0)
        inArrayXTarget = mNLocal;
    else {
        inArrayXTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), NPY_DOUBLE);
        if (inArrayXTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayXTarget, mNLocal) < 0) {
            Py_DECREF(inArrayXTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayXTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mNLocal) == 0)
        inArrayYTarget = mELocal;
    else {
        inArrayYTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mELocal), PyArray_SHAPE(mELocal), NPY_DOUBLE);
        if (inArrayYTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayYTarget, mELocal) < 0) {
            Py_DECREF(inArrayYTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayYTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mDLocal) == 0)
        inArrayZTarget = mDLocal;
    else {
        inArrayZTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mDLocal), PyArray_SHAPE(mDLocal), NPY_DOUBLE);
        if (inArrayZTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayZTarget, mDLocal) < 0) {
            Py_DECREF(inArrayZTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayZTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLatOrigin) == 0)
        inArrayLatOrigin = radLatOrigin;
    else {
        inArrayLatOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLatOrigin), PyArray_SHAPE(radLatOrigin), NPY_DOUBLE);
        if (inArrayLatOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLatOrigin, radLatOrigin) < 0) {
            Py_DECREF(inArrayLatOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLatOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLonOrigin) == 0)
        inArrayLonOrigin = radLonOrigin;
    else {
        inArrayLonOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLonOrigin), PyArray_SHAPE(radLonOrigin), NPY_DOUBLE);
        if (inArrayLonOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLonOrigin, radLonOrigin) < 0) {
            Py_DECREF(inArrayLonOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLonOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAltOrigin) == 0)
        inArrayAltOrigin = mAltOrigin;
    else {
        inArrayAltOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAltOrigin), PyArray_SHAPE(mAltOrigin), NPY_DOUBLE);
        if (inArrayAltOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAltOrigin, mAltOrigin) < 0) {
            Py_DECREF(inArrayAltOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAltOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayXTarget) == PyArray_Size((PyObject*)inArrayAltOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
    case NPY_DOUBLE:
        ENU2ECEFDoubleUnrolled(
            (double*)PyArray_DATA(inArrayLatOrigin), (double*)PyArray_DATA(inArrayLonOrigin), (double*)PyArray_DATA(inArrayAltOrigin), (double*)PyArray_DATA(inArrayXTarget), (double*)PyArray_DATA(inArrayYTarget), (double*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
        break;
    case NPY_FLOAT:
        ENU2ECEFFloatUnrolled(
            (float*)PyArray_DATA(inArrayLatOrigin), (float*)PyArray_DATA(inArrayLonOrigin), (float*)PyArray_DATA(inArrayAltOrigin), (float*)PyArray_DATA(inArrayXTarget), (float*)PyArray_DATA(inArrayYTarget), (float*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (float)(a), (float)(b), (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }

    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(inArrayLatOrigin);
        Py_DECREF(inArrayLonOrigin);
        Py_DECREF(inArrayAltOrigin);
        Py_DECREF(inArrayXTarget);
        Py_DECREF(inArrayYTarget);
        Py_DECREF(inArrayZTarget);
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
            "O!O!dd",
            &PyArray_Type,
            &rrmLLALocalOrigin,
            &PyArray_Type,
            &mmmLocal,
            &a,
            &b))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmLLALocalOrigin)) || !(PyArray_ISCONTIGUOUS(mmmLocal))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
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

    // ensure matching floating point types
    PyArrayObject *inArrayLocal, *inArrayOrigin;
    if (((PyArray_TYPE(rrmLLALocalOrigin) == NPY_FLOAT) && (PyArray_TYPE(mmmLocal) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmLLALocalOrigin) == 0)) {
        inArrayOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLALocalOrigin), PyArray_SHAPE(rrmLLALocalOrigin), NPY_DOUBLE);
        if (inArrayOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayOrigin, rrmLLALocalOrigin) < 0) {
            Py_DECREF(inArrayOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayOrigin = rrmLLALocalOrigin;
    if (((PyArray_TYPE(mmmLocal) == NPY_FLOAT) && (PyArray_TYPE(rrmLLALocalOrigin) == NPY_DOUBLE)) || (PyArray_ISFLOAT(mmmLocal) == 0)) {
        inArrayLocal = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmLocal), PyArray_SHAPE(mmmLocal), NPY_DOUBLE);
        if (inArrayLocal == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLocal, mmmLocal) < 0) {
            Py_DECREF(inArrayLocal);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLocal))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayLocal = mmmLocal;

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayLocal), PyArray_SHAPE(inArrayLocal), PyArray_TYPE(inArrayLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ENU2ECEFDoubleRolled(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ENU2ECEFFloatRolled(
            (float*)PyArray_DATA(inArrayOrigin), (float*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (float)a, (float)b, (float*)PyArray_DATA(result_array));
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
            "O!O!",
            &PyArray_Type,
            &rrmLLALocalOrigin,
            &PyArray_Type,
            &mmmLocal))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmLLALocalOrigin)) || !(PyArray_ISCONTIGUOUS(mmmLocal))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
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

    // ensure matching floating point types
    PyArrayObject *inArrayLocal, *inArrayOrigin;
    if (((PyArray_TYPE(rrmLLALocalOrigin) == NPY_FLOAT) && (PyArray_TYPE(mmmLocal) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmLLALocalOrigin) == 0)) {
        inArrayOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLALocalOrigin), PyArray_SHAPE(rrmLLALocalOrigin), NPY_DOUBLE);
        if (inArrayOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayOrigin, rrmLLALocalOrigin) < 0) {
            Py_DECREF(inArrayOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayOrigin = rrmLLALocalOrigin;
    if (((PyArray_TYPE(mmmLocal) == NPY_FLOAT) && (PyArray_TYPE(rrmLLALocalOrigin) == NPY_DOUBLE)) || (PyArray_ISFLOAT(mmmLocal) == 0)) {
        inArrayLocal = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmLocal), PyArray_SHAPE(mmmLocal), NPY_DOUBLE);
        if (inArrayLocal == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLocal, mmmLocal) < 0) {
            Py_DECREF(inArrayLocal);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLocal))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayLocal = mmmLocal;

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayLocal), PyArray_SHAPE(inArrayLocal), PyArray_TYPE(inArrayLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ENU2ECEFvDoubleRolled(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ENU2ECEFvFloatRolled(
            (float*)PyArray_DATA(inArrayOrigin), (float*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(result_array));
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
    PyArrayObject *radLatOrigin, *radLonOrigin, *mAltOrigin, *mNLocal, *mELocal, *mDLocal;

    // checks
    if (!PyArg_ParseTuple(args,
        "O!O!O!O!O!O!",
        &PyArray_Type,
        &radLatOrigin,
        &PyArray_Type,
        &radLonOrigin,
        &PyArray_Type,
        &mAltOrigin,
        &PyArray_Type,
        &mNLocal,
        &PyArray_Type,
        &mELocal,
        &PyArray_Type,
        &mDLocal))
    return NULL;
    if (!((PyArray_ISCONTIGUOUS(radLatOrigin)) && (PyArray_ISCONTIGUOUS(radLonOrigin)) && (PyArray_ISCONTIGUOUS(mAltOrigin)) && (PyArray_ISCONTIGUOUS(mNLocal)) && (PyArray_ISCONTIGUOUS(mELocal)) && (PyArray_ISCONTIGUOUS(mDLocal)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(radLatOrigin) == PyArray_TYPE(radLonOrigin)) && (PyArray_TYPE(radLonOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mAltOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mELocal) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mNLocal) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mDLocal) == PyArray_TYPE(radLatOrigin)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(mELocal) == PyArray_SIZE(mNLocal)) && (PyArray_SIZE(mELocal) == PyArray_SIZE(mDLocal)))) {
        PyErr_SetString(PyExc_ValueError, "Input target arrays must be the same size.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLatOrigin) == PyArray_SIZE(radLonOrigin)) && (PyArray_SIZE(radLatOrigin) == PyArray_SIZE(mAltOrigin)))) {
        PyErr_SetString(PyExc_ValueError, "Input origin arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayXTarget, *inArrayYTarget, *inArrayZTarget, *inArrayLatOrigin, *inArrayLonOrigin, *inArrayAltOrigin;
    if (PyArray_ISINTEGER(mNLocal) == 0)
        inArrayXTarget = mNLocal;
    else {
        inArrayXTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), NPY_DOUBLE);
        if (inArrayXTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayXTarget, mNLocal) < 0) {
            Py_DECREF(inArrayXTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayXTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mNLocal) == 0)
        inArrayYTarget = mELocal;
    else {
        inArrayYTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mELocal), PyArray_SHAPE(mELocal), NPY_DOUBLE);
        if (inArrayYTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayYTarget, mELocal) < 0) {
            Py_DECREF(inArrayYTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayYTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mDLocal) == 0)
        inArrayZTarget = mDLocal;
    else {
        inArrayZTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mDLocal), PyArray_SHAPE(mDLocal), NPY_DOUBLE);
        if (inArrayZTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayZTarget, mDLocal) < 0) {
            Py_DECREF(inArrayZTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayZTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLatOrigin) == 0)
        inArrayLatOrigin = radLatOrigin;
    else {
        inArrayLatOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLatOrigin), PyArray_SHAPE(radLatOrigin), NPY_DOUBLE);
        if (inArrayLatOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLatOrigin, radLatOrigin) < 0) {
            Py_DECREF(inArrayLatOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLatOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLonOrigin) == 0)
        inArrayLonOrigin = radLonOrigin;
    else {
        inArrayLonOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLonOrigin), PyArray_SHAPE(radLonOrigin), NPY_DOUBLE);
        if (inArrayLonOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLonOrigin, radLonOrigin) < 0) {
            Py_DECREF(inArrayLonOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLonOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAltOrigin) == 0)
        inArrayAltOrigin = mAltOrigin;
    else {
        inArrayAltOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAltOrigin), PyArray_SHAPE(mAltOrigin), NPY_DOUBLE);
        if (inArrayAltOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAltOrigin, mAltOrigin) < 0) {
            Py_DECREF(inArrayAltOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAltOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayXTarget) == PyArray_Size((PyObject*)inArrayAltOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
        case NPY_DOUBLE:
            ENU2ECEFvDoubleUnrolled(
                (double*)PyArray_DATA(inArrayLatOrigin), (double*)PyArray_DATA(inArrayLonOrigin), (double*)PyArray_DATA(inArrayAltOrigin), (double*)PyArray_DATA(inArrayXTarget), (double*)PyArray_DATA(inArrayYTarget), (double*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
            break;
        case NPY_FLOAT:
            ENU2ECEFvFloatUnrolled(
                (float*)PyArray_DATA(inArrayLatOrigin), (float*)PyArray_DATA(inArrayLonOrigin), (float*)PyArray_DATA(inArrayAltOrigin), (float*)PyArray_DATA(inArrayXTarget), (float*)PyArray_DATA(inArrayYTarget), (float*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                "Only 32 and 64 bit float types or all integer are accepted.");
            return NULL;
        }
    
        // output
        PyObject* tuple = PyTuple_New(3);
        if (!tuple){
            Py_DECREF(inArrayLatOrigin);
            Py_DECREF(inArrayLonOrigin);
            Py_DECREF(inArrayAltOrigin);
            Py_DECREF(inArrayXTarget);
            Py_DECREF(inArrayYTarget);
            Py_DECREF(inArrayZTarget);
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
        "O!O!O!O!O!O!",
        &PyArray_Type,
        &radLatOrigin,
        &PyArray_Type,
        &radLonOrigin,
        &PyArray_Type,
        &mAltOrigin,
        &PyArray_Type,
        &mNLocal,
        &PyArray_Type,
        &mELocal,
        &PyArray_Type,
        &mDLocal))
    return NULL;
    if (!((PyArray_ISCONTIGUOUS(radLatOrigin)) && (PyArray_ISCONTIGUOUS(radLonOrigin)) && (PyArray_ISCONTIGUOUS(mAltOrigin)) && (PyArray_ISCONTIGUOUS(mNLocal)) && (PyArray_ISCONTIGUOUS(mELocal)) && (PyArray_ISCONTIGUOUS(mDLocal)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(radLatOrigin) == PyArray_TYPE(radLonOrigin)) && (PyArray_TYPE(radLonOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mAltOrigin) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mELocal) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mNLocal) == PyArray_TYPE(radLatOrigin)) && (PyArray_TYPE(mDLocal) == PyArray_TYPE(radLatOrigin)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(mELocal) == PyArray_SIZE(mNLocal)) && (PyArray_SIZE(mELocal) == PyArray_SIZE(mDLocal)))) {
        PyErr_SetString(PyExc_ValueError, "Input target arrays must be the same size.");
        return NULL;
    }
    if (!((PyArray_SIZE(radLatOrigin) == PyArray_SIZE(radLonOrigin)) && (PyArray_SIZE(radLatOrigin) == PyArray_SIZE(mAltOrigin)))) {
        PyErr_SetString(PyExc_ValueError, "Input origin arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayXTarget, *inArrayYTarget, *inArrayZTarget, *inArrayLatOrigin, *inArrayLonOrigin, *inArrayAltOrigin;
    if (PyArray_ISINTEGER(mNLocal) == 0)
        inArrayXTarget = mNLocal;
    else {
        inArrayXTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mNLocal), PyArray_SHAPE(mNLocal), NPY_DOUBLE);
        if (inArrayXTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayXTarget, mNLocal) < 0) {
            Py_DECREF(inArrayXTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayXTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mNLocal) == 0)
        inArrayYTarget = mELocal;
    else {
        inArrayYTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mELocal), PyArray_SHAPE(mELocal), NPY_DOUBLE);
        if (inArrayYTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayYTarget, mELocal) < 0) {
            Py_DECREF(inArrayYTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayYTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mDLocal) == 0)
        inArrayZTarget = mDLocal;
    else {
        inArrayZTarget = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mDLocal), PyArray_SHAPE(mDLocal), NPY_DOUBLE);
        if (inArrayZTarget == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayZTarget, mDLocal) < 0) {
            Py_DECREF(inArrayZTarget);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayZTarget))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLatOrigin) == 0)
        inArrayLatOrigin = radLatOrigin;
    else {
        inArrayLatOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLatOrigin), PyArray_SHAPE(radLatOrigin), NPY_DOUBLE);
        if (inArrayLatOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLatOrigin, radLatOrigin) < 0) {
            Py_DECREF(inArrayLatOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLatOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radLonOrigin) == 0)
        inArrayLonOrigin = radLonOrigin;
    else {
        inArrayLonOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radLonOrigin), PyArray_SHAPE(radLonOrigin), NPY_DOUBLE);
        if (inArrayLonOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLonOrigin, radLonOrigin) < 0) {
            Py_DECREF(inArrayLonOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLonOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mAltOrigin) == 0)
        inArrayAltOrigin = mAltOrigin;
    else {
        inArrayAltOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mAltOrigin), PyArray_SHAPE(mAltOrigin), NPY_DOUBLE);
        if (inArrayAltOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAltOrigin, mAltOrigin) < 0) {
            Py_DECREF(inArrayAltOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAltOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *mX, *mY, *mZ;
    mX = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mY = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    mZ = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayXTarget), PyArray_SHAPE(inArrayXTarget), PyArray_TYPE(inArrayXTarget));
    if ((mX == NULL) || (mY == NULL) || (mZ == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayXTarget);
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayXTarget) == PyArray_Size((PyObject*)inArrayAltOrigin));

    // run function
    switch (PyArray_TYPE(mX)) {
        case NPY_DOUBLE:
            NED2ECEFvDoubleUnrolled(
                (double*)PyArray_DATA(inArrayLatOrigin), (double*)PyArray_DATA(inArrayLonOrigin), (double*)PyArray_DATA(inArrayAltOrigin), (double*)PyArray_DATA(inArrayXTarget), (double*)PyArray_DATA(inArrayYTarget), (double*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(mX), (double*)PyArray_DATA(mY), (double*)PyArray_DATA(mZ));
            break;
        case NPY_FLOAT:
            NED2ECEFvFloatUnrolled(
                (float*)PyArray_DATA(inArrayLatOrigin), (float*)PyArray_DATA(inArrayLonOrigin), (float*)PyArray_DATA(inArrayAltOrigin), (float*)PyArray_DATA(inArrayXTarget), (float*)PyArray_DATA(inArrayYTarget), (float*)PyArray_DATA(inArrayZTarget), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(mX), (float*)PyArray_DATA(mY), (float*)PyArray_DATA(mZ));
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                "Only 32 and 64 bit float types or all integer are accepted.");
            return NULL;
        }
    
        // output
        PyObject* tuple = PyTuple_New(3);
        if (!tuple){
            Py_DECREF(inArrayLatOrigin);
            Py_DECREF(inArrayLonOrigin);
            Py_DECREF(inArrayAltOrigin);
            Py_DECREF(inArrayXTarget);
            Py_DECREF(inArrayYTarget);
            Py_DECREF(inArrayZTarget);
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
            "O!O!",
            &PyArray_Type,
            &rrmLLALocalOrigin,
            &PyArray_Type,
            &mmmLocal))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmLLALocalOrigin)) || !(PyArray_ISCONTIGUOUS(mmmLocal))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
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

    // ensure matching floating point types
    PyArrayObject *inArrayLocal, *inArrayOrigin;
    if (((PyArray_TYPE(rrmLLALocalOrigin) == NPY_FLOAT) && (PyArray_TYPE(mmmLocal) == NPY_DOUBLE)) || (PyArray_ISFLOAT(rrmLLALocalOrigin) == 0)) {
        inArrayOrigin = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmLLALocalOrigin), PyArray_SHAPE(rrmLLALocalOrigin), NPY_DOUBLE);
        if (inArrayOrigin == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayOrigin, rrmLLALocalOrigin) < 0) {
            Py_DECREF(inArrayOrigin);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayOrigin))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayOrigin = rrmLLALocalOrigin;
    if (((PyArray_TYPE(mmmLocal) == NPY_FLOAT) && (PyArray_TYPE(rrmLLALocalOrigin) == NPY_DOUBLE)) || (PyArray_ISFLOAT(mmmLocal) == 0)) {
        inArrayLocal = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmLocal), PyArray_SHAPE(mmmLocal), NPY_DOUBLE);
        if (inArrayLocal == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayLocal, mmmLocal) < 0) {
            Py_DECREF(inArrayLocal);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayLocal))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    } else
        inArrayLocal = mmmLocal;

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayLocal), PyArray_SHAPE(inArrayLocal), PyArray_TYPE(inArrayLocal));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSIN3D;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        NED2ECEFvDoubleRolled(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        NED2ECEFvFloatRolled(
            (float*)PyArray_DATA(inArrayOrigin), (float*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (float*)PyArray_DATA(result_array));
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
    // checks
    if (!PyArg_ParseTuple(args,
        "O!O!O!",
        &PyArray_Type,
        &mE,
        &PyArray_Type,
        &mN,
        &PyArray_Type,
        &mU
        ))
    return NULL;
    if (!((PyArray_ISCONTIGUOUS(mE)) && (PyArray_ISCONTIGUOUS(mN)) && (PyArray_ISCONTIGUOUS(mU)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(mE) == PyArray_TYPE(mN)) && (PyArray_TYPE(mE) == PyArray_TYPE(mU)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(mE) == PyArray_SIZE(mN)) && (PyArray_SIZE(mE) == PyArray_SIZE(mU)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayE, *inArrayN, *inArrayU;
    if (PyArray_ISINTEGER(mE) == 0)
        inArrayE = mE;
    else {
        inArrayE = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mE), PyArray_SHAPE(mE), NPY_DOUBLE);
        if (inArrayE == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayE, mE) < 0) {
            Py_DECREF(inArrayE);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayE))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mN) == 0)
        inArrayN = mN;
    else {
        inArrayN = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mN), PyArray_SHAPE(mN), NPY_DOUBLE);
        if (inArrayN == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayN, mN) < 0) {
            Py_DECREF(inArrayN);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayN))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mU) == 0)
        inArrayU = mU;
    else {
        inArrayU = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mU), PyArray_SHAPE(mU), NPY_DOUBLE);
        if (inArrayU == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayU, mU) < 0) {
            Py_DECREF(inArrayU);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayU))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *radAz, *radEl, *mRange;
    radAz = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayE), PyArray_SHAPE(inArrayE), PyArray_TYPE(inArrayE));
    radEl = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayE), PyArray_SHAPE(inArrayE), PyArray_TYPE(inArrayE));
    mRange = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayE), PyArray_SHAPE(inArrayE), PyArray_TYPE(inArrayE));
    if ((radAz == NULL) || (radEl == NULL) || (mRange == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayE);

    // run function
    switch (PyArray_TYPE(mE)) {
        case NPY_DOUBLE:
            ENU2AERDoubleUnrolled(
                (double*)PyArray_DATA(inArrayE), (double*)PyArray_DATA(inArrayN), (double*)PyArray_DATA(inArrayU), nPoints, (double*)PyArray_DATA(radAz), (double*)PyArray_DATA(radEl), (double*)PyArray_DATA(mRange));
            break;
        case NPY_FLOAT:
            ENU2AERFloatUnrolled(
                (float*)PyArray_DATA(inArrayE), (float*)PyArray_DATA(inArrayN), (float*)PyArray_DATA(inArrayU), nPoints, (float*)PyArray_DATA(radAz), (float*)PyArray_DATA(radEl), (float*)PyArray_DATA(mRange));
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                "Only 32 and 64 bit float types or all integer are accepted.");
            return NULL;
        }
    
        // output
        PyObject* tuple = PyTuple_New(3);
        if (!tuple){
            Py_DECREF(inArrayE);
            Py_DECREF(inArrayN);
            Py_DECREF(inArrayU);
            Py_DECREF(mE);
            Py_DECREF(mN);
            Py_DECREF(mU);
            Py_DECREF(radAz);
            Py_DECREF(radEl);
            Py_DECREF(mRange);
            return NULL;
        }
        PyTuple_SetItem(tuple, 0, (PyObject*)radAz);
        PyTuple_SetItem(tuple, 1, (PyObject*)radEl);
        PyTuple_SetItem(tuple, 2, (PyObject*)mRange);
        return tuple;}

static PyObject*
ENU2AERRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* mmmENU;

    // checks
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &mmmENU))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(mmmENU))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(mmmENU) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    PyArrayObject* inArray;
    if (PyArray_ISINTEGER(mmmENU) == 0)
        inArray = mmmENU;
    else {
        inArray = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmENU), PyArray_SHAPE(mmmENU), NPY_DOUBLE);
        if (inArray == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArray, mmmENU) < 0) {
            Py_DECREF(inArray);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArray))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArray), PyArray_SHAPE(inArray), PyArray_TYPE(inArray));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArray) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ENU2AERDoubleRolled((double*)PyArray_DATA(inArray), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ENU2AERFloatRolled((float*)PyArray_DATA(inArray), nPoints, (float*)PyArray_DATA(result_array));
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
    // checks
    if (!PyArg_ParseTuple(args,
        "O!O!O!",
        &PyArray_Type,
        &mN,
        &PyArray_Type,
        &mE,
        &PyArray_Type,
        &mD
        ))
    return NULL;
    if (!((PyArray_ISCONTIGUOUS(mN)) && (PyArray_ISCONTIGUOUS(mE)) && (PyArray_ISCONTIGUOUS(mD)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(mN) == PyArray_TYPE(mE)) && (PyArray_TYPE(mN) == PyArray_TYPE(mD)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(mN) == PyArray_SIZE(mE)) && (PyArray_SIZE(mN) == PyArray_SIZE(mD)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayE, *inArrayN, *inArrayU;
    if (PyArray_ISINTEGER(mN) == 0)
        inArrayE = mN;
    else {
        inArrayE = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mN), PyArray_SHAPE(mN), NPY_DOUBLE);
        if (inArrayE == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayE, mN) < 0) {
            Py_DECREF(inArrayE);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayE))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mE) == 0)
        inArrayN = mE;
    else {
        inArrayN = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mE), PyArray_SHAPE(mE), NPY_DOUBLE);
        if (inArrayN == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayN, mE) < 0) {
            Py_DECREF(inArrayN);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayN))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mD) == 0)
        inArrayU = mD;
    else {
        inArrayU = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mD), PyArray_SHAPE(mD), NPY_DOUBLE);
        if (inArrayU == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayU, mD) < 0) {
            Py_DECREF(inArrayU);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayU))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *radAz, *radEl, *mRange;
    radAz = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayE), PyArray_SHAPE(inArrayE), PyArray_TYPE(inArrayE));
    radEl = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayE), PyArray_SHAPE(inArrayE), PyArray_TYPE(inArrayE));
    mRange = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayE), PyArray_SHAPE(inArrayE), PyArray_TYPE(inArrayE));
    if ((radAz == NULL) || (radEl == NULL) || (mRange == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayE);

    // run function
    switch (PyArray_TYPE(mN)) {
        case NPY_DOUBLE:
            NED2AERDoubleUnrolled(
                (double*)PyArray_DATA(inArrayE), (double*)PyArray_DATA(inArrayN), (double*)PyArray_DATA(inArrayU), nPoints, (double*)PyArray_DATA(radAz), (double*)PyArray_DATA(radEl), (double*)PyArray_DATA(mRange));
            break;
        case NPY_FLOAT:
            NED2AERFloatUnrolled(
                (float*)PyArray_DATA(inArrayE), (float*)PyArray_DATA(inArrayN), (float*)PyArray_DATA(inArrayU), nPoints, (float*)PyArray_DATA(radAz), (float*)PyArray_DATA(radEl), (float*)PyArray_DATA(mRange));
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                "Only 32 and 64 bit float types or all integer are accepted.");
            return NULL;
        }
    
        // output
        PyObject* tuple = PyTuple_New(3);
        if (!tuple){
            Py_DECREF(inArrayE);
            Py_DECREF(inArrayN);
            Py_DECREF(inArrayU);
            Py_DECREF(mN);
            Py_DECREF(mE);
            Py_DECREF(mD);
            Py_DECREF(radAz);
            Py_DECREF(radEl);
            Py_DECREF(mRange);
            return NULL;
        }
        PyTuple_SetItem(tuple, 0, (PyObject*)radAz);
        PyTuple_SetItem(tuple, 1, (PyObject*)radEl);
        PyTuple_SetItem(tuple, 2, (PyObject*)mRange);
        return tuple;}

static PyObject*
NED2AERRolledWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* mmmNED;

    // checks
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &mmmNED))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(mmmNED))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(mmmNED) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    PyArrayObject* inArray;
    if (PyArray_ISINTEGER(mmmNED) == 0)
        inArray = mmmNED;
    else {
        inArray = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mmmNED), PyArray_SHAPE(mmmNED), NPY_DOUBLE);
        if (inArray == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArray, mmmNED) < 0) {
            Py_DECREF(inArray);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArray))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArray), PyArray_SHAPE(inArray), PyArray_TYPE(inArray));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArray) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        NED2AERDoubleRolled((double*)PyArray_DATA(inArray), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        NED2AERFloatRolled((float*)PyArray_DATA(inArray), nPoints, (float*)PyArray_DATA(result_array));
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
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &rrmAER))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmAER))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmAER) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    PyArrayObject* inArray;
    if (PyArray_ISINTEGER(rrmAER) == 0)
        inArray = rrmAER;
    else {
        inArray = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmAER), PyArray_SHAPE(rrmAER), NPY_DOUBLE);
        if (inArray == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArray, rrmAER) < 0) {
            Py_DECREF(inArray);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArray))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArray), PyArray_SHAPE(inArray), PyArray_TYPE(inArray));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArray) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        AER2NEDDoubleRolled((double*)PyArray_DATA(inArray), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        AER2NEDFloatRolled((float*)PyArray_DATA(inArray), nPoints, (float*)PyArray_DATA(result_array));
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
    if (!PyArg_ParseTuple(args,
        "O!O!O!",
        &PyArray_Type,
        &radAz,
        &PyArray_Type,
        &radEl,
        &PyArray_Type,
        &mRange
        ))
    return NULL;
    if (!((PyArray_ISCONTIGUOUS(radAz)) && (PyArray_ISCONTIGUOUS(radEl)) && (PyArray_ISCONTIGUOUS(mRange)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(radAz) == PyArray_TYPE(radEl)) && (PyArray_TYPE(radAz) == PyArray_TYPE(mRange)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(radAz) == PyArray_SIZE(radEl)) && (PyArray_SIZE(radAz) == PyArray_SIZE(mRange)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayAz, *inArrayEl, *inArrayRange;
    if (PyArray_ISINTEGER(radAz) == 0)
        inArrayAz = radAz;
    else {
        inArrayAz = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radAz), PyArray_SHAPE(radAz), NPY_DOUBLE);
        if (inArrayAz == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAz, radAz) < 0) {
            Py_DECREF(inArrayAz);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAz))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radEl) == 0)
        inArrayEl = radEl;
    else {
        inArrayEl = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radEl), PyArray_SHAPE(radEl), NPY_DOUBLE);
        if (inArrayEl == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayEl, radEl) < 0) {
            Py_DECREF(inArrayEl);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayEl))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mRange) == 0)
        inArrayRange = mRange;
    else {
        inArrayRange = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mRange), PyArray_SHAPE(mRange), NPY_DOUBLE);
        if (inArrayRange == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayRange, mRange) < 0) {
            Py_DECREF(inArrayRange);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayRange))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *mN, *mE, *mD;
    mN = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayAz), PyArray_SHAPE(inArrayAz), PyArray_TYPE(inArrayAz));
    mE = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayAz), PyArray_SHAPE(inArrayAz), PyArray_TYPE(inArrayAz));
    mD = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayAz), PyArray_SHAPE(inArrayAz), PyArray_TYPE(inArrayAz));
    if ((mN == NULL) || (mE == NULL) || (mD == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayAz);

    // run function
    switch (PyArray_TYPE(mN)) {
    case NPY_DOUBLE:
        AER2NEDDoubleUnrolled(
            (double*)PyArray_DATA(inArrayAz), (double*)PyArray_DATA(inArrayEl), (double*)PyArray_DATA(inArrayRange), nPoints, (double*)PyArray_DATA(mN), (double*)PyArray_DATA(mE), (double*)PyArray_DATA(mD));
        break;
    case NPY_FLOAT:
        AER2NEDFloatUnrolled(
            (float*)PyArray_DATA(inArrayAz), (float*)PyArray_DATA(inArrayEl), (float*)PyArray_DATA(inArrayRange), nPoints, (float*)PyArray_DATA(mN), (float*)PyArray_DATA(mE), (float*)PyArray_DATA(mD));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    
    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(inArrayAz);
        Py_DECREF(inArrayEl);
        Py_DECREF(inArrayRange);
        Py_DECREF(mN);
        Py_DECREF(mE);
        Py_DECREF(mD);
        Py_DECREF(radAz);
        Py_DECREF(radEl);
        Py_DECREF(mRange);
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
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &rrmAER))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmAER))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmAER) % NCOORDSIN3D) != 0) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a multiple of 3.");
        return NULL;
    }

    PyArrayObject* inArray;
    if (PyArray_ISINTEGER(rrmAER) == 0)
        inArray = rrmAER;
    else {
        inArray = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(rrmAER), PyArray_SHAPE(rrmAER), NPY_DOUBLE);
        if (inArray == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArray, rrmAER) < 0) {
            Py_DECREF(inArray);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArray))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArray), PyArray_SHAPE(inArray), PyArray_TYPE(inArray));
    if (result_array == NULL)
        return NULL;
    long nPoints = (int)PyArray_SIZE(inArray) / NCOORDSIN3D;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        AER2ENUDoubleRolled((double*)PyArray_DATA(inArray), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        AER2ENUFloatRolled((float*)PyArray_DATA(inArray), nPoints, (float*)PyArray_DATA(result_array));
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
    if (!PyArg_ParseTuple(args,
        "O!O!O!",
        &PyArray_Type,
        &radAz,
        &PyArray_Type,
        &radEl,
        &PyArray_Type,
        &mRange
        ))
    return NULL;
    if (!((PyArray_ISCONTIGUOUS(radAz)) && (PyArray_ISCONTIGUOUS(radEl)) && (PyArray_ISCONTIGUOUS(mRange)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be C contiguous.");
        return NULL;
    }
    if (!((PyArray_TYPE(radAz) == PyArray_TYPE(radEl)) && (PyArray_TYPE(radAz) == PyArray_TYPE(mRange)))) {
            PyErr_SetString(PyExc_ValueError, "Input arrays must of the same type.");
        return NULL;
    }
    if (!((PyArray_SIZE(radAz) == PyArray_SIZE(radEl)) && (PyArray_SIZE(radAz) == PyArray_SIZE(mRange)))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be the same size.");
        return NULL;
    }

    // ensure floating point type
    PyArrayObject *inArrayAz, *inArrayEl, *inArrayRange;
    if (PyArray_ISINTEGER(radAz) == 0)
        inArrayAz = radAz;
    else {
        inArrayAz = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radAz), PyArray_SHAPE(radAz), NPY_DOUBLE);
        if (inArrayAz == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayAz, radAz) < 0) {
            Py_DECREF(inArrayAz);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayAz))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(radEl) == 0)
        inArrayEl = radEl;
    else {
        inArrayEl = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(radEl), PyArray_SHAPE(radEl), NPY_DOUBLE);
        if (inArrayEl == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayEl, radEl) < 0) {
            Py_DECREF(inArrayEl);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayEl))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }
    if (PyArray_ISINTEGER(mRange) == 0)
        inArrayRange = mRange;
    else {
        inArrayRange = (PyArrayObject*)PyArray_SimpleNew(
            PyArray_NDIM(mRange), PyArray_SHAPE(mRange), NPY_DOUBLE);
        if (inArrayRange == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create new array.");
            return NULL;
        }
        if (PyArray_CopyInto(inArrayRange, mRange) < 0) {
            Py_DECREF(inArrayRange);
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy data to new array.");
            return NULL;
        }
        if (!(PyArray_ISCONTIGUOUS(inArrayRange))) {
            PyErr_SetString(PyExc_ValueError, "Created array is not C contiguous.");
            return NULL;
        }
    }

    // prepare inputs
    PyArrayObject *mE, *mN, *mU;
    mE = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayAz), PyArray_SHAPE(inArrayAz), PyArray_TYPE(inArrayAz));
    mN = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayAz), PyArray_SHAPE(inArrayAz), PyArray_TYPE(inArrayAz));
    mU = (PyArrayObject*)PyArray_SimpleNew(
        PyArray_NDIM(inArrayAz), PyArray_SHAPE(inArrayAz), PyArray_TYPE(inArrayAz));
    if ((mE == NULL) || (mN == NULL) || (mU == NULL)) {
        PyErr_SetString(PyExc_ValueError, "Failed to initialise output arrays.");
        return NULL;
    }
    long nPoints = (int)PyArray_SIZE(inArrayAz);

    // run function
    switch (PyArray_TYPE(mE)) {
    case NPY_DOUBLE:
        AER2ENUDoubleUnrolled(
            (double*)PyArray_DATA(inArrayAz), (double*)PyArray_DATA(inArrayEl), (double*)PyArray_DATA(inArrayRange), nPoints, (double*)PyArray_DATA(mE), (double*)PyArray_DATA(mN), (double*)PyArray_DATA(mU));
        break;
    case NPY_FLOAT:
        AER2ENUFloatUnrolled(
            (float*)PyArray_DATA(inArrayAz), (float*)PyArray_DATA(inArrayEl), (float*)PyArray_DATA(inArrayRange), nPoints, (float*)PyArray_DATA(mE), (float*)PyArray_DATA(mN), (float*)PyArray_DATA(mU));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    
    // output
    PyObject* tuple = PyTuple_New(3);
    if (!tuple){
        Py_DECREF(inArrayAz);
        Py_DECREF(inArrayEl);
        Py_DECREF(inArrayRange);
        Py_DECREF(mE);
        Py_DECREF(mN);
        Py_DECREF(mU);
        Py_DECREF(radAz);
        Py_DECREF(radEl);
        Py_DECREF(mRange);
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
