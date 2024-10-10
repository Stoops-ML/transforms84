#include <Python.h>
#include <float.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <omp.h>

#include "definitions.h"

/*
Geodetic to ECEF transformation of float precision.
https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates

@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
@param int nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmXYZ array of size nx3 X, Y, Z [rad, rad, m]
*/
void geodetic2ECEFFloat(const float* rrmLLA,
    int nPoints,
    float a,
    float b,
    float* mmmXYZ)
{
    float e2 = 1 - (b * b) / (a * a);
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
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
@param int nPoints Number of LLA points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmXYZ array of size nx3 X, Y, Z [m, m, m]
*/
void geodetic2ECEFDouble(const double* rrmLLA,
    int nPoints,
    double a,
    double b,
    double* mmmXYZ)
{
    double e2 = 1 - (b * b) / (a * a);
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
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
@param int nPoints Number of ECEF points
@param double a semi-major axis
@param double b semi-minor axis
@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
*/
void ECEF2geodeticFloat(const float* mmmXYZ,
    int nPoints,
    float a,
    float b,
    float* rrmLLA)
{
    int iPoint;
    float half = 0.5;
    float e2 = ((a * a) - (b * b)) / (a * a);
    float ed2 = ((a * a) - (b * b)) / (b * b);
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
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
@param int nPoints Number of ECEF points
@param double a semi-major axis
@param double b semi-minor axis
@param double *rrmLLA array of size nx3 latitude (phi), longitude (gamma),
height (h) [rad, rad, m]
*/
void ECEF2geodeticDouble(const double* mmmXYZ,
    int nPoints,
    double a,
    double b,
    double* rrmLLA)
{
    double e2 = ((a * a) - (b * b)) / (a * a);
    double ed2 = ((a * a) - (b * b)) / (b * b);
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
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
@param int nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void ECEF2ENUFloat(const float* rrmLLALocalOrigin,
    const float* mmmXYZTarget,
    int nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mmmLocal)
{
    int nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mmmXYZLocalOrigin = (float*)malloc(nOriginPoints * NCOORDSINPOINT * sizeof(float));
    geodetic2ECEFFloat(rrmLLALocalOrigin, nOriginPoints, (float)(a), (float)(b), mmmXYZLocalOrigin);
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
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
@param int nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void ECEF2ENUDouble(const double* rrmLLALocalOrigin,
    const double* mmmXYZTarget,
    int nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mmmLocal)
{
    int nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mmmXYZLocalOrigin = (double*)malloc(nOriginPoints * NCOORDSINPOINT * sizeof(double));
    geodetic2ECEFDouble(
        rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
        double DeltaX = mmmXYZTarget[iTarget + 0] - mmmXYZLocalOrigin[iOrigin + 0];
        double DeltaY = mmmXYZTarget[iTarget + 1] - mmmXYZLocalOrigin[iOrigin + 1];
        double DeltaZ = mmmXYZTarget[iTarget + 2] - mmmXYZLocalOrigin[iOrigin + 2];
        mmmLocal[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY;
        mmmLocal[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + cos(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
        mmmLocal[iTarget + 2] = cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + sin(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
    }
    free(mmmXYZLocalOrigin);
}

void ECEF2NEDFloat(const float* rrmLLALocalOrigin,
    const float* mmmXYZTarget,
    int nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mmmLocal)
{
    int nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mmmXYZLocalOrigin = (float*)malloc(nOriginPoints * NCOORDSINPOINT * sizeof(float));
    geodetic2ECEFFloat(
        rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
        float DeltaX = mmmXYZTarget[iTarget + 0] - mmmXYZLocalOrigin[iOrigin + 0];
        float DeltaY = mmmXYZTarget[iTarget + 1] - mmmXYZLocalOrigin[iOrigin + 1];
        float DeltaZ = mmmXYZTarget[iTarget + 2] - mmmXYZLocalOrigin[iOrigin + 2];
        mmmLocal[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + cosf(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
        mmmLocal[iTarget + 1] = -sinf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + cosf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY;
        mmmLocal[iTarget + 2] = -cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
    }
    free(mmmXYZLocalOrigin);
}

void ECEF2NEDDouble(const double* rrmLLALocalOrigin,
    const double* mmmXYZTarget,
    int nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mmmLocal)
{
    int nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mmmXYZLocalOrigin = (double*)malloc(nOriginPoints * NCOORDSINPOINT * sizeof(double));
    geodetic2ECEFDouble(
        rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
        double DeltaX = mmmXYZTarget[iTarget + 0] - mmmXYZLocalOrigin[iOrigin + 0];
        double DeltaY = mmmXYZTarget[iTarget + 1] - mmmXYZLocalOrigin[iOrigin + 1];
        double DeltaZ = mmmXYZTarget[iTarget + 2] - mmmXYZLocalOrigin[iOrigin + 2];
        mmmLocal[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + cos(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
        mmmLocal[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY;
        mmmLocal[iTarget + 2] = -cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * DeltaX + -cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * DeltaY + -sin(rrmLLALocalOrigin[iOrigin + 0]) * DeltaZ;
    }
    free(mmmXYZLocalOrigin);
}

void ECEF2NEDvFloat(const float* rrmLLALocalOrigin,
    const float* mmmXYZTarget,
    int nTargets,
    int isOriginSizeOfTargets,
    float* mmmLocal)
{
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
        mmmLocal[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
        mmmLocal[iTarget + 1] = -sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1];
        mmmLocal[iTarget + 2] = -cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
    }
}

void ECEF2NEDvDouble(const double* rrmLLALocalOrigin,
    const double* mmmXYZTarget,
    int nTargets,
    int isOriginSizeOfTargets,
    double* mmmLocal)
{
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
        mmmLocal[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + cos(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
        mmmLocal[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1];
        mmmLocal[iTarget + 2] = -cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
    }
}

void ECEF2ENUvFloat(const float* rrmLLALocalOrigin,
    const float* mmmXYZTarget,
    int nTargets,
    int isOriginSizeOfTargets,
    float* mmmLocal)
{
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
        mmmLocal[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1];
        mmmLocal[iTarget + 1] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
        mmmLocal[iTarget + 2] = cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + sinf(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
    }
}

void ECEF2ENUvDouble(const double* rrmLLALocalOrigin,
    const double* mmmXYZTarget,
    int nTargets,
    int isOriginSizeOfTargets,
    double* mmmLocal)
{
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
        mmmLocal[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1];
        mmmLocal[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + cos(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
        mmmLocal[iTarget + 2] = cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmXYZTarget[iTarget + 1] + sin(rrmLLALocalOrigin[iOrigin + 0]) * mmmXYZTarget[iTarget + 2];
    }
}

void ENU2ECEFvFloat(const float* rrmLLALocalOrigin,
    const float* mmmTargetLocal,
    int nTargets,
    int isOriginSizeOfTargets,
    float* mmmXYZTarget)
{
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
        mmmXYZTarget[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 1] = cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 2] = cosf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 1] + sinf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2];
    }
}

void NED2ECEFvFloat(const float* rrmLLALocalOrigin,
    const float* mmmTargetLocal,
    int nTargets,
    int isOriginSizeOfTargets,
    float* mmmXYZTarget)
{
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
        mmmXYZTarget[iTarget + 0] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cosf(rrmLLALocalOrigin[iOrigin + 0]) * cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 1] = -sinf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + cosf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cosf(rrmLLALocalOrigin[iOrigin + 0]) * sinf(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 2] = cosf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 0] + -sinf(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2];
    }
}

void NED2ECEFvDouble(const double* rrmLLALocalOrigin,
    const double* mmmTargetLocal,
    int nTargets,
    int isOriginSizeOfTargets,
    double* mmmXYZTarget)
{
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
        mmmXYZTarget[iTarget + 0] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cos(rrmLLALocalOrigin[iOrigin + 0]) * cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 1] = -sin(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 0] + cos(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 1] + -cos(rrmLLALocalOrigin[iOrigin + 0]) * sin(rrmLLALocalOrigin[iOrigin + 1]) * mmmTargetLocal[iTarget + 2];
        mmmXYZTarget[iTarget + 2] = cos(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 0] + -sin(rrmLLALocalOrigin[iOrigin + 0]) * mmmTargetLocal[iTarget + 2];
    }
}

void ENU2ECEFvDouble(const double* rrmLLALocalOrigin,
    const double* mmmTargetLocal,
    int nTargets,
    int isOriginSizeOfTargets,
    double* mmmXYZTarget)
{
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
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
@param int nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void NED2ECEFFloat(const float* rrmLLALocalOrigin,
    const float* mmmTargetLocal,
    int nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mmmXYZTarget)
{
    int nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mmmXYZLocalOrigin = (float*)malloc(nOriginPoints * NCOORDSINPOINT * sizeof(float));
    geodetic2ECEFFloat(rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
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
@param int nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void NED2ECEFDouble(const double* rrmLLALocalOrigin,
    const double* mmmTargetLocal,
    int nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mmmXYZTarget)
{
    int nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mmmXYZLocalOrigin = (double*)malloc(nOriginPoints * NCOORDSINPOINT * sizeof(double));
    geodetic2ECEFDouble(rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
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
@param int nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void ENU2ECEFFloat(const float* rrmLLALocalOrigin,
    const float* mmmTargetLocal,
    int nTargets,
    int isOriginSizeOfTargets,
    float a,
    float b,
    float* mmmXYZTarget)
{
    int nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    float* mmmXYZLocalOrigin = (float*)malloc(nOriginPoints * NCOORDSINPOINT * sizeof(float));
    geodetic2ECEFFloat(rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
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
@param int nPoints Number of target points
@param double a semi-major axis
@param double b semi-minor axis
@param double *mmmXYZTarget array of size nx3 of target point X, Y, Z [m, m, m]
*/
void ENU2ECEFDouble(const double* rrmLLALocalOrigin,
    const double* mmmTargetLocal,
    int nTargets,
    int isOriginSizeOfTargets,
    double a,
    double b,
    double* mmmXYZTarget)
{
    int nOriginPoints = (nTargets - 1) * isOriginSizeOfTargets + 1;
    double* mmmXYZLocalOrigin = (double*)malloc(nOriginPoints * NCOORDSINPOINT * sizeof(double));
    geodetic2ECEFDouble(
        rrmLLALocalOrigin, nOriginPoints, a, b, mmmXYZLocalOrigin);
    int iPoint;
#pragma omp parallel for if (nTargets > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nTargets; ++iPoint) {
        int iTarget = iPoint * NCOORDSINPOINT;
        int iOrigin = iTarget * isOriginSizeOfTargets;
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
@param int nPoints Number of target points
@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void NED2AERFloat(const float* mmmENU, int nPoints, float* rrmAER)
{
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
        rrmAER[i + 0] = atan2f(mmmENU[i + 1], mmmENU[i + 0]);
        if (rrmAER[i + 0] < 0)
            rrmAER[i + 0] = rrmAER[i + 0] + (2.0f * PIf);
        rrmAER[i + 2] = sqrtf(mmmENU[i + 0] * mmmENU[i + 0] + mmmENU[i + 1] * mmmENU[i + 1] + mmmENU[i + 2] * mmmENU[i + 2]);
        rrmAER[i + 1] = asinf(-mmmENU[i + 2] / rrmAER[i + 2]);
    }
}

/*
ENU to AER transformation of float precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf <-
includes additional errors and factors that could be implemented
https://www.lddgo.net/en/coordinate/ecef-enu

@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param int nPoints Number of target points
@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void NED2AERDouble(const double* mmmNED, int nPoints, double* rrmAER)
{
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
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
@param int nPoints Number of target points
@param float *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void ENU2AERFloat(const float* mmmNED, int nPoints, float* rrmAER)
{
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
        rrmAER[i + 0] = atan2f(mmmNED[i + 0], mmmNED[i + 1]);
        if (rrmAER[i + 0] < 0)
            rrmAER[i + 0] = rrmAER[i + 0] + (2.0f * PIf);
        rrmAER[i + 2] = sqrtf(mmmNED[i + 0] * mmmNED[i + 0] + mmmNED[i + 1] * mmmNED[i + 1] + mmmNED[i + 2] * mmmNED[i + 2]);
        rrmAER[i + 1] = asinf(mmmNED[i + 2] / rrmAER[i + 2]);
    }
}

/*
ENU to AER transformation of double precision.
https://x-lumin.com/wp-content/uploads/2020/09/Coordinate_Transforms.pdf
https://www.lddgo.net/en/coordinate/ecef-enu

@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
@param int nPoints Number of target points
@param double *rrmAER array of size nx3 of target point azimuth, elevation,
range [rad, rad, m]
*/
void ENU2AERDouble(const double* mmmENU, int nPoints, double* rrmAER)
{
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
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
@param int nPoints Number of target points
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2NEDFloat(const float* rrmAER, int nPoints, float* mmmNED)
{
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
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
@param int nPoints Number of target points
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2NEDDouble(const double* rrmAER, int nPoints, double* mmmNED)
{
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
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
@param int nPoints Number of target points
@param float *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2ENUFloat(const float* rrmAER, int nPoints, float* mmmENU)
{
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
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
@param int nPoints Number of target points
@param double *mmmLocal array of size nx3 X, Y, Z [m, m, m]
*/
void AER2ENUDouble(const double* rrmAER, int nPoints, double* mmmENU)
{
    int iPoint;
#pragma omp parallel for if (nPoints > omp_get_num_procs() * THREADING_CORES_MULTIPLIER)
    for (iPoint = 0; iPoint < nPoints; ++iPoint) {
        int i = iPoint * NCOORDSINPOINT;
        mmmENU[i + 0] = cos(rrmAER[i + 1]) * sin(rrmAER[i + 0]) * rrmAER[i + 2];
        mmmENU[i + 1] = cos(rrmAER[i + 1]) * cos(rrmAER[i + 0]) * rrmAER[i + 2];
        mmmENU[i + 2] = sin(rrmAER[i + 1]) * rrmAER[i + 2];
    }
}

static PyObject*
geodetic2ECEFWrapper(PyObject* self, PyObject* args)
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
    if ((PyArray_SIZE(rrmLLA) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArray) / NCOORDSINPOINT;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        geodetic2ECEFDouble((double*)PyArray_DATA(inArray), nPoints, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        geodetic2ECEFFloat((float*)PyArray_DATA(inArray), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
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
    PyArrayObject* mmmXYZ;
    double a, b;

    // checks
    if (!PyArg_ParseTuple(args, "O!dd", &PyArray_Type, &mmmXYZ, &a, &b))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(mmmXYZ))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(mmmXYZ) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArray) / NCOORDSINPOINT;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2geodeticDouble((double*)PyArray_DATA(inArray), nPoints, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2geodeticFloat((float*)PyArray_DATA(inArray), nPoints, (float)(a), (float)(b), (float*)PyArray_DATA(result_array));
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
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmXYZTarget)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmXYZTarget)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSINPOINT) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmXYZTarget))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSINPOINT) != 0 || (PyArray_SIZE(mmmXYZTarget) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSINPOINT;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2ENUDouble(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2ENUFloat(
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
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmXYZTarget)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmXYZTarget)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSINPOINT) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmXYZTarget))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSINPOINT) != 0 || (PyArray_SIZE(mmmXYZTarget) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSINPOINT;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2NEDDouble(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2NEDFloat(
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
ECEF2NEDvWrapper(PyObject* self, PyObject* args)
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
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmXYZTarget)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmXYZTarget)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSINPOINT) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmXYZTarget))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSINPOINT) != 0 || (PyArray_SIZE(mmmXYZTarget) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSINPOINT;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2NEDvDouble(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2NEDvFloat(
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
ECEF2ENUvWrapper(PyObject* self, PyObject* args)
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
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmXYZTarget)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmXYZTarget)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSINPOINT) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmXYZTarget))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSINPOINT) != 0 || (PyArray_SIZE(mmmXYZTarget) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSINPOINT;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ECEF2ENUvDouble(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ECEF2ENUvFloat(
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
NED2ECEFWrapper(PyObject* self, PyObject* args)
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
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmLocal)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmLocal)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSINPOINT) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmLocal))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSINPOINT) != 0 || (PyArray_SIZE(mmmLocal) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSINPOINT;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        NED2ECEFDouble(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        NED2ECEFFloat(
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
ENU2ECEFWrapper(PyObject* self, PyObject* args)
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
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmLocal)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmLocal)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSINPOINT) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmLocal))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSINPOINT) != 0 || (PyArray_SIZE(mmmLocal) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSINPOINT;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ENU2ECEFDouble(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, a, b, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ENU2ECEFFloat(
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
ENU2ECEFvWrapper(PyObject* self, PyObject* args)
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
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmLocal)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmLocal)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSINPOINT) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmLocal))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSINPOINT) != 0 || (PyArray_SIZE(mmmLocal) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSINPOINT;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ENU2ECEFvDouble(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ENU2ECEFvFloat(
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
    if (!((PyArray_NDIM(rrmLLALocalOrigin) == PyArray_NDIM(mmmLocal)) && (PyArray_SIZE(rrmLLALocalOrigin) == PyArray_SIZE(mmmLocal)) || ((PyArray_Size((PyObject*)rrmLLALocalOrigin) == NCOORDSINPOINT) && (PyArray_SIZE(rrmLLALocalOrigin) < PyArray_SIZE(mmmLocal))))) {
        PyErr_SetString(PyExc_ValueError,
            "Input arrays must have matching size and dimensions or "
            "the origin must be of size three.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmLLALocalOrigin) % NCOORDSINPOINT) != 0 || (PyArray_SIZE(mmmLocal) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArrayLocal) / NCOORDSINPOINT;
    int isOriginSizeOfTargets = (PyArray_Size((PyObject*)inArrayOrigin) == PyArray_Size((PyObject*)inArrayLocal));

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        NED2ECEFvDouble(
            (double*)PyArray_DATA(inArrayOrigin), (double*)PyArray_DATA(inArrayLocal), nPoints, isOriginSizeOfTargets, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        NED2ECEFvFloat(
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
ENU2AERWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* mmmENU;

    // checks
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &mmmENU))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(mmmENU))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(mmmENU) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArray) / NCOORDSINPOINT;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        ENU2AERDouble((double*)PyArray_DATA(inArray), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        ENU2AERFloat((float*)PyArray_DATA(inArray), nPoints, (float*)PyArray_DATA(result_array));
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
    PyArrayObject* mmmNED;

    // checks
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &mmmNED))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(mmmNED))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(mmmNED) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArray) / NCOORDSINPOINT;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        NED2AERDouble((double*)PyArray_DATA(inArray), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        NED2AERFloat((float*)PyArray_DATA(inArray), nPoints, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
AER2NEDWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* rrmAER;

    // checks
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &rrmAER))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmAER))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmAER) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArray) / NCOORDSINPOINT;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        AER2NEDDouble((double*)PyArray_DATA(inArray), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        AER2NEDFloat((float*)PyArray_DATA(inArray), nPoints, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
}

static PyObject*
AER2ENUWrapper(PyObject* self, PyObject* args)
{
    PyArrayObject* rrmAER;

    // checks
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &rrmAER))
        return NULL;
    if (!(PyArray_ISCONTIGUOUS(rrmAER))) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must be a C contiguous.");
        return NULL;
    }
    if ((PyArray_SIZE(rrmAER) % NCOORDSINPOINT) != 0) {
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
    int nPoints = (int)PyArray_SIZE(inArray) / NCOORDSINPOINT;

    // run function
    switch (PyArray_TYPE(result_array)) {
    case NPY_DOUBLE:
        AER2ENUDouble((double*)PyArray_DATA(inArray), nPoints, (double*)PyArray_DATA(result_array));
        break;
    case NPY_FLOAT:
        AER2ENUFloat((float*)PyArray_DATA(inArray), nPoints, (float*)PyArray_DATA(result_array));
        break;
    default:
        PyErr_SetString(PyExc_ValueError,
            "Only 32 and 64 bit float types or all integer are accepted.");
        return NULL;
    }
    return (PyObject*)result_array;
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
