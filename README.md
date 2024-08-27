# transforms84
![PyPI - Version](https://img.shields.io/pypi/v/transforms84)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/transforms84)
![Codecov](https://img.shields.io/codecov/c/gh/Stoops-ML/transforms84)
![PyPI - License](https://img.shields.io/pypi/l/transforms84)

Small geographic coordinate systems Python library with a few additional helper functions.

This package focuses on performance, correct matrix shapes, single to many point functions and many to many point functions

This package focuses on:
1. Performance
2. Input and output coordinates of ideal mathematical shapes. Ideally, all coordinates should be of shapes (3,1) or (nPoints,3,1), but shapes (3,) and (nPoints,3) are supported too.
3. Functions that adapt to differing input matrices shapes: one-to-one, many-to-many and one-to-many points. See [here](examples/example1.ipynb) for an example.

## Installation
`pip install transforms84`

## Examples
See the Jupyter notebooks in [examples](examples) to see how to use the transform84. Run `pip install transforms84[examples]` to run the examples locally.

## Operations
### Transformations
The following transformations have been implemented:
- geodetic &rarr; ECEF
- ECEF &rarr; geodetic
- ECEF &rarr; ENU
- ENU &rarr; ECEF
- ENU &rarr; AER
- AER &rarr; ENU

### Distances
The following distance formulae have been implemented:
- Haversine

### Helpers
The following functions have been implemented:
- Angular difference (smallest and largest)
- [rad, rad, X] &rarr; [deg, deg, X]
- [deg, deg, X] &rarr; [rad, rad, X]

## Contributing
PRs are always welcome and appreciated!
