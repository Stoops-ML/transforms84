# transforms84
Small geographic coordinate systems Python library with a few additional helper functions.

See the Jupyter notebooks in [examples](examples) to see how to use the transform84.

## Installation
`pip install transforms84`

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
