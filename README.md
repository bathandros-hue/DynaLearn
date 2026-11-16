# DynaLearn

Identify damping and stiffness in a single-degree-of-freedom mechanical oscillator using measured force, displacement, velocity and acceleration. Includes a fourth-order Runge–Kutta simulator, physically constrained least squares, residual metrics, and bootstrap uncertainty estimates.

## Quickstart

Python 3.10+ and NumPy >=1.26:

    python -m unittest discover -s tests -v
    python -m dynalearn.cli
    python -m dynalearn.cli --export-demo simulated.csv
    python -m dynalearn.cli --csv simulated.csv --mass 2

The default demo synthesizes motion using m=2 kg, c=1.2 N*s/m, k=25 N/m, with measurement noise added to acceleration.

## Governing equation

    m*x_ddot + c*x_dot + k*x = F(t)
    F(t) - m*x_ddot = [x_dot, x] * [c, k]

Ordinary least squares fits damping c and stiffness k. The default constrained solution enforces c >= 0 and k >= 0, enumerating valid active-set boundaries for the two-parameter problem.

## CSV format

Required columns, SI units:

- displacement (m)
- velocity (m/s)
- acceleration (m/s^2)
- force (N)

Known mass is supplied separately in kg. The time column is optional for fitting. Signals should be synchronized and have meaningful excitation.

## Uncertainty caveat

The i.i.d. paired bootstrap used for illustration does not account for time-series autocorrelation, errors in displacement or velocity, or systematic sensor bias. Its confidence intervals can be optimistic; consider block bootstrap or a full state-space estimator for experimental data.

## Limitations

Single degree of freedom, known constant mass, linear model, directly observed acceleration. Identifiability requires sufficiently independent displacement and velocity. No nonlinear friction, drift estimation, or sensor fusion.

## GitHub topics

system-identification, mechanical-engineering, physics-informed, numerical-methods, dynamics, python

## License

MIT; see LICENSE.
