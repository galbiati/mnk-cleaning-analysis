#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wei Ji Ma Lab
# By: Gianni Galbiati

# Standard libraries

# External libraries
from pymc3 import Model
from pymc3 import Deterministic, HalfStudentT, Normal, Poisson

from theano import tensor as T

# Internal libraries


def build_poisson_model(exogenous_df, endogenous_values):
    """Build a Poisson regression model

    For replacing Chi-Sq tests
    (see Krushke DBA chapter 22)
    (pymc3 example: https://docs.pymc.io/notebooks/GLM-poisson-regression.html)

    TODO: even though I've tried to make this pretty general,
          it's still for a 2x3 design unfortunately
          revisit this to generalize to an NxM design someday!

    Arguments
    ---------
    exogenous_df : pandas.DataFrame
        a dataframe containing indicators for the exogenous variables
        this df is expected to be produced using patsy

    endogenous_values : numpy.ndarray
        an array of count data corresponding to rows of exogenous_df

    Returns
    -------
    model : pymc3.Model()
        pymc model object
    """

    with Model() as model:
        # Create placeholder lists for PyMC RVs
        sds = []
        coeffs = []

        # Create intercept coefficient; no prior on variance
        b0 = Normal('intercept', mu=0, sd=10)
        coeffs.append(b0)

        # Create a coefficient for all other indicators with prior on variance
        for c in exogenous_df.columns[1:]:
            # Theano hates ":"
            variable_name = c.replace(':', '_')

            # Prior on coefficient variance/precision
            sd = HalfStudentT(f'sd_{variable_name}', nu=2, lam=.001)

            # Unnormalized coefficient
            b = Normal(f'b_{variable_name}', mu=0, tau=1 / sd ** 2)

            sds.append(sd)
            coeffs.append(b)

        # Coefficients for each type of indicator
        # must sum to zero for identifiability,
        # so final coefficient can be computed from the other two
        f = exogenous_df.columns[-1].replace(':', '_')
        f0 = f.split('[')[0]
        f1 = f.split(']')[1]
        f = '[T.1]'.join([f0, f1])

        b6 = Deterministic(f'b_{f}', -coeffs[-2] - coeffs[-1])

        # We're using Laplacian approximation to prior,
        # so softmax indicators with coefficients
        theta = T.dot(exogenous_df.values, T.stack(coeffs))
        poisson_mu = T.exp(theta)

        y = Poisson('y', mu=poisson_mu, observed=endogenous_values)

    return model
