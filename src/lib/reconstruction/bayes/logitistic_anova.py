#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wei Ji Ma Lab
# By: Gianni Galbiati

# Standard libraries

# External libraries
import numpy as np

from theano import Tensor as T

from pymc3 import Model
from pymc3 import Bernoulli, Beta, Deterministic, Gamma, HalfStudentT, Normal

# Internal libraries


def build_logistic_anova_model(exogenous_df, endogenous_values):
    """Build a Bayesian logistic ANOVA model.

    From Krushke's textbook, p 457

    Model spec:
    (! used to indicate hyperparamter)

    z ~ Bernoulli(theta)
    theta ~ Beta(alpha, beta)
    alpha = mu * kappa
    beta = (1 - mu) * kappa
    k ~ Gamma(scale!, rate!)
    mu = sigmoid(BX + I)
    I ~ Normal(mu_I, sigma_I)
    B ~ Normal(0, sigma_B)
    sigma_B ~ HalfT(param!)

    Arguments
    ---------
    data_dict : dict
        {'group0_name': {'x': <numpy data for individual IDs>,
                        {'y': <numpy data for target variables>},
         'group1_name': <etc>}

    Returns
    -------
    model : pymc3.Model
        model object ready for sampling
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

        # Estimate mean of beta prior as sigmoid
        mu = T.nnet.sigmoid(T.dot(exogenous_df.values, T.stack(coeffs)))
        mu = Deterministic('mu', mu)

        kappa = Gamma('beta_variance', shape=10, rate=10)
        alpha = Deterministic('alpha', mu * kappa)
        beta = Deterministic('beta', (1 - mu) * kappa)

        theta = Beta('beta', alpha=alpha, beta=beta)
        y = Bernoulli('tagets', p=theta, observed=endogenous_values)

        # Measures


        return model

