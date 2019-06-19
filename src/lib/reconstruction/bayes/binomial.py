#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Wei Ji Ma Lab
# By: Gianni Galbiati

# Standard libraries

# External libraries
import numpy as np

from pymc3 import Model
from pymc3 import Bernoulli, Beta, Deterministic, Gamma

# Internal libraries


def build_binomial_model(data_dict):
    """Builds hierarchical Bayesian model for binomial group differences

    From Krushke's textbook
    TODO: page number and further explanation when you have time

    Models using Bernoulli outcome instead of binomial
    (I believe there are supposed to be advantages to this when using NUTS)
    TODO: are there really?

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

    # Unpack data dict for easier reading
    group0_name, group1_name = list(data_dict.keys())
    data0 = data_dict[group0_name]
    x0 = data0['x']
    y0 = data0['y']

    data1 = data_dict[group1_name]
    x1 = data1['x']
    y1 = data1['y']

    with Model() as model:
        # Population prior
        # In hierarchiacl binomial estimation,
        # group level means/variances will be drawn from a global prior

        # Priors for population means
        mu = Beta('mu', alpha=1, beta=1)
        kappa = Gamma('kappa', mu=10, sd=10)

        # Priors for population variances
        shape = Gamma('shape', mu=10, sd=10)
        rate = Gamma('rate', mu=10, sd=10)

        # Per-group priors
        # Mean and variance for each subject will be drawn from group priors
        mu0 = Beta(f'mu_{group0_name}',
                   alpha=mu * kappa, beta=(1 - mu) * kappa)

        kappa0 = Gamma(f'kappa_{group0_name}', alpha=shape, beta=rate)

        mu1 = Beta(f'mu_{group1_name}',
                   alpha=mu * kappa, beta=(1 - mu) * kappa)

        kappa1 = Gamma(f'kappa_{group1_name}', alpha=shape, beta=rate)

        # Per-individual priors
        # Error rates for each individual will be drawn from individual priors
        # Shape provides number of individuals
        p0 = Beta(f'p_{group0_name}',
                  alpha=mu0 * kappa0,
                  beta=(1 - mu0) * kappa0,
                  shape=len(np.unique(x0)))

        p1 = Beta(f'p_{group1_name}',
                  alpha=mu1 * kappa1,
                  beta=(1 - mu1) * kappa1,
                  shape=len(np.unique(x1)))

        # Likelihoods
        # Index probability variables with observed individual IDs
        targets0 = Bernoulli('targets0', p=p0[x0], observed=y0)
        targets1 = Bernoulli('targets1', p=p1[x1], observed=y1)

        # Measures
        difference_in_means = Deterministic('difference in means', mu0 - mu1)

        difference_in_variances = Deterministic('difference in variances',
                                                kappa0 - kappa1)

        effect_size = difference_in_means / np.sqrt((kappa0 + kappa1) / 2)
        effect_size = Deterministic('effect size', effect_size)

        return model
