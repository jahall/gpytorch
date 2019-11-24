#!/usr/bin/env python3

import math

import torch

from ..constraints import Positive
from .kernel import Kernel


class ChangePointsKernel(Kernel):
    r"""
    The ChangePoints kernel defines a fixed number of change-points along a 1d
    input space where different kernels govern different parts of the space.

    The kernel is by multiplication and addition of the base kernels with
    sigmoid functions (:math:`\sigma`). A single change-point kernel is defined as:

    .. math::

       \begin{equation*}
          k_{\text{CP}}(x_1, x_2) =
          k_1(x_1, x_2) (1 - \sigma(x_1)) (1 - \sigma(x_2))
          + k_2(x_1, x_2) \sigma(x_1) \sigma(x_2)
       \end{equation*}

    where :math:`k_1` is deactivated around the change-point and :math:`k_2` is activated.
    The single change-point version can be found in `Automatic Construction and
    Natural-language Description of Nonparametric Regression Models`_. Each sigmoid
    is a logistic function defined as:

    .. math::

       \begin{equation*}
          \sigma(x) = \frac{1}{1 + \exp\right(-s(x - x_0)\left)}
       \end{equation*}

    parameterized by location :math:`x_0` and steepness "s".

    .. note::
        This kernel can only operate on one dimensional data, therefore either the dimension
        of the input must be one or :attr:`active_dims` must be a tuple of one int.

    Args:
        :attr:`active_dims` (tuple of ints, optional):
            Set this to specify which dimension to apply the kernel to. If unspecified the dimension
            of the input must be one. Default: `None`.
        :attr:`locations_prior` (Prior, optional):
            Set this if you want to apply a prior to the locations parameter.  Default: `None`.
        :attr:`locations_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the value of the locations. Default: `None`.
        :attr:`steepness_prior` (Prior, optional):
            Set this if you want to apply a prior to the steepness parameter.  Default: `None`.
        :attr:`steepness_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the value of the steepness. Default: `Positive`.

    Attributes:
        :attr:`locations` (Tensor):
            The locations parameter. Size = `*batch_shape x n_cp x 1`.
        :attr:`steepness` (Tensor):
            The steepness parameter. Size = `*batch_shape x n_cp x 1`.

    .. _Automatic Construction and Natural-language Description of Nonparametric Regression Models:
        https://arxiv.org/abs/1402.4304.pdf
    """

    def __init__(
        self,
        locations_prior=None,
        locations_constraint=None,
        steepness_prior=None,
        steepness_constraint=None,
        active_dims=None,
        **kwargs,
    ):
        if active_dims is not None and len(active_dims) != 1:
            raise ValueError("ChangePointsKernel can only operate on one dimension")

        super(ChangePointsKernel, self).__init__(active_dims=active_dims, **kwargs)
        if steepness_constraint is None:
            steepness_constraint = Positive()

        # TODO...do the rest
        # self.register_parameter(
        #    name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        # )

        # if period_length_prior is not None:
        #    self.register_prior(
        #        "period_length_prior",
        #        period_length_prior,
        #        lambda: self.period_length,
        #        lambda v: self._set_period_length(v),
        #    )

        # self.register_constraint("raw_period_length", period_length_constraint)

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value):
        self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)
        self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.period_length)
        x2_ = x2.div(self.period_length)
        diff = self.covar_dist(x1_, x2_, diag=diag, **params)
        res = torch.sin(diff.mul(math.pi)).pow(2).mul(-2 / self.lengthscale).exp_()
        if diff.ndimension() == 2 or diag:
            res = res.squeeze(0)
        return res
