#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at:
# https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import logging

from cil.optimisation.operators import (
    LinearOperator,
    IdentityOperator,
    BlockOperator,
    DiagonalOperator,
)

log = logging.getLogger(__name__)

VALID_FORMS = ('auto', 'standard', 'block')


def _geometry_size(geometry):
    """Total number of elements in a geometry, block-aware."""
    if hasattr(geometry, 'geometries'):
        return sum(_geometry_size(g) for g in geometry.geometries)
    size = 1
    for length in geometry.shape:
        size *= length
    return size


def _is_standard_form_capable(struct_operator):
    r"""
    True when :math:`L^{-1}` exists *and* equals :math:`L^*`.

    ``is_orthogonal()`` alone is not enough: it reports a property of the
    wavelet filter bank, not of the operator as a map between the two
    geometries, and a ``WaveletOperator`` that pads at the boundary is a *tall
    isometry* — :math:`L^*L = I`, so :math:`L^*` is a left inverse, but
    :math:`LL^* \neq I` and no true inverse exists. The standard form applies
    :math:`(WL)^{-1}` to a solver iterate that is a general element of
    ``Range(L)``, not one known to lie in the image of :math:`L`, so a left
    inverse is not sufficient. Requiring ``r == m`` on top of orthogonality is
    what makes :math:`L` invertible, which in practice admits :math:`L = I`
    and orthogonal wavelets with ``bnd_cond='periodization'``.
    """
    if struct_operator is None:
        return True
    if not struct_operator.is_orthogonal():
        return False
    return (_geometry_size(struct_operator.domain_geometry())
            == _geometry_size(struct_operator.range_geometry()))


def resolve_form(form='auto', solver=None, struct_operator=None,
                 weighted=False, zero_initial=True):
    r"""
    Answer ``form='auto'``: ``'standard'`` or ``'block'``.

    The rule :func:`create_tikhonov_operator` applies, exposed on its own so
    that it can be asked without building an operator -- the answer depends on
    :math:`L` and on the solver, not on the size of anything. An explicit
    ``'standard'`` or ``'block'`` is returned unchanged.

    Parameters
    ----------
    form : {'auto', 'standard', 'block'}, default 'auto'
        The form asked for.
    solver : class or Algorithm, optional
        The solver that will iterate on the result, read only for its
        ``warm_starts_in_standard_form``. That is a class attribute rather than
        the derived ``supports_warm_start`` property precisely so the question
        can be asked before an instance exists, so the class will do as well as
        an instance. ``None`` assumes the permissive answer.
    struct_operator : LinearOperator, optional
        :math:`L`. ``None`` means the identity.
    weighted : bool, default False
        Whether an IRLS reweighting loop is coming.
    zero_initial : bool, default True
        Whether the caller starts from zero.

    Returns
    -------
    {'standard', 'block'}
        Which form to build. Never ``'auto'``.

    Examples
    --------
    >>> from cil.optimisation.algorithms import CGLS, LSQR
    >>> resolve_form('auto', LSQR, weighted=True)
    'block'
    >>> resolve_form('auto', CGLS, weighted=True)
    'standard'
    """
    if form not in VALID_FORMS:
        raise ValueError(
            "form must be one of {}, got {!r}.".format(VALID_FORMS, form))

    if form != 'auto':
        return form

    if not _is_standard_form_capable(struct_operator):
        log.info("form='auto' resolved to 'block': (WL)^-1 does not exist.")
        return 'block'

    if getattr(solver, 'warm_starts_in_standard_form', True):
        # The standard form is the cheaper one, and a solver that runs on the
        # true regularised normal equations is free to use it from anywhere,
        # IRLS included.
        log.info("form='auto' resolved to 'standard'.")
        return 'standard'

    if weighted:
        # IRLS re-solves from the previous outer iterate, and this solver damps
        # the step rather than the solution, so only the block form imposes the
        # penalty on what is actually being minimised.
        log.info("form='auto' resolved to 'block': IRLS warm starts, which "
                 "this solver does not support in the standard form.")
        return 'block'

    if zero_initial:
        # Damping the step and damping the solution coincide at x_0 = 0.
        log.info("form='auto' resolved to 'standard': the solve starts from "
                 "zero.")
        return 'standard'

    log.info("form='auto' resolved to 'block': a non-zero start, which this "
             "solver does not support in the standard form.")
    return 'block'


def create_tikhonov_operator(operator, solution_geometry=None,
                             struct_operator=None, regalpha=0.0, form='auto',
                             weighted=False, solver=None, zero_initial=True,
                             tmp_domain=None, tmp_range_struct=None):
    r"""
    Build the operator :math:`K` that a Krylov solver iterates on.

    The generalised Tikhonov problem

    .. math::

        \min_u \|A u - b\|_2^2 + \alpha^2 \|W L u\|_2^2

    can be handed to LSQR or CGLS in more than one way, and the choices are not
    interchangeable. This factory owns that decision, so the solvers do not have
    to.

    Parameters
    ----------
    operator : LinearOperator
        The forward model :math:`A`.
    solution_geometry : cil geometry, optional
        The physical solution space. Defaults to ``operator.domain_geometry()``.
    struct_operator : LinearOperator, optional
        The structural operator :math:`L`. ``None`` means the identity.
    regalpha : float, default 0.0
        The regularisation parameter :math:`\alpha`. Zero means no
        regularisation, and the bare ``operator`` is returned unwrapped.
    form : {'auto', 'standard', 'block'}, default 'auto'
        Which change of variable to use. See the table below.
    weighted : bool, default False
        Whether to allocate the diagonal weight operator :math:`W` up front.
        The alternative is :meth:`WeightedStructOperator.enable_weights`
        afterwards.
    solver, zero_initial
        Passed to :func:`resolve_form`, and so consulted only by
        ``form='auto'``.
    tmp_domain, tmp_range_struct : DataContainer, optional
        Scratch buffers to borrow instead of allocating. Leave as ``None`` and
        the operator allocates only what its configuration actually needs.

    Returns
    -------
    LinearOperator
        ``operator`` itself when ``regalpha == 0``, otherwise a
        :class:`TikhonovOperator` or a :class:`BlockTikhonovOperator`. The
        result carries a ``form`` attribute; use ``getattr(op, 'form', 'none')``
        to cover the unwrapped case.

    Notes
    -----
    Dispatch:

    ==============  ==============  ========================  ==========================  ==============================
    ``regalpha``    ``form``        :math:`L`                 ``solver``                  result
    ==============  ==============  ========================  ==========================  ==============================
    ``== 0``        any             any                       any                         ``operator``, unwrapped
    ``> 0``         ``'auto'``      otherwise                 any                         :class:`BlockTikhonovOperator`
    ``> 0``         ``'auto'``      ``None`` or invertible    ``CGLS``, or ``None``       :class:`TikhonovOperator`
    ``> 0``         ``'auto'``      ``None`` or invertible    ``LSQR``, zero start, no W  :class:`TikhonovOperator`
    ``> 0``         ``'auto'``      ``None`` or invertible    ``LSQR``, otherwise         :class:`BlockTikhonovOperator`
    ``> 0``         ``'block'``     any                       any                         :class:`BlockTikhonovOperator`
    ``> 0``         ``'standard'``  ``None`` or invertible    any                         :class:`TikhonovOperator`
    ``> 0``         ``'standard'``  otherwise                 any                         ``ValueError``
    ==============  ==============  ========================  ==========================  ==============================

    The standard form is the cheaper of the two — one space of size :math:`r`
    rather than a stack of size :math:`m + r`, collapsing to :math:`K = A` for
    :math:`L = I` — so ``'auto'`` prefers it wherever it is correct, which
    takes an invertible :math:`L` (:func:`_is_standard_form_capable`) and a
    solver that stays correct there from where the caller starts
    (:func:`resolve_form`). Everything ``'auto'`` excludes is *handled*, not
    rejected: the block form imposes the same penalty
    :math:`\alpha^2\|WLu\|^2` without ever forming an inverse. Only an explicit
    ``form='standard'`` raises.
    """
    if form not in VALID_FORMS:
        raise ValueError(
            "form must be one of {}, got {!r}.".format(VALID_FORMS, form))

    if regalpha < 0:
        raise ValueError(
            "regalpha must be non-negative, got {}.".format(regalpha))

    if solution_geometry is None:
        solution_geometry = operator.domain_geometry()

    if regalpha == 0:
        if form != 'auto':
            log.info("regalpha is 0, so form=%r is ignored and the operator is "
                     "used unregularised.", form)
        return operator

    form = resolve_form(form, solver, struct_operator=struct_operator,
                        weighted=weighted, zero_initial=zero_initial)

    if form == 'standard':
        return TikhonovOperator(
            operator,
            solution_geometry,
            struct_operator=struct_operator,
            regalpha=regalpha,
            weighted=weighted,
            tmp_domain=tmp_domain,
            tmp_range_struct=tmp_range_struct,
        )

    return BlockTikhonovOperator(
        operator,
        solution_geometry,
        struct_operator=struct_operator,
        regalpha=regalpha,
        weighted=weighted,
        tmp_domain=tmp_domain,
        tmp_range_struct=tmp_range_struct,
    )


class WeightedStructOperator(LinearOperator):
    r"""
    Weighted structural operator for IRLS.

    Composes a structural regularisation operator :math:`L` with a diagonal
    weight operator :math:`W`:

    .. math::

        L_{\text{weighted}} = W L.

    During IRLS the weights approximate an L1 penalty:

    .. math::

        w_k = (|L u_{k-1}|^2 + \tau_k^2)^{-1/4}.

    For a block-valued structural operator, such as a gradient operator, the
    weights live in ``Range(L)`` and are therefore block-valued. This is handled
    by the block-aware ``DiagonalOperator``.

    Parameters
    ----------
    domain_geometry : cil geometry
        The physical solution space, ``Domain(L)``.
    struct_operator : LinearOperator, optional
        :math:`L`. ``None`` means the identity, and is treated as a genuinely
        absent stage rather than as a multiplication by one: it removes both an
        operator call and a staging buffer.
    weighted : bool, default False
        Allocate :math:`W` immediately. Equivalent to calling
        :meth:`enable_weights` straight after construction.
    tmp_range_struct : DataContainer, optional
        Scratch in ``Range(L)`` to borrow instead of allocating.

    Notes
    -----
    **Unweighted mode.** With ``weighted=False`` the weight operator is ``None``
    and every method skips the :math:`W` stage entirely, rather than multiplying
    by an identity.

    **Two of the four maps are conditional.** :meth:`direct` and :meth:`adjoint`
    work for any :math:`L`. :meth:`inverse` and :meth:`inverse_adjoint` apply
    :math:`L^{-1}` as :math:`L^*` and so require an orthogonal :math:`L`; they
    raise otherwise. Check with :meth:`is_invertible` before calling. The block
    form uses only the first two, which is why it accepts any :math:`L`.

    Geometries
    ----------
    Domain:
        Domain(L), the physical solution space.

    Range:
        Range(L), the weighted structure space.
    """

    def __init__(self, domain_geometry, struct_operator=None, weighted=False,
                 tmp_range_struct=None):
        # An absent L and an explicit IdentityOperator are the same thing, and
        # both let us skip a stage. Keep a real operator around for the paths
        # that want to call it uniformly, but remember which case we are in.
        self.struct_is_identity = (struct_operator is None
                                   or isinstance(struct_operator,
                                                 IdentityOperator))
        if struct_operator is None:
            struct_operator = IdentityOperator(domain_geometry)

        self.struct_operator = struct_operator
        self.weight_operator = None
        self.tmp_range_struct = tmp_range_struct

        super(WeightedStructOperator, self).__init__(
            domain_geometry=domain_geometry,
            range_geometry=struct_operator.range_geometry(),
        )

        if weighted:
            self.enable_weights()

    # ------------------------------------------------------------------ #
    # weights
    # ------------------------------------------------------------------ #

    def enable_weights(self):
        r"""
        Allocate :math:`W`, and the staging buffer :math:`WL` needs, once.

        Idempotent. This is the only allocation in the whole solve that happens
        outside ``set_up``: IRLS is attached to an already-constructed inner
        solver, so it cannot pass ``weighted=True`` down into the factory. It
        calls this instead, once, and mutates the weights in place from then on.

        Returns
        -------
        DataContainer
            The mutable weights container, filled with ones.
        """
        if self.weight_operator is not None:
            return self.weights

        range_geometry = self.range_geometry()
        self.weight_operator = DiagonalOperator(
            range_geometry.allocate(1.0),
            domain_geometry=range_geometry,
        )

        # Only a genuine two-stage WL needs an intermediate. Either factor being
        # the identity collapses the composition to one stage.
        if self.tmp_range_struct is None and not self.struct_is_identity:
            self.tmp_range_struct = range_geometry.allocate(0)

        return self.weights

    @property
    def is_weighted(self):
        """True once :math:`W` exists."""
        return self.weight_operator is not None

    @property
    def weights(self):
        """
        The mutable diagonal weights container, living in ``Range(L)``.

        ``None`` in unweighted mode. Block-valued for gradient-like operators.
        Modify in place (``fill``); rebinding the returned object has no effect
        on the operator.
        """
        if self.weight_operator is None:
            return None
        return self.weight_operator.diagonal

    @staticmethod
    def _fill_in_place(target, values):
        """
        Recurse to the leaves before filling, at whatever depth.

        ``BlockDataContainer.fill`` is not itself recursive: given a scalar it
        silently leaves a nested child untouched. A single level of descent
        covers a flat Range(L) -- the gradient's -- but not the leaves below
        it when :math:`L` is a stack of struct operators and Range(L) is a
        block of blocks, so descend all the way.
        """
        if not hasattr(target, "containers"):
            target.fill(values)
            return
        source_is_block = hasattr(values, "containers")
        for i in range(len(target)):
            WeightedStructOperator._fill_in_place(
                target[i], values[i] if source_is_block else values)

    def set_weights(self, values):
        """
        Copy ``values`` into the weights in place, block-aware.

        ``values`` may be a scalar, a leaf container, or a container matching
        the shape of Range(L).
        """
        target = self.weights
        if target is None:
            raise RuntimeError(
                "This operator is unweighted. Call enable_weights() first.")
        self._fill_in_place(target, values)
        return target

    # ------------------------------------------------------------------ #
    # capability
    # ------------------------------------------------------------------ #

    def struct_is_orthogonal(self):
        r"""
        True when :math:`L^*L = I`, as :math:`L` itself reports it.

        Note that this is weaker than invertibility for a non-square :math:`L`.
        Use :meth:`is_invertible` to decide whether :meth:`inverse` may be
        called.
        """
        return self.struct_operator.is_orthogonal()

    def is_invertible(self):
        r"""
        True when :meth:`inverse` and :meth:`inverse_adjoint` can be used.

        :math:`W` is a strictly positive diagonal and so always invertible,
        which leaves :math:`L`. The only inverse this class can form is
        :math:`L^*` — no operator in CIL exposes an ``inverse()`` method — so
        :math:`L` must be orthogonal *and* square, not merely a tall
        isometry. See :func:`_is_standard_form_capable`.
        """
        return _is_standard_form_capable(self.struct_operator)

    def _require_invertible(self, what):
        if self.is_invertible():
            return

        name = type(self.struct_operator).__name__
        if not self.struct_is_orthogonal():
            reason = ("{} reports is_orthogonal() == False. CIL's biorthogonal "
                      "wavelets (bior*, rbio*) fall here: they are invertible, "
                      "but L^-1 != L^* and CIL exposes no inverse() to use "
                      "instead.".format(name))
        else:
            reason = (
                "{} is orthogonal but not square: it maps {} elements to {}. "
                "L^*L = I, so L^* is only a left inverse, and LL^* != I. For a "
                "WaveletOperator this is boundary padding; bnd_cond="
                "'periodization' keeps the coefficient space the same size as "
                "the image and does give a true inverse.".format(
                    name,
                    _geometry_size(self.struct_operator.domain_geometry()),
                    _geometry_size(self.struct_operator.range_geometry())))

        raise ValueError(
            "{} needs (W L)^-1, which is formed as L^* W^-1 and so requires L "
            "to be orthogonal and square. {} Use form='block', which never "
            "forms an inverse.".format(what, reason))

    def is_identity(self):
        r"""True when :math:`WL = I`, so the operator can be skipped outright."""
        return self.struct_is_identity and not self.is_weighted

    # ------------------------------------------------------------------ #
    # the four maps
    # ------------------------------------------------------------------ #

    def direct(self, x, out=None):
        r"""
        Return :math:`W L x`.

        Never needs staging: both stages can write through ``out``, because
        ``W`` maps ``Range(L)`` to itself.
        """
        if out is None:
            out = self.range_geometry().allocate(0)

        self.struct_operator.direct(x, out=out)
        if self.weight_operator is not None:
            self.weight_operator.direct(out, out=out)

        return out

    def adjoint(self, x, out=None):
        r"""
        Return :math:`L^* W^* x`.

        Needs staging only when both factors are non-trivial, since ``W`` acts
        in ``Range(L)`` and ``L^*`` lands in ``Domain(L)``.
        """
        if out is None:
            out = self.domain_geometry().allocate(0)

        if self.weight_operator is None:
            return self.struct_operator.adjoint(x, out=out)

        if self.struct_is_identity:
            # Domain(L) is Range(L); L^* is a no-op.
            return self.weight_operator.adjoint(x, out=out)

        self.weight_operator.adjoint(x, out=self.tmp_range_struct)
        return self.struct_operator.adjoint(self.tmp_range_struct, out=out)

    def inverse(self, x, out=None):
        r"""
        Return :math:`y` such that :math:`W L y = x`. Orthogonal :math:`L` only.

        Exact, not an approximation: :math:`(WL)^{-1} = L^{-1} W^{-1}` holds for
        any invertible :math:`W` and :math:`L`, with no commutation assumption.
        :math:`L^{-1}` is applied as :math:`L^*`, which is why orthogonality is
        required rather than mere invertibility.

        :math:`W^{-1}` is an element-wise division, done here rather than
        delegated, because ``DiagonalOperator`` has no ``inverse``. The IRLS
        weights are :math:`(|Lu|^2 + \tau^2)^{-1/4} \in (0, \tau^{-1/2}]`,
        strictly positive by construction, so the division is always safe and
        no pseudo-inverse masking is wanted: masking would silently turn
        :math:`(WL)^{-1}` into a projector.
        """
        self._require_invertible("inverse()")

        if out is None:
            out = self.domain_geometry().allocate(0)

        if self.weight_operator is None:
            return self.struct_operator.adjoint(x, out=out)

        if self.struct_is_identity:
            # L^{-1} is a no-op, so divide straight into out.
            x.divide(self.weights, out=out)
            return out

        x.divide(self.weights, out=self.tmp_range_struct)
        return self.struct_operator.adjoint(self.tmp_range_struct, out=out)

    def inverse_adjoint(self, x, out=None):
        r"""
        Return :math:`y` such that :math:`(W L)^* y = x`. Orthogonal :math:`L`
        only.

        Also exact: :math:`(WL)^{-*} = W^{-*} L^{-*}`, and :math:`W` is real
        diagonal so :math:`W^{-*} = W^{-1}`. With :math:`L` orthogonal,
        :math:`L^{-*} = L`. Needs no staging: :math:`L^{-*}` lands in
        ``Range(L)``, which is where :math:`W^{-1}` acts.
        """
        self._require_invertible("inverse_adjoint()")

        if out is None:
            out = self.range_geometry().allocate(0)

        self.struct_operator.direct(x, out=out)
        if self.weight_operator is not None:
            out.divide(self.weights, out=out)

        return out


class BlockTikhonovOperator(BlockOperator):
    r"""
    Block Tikhonov operator for general L2 and L1 regularisation.

    Performs regularisation directly in the physical solution space
    ``u``. It represents

    .. math::

        K =
        \begin{bmatrix}
            A \\
            \alpha W L
        \end{bmatrix}

    This is the only valid choice when :math:`L` is not invertible, since it
    never forms :math:`L^{-1}`.

    Geometries
    ----------
    Domain:
        Domain(A), the physical solution space.

    Range:
        BlockGeometry(Range(A), Range(L)).
    """

    form = 'block'

    def __init__(self, operator, solution_geometry, struct_operator=None,
                 regalpha=0.0, weighted=False, tmp_domain=None,
                 tmp_range_struct=None):
        self.operator = operator

        self.reg_operator = WeightedStructOperator(
            domain_geometry=solution_geometry,
            struct_operator=struct_operator,
            weighted=weighted,
            tmp_range_struct=tmp_range_struct,
        )
        self.struct_operator = self.reg_operator.struct_operator
        self.scaled_reg_operator = regalpha * self.reg_operator

        # Accumulator for the allocation-free adjoint below. Needed whenever the
        # second row is more than a plain scaling of x[1], i.e. whenever
        # WL != I. See the note in `adjoint` on why this container cannot be
        # avoided without lending a buffer from the algorithm.
        if tmp_domain is None and not self.reg_operator.is_identity():
            tmp_domain = solution_geometry.allocate(0)
        self.tmp_domain = tmp_domain

        super(BlockTikhonovOperator, self).__init__(
            self.operator,
            self.scaled_reg_operator,
        )

    @property
    def regalpha(self):
        """The current Tikhonov regularisation parameter."""
        return self.scaled_reg_operator.scalar

    @regalpha.setter
    def regalpha(self, value):
        """Update the regularisation parameter in place."""
        self.scaled_reg_operator.scalar = value

    @property
    def weights(self):
        """
        The mutable diagonal IRLS weights, or ``None`` when unweighted.

        The weights live in Range(L), not necessarily in Domain(A).
        """
        return self.reg_operator.weights

    def struct_direct(self, x, out=None):
        r"""
        Apply :math:`L` alone -- not :math:`WL` -- mapping a physical solution
        into ``Range(L)``, where the weights live.

        With :attr:`weights` and :meth:`enable_weights`, this is the named
        surface an outer reweighting loop drives the operator through: IRLS
        approximates :math:`\|Lu\|_1`, so its weights are a function of
        :math:`Lu`, never of :math:`WLu`.
        """
        return self.struct_operator.direct(x, out=out)

    def enable_weights(self):
        """
        Allocate the IRLS weights, and the accumulator the adjoint then needs.

        See :meth:`WeightedStructOperator.enable_weights`.
        """
        weights = self.reg_operator.enable_weights()
        if self.tmp_domain is None:
            # WL is no longer the identity, so the second row stops being a
            # plain scaling and the adjoint needs somewhere to stage it.
            self.tmp_domain = self.reg_operator.domain_geometry().allocate(0)
        return weights

    def adjoint(self, x, out=None):
        r"""
        Return :math:`K^* x = A^* x_0 + \alpha (W L)^* x_1`.

        Overrides :meth:`BlockOperator.adjoint`, which calls each sub-operator
        after the first *without* ``out=`` and then ``+=`` the result,
        allocating a full domain container on every call. LSQR and CGLS call
        the adjoint once per iteration, so that is a container per iteration
        for the length of the solve.

        The accumulator this needs costs one domain container whenever
        :math:`WL \neq I`. That is unavoidable without lending a buffer from
        the algorithm: ``out`` already holds :math:`A^* x_0`, and :math:`x_1`
        cannot be clobbered because the solver reads it again on the next
        iteration.

        Only the two-row Tikhonov shape is handled here; anything else falls
        back to the base class.
        """
        if self.shape != (2, 1):
            return super(BlockTikhonovOperator, self).adjoint(x, out=out)

        if out is None:
            out = self.domain_geometry().allocate(0)

        self.operator.adjoint(x[0], out=out)

        if self.tmp_domain is None:
            # WL = I, so the second row is a plain scaling and needs no
            # accumulator at all.
            return out.sapyb(1.0, x[1], self.regalpha, out=out)

        self.reg_operator.adjoint(x[1], out=self.tmp_domain)
        return out.sapyb(1.0, self.tmp_domain, self.regalpha, out=out)


class TikhonovOperator(LinearOperator):
    r"""
    Standard-form transformation operator.

    Transforms the generalised Tikhonov problem into standard form by mapping
    the Krylov search space out of the physical domain and into the regularised
    structure space:

    .. math::

        K = A L_{\text{weighted}}^{-1} = A (W L)^{-1}

    The solver operates on :math:`x`; once it has converged, the physical
    solution is recovered through

    .. math::

        u = L_{\text{weighted}}^{-1} x.

    :math:`\alpha` is **not** applied by this operator. In standard form it
    enters the solver's scalar recurrence (LSQR) or the normal-equations term
    (CGLS); it is stored here only so callers can check the two agree.

    Requires an **orthogonal** :math:`L`, since :math:`(WL)^{-1}` is formed as
    :math:`L^* W^{-1}`.

    .. warning::

        LSQR in this form damps the *step*, so a non-zero ``initial`` makes it
        minimise :math:`\|A(x_0 + \delta) - b\|^2 + \alpha^2\|WL\delta\|^2`,
        which is not the regularised problem. Use the block form to warm start.
        CGLS is unaffected: it subtracts the :math:`\alpha^2 x` term explicitly,
        so it solves the true regularised normal equations from any start.

    Geometries
    ----------
    Domain:
        Range(L), the regularised structure space.

    Range:
        Range(A), the physical data space.
    """

    form = 'standard'

    def __init__(self, operator, solution_geometry, struct_operator=None,
                 regalpha=0.0, weighted=False, tmp_domain=None,
                 tmp_range_struct=None):
        self.operator = operator
        self.regalpha = regalpha

        self.reg_operator = WeightedStructOperator(
            domain_geometry=solution_geometry,
            struct_operator=struct_operator,
            weighted=weighted,
            tmp_range_struct=tmp_range_struct,
        )
        self.struct_operator = self.reg_operator.struct_operator

        self.reg_operator._require_invertible(
            "The standard-form transformation K = A (W L)^-1")

        # With WL = I the transformation is K = A exactly, so there is nothing
        # to stage and the whole buffer can go. This is what makes the
        # degenerate regularised path cost the same as unregularised LSQR.
        if tmp_domain is None and not self.reg_operator.is_identity():
            tmp_domain = solution_geometry.allocate(0)
        self.tmp_domain = tmp_domain

        super(TikhonovOperator, self).__init__(
            domain_geometry=self.reg_operator.range_geometry(),
            range_geometry=operator.range_geometry(),
        )

    @property
    def weights(self):
        """
        The mutable diagonal IRLS weights, or ``None`` when unweighted.

        The weights live in Range(L), not necessarily in Domain(A).
        """
        return self.reg_operator.weights

    def struct_direct(self, x, out=None):
        r"""
        Apply :math:`L` alone -- not :math:`WL` -- mapping a physical solution
        into ``Range(L)``, where the weights live.

        With :attr:`weights` and :meth:`enable_weights`, this is the named
        surface an outer reweighting loop drives the operator through: IRLS
        approximates :math:`\|Lu\|_1`, so its weights are a function of
        :math:`Lu`, never of :math:`WLu`.
        """
        return self.struct_operator.direct(x, out=out)

    def enable_weights(self):
        """Allocate the IRLS weights, and the scratch the transformation then needs."""
        weights = self.reg_operator.enable_weights()
        if self.tmp_domain is None:
            # WL is no longer the identity, so the staging buffer is needed.
            self.tmp_domain = self.reg_operator.domain_geometry().allocate(0)
        return weights

    def direct(self, x, out=None):
        r"""Apply :math:`K x = A (W L)^{-1} x`."""
        if out is None:
            out = self.range_geometry().allocate(0)

        if self.tmp_domain is None:
            # WL = I, so K = A.
            return self.operator.direct(x, out=out)

        self.reg_operator.inverse(x, out=self.tmp_domain)
        return self.operator.direct(self.tmp_domain, out=out)

    def adjoint(self, x, out=None):
        r"""Apply :math:`K^* x = (W L)^{-*} A^* x`."""
        if out is None:
            out = self.domain_geometry().allocate(0)

        if self.tmp_domain is None:
            return self.operator.adjoint(x, out=out)

        self.operator.adjoint(x, out=self.tmp_domain)
        return self.reg_operator.inverse_adjoint(self.tmp_domain, out=out)
