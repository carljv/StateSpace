# ---------------------------------------------------------
# StateSpace.jl
# Types and functions for making State Space Time Series Models.
#
# (Linear) State Space Models can be used to represent a large
# number of common time series, including trends, seasonality,
# and ARIMA patterns.
#
# References:
#   David Brockwell and Richard Davis,
#     /Time Series Theory and Methods/, ch. 12
#
# C. Vogel     Nov 2013
# ---------------------------------------------------------
module StateSpaceModels

typealias FP FloatingPoint

abstract StateSpace

# A *Linear State Space Model* has the form:
#
#    y[t] = G * x[t] + v[t]
#    x[t+1] = F * x[t] + w[t] ,
#
# where x is nx1, y is mx1, (n, m >= 1) and therefore
# F is nxn, G is mxn, v is nx1, and w is nx1
# v[t] and w[t] are random noise vectors with mean zero
# and
#  E(w*w') = Q; E(v*v') = R.
# w and v are assumed to be uncorrelated, which is ok for
# the common time series models, but this can be relaxed.
#
# (ASIDE: To do this, imagine the stacked random vector
# [w v]' which is (n+m)x1 and a big covariance matrix:
#
# [Q S;  = E([w v]' * [w' v'])
#  S'R]
#
# This matrix is (n+m)x(n+m), meaning S is an mxn
# that reflects covariance between w an v -- E(w*v')
immutable LinearStateSpace{T<:FP} <: StateSpace
    F::Union(T, Matrix{T})
    Q::Union(T, Matrix{T})
    G::Union(T, Array{T})
    R::Union(T, Matrix{T})
end

# Hacking the Cholesky Decomposition to deal with
# deterministic elements in the SS model:
#
# In some State Space models, elements of x or y can be
# deterministic, meaning that the corresponding row of
# w or v is deterministic (or has zero variance).
# We can reflect this by:
# (1) computing the random and deterministic parts of the
# model separately (so, e.g. if x is nx1 with 1
# deterministic component, then w is only (n-1)x1 and
# Q is (n-1)x(n-1).
#
# This is a little gross, so we could also try:
# (2) making Q and R have a column/row pair associated
# with this element that is are all zeros.
#
# This latter technique runs into problems, because we
# expect the vcov matrix to be positive definite for
# Cholesky decompositions. We can hack this by just
# plugging in a number on the diagonals of Q and/or R
# where there is a deterministic component. So long
# as the row and column of that diagonal are zero
# everywhere else, the rest of the resulting
# decomposition won't be affected, and we can just
# replace the diagonals we modified with zero in
# the resulting decomposition matrix.


# Let's just make a scalar version that mimics the square root.
# This means we don't have to make a distinction between
# scalar and vector w and/or v in our models.
modified_cholesky{T<:Number}(v::T) = sqrt(float(v))

function modified_cholesky{T<:FloatingPoint}(Q::Matrix{T})
    n = size(Q, 1)
    # Replace zeros along the diagonal with the mean of the non-zero
    # elements, or 1 if all diagonals are zero.
    stub_val = nnz(diag(Q)) == 0 ? 1:  mean(diag(Q)[diag(Q) .!= 0.])
    zero_idx = Int[]
    for i in 1:n
        if Q[i,i] == 0.
            all(Q[:, i] .== 0.) && all(Q[i, :] .== 0) ||
                error("Variables with 0 variance must have 0 covariances.")
            push!(zero_idx, i)
            setindex!(Q, stub_val, i, i)
        end
    end

    QL = chol(Q, :L)
    # Set the diagonals we messed with back to zero in the decomposition.
    for i in zero_idx
        setindex!(Q, 0., i, i)
        setindex!(QL, 0., i, i)
    end
    QL
end

# Iterating the state over time
# ---------------------------------------------
# F, G, Q, R parameters are specified, the
# following functions produce an iterator
# that will generate a sequence of states
# x[1], x[2], ... according to the state
# equation x[t+1] = F * x[t] + w[t].
# w[t] is assumed to be Normal(0, Q), though
# theoretically, the model accomodates
# w[t] to be generic white noise.
function state_updater(ss::LinearStateSpace)
    QL = modified_cholesky(ss.Q)
    function update(x::Vector)
        newx = ss.F * x + QL * randn(size(x, 1))
        newx = size(newx) == (1, 1) ? newx[1] : newx
        newx
    end
    update
end

function state_iterator(ss::LinearStateSpace, x)
    update = state_updater(ss)
    println(update(x))
    function _it()
        state = x
        while true
            newstate = update(state)
            state = newstate
            produce(newstate)
        end
    end
    @task _it()
end


# The observation function
#    y[t] = G * x[t] + v[t], v[t] ~ N(0, R)
# The result of obersver is a function that takes x[t],
# as its argument, not the y[t] resulting from a specific x[t]
# (i.e., this is just a closure around G and R.
function observer(ss::LinearStateSpace)
    RL = modified_cholesky(ss.R)
    function(x::Vector)
        y = ss.G * x + RL * randn(size(RL, 1))
        y = size(y) == (1, 1) || size(y) == (1,) ? y[1] : y
    end
end

# Generate a simulated series y[t] for t = 1, ..., n, given
# an initial value x[1].
function simulate(ss::LinearStateSpace, n::Int, init)
    states = state_iterator(ss, init)
    observe = observer(ss)
    leny = size(ss.G, 1)
    y = leny == 1 ? zeros(n) : zeros(n, leny)
    for i = 1:n
        y[i, :] = observe(consume(states))
    end
    y
end

# Functions/operators to combine Linerar State Space Models.
# State Space Models can be summed or stacked.
# The sum of two LSSMs is also an LSSM, as are two
# stacked LSSMs.
# e.g. if M1, and M2 are two LSSMS, denoted respectively,
# M1 = LSSM{F1, G1, Q1, R1}; M2 = LSSM{F2, G2, Q2, R2},
# then M1 + M2 = LSSM{F12, G12, Q12, R12},
# and [M1, M2]' = LSSM{F12*, G12*, Q12*, R12*}.
#
# These operation allow for abstract operations on the
# models, to get new models.  This is an alternative
# to performing operations on the
function blockdiag{T<:Number}(M::Matrix{T}, N::Matrix{T})
    rm = size(M, 1); cm = size(M, 2)
    rn = size(N, 1); cn = size(N, 2)
    P = zeros(rm + rn, cm + cn)
    P[1:rm, 1:cm] = M
    P[(rm+1):end, (cm+1):end] = N
    P
end

function blockdiag{T<:Number}(M::T, N::T)
    [M 0; 0 N]
end

function blockdiag{T<:Number}(M::Matrix{T}, N::T)
    rm = size(M, 1); cm = size(M, 2)
    P = zeros(rm + 1, cm + 1)
    P[1:rm, 1:cm] = M
    P[end, end] = N
    P
end

function blockdiag{T<:Number}(M::T, N::Matrix{T})
    rn = size(N, 1); cn = size(N, 2)
    P = zeros(rn + 1, cn + 1)
    P[1, 1] = M
    P[2:end, 2:end] = N
    P
end

function +(ss1::LinearStateSpace, ss2::LinearStateSpace)
    size(ss1.G, 1) == size(ss2.G, 1) ||
        error("G matrices must have same no. of rows.")
    size(ss1.R, 1) == size(ss2.R, 1) ||
        error("R matrices must be the same size.")

    F = blockdiag(ss1.F, ss2.F)
    Q = blockdiag(ss1.Q, ss2.Q)
    G = hcat(ss1.G, ss2.G)
    R = ss1.R + ss2.R

    LinearStateSpace(F, Q, G, R)
end

function vcat(ss1::LinearStateSpace, ss2::LinearStateSpace)
    F = blockdiag(ss1.F, ss2.F)
    Q = blockdiag(ss1.Q, ss2.Q)
    G = blockdiag(ss1.G, ss2.G)
    R = blockdiag(ss1.R, ss2.R)

    LinearStateSpace(F, Q, G, R)
end

end # module


# ---------------------------------------------------------
# EXAMPLES - just for illustration, not useful otherwise
# ---------------------------------------------------------

module StateSpaceExamples
using StateSpaceModels
# Varying linear trend model
# z[t] = z[t-1] + b + s * e[t]; e[t] ~ N(0, 1)
#
# State Space System
# X[t] = [x[t], b]
# X[t+1] = [1. 1.; 0 0] * X[t] + [s 0.; 0. 0.] * v[t+1]; v[t] ~ N(0, I_1)
# Y[t] = [1. 0.] * X[t] + 0. * w[t]; w[t] ~ N(0, 1)

F = [1.  1.;
     0.  1.]

Q = [.25  0.;
     0.  0.]

G = [1. 0.]
R = .0

x0_tr = [0., .5]
tr = StateSpaceModels.LinearStateSpace(F, Q, G, R)
y_tr = StateSpaceModels.simulate(tr, 200, x0_tr)


# Seasonal Time Series (noiseless) w/ period d=12
#
# s[t+1] = -(s[t] + s[t-1] + ... + s[t-d+2])
#
# X[t] = [s[t], s[t-1], ..., s[t-d+2]]
# X[t+1] = [-1 -1 ... -1; 1 0 ... 0; ... ; 0 ... 1 0] * X[t] + 0 * v[t+1]
# Y[t] = [1 0 ... 0] * X[t] + 0 * w[t]

d = 12
F1 = vcat(-1 * ones(d-2)', eye(d-2)[1:(end-1), :])
Q1 = zeros(d-2, d-2)

G1 = [1. zeros(d-3)']
R1 = 0.

sts = StateSpaceModels.LinearStateSpace(F1, Q1, G1, R1)

x0_sts = randn(d-2) * 5
y_sts = StateSpaceModels.simulate(sts, 200, x0_sts)

# We can add the seaonsal model to the trend model
# to get a time series with trend and seasonal
# components
strts = tr + sts
y_strts = StateSpaceModels.simulate(strts, 200, [x0_tr, x0_sts])



# AR(1) Series
#
# X[t] = [x[t], x[t-1]]
# X[t+1] = [p 0; 1 0] * X[t] + [s 0; 0 0] * v[t+1]
# Y[t] = [1 0] * X[t] + 0 * w[t]

F2 = [.75 0; 1 0]
Q2 = [.15 0; 0 0]
G2 = [1. 0.]
R2 = 0.

ar1 = StateSpaceModels.LinearStateSpace(F2, Q2, G2, R2)

end # module
