const HAS_REACTANT = Base.find_package("Reactant") !== nothing

if HAS_REACTANT
    using Reactant
end

using GeoVI
using Test
using LinearAlgebra
using Optimisers
using Random
using Statistics: mean

# The fitted output is an `AbstractVariationalDistribution`; its center is `.mean`.
_post_mean(q) = q.mean

function _linear_gaussian_setup(rng; D = 100, M = 50, σ² = 0.25)
    A = randn(rng, M, D) ./ sqrt(D)
    ξ_true = randn(rng, D)
    data = A * ξ_true .+ sqrt(σ²) .* randn(rng, M)
    precision = fill(1 / σ², M)
    Σ_post = inv(I + (A' * A) ./ σ²)
    μ_post = Σ_post * (A' * data) ./ σ²
    return (; A, data, precision, ξ_true, σ², Σ_post, μ_post)
end

@testset "GeoVI.jl" begin
    @testset "array utilities" begin
        x = [1.0, 2.0]
        y = zero(x)

        @test y == zeros(2)

        shifted = 2 .* x .+ y
        @test shifted == 2 .* x

        expected_inner = sum(abs2, x)
        @test dot(x, x) ≈ expected_inner
        @test norm(x) ≈ sqrt(expected_inner)

        noise = randn_like(MersenneTwister(12), x)
        @test size(noise) == size(x)
        @test eltype(noise) == eltype(x)

        x32 = Float32[1, 2]
        noise32 = randn_like(MersenneTwister(13), x32)
        @test eltype(noise32) == Float32
    end

    @testset "Gaussian likelihood" begin
        data = [2.0, -1.0]
        precision = [4.0, 9.0]
        lh = GaussianLikelihood(data; precision = precision)

        y = [1.5, -2.0]
        v = [0.3, -0.5]
        resid = data - y

        @test logdensity(lh, y) ≈ -0.5 * sum(precision .* resid .^ 2)
        @test energy(lh, y) ≈ -logdensity(lh, y)
        @test lh(y) ≈ logdensity(lh, y)
        @test normalized_residual(lh, y) ≈ sqrt.(precision) .* resid
        @test transformation(lh, y) ≈ sqrt.(precision) .* y
        @test leftsqrtfishermetric(lh, y, v) ≈ sqrt.(precision) .* v
        @test leftsqrtmetric(lh, y, v) ≈ sqrt.(precision) .* v
        @test rightsqrtfishermetric(lh, y, v) ≈ sqrt.(precision) .* v
        @test rightsqrtmetric(lh, y, v) ≈ sqrt.(precision) .* v
        @test fishermetric(lh, y, v) ≈ precision .* v
        @test metric(lh, y, v) ≈ fishermetric(lh, y, v)

        mat_precision = Diagonal([4.0, 9.0])
        mat_sqrt = Diagonal([2.0, 3.0])
        mat_lh = GaussianLikelihood(data; precision = mat_precision, sqrt_precision = mat_sqrt)
        @test fishermetric(mat_lh, y, v) ≈ mat_precision * v

        @test_throws ArgumentError GaussianLikelihood(data; precision = x -> 2 .* x)
    end

    @testset "likelihood composition" begin
        data = [1.0, -2.0]
        precision = [3.0, 5.0]
        base = GaussianLikelihood(data; precision = precision)

        A = [1.0 2.0; -1.0 0.5]
        forward(x) = A * x
        pushforward(x, v) = A * v
        pullback(x, η) = A' * η

        x = [0.4, -0.7]
        v = [1.2, -0.5]
        η = [0.3, 2.0]
        y = forward(x)

        manual = compose(base, forward; pushforward = pushforward, pullback = pullback)

        @test logdensity(manual, x) ≈ logdensity(base, y)
        @test normalized_residual(manual, x) ≈ normalized_residual(base, y)
        @test transformation(manual, x) ≈ transformation(base, y)
        @test rightsqrtmetric(manual, x, v) ≈ rightsqrtmetric(base, y, A * v)
        @test leftsqrtmetric(manual, x, η) ≈ A' * leftsqrtmetric(base, y, η)
        @test fishermetric(manual, x, v) ≈ A' * (precision .* (A * v))

        linearize(x) = (
            value = forward(x),
            pushforward = v -> A * v,
            pullback = η -> A' * η,
        )
        bundled = compose(base, forward; linearize = linearize)
        @test rightsqrtmetric(bundled, x, v) ≈ rightsqrtmetric(base, y, A * v)
        @test leftsqrtmetric(bundled, x, η) ≈ A' * leftsqrtmetric(base, y, η)
        @test fishermetric(bundled, x, v) ≈ A' * (precision .* (A * v))

        struct ToyLinearization{T}
            value::T
            jacobian::Matrix{Float64}
        end
        GeoVI.pushforward(lin::ToyLinearization, v::AbstractArray) = lin.jacobian * v
        GeoVI.pullback(lin::ToyLinearization, η::AbstractArray) = lin.jacobian' * η
        method_based = compose(base, forward; linearize = x -> ToyLinearization(forward(x), A))
        @test rightsqrtmetric(method_based, x, v) ≈ rightsqrtmetric(base, y, A * v)
        @test leftsqrtmetric(method_based, x, η) ≈ A' * leftsqrtmetric(base, y, η)
        @test fishermetric(method_based, x, v) ≈ A' * (precision .* (A * v))

        automatic = compose(base, forward)
        @test logdensity(automatic, x) ≈ logdensity(base, y)
        @test normalized_residual(automatic, x) ≈ normalized_residual(base, y)
        @test transformation(automatic, x) ≈ transformation(base, y)
        @test rightsqrtmetric(automatic, x, v) ≈ rightsqrtmetric(base, y, A * v) atol = 1.0e-6 rtol = 1.0e-6
        @test leftsqrtmetric(automatic, x, η) ≈ A' * leftsqrtmetric(base, y, η) atol = 1.0e-6 rtol = 1.0e-6
        @test fishermetric(automatic, x, v) ≈ A' * (precision .* (A * v)) atol = 1.0e-5 rtol = 1.0e-5

        noauto = compose(base, forward; adtype = GeoVI.ADTypes.NoAutoDiff())
        @test_throws ArgumentError rightsqrtmetric(noauto, x, v)
        @test_throws ArgumentError leftsqrtmetric(noauto, x, η)
        @test_throws ArgumentError compose(base, forward; linearize = linearize, pushforward = pushforward)
    end

    @testset "linearization interface" begin
        A = [2.0 -1.0; 0.5 3.0]
        forward(x) = A * x
        x = [0.3, -0.8]
        v = [1.1, -0.4]
        η = [0.7, -1.5]

        finite_diff = GeoVI._automatic_linearize(GeoVI.ADTypes.AutoFiniteDiff(), forward, x)
        @test finite_diff.value ≈ forward(x)
        @test GeoVI.pushforward(finite_diff, v) ≈ A * v atol = 1.0e-6 rtol = 1.0e-6
        @test GeoVI.pullback(finite_diff, η) ≈ A' * η atol = 1.0e-6 rtol = 1.0e-6
    end

    @testset "exponential-family likelihoods" begin
        v = [0.3, -0.5]
        eps = 1.0e-6

        poisson = PoissonLikelihood([2.0, 4.0]; weight = [1.5, 0.5])
        ηp = log.([3.0, 5.0])
        λ = exp.(ηp)
        @test logdensity(poisson, ηp) ≈ -sum([1.5, 0.5] .* (λ .- [2.0, 4.0] .* ηp))
        @test normalized_residual(poisson, ηp) ≈ sqrt.([1.5, 0.5]) .* ([2.0, 4.0] .- λ) ./ sqrt.(λ)
        @test leftsqrtmetric(poisson, ηp, v) ≈ sqrt.([1.5, 0.5] .* λ) .* v
        @test rightsqrtmetric(poisson, ηp, v) ≈ sqrt.([1.5, 0.5] .* λ) .* v
        @test fishermetric(poisson, ηp, v) ≈ ([1.5, 0.5] .* λ) .* v
        @test (
            transformation(poisson, ηp .+ eps .* v) .- transformation(poisson, ηp .- eps .* v)
        ) ./ (2 * eps) ≈ rightsqrtmetric(poisson, ηp, v) atol = 1.0e-6 rtol = 1.0e-6
        @test -leftsqrtmetric(poisson, ηp, normalized_residual(poisson, ηp)) ≈
            [1.5, 0.5] .* (λ .- [2.0, 4.0])

        bernoulli = BernoulliLikelihood([1.0, 0.0]; weight = [2.0, 0.75])
        ηb = [0.3, -0.4]
        p = 1 ./ (1 .+ exp.(-ηb))
        @test logdensity(bernoulli, ηb) ≈
            -sum([2.0, 0.75] .* (log1p.(exp.(ηb)) .- [1.0, 0.0] .* ηb))
        @test fishermetric(bernoulli, ηb, v) ≈ ([2.0, 0.75] .* p .* (1 .- p)) .* v
        @test (
            transformation(bernoulli, ηb .+ eps .* v) .- transformation(bernoulli, ηb .- eps .* v)
        ) ./ (2 * eps) ≈ rightsqrtmetric(bernoulli, ηb, v) atol = 1.0e-6 rtol = 1.0e-6
        @test -leftsqrtmetric(bernoulli, ηb, normalized_residual(bernoulli, ηb)) ≈
            [2.0, 0.75] .* (p .- [1.0, 0.0])

        binomial = BinomialLikelihood([3.0, 1.0]; trials = [5.0, 2.0], weight = [1.0, 0.5])
        ηn = [0.2, -0.1]
        q = 1 ./ (1 .+ exp.(-ηn))
        μ = [5.0, 2.0] .* q
        @test logdensity(binomial, ηn) ≈
            -sum([1.0, 0.5] .* ([5.0, 2.0] .* log1p.(exp.(ηn)) .- [3.0, 1.0] .* ηn))
        @test fishermetric(binomial, ηn, v) ≈
            ([1.0, 0.5] .* [5.0, 2.0] .* q .* (1 .- q)) .* v
        @test (
            transformation(binomial, ηn .+ eps .* v) .- transformation(binomial, ηn .- eps .* v)
        ) ./ (2 * eps) ≈ rightsqrtmetric(binomial, ηn, v) atol = 1.0e-6 rtol = 1.0e-6
        @test -leftsqrtmetric(binomial, ηn, normalized_residual(binomial, ηn)) ≈
            [1.0, 0.5] .* (μ .- [3.0, 1.0])

        @test_throws ArgumentError BernoulliLikelihood([0.0, 0.5])
        @test_throws ArgumentError BinomialLikelihood([2.0]; trials = [1.0])
    end

    @testset "samples" begin
        position = [10.0, 20.0]
        residuals = [1.0 2.0; 3.0 4.0]
        samples = Samples(position, residuals; keys = [:a, :b])

        @test length(samples) == 2
        @test posterior_samples(samples) ≈ [11.0 22.0; 13.0 24.0]
        @test samples[1] ≈ [11.0, 22.0]
        @test samples[2] ≈ [13.0, 24.0]
        @test collect(samples)[2] ≈ [13.0, 24.0]

        shifted = recenter(samples, [11.0, 21.0])
        @test shifted.position == [11.0, 21.0]
        @test posterior_samples(shifted) ≈ posterior_samples(samples)
        @test shifted.residuals ≈ [0.0 1.0; 2.0 3.0]
    end

    @testset "VI surface" begin
        est = MCEstimator(n_samples = 6, mirrored = true)
        @test est.n_samples == 6
        @test est.mirrored
        @test GeoVI._n_base_draws(est) == 3
        @test_throws ArgumentError MCEstimator(n_samples = 3, mirrored = true)
        @test GeoVI._n_base_draws(MCEstimator(n_samples = 5, mirrored = false)) == 5

        # the estimator extension surface: the loop consults accessors, not fields,
        # and a fieldless custom estimator gets a clear error naming what to implement.
        @test GeoVI._n_stored_samples(est) == 6
        @test GeoVI._mirrored(est)
        @test GeoVI._n_stored_samples(MCEstimator()) == 8   # default is VI, not MAP
        struct FieldlessEstimator <: GeoVI.AbstractEstimator end
        @test_throws ArgumentError GeoVI._n_stored_samples(FieldlessEstimator())
        @test_throws ArgumentError GeoVI._mirrored(FieldlessEstimator())
        @test_throws ArgumentError GeoVI._n_base_draws(FieldlessEstimator())

        @test GeoVI._infer_adtype(GeoVI.ADTypes.AutoFiniteDiff(), [1.0, 2.0]) isa
            GeoVI.ADTypes.AutoFiniteDiff
        @test GeoVI._value_and_gradient(
            GeoVI.ADTypes.AutoFiniteDiff(), x -> sum(abs2, x), [1.0, 2.0]
        )[2] ≈ [2.0, 4.0] atol = 1.0e-5

        @test NewtonCG(maxiter = 7, cg_rtol = 1.0e-9).maxiter == 7
        @test NewtonCG(cg_rtol = 1.0e-9).cg.rtol == 1.0e-9
        @test_throws ArgumentError NewtonCG(maxiter = -1)

        simple_lh = GaussianLikelihood([0.0]; precision = [1.0])
        problem = VariationalProblem(
            simple_lh,
            [0.0];
            family = MGVIFamily(),
            divergence = ReverseKL(),
            estimator = est,
            optimizer = NewtonCG(),
        )
        @test problem.adtype isa GeoVI.ADTypes.AutoFiniteDiff
        @test GeoVI._n_base_draws(problem) == 3
        @test problem.optimizer.maxiter == 20

        rng, state = init(MersenneTwister(2), problem)
        @test state isa VIState
        @test rng isa MersenneTwister
        @test size(state.residuals) == (6, 1)

        @test_throws ArgumentError update_nonlinear_residual(simple_lh, [0.0], [0.0])
        @test_throws ArgumentError VariationalProblem(
            simple_lh, [0.0]; divergence = GeoVI.ForwardKL(), optimizer = NewtonCG()
        )
        @test_throws ArgumentError VariationalProblem(
            simple_lh, [0.0]; divergence = ReverseKL(), optimizer = :adam
        )

        # the positional constructor validates too (no bypass around _require_supported)
        @test_throws ArgumentError VariationalProblem(
            simple_lh, [0.0], MeanFieldGaussian(), ReverseKL(), MCEstimator(),
            NewtonCG(), GeoVI.ADTypes.AutoFiniteDiff(),
        )

        # a structured-θ family with no samples must fail loudly at construction:
        # the sample-free MAP objective is undefined off the latent point.
        struct StructuredThetaFam <: GeoVI.AbstractVariationalFamily end
        GeoVI.init_params(::StructuredThetaFam, x) = (; mean = copy(x))
        @test_throws ArgumentError VariationalProblem(
            simple_lh, [0.0]; family = StructuredThetaFam(),
            estimator = MCEstimator(n_samples = 0), optimizer = Optimisers.Descent(0.1),
        )

        # initial Samples must not carry residuals (step_vi! redraws them, so they
        # could never be used — rejected at construction)
        @test_throws ArgumentError VariationalProblem(
            simple_lh, Samples([0.0], reshape([0.25, -0.25], (2, 1)); keys = nothing);
            family = MGVIFamily(),
            estimator = MCEstimator(n_samples = 2), optimizer = NewtonCG(),
        )
        # a position-only Samples is fine, and the buffer starts zero-filled (no
        # "residuals not yet drawn" state exists after init)
        zerofill_problem = VariationalProblem(
            simple_lh, Samples([0.0], nothing; keys = nothing); family = MGVIFamily(),
            estimator = MCEstimator(n_samples = 2), optimizer = NewtonCG(),
        )
        _, zerofill_state = init(MersenneTwister(3), zerofill_problem)
        @test zerofill_state.residuals == zeros(2, 1)

        nd_problem = VariationalProblem(
            simple_lh,
            [0.0];
            family = GeoVIFamily(),
            divergence = ReverseKL(),
            estimator = MCEstimator(n_samples = 2),
            optimizer = NewtonCG(),
            adtype = GeoVI.ADTypes.NoAutoDiff(),
        )
        @test_throws ArgumentError fit(nd_problem, 1; rng = MersenneTwister(1))
    end

    @testset "conjugate gradient stopping" begin
        # SPD operator A = AᵀΣA + I (the posterior-metric form).
        rng = MersenneTwister(0xc6)
        D, M = 80, 40
        B = randn(rng, M, D) ./ sqrt(D)
        prec = fill(4.0, M)
        op = v -> (transpose(B) * (prec .* (B * v))) .+ v
        b = randn(rng, D)
        xstar = (transpose(B) * Diagonal(prec) * B + I) \ b

        @test ConjugateGradient(absdelta = 1.0e-3).absdelta == 1.0e-3
        @test ConjugateGradient().absdelta === nothing

        # Residual-tolerance stop reaches the true solution.
        x, info = GeoVI.solve(ConjugateGradient(rtol = 1.0e-10), op, b)
        @test info.converged
        @test x ≈ xstar atol = 1.0e-6 rtol = 1.0e-6

        # The absdelta energy-decrease criterion stops earlier than a tight
        # residual tolerance, while still descending toward the solution.
        x_full, info_full = GeoVI.solve(ConjugateGradient(rtol = 1.0e-14, maxiter = 500), op, b)
        x_ad, info_ad = GeoVI.solve(
            ConjugateGradient(rtol = 1.0e-14, maxiter = 500), op, b; absdelta = 1.0e-2
        )
        @test info_ad.iterations < info_full.iterations
        @test info_ad.converged
        @test norm(op(x_ad) .- b) < norm(b)

        # `nothing` tolerances mean "criterion absent": with only `atol` set the
        # solve stops at that absolute residual norm.
        x_nt, info_nt = GeoVI.solve(ConjugateGradient(rtol = nothing, atol = 1.0e-6), op, b)
        @test info_nt.converged
        @test norm(op(x_nt) .- b) <= 1.0e-5

        # NewtonCG inner-CG tolerances: the default is `nothing` (pure
        # Eisenstat–Walker forcing). Explicit `cg_rtol`/`cg_atol` act as a FLOOR on
        # the inner residual target (a cap on tightness, guarding against
        # over-solving near the optimum); they can only LOOSEN the inner solve, never
        # tighten it below the forcing target.
        @test NewtonCG().cg.rtol === nothing
        @test NewtonCG().cg.atol === nothing
        quad_fg = x -> (0.5 * real(dot(x, op(x))) - real(dot(b, x)), op(x) .- b)
        run_newton = opt -> GeoVI._optimize(
            opt, zeros(D);
            fun_and_grad = quad_fg, metricp = (_ -> op),
            GeoVI._optimizer_kwargs(opt)...,
        )
        res_forcing = run_newton(NewtonCG(maxiter = 1))
        # A `cg_rtol` below the forcing target is a no-op: it cannot tighten the inner
        # solve, so a single Newton step costs the same inner matvecs as pure forcing.
        res_tight = run_newton(NewtonCG(maxiter = 1, cg_rtol = 1.0e-12, cg_maxiter = 500))
        @test res_tight.hessian_evaluations == res_forcing.hessian_evaluations
        # Forcing alone still drives the optimizer to the Newton point A⁻¹b: over
        # successive steps ‖g‖→0 tightens the forcing until x lands on xstar.
        res_converged = run_newton(NewtonCG(maxiter = 50))
        @test res_converged.x ≈ xstar atol = 1.0e-6 rtol = 1.0e-6
    end

    @testset "MGVI linear residuals" begin
        precision = [3.0, 5.0]
        base = GaussianLikelihood([0.0, 0.0]; precision = precision)

        A = [1.0 2.0; -1.0 0.5]
        forward(x) = A * x
        pushforward(x, v) = A * v
        pullback(x, η) = A' * η

        lh = compose(base, forward; pushforward = pushforward, pullback = pullback)
        xi = [0.2, -0.1]

        posterior_metric = I + A' * Diagonal(precision) * A

        rng_metric = MersenneTwister(11)
        metric_draw = draw_metric_sample(lh, xi, rng_metric)

        rng_residual = MersenneTwister(11)
        residual_draw = draw_linear_residual(
            lh,
            xi,
            rng_residual;
            cg_rtol = 1.0e-12,
            cg_maxiter = 10,
        )
        @test residual_draw.info.converged
        @test residual_draw.info.iterations > 0
        @test residual_draw.residual ≈ posterior_metric \ metric_draw.metric atol = 1.0e-10 rtol = 1.0e-10

        @test_throws ErrorException draw_linear_residual(
            lh,
            xi,
            MersenneTwister(11);
            cg_maxiter = 0,
        )

        stalled_draw = draw_linear_residual(
            lh,
            xi,
            MersenneTwister(11);
            cg_maxiter = 0,
            throw_on_failure = false,
        )
        @test !stalled_draw.info.converged
        @test stalled_draw.info.iterations == 0
        @test size(stalled_draw.residual) == size(xi)

        n_draws = 4_000
        draws = Matrix{eltype(xi)}(undef, 2, n_draws)
        rng = MersenneTwister(23)
        for i in 1:n_draws
            draw = draw_linear_residual(lh, xi, rng; cg_rtol = 1.0e-10, cg_maxiter = 10)
            @test draw.info.converged
            draws[:, i] = draw.residual
        end

        mean_draw = vec(sum(draws; dims = 2) ./ n_draws)
        centered = draws .- reshape(mean_draw, :, 1)
        empirical_cov = centered * centered' / (n_draws - 1)
        analytic_cov = inv(Matrix(posterior_metric))

        @test norm(mean_draw) < 0.06
        @test empirical_cov ≈ analytic_cov atol = 0.035 rtol = 0.15
    end

    @testset "geoVI nonlinear residuals" begin
        precision = [3.0, 5.0]
        base = GaussianLikelihood([0.0, 0.0]; precision = precision)

        A = [1.0 2.0; -1.0 0.5]
        forward(x) = A * x
        pushforward(x, v) = A * v
        pullback(x, η) = A' * η

        lh = compose(base, forward; pushforward = pushforward, pullback = pullback)
        xi = [0.2, -0.1]

        rng = MersenneTwister(11)
        metric_sample = draw_metric_sample(lh, xi, rng)
        linear_draw = draw_linear_residual(
            lh,
            xi,
            metric_sample;
            cg_rtol = 1.0e-12,
            cg_maxiter = 10,
        )

        @test linear_draw.info.converged

        update = update_nonlinear_residual(
            lh,
            xi,
            linear_draw;
            optimizer_options = (; xtol = 1.0e-10, cg_rtol = 1.0e-12, cg_maxiter = 10),
        )

        @test update.result.converged
        @test update.residual ≈ linear_draw.residual atol = 1.0e-12 rtol = 1.0e-12
        @test norm(update.result.gradient) < 1.0e-12
        @test update.result.value < 1.0e-24

        mirrored_draw = draw_residual(
            lh,
            xi,
            MersenneTwister(11);
            draw_linear_kwargs = (; cg_rtol = 1.0e-12, cg_maxiter = 10),
            optimizer_options = (; xtol = 1.0e-10, cg_rtol = 1.0e-12, cg_maxiter = 10),
        )

        @test size(mirrored_draw.residuals) == (2, length(xi))
        @test mirrored_draw.linear.info.converged
        @test mirrored_draw.positive.result.converged
        @test mirrored_draw.negative.result.converged
        @test mirrored_draw.residuals[1, :] ≈ linear_draw.residual atol = 1.0e-12 rtol = 1.0e-12
        @test mirrored_draw.residuals[2, :] ≈ -linear_draw.residual atol = 1.0e-12 rtol = 1.0e-12
        @test norm(mirrored_draw.residuals[1, :] + mirrored_draw.residuals[2, :]) < 1.0e-12

        adam_initial_residual = linear_draw.residual .+ [0.25, -0.2]
        trafo_at_point = transformation(lh, xi)
        adam_initial_value, adam_initial_gradient = GeoVI._nonlinear_residual_value_and_gradient(
            lh,
            xi,
            trafo_at_point,
            metric_sample.metric,
            xi .+ adam_initial_residual,
        )
        adam_update = update_nonlinear_residual(
            lh,
            xi,
            adam_initial_residual;
            metric_sample = metric_sample,
            optimizer = Optimisers.Adam(0.05),
            optimizer_options = (; maxiter = 400, miniter = 50, xtol = 1.0e-10, absdelta = 1.0e-12),
        )

        @test adam_update.result.converged
        @test adam_update.result.optimizer_state !== nothing
        @test adam_update.result.value < adam_initial_value
        @test norm(adam_update.result.gradient) < norm(adam_initial_gradient)
        @test adam_update.residual ≈ linear_draw.residual atol = 1.0e-3 rtol = 1.0e-3

        toy_base = GaussianLikelihood([0.0]; precision = [4.0])
        toy_forward(x) = x .+ 0.25 .* x .^ 3
        toy_jac(x) = 1 .+ 0.75 .* x .^ 2
        toy_pushforward(x, v) = toy_jac(x) .* v
        toy_pullback(x, η) = toy_jac(x) .* η

        toy_lh = compose(
            toy_base,
            toy_forward;
            pushforward = toy_pushforward,
            pullback = toy_pullback,
        )
        toy_xi = [0.35]

        toy_rng = MersenneTwister(7)
        toy_metric_sample = draw_metric_sample(toy_lh, toy_xi, toy_rng)
        toy_linear_draw = draw_linear_residual(
            toy_lh,
            toy_xi,
            toy_metric_sample;
            cg_rtol = 1.0e-12,
            cg_maxiter = 20,
        )

        @test toy_linear_draw.info.converged

        toy_trafo_at_point = transformation(toy_lh, toy_xi)
        initial_value, initial_gradient = GeoVI._nonlinear_residual_value_and_gradient(
            toy_lh,
            toy_xi,
            toy_trafo_at_point,
            toy_metric_sample.metric,
            toy_xi .+ toy_linear_draw.residual,
        )

        toy_update = update_nonlinear_residual(
            toy_lh,
            toy_xi,
            toy_linear_draw;
            optimizer_options = (; maxiter = 20, xtol = 1.0e-10, cg_rtol = 1.0e-12, cg_maxiter = 20),
        )

        @test toy_update.result.converged
        @test toy_update.result.value < initial_value
        @test norm(toy_update.result.gradient) < norm(initial_gradient)
        @test abs(toy_update.residual[1] - toy_linear_draw.residual[1]) > 1.0e-4

        skipped_update = update_nonlinear_residual(
            toy_lh,
            toy_xi,
            toy_linear_draw;
            optimizer_options = (; maxiter = 0),
        )

        @test skipped_update.residual == toy_linear_draw.residual
        @test skipped_update.result.skipped
        @test skipped_update.result.iterations == 0
    end

    @testset "outer VI loop" begin
        lh = GaussianLikelihood([2.0]; precision = [4.0])
        xi0 = [0.0]
        analytic_mean = [1.6]

        mgvi_family = MGVIFamily(solver = ConjugateGradient(rtol = 1.0e-12, maxiter = 10))
        outer = NewtonCG(maxiter = 12, xtol = 1.0e-10, cg_rtol = 1.0e-12, cg_maxiter = 10)
        est = MCEstimator(n_samples = 8, mirrored = true)

        mgvi_problem = VariationalProblem(
            lh,
            xi0;
            family = mgvi_family,
            divergence = ReverseKL(),
            estimator = est,
            optimizer = outer,
        )

        # in-place step_vi! mutates one VIState
        rng_step, step_state = init(MersenneTwister(5), mgvi_problem)
        step_vi!(rng_step, mgvi_problem, step_state)
        @test size(step_state.residuals, 1) == 8
        @test distribution(mgvi_problem, step_state) isa FisherGaussianDistribution

        # fit returns the fitted variational distribution
        mgvi_post = fit(mgvi_problem, 3; rng = MersenneTwister(5))
        @test mgvi_post isa AbstractVariationalDistribution
        @test _post_mean(mgvi_post) ≈ analytic_mean atol = 0.2 rtol = 0.0
        @test size(rand(MersenneTwister(7), mgvi_post, 8)) == (8, 1)   # draw fresh samples

        # the explicit loop reuses one VIState and matches fit bit-for-bit
        rng, state = init(MersenneTwister(5), mgvi_problem)
        for _ in 1:3
            step_vi!(rng, mgvi_problem, state)
        end
        @test _post_mean(distribution(mgvi_problem, state)) ≈ _post_mean(mgvi_post) atol = 1.0e-12 rtol = 1.0e-12

        # rand draws arbitrary new samples from the fitted distribution
        draws = rand(MersenneTwister(7), mgvi_post, 64)
        @test size(draws) == (64, 1)

        # geoVI: linear draw + nonlinear curve
        geovi_family = GeoVIFamily(
            solver = ConjugateGradient(rtol = 1.0e-12, maxiter = 10),
            curve = NewtonCG(maxiter = 4, xtol = 1.0e-10, cg_rtol = 1.0e-12, cg_maxiter = 10),
        )
        geovi_problem = VariationalProblem(
            lh,
            xi0;
            family = geovi_family,
            divergence = ReverseKL(),
            estimator = est,
            optimizer = outer,
        )
        rng_g, geovi_state = init(MersenneTwister(5), geovi_problem)
        step_vi!(rng_g, geovi_problem, geovi_state)
        @test geovi_problem.family isa GeoVIFamily
        geovi_post = fit(geovi_problem, 3; rng = MersenneTwister(5))
        @test _post_mean(geovi_post) ≈ analytic_mean atol = 0.2 rtol = 0.0

        # A bare Optimisers rule is the outer optimizer: each step_vi! takes one
        # gradient step, and the user owns the iteration count. With no samples
        # this optimizes the latent mean (MAP) directly.
        adam_problem = VariationalProblem(
            lh,
            xi0;
            family = MGVIFamily(),
            divergence = ReverseKL(),
            estimator = MCEstimator(n_samples = 0),
            optimizer = Optimisers.Adam(0.05),
        )
        adam_post = fit(adam_problem, 400; rng = MersenneTwister(11))
        @test _post_mean(adam_post) ≈ analytic_mean atol = 1.0e-2 rtol = 0.0

        # Adam optimizer state persists across steps: two steps (carrying
        # momentum) differ from a fresh single step at the same position.
        adam_step_problem = VariationalProblem(
            lh,
            xi0;
            family = MGVIFamily(),
            divergence = ReverseKL(),
            estimator = MCEstimator(n_samples = 0),
            optimizer = Optimisers.Adam(0.05),
        )
        rng_s, s = init(MersenneTwister(12), adam_step_problem)
        step_vi!(rng_s, adam_step_problem, s)
        pos1 = copy(s.position)
        @test s.optimizer_state !== nothing
        step_vi!(rng_s, adam_step_problem, s)
        @test s.optimizer_state !== nothing
        with_momentum = s.position[1]
        rng_f, sf = init(MersenneTwister(12), adam_step_problem)
        copyto!(sf.position, pos1)
        step_vi!(rng_f, adam_step_problem, sf)
        @test abs(with_momentum - sf.position[1]) > 1.0e-8

        # family × divergence joint dispatch
        @test GeoVI._fdivergence_value(MGVIFamily(), ReverseKL(), lh, xi0, nothing) ≈
            GeoVI._negative_logposterior(lh, xi0)
        @test_throws ArgumentError GeoVI._fdivergence_value(
            MGVIFamily(), GeoVI.ForwardKL(), lh, xi0, nothing
        )
    end

    @testset "VI phases (draw_samples! / update!)" begin
        D, M = 40, 20
        setup = _linear_gaussian_setup(MersenneTwister(0x5eed); D = D, M = M, σ² = 0.25)
        lh = compose(GaussianLikelihood(setup.data; precision = setup.precision), ξ -> setup.A * ξ)
        solver = ConjugateGradient(rtol = 1.0e-10, maxiter = 200)
        outer = NewtonCG(maxiter = 20, xtol = 1.0e-9, cg_rtol = 1.0e-10, cg_maxiter = 200)
        est = MCEstimator(n_samples = 32, mirrored = true)

        # The residual buffer is preallocated at init: (n_samples, latent...).
        mgvi = VariationalProblem(lh, zeros(D); family = MGVIFamily(solver = solver), estimator = est, optimizer = outer)
        _, st = init(MersenneTwister(1), mgvi)
        @test size(st.residuals) == (32, D)

        geovi = VariationalProblem(
            lh, zeros(D);
            family = GeoVIFamily(solver = solver, curve = NewtonCG(cg_rtol = 1.0e-10, cg_maxiter = 200)),
            estimator = est, optimizer = outer,
        )

        # `draw_samples!` is the stochastic phase: it fills the residual buffer, and two
        # successive draws (fresh noise) generally differ.
        rng, s = init(MersenneTwister(7), geovi)
        GeoVI.draw_samples!(rng, geovi, s)
        @test size(s.residuals) == (32, D)
        r1 = copy(s.residuals)
        GeoVI.draw_samples!(rng, geovi, s)
        @test !(s.residuals ≈ r1)

        # The loop is `draw_samples! → update!`: draw a fresh set each iteration, optimize
        # the mean against it (NewtonCG to convergence), and the mean reaches the posterior.
        rng2, s2 = init(MersenneTwister(9), geovi)
        for _ in 1:8
            GeoVI.draw_samples!(rng2, geovi, s2)
            GeoVI.update!(geovi, s2)
        end
        @test _post_mean(distribution(geovi, s2)) ≈ setup.μ_post atol = 0.15 rtol = 0.0

        # `step_vi!` (a fresh draw_samples!+update! per call) also converges.
        post = fit(geovi, 8; rng = MersenneTwister(9))
        @test _post_mean(post) ≈ setup.μ_post atol = 0.15 rtol = 0.0
    end

    @testset "linearization caching (perf regression)" begin
        # A counting `linearize` measures how often the forward model is
        # re-linearized. The cached `_at_point` handles must make constructions
        # scale with (samples × Newton iterations), NOT with
        # (samples × Newton iterations × CG matvecs).
        D, M = 24, 16
        setup = _linear_gaussian_setup(MersenneTwister(0xcafe); D = D, M = M, σ² = 0.25)
        lin_count = Ref(0)
        counting_linearize = x -> begin
            lin_count[] += 1
            (value = setup.A * x, pushforward = v -> setup.A * v, pullback = η -> setup.A' * η)
        end
        lh = compose(
            GaussianLikelihood(setup.data; precision = setup.precision), ξ -> setup.A * ξ;
            linearize = counting_linearize,
        )

        n_samples = 4
        newton_maxiter = 3
        problem = VariationalProblem(
            lh, zeros(D);
            family = MGVIFamily(solver = ConjugateGradient(rtol = 1.0e-10, maxiter = 100)),
            estimator = MCEstimator(n_samples = n_samples, mirrored = true),
            optimizer = NewtonCG(maxiter = newton_maxiter, xtol = 1.0e-9, cg_maxiter = 100),
        )
        rng, st = init(MersenneTwister(11), problem)

        lin_count[] = 0
        step_vi!(rng, problem, st)
        # Calibrated (measured = budget, deterministic): the draw pins exactly 2 per
        # base draw (metric-sample lift + the pinned CG operator); the update pins
        # n_samples handles per Newton iteration (the field evaluated once per
        # iteration). Line searches and CG matvecs pin nothing, so the count is
        # bounded by the maxiters — uncached it would scale with CG matvecs,
        # an order of magnitude larger.
        n_base = n_samples ÷ 2
        budget = 2 * n_base + newton_maxiter * n_samples
        @test lin_count[] <= budget

        # geoVI adds the nonlinear curve: still bounded by (curve Newton iters ×
        # per-iteration pins), independent of CG iteration counts.
        curve_maxiter = 4
        geovi_problem = VariationalProblem(
            lh, zeros(D);
            family = GeoVIFamily(
                solver = ConjugateGradient(rtol = 1.0e-10, maxiter = 100),
                curve = NewtonCG(maxiter = curve_maxiter, cg_rtol = 1.0e-10, cg_maxiter = 100),
            ),
            estimator = MCEstimator(n_samples = n_samples, mirrored = true),
            optimizer = NewtonCG(maxiter = newton_maxiter, xtol = 1.0e-9, cg_maxiter = 100),
        )
        rng_g, st_g = init(MersenneTwister(11), geovi_problem)
        lin_count[] = 0
        step_vi!(rng_g, geovi_problem, st_g)
        # Calibrated (measured = budget, deterministic): the curve adds exactly 2 pins
        # per curve-Newton iteration per base draw on top of the MGVI step; its line
        # searches pin nothing.
        curve_budget = budget + 2 * curve_maxiter * n_base
        @test lin_count[] <= curve_budget

        # And the fit still converges to the analytic posterior with caching on.
        post = fit(problem, 6; rng = MersenneTwister(2))
        @test _post_mean(post) ≈ setup.μ_post atol = 0.15 rtol = 0.0
    end

    @testset "linear-Gaussian conjugate end-to-end" begin
        rng = MersenneTwister(0x000a11ce)
        D, M = 100, 50
        setup = _linear_gaussian_setup(rng; D = D, M = M, σ² = 0.25)
        forward = ξ -> setup.A * ξ
        lh = compose(GaussianLikelihood(setup.data; precision = setup.precision), forward)
        xi0 = zeros(D)

        n_samples = 128
        est = MCEstimator(n_samples = n_samples, mirrored = true)
        solver = ConjugateGradient(rtol = 1.0e-10, maxiter = 200)
        outer = NewtonCG(maxiter = 20, xtol = 1.0e-9, cg_rtol = 1.0e-10, cg_maxiter = 200)

        families = (
            MGVIFamily(solver = solver),
            GeoVIFamily(solver = solver, curve = NewtonCG(cg_rtol = 1.0e-10, cg_maxiter = 200)),
        )
        for family in families
            problem = VariationalProblem(
                lh,
                xi0;
                family = family,
                divergence = ReverseKL(),
                estimator = est,
                optimizer = outer,
            )
            post = fit(problem, 8; rng = MersenneTwister(2025))
            @test _post_mean(post) ≈ setup.μ_post atol = 0.1 rtol = 0.0

            # `rand` from the fitted distribution recovers the posterior moments.
            # Independent (non-mirrored) draws carry Monte-Carlo noise, so the
            # sample-mean norm is bounded by the per-dimension MC error rather
            # than a tight elementwise tolerance.
            n_rand = 512
            rdraws = rand(MersenneTwister(99), post, n_rand)
            mc_err = sqrt(tr(setup.Σ_post) / n_rand)
            @test norm(vec(mean(rdraws; dims = 1)) .- _post_mean(post)) < 4 * mc_err
            rcentered = rdraws .- mean(rdraws; dims = 1)
            @test sum(abs2, rcentered) / n_rand ≈ tr(setup.Σ_post) rtol = 0.3
        end

        # geoVI with the NIFTy-style CG energy-decrease coupling switched on:
        # a calibrated `absdelta = delta·D` on both the curve and the outer
        # Newton makes the inner CG stop on quadratic-energy progress. This must
        # stay stable (no line-search failure) and still recover the posterior.
        ad = 1.0e-4 * D
        coupled = GeoVIFamily(
            solver = solver, curve = NewtonCG(absdelta = ad, cg_rtol = 1.0e-10, cg_maxiter = 200)
        )
        coupled_problem = VariationalProblem(
            lh,
            xi0;
            family = coupled,
            divergence = ReverseKL(),
            estimator = est,
            optimizer = NewtonCG(maxiter = 20, xtol = 1.0e-9, absdelta = ad, cg_rtol = 1.0e-10, cg_maxiter = 200),
        )
        coupled_post = fit(coupled_problem, 8; rng = MersenneTwister(2025))
        @test _post_mean(coupled_post) ≈ setup.μ_post atol = 0.1 rtol = 0.0

        # The `delta` convenience (per-d.o.f. tolerance) must be exactly
        # equivalent to `absdelta = delta·length(x0)`.
        delta = 1.0e-4
        @test_throws ArgumentError NewtonCG(absdelta = 1.0, delta = 1.0)
        delta_fam = GeoVIFamily(
            solver = solver, curve = NewtonCG(delta = delta, cg_rtol = 1.0e-10, cg_maxiter = 200)
        )
        delta_problem = VariationalProblem(
            lh,
            xi0;
            family = delta_fam,
            divergence = ReverseKL(),
            estimator = est,
            optimizer = NewtonCG(maxiter = 20, xtol = 1.0e-9, delta = delta, cg_rtol = 1.0e-10, cg_maxiter = 200),
        )
        delta_post = fit(delta_problem, 8; rng = MersenneTwister(2025))
        @test _post_mean(delta_post) ≈ _post_mean(coupled_post) atol = 1.0e-12 rtol = 1.0e-12
    end

    @testset "θ-container generalization + MeanFieldGaussian ADVI" begin
        # ── interface defaults: for MGVI/geoVI θ *is* the latent point ──
        xi = [0.3, -0.7, 1.2]
        r = [0.1, 0.2, -0.1]
        # default transport_and_logjac: ξ = θ .+ residual, logjac = 0 (fixed-metric)
        @test GeoVI.transport_and_logjac(MGVIFamily(), xi, r) == (r .+ xi, zero(eltype(r)))
        ip = GeoVI.init_params(MGVIFamily(), xi)
        @test ip == xi && ip !== xi   # a fresh copy

        # ── tree-generic optimizer helpers reduce to the array ops ──
        a = [1.0, 2.0, 3.0]
        b = [0.5, 1.0, 0.5]
        @test GeoVI._param_sub(a, b) == a .- b
        @test GeoVI._param_norm(a) ≈ norm(a)
        @test GeoVI._param_all_finite(a)
        @test !GeoVI._param_all_finite([1.0, Inf])
        θa = (; mean = a, logstd = b)
        θb = (; mean = b, logstd = a)
        d = GeoVI._param_sub(θa, θb)
        @test d.mean == a .- b && d.logstd == b .- a
        @test GeoVI._param_norm(θa) ≈ sqrt(norm(a)^2 + norm(b)^2)
        @test GeoVI._param_all_finite(θa)
        @test !GeoVI._param_all_finite((; mean = a, logstd = [Inf, 0.0, 0.0]))

        # ── mean-field ADVI recovers the linear-Gaussian closed form ──
        rng = MersenneTwister(0x9b16)
        D, M = 24, 14
        setup = _linear_gaussian_setup(rng; D = D, M = M, σ² = 0.25)
        lh = compose(GaussianLikelihood(setup.data; precision = setup.precision), ξ -> setup.A * ξ)
        xi0 = zeros(D)

        mf = VariationalProblem(
            lh, xi0;
            family = MeanFieldGaussian(),
            divergence = ReverseKL(),
            estimator = MCEstimator(n_samples = 128, mirrored = true),
            optimizer = Optimisers.Adam(0.05),
        )
        # θ is the structured `(; mean, logstd)` NamedTuple; no metric-tangent noise.
        rng_i, st = init(MersenneTwister(0xfeed), mf)
        @test st.position isa NamedTuple
        @test keys(st.position) == (:mean, :logstd)
        # mean-field: the residual buffer is latent-shaped white noise (n_samples, latent).
        @test st.residuals isa AbstractArray
        @test size(st.residuals) == (128, D)

        post = fit(mf, 3000; rng = MersenneTwister(0xfeed))
        # Mean-field recovers the exact posterior mean; its marginal σ is the
        # inverse-sqrt of the posterior PRECISION diagonal (it underestimates the
        # true marginal variance — a known property of mean-field).
        σ_expected = 1 ./ sqrt.(diag(I + setup.A' * Diagonal(setup.precision) * setup.A))
        @test post isa DiagonalGaussian
        @test _post_mean(post) isa AbstractVector       # the latent mean
        @test _post_mean(post) ≈ setup.μ_post atol = 0.05 rtol = 0.0
        @test exp.(post.logstd) ≈ σ_expected rtol = 0.15

        # draw fresh samples from the fitted diagonal Gaussian → recovers the marginal σ
        rdraws = rand(MersenneTwister(3), post, 4000)
        @test size(rdraws) == (4000, D)
        emp_var = vec(sum(abs2, rdraws .- mean(rdraws; dims = 1); dims = 1)) ./ 4000
        @test emp_var ≈ σ_expected .^ 2 rtol = 0.2

        # ── a NamedTuple-θ `step_vi!` moves BOTH leaves and threads the state ──
        rng_s, s = init(MersenneTwister(123), mf)
        mean0 = copy(s.position.mean)
        logstd0 = copy(s.position.logstd)
        step_vi!(rng_s, mf, s)
        @test s.position.mean != mean0
        @test s.position.logstd != logstd0
        @test s.optimizer_state !== nothing

        # ── guards: mean-field needs an Optimisers rule and n_samples > 0 ──
        @test_throws ArgumentError VariationalProblem(
            lh, xi0; family = MeanFieldGaussian(), optimizer = NewtonCG()
        )
        @test_throws ArgumentError VariationalProblem(
            lh, xi0; family = MeanFieldGaussian(),
            estimator = MCEstimator(n_samples = 0), optimizer = Optimisers.Adam(0.05),
        )
    end

    @testset "custom pushforward family (minimal interface)" begin
        # A brand-new family that is NOT one of the built-ins, implementing ONLY the
        # general pushforward surface — `init_params` + `transport_and_logjac` — and no
        # Fisher-Gaussian / metric machinery. It is a diagonal Gaussian with a single
        # *shared* scalar log-scale (distinct from mean-field's per-coordinate σ), which
        # exercises a structured θ reconstructed through `transport_and_logjac`. Proving it
        # runs through `fit` with a bare `Optimisers.jl` rule is the point of the refactor.
        struct ScalarScaleGaussian <: GeoVI.AbstractVariationalFamily end
        GeoVI.init_params(::ScalarScaleGaussian, x) =
            (; mean = copy(x), logs = fill(zero(eltype(x)), 1))
        # ξ = μ + σ⊙ε with σ = exp(logs) (shared scalar); log-Jacobian = log|det diag(σ)| = D·logs.
        GeoVI.transport_and_logjac(::ScalarScaleGaussian, θ, ε) =
            (θ.mean .+ exp(θ.logs[1]) .* ε, length(θ.mean) * θ.logs[1])
        # the fitted distribution (the first-class output) is the family's own type:
        struct ScalarScaleDist{V, T} <: GeoVI.AbstractVariationalDistribution
            mean::V
            logscale::T
        end
        GeoVI.distribution(::ScalarScaleGaussian, θ, lh) = ScalarScaleDist(θ.mean, θ.logs[1])
        Base.rand(rng::AbstractRNG, d::ScalarScaleDist) =
            d.mean .+ exp(d.logscale) .* randn_like(rng, d.mean)

        rng = MersenneTwister(0x515c)
        D, M = 20, 12
        setup = _linear_gaussian_setup(rng; D = D, M = M, σ² = 0.25)
        lh = compose(GaussianLikelihood(setup.data; precision = setup.precision), ξ -> setup.A * ξ)

        problem = VariationalProblem(
            lh, zeros(D);
            family = ScalarScaleGaussian(),
            estimator = MCEstimator(n_samples = 64, mirrored = true),
            optimizer = Optimisers.Adam(0.05),
        )   # no solver, no metric — the default `draw_samples!` supplies white noise.

        # Init allocates the latent-shaped residual buffer (white noise; no metric tangent).
        rng_i, st = init(MersenneTwister(0x01), problem)
        @test st.position isa NamedTuple && keys(st.position) == (:mean, :logs)
        @test st.residuals isa AbstractArray && size(st.residuals) == (64, D)

        post = fit(problem, 3000; rng = MersenneTwister(0x01))
        @test post isa ScalarScaleDist
        @test post.mean ≈ setup.μ_post atol = 0.05 rtol = 0.0
        # draw fresh samples from the fitted custom distribution
        draws = rand(MersenneTwister(5), post, 64)
        @test size(draws) == (64, D)
    end

    @testset "Reactant extension" begin
        if !HAS_REACTANT
            @info "Skipping Reactant tests because `Reactant` is not available in the active environment."
        else
            rng = MersenneTwister(0x0b0b)
            D, M = 64, 32
            setup = _linear_gaussian_setup(rng; D = D, M = M, σ² = 0.25)
            A_r = Reactant.to_rarray(Float32.(setup.A))
            data_r = Reactant.to_rarray(Float32.(setup.data))
            precision_r = Reactant.to_rarray(Float32.(setup.precision))
            xi0_r = Reactant.to_rarray(zeros(Float32, D))
            forward = ξ -> A_r * ξ
            lh = compose(
                GaussianLikelihood(data_r; precision = precision_r),
                forward;
                adtype = GeoVI.ADTypes.AutoEnzyme(),
            )

            n_samples = 128
            problem = VariationalProblem(
                lh,
                xi0_r;
                family = MGVIFamily(solver = ConjugateGradient(rtol = 1.0f-6, maxiter = 200)),
                divergence = ReverseKL(),
                estimator = MCEstimator(n_samples = n_samples, mirrored = true),
                optimizer = NewtonCG(maxiter = 20, xtol = 1.0f-6),
                adtype = GeoVI.ADTypes.AutoEnzyme(),
            )

            # init wraps the host RNG into a ReactantRNG for the compiled path.
            rng, state = init(MersenneTwister(0xfeed), problem)
            @test rng isa Reactant.ReactantRNG

            # `fit` compiles `step_vi!` once (via `_run_vi!(::AutoReactant,...)`)
            # and loops the compiled thunk.
            post = fit(problem, 8; rng = MersenneTwister(0xfeed))
            @test post isa FisherGaussianDistribution
            position_host = Array(_post_mean(post))
            @test position_host ≈ Float32.(setup.μ_post) atol = 0.2 rtol = 0.0
            # (drawing fresh MGVI samples from the fitted distribution is a CG solve, which
            # under Reactant needs compilation; the sample-moment recovery is covered on the
            # CPU path above, so here we only check the fitted mean.)

            # The user can compile `step_vi!` themselves and loop the compiled thunk.
            rng2, state2 = init(MersenneTwister(0xfeed), problem)
            cstep = Reactant.@compile step_vi!(rng2, problem, state2)
            for _ in 1:8
                cstep(rng2, problem, state2)
            end
            @test Array(state2.position) ≈ position_host atol = 1.0f-4 rtol = 0.0

            # ── mean-field ADVI under Reactant ──
            # The generic NamedTuple θ `(; mean, logstd)` is allocated and wrapped
            # at `init`, `@compile step_vi!` traces it, and the rule-optimizer loop
            # (`_run_optimizer_rule`) compiles now that its body is branchless
            # (`_select_pick`/`_select_scalar`, like the NewtonCG loop) rather than
            # rebinding the carried iterate inside `@trace if`. Both leaves of θ
            # must advance and recover the posterior moments.
            xi0_mf = Reactant.to_rarray(zeros(Float32, D))
            mf_problem = VariationalProblem(
                lh, xi0_mf;
                family = MeanFieldGaussian(),
                divergence = ReverseKL(),
                estimator = MCEstimator(n_samples = n_samples, mirrored = true),
                optimizer = Optimisers.Adam(0.05),
                adtype = GeoVI.ADTypes.AutoEnzyme(),
            )
            rng_mf, state_mf = init(MersenneTwister(0xabcd), mf_problem)
            @test state_mf.position isa NamedTuple   # generic θ container under Reactant
            @test keys(state_mf.position) == (:mean, :logstd)

            mf_post = fit(mf_problem, 3000; rng = MersenneTwister(0xabcd))
            σ_expected = 1 ./ sqrt.(diag(I + setup.A' * Diagonal(setup.precision) * setup.A))
            @test mf_post isa DiagonalGaussian
            @test Array(_post_mean(mf_post)) ≈ Float32.(setup.μ_post) atol = 0.2 rtol = 0.0
            @test Array(exp.(mf_post.logstd)) ≈ Float32.(σ_expected) rtol = 0.35

            # ── optimizer state must propagate across compiled step_vi! calls ──
            # `update!` REASSIGNS `state.optimizer_state` (unlike `position`,
            # which is written in place via `fmap(copyto!, ...)`). This guards
            # that the updated Adam moments actually reach the host `VIState`
            # between compiled calls instead of being silently frozen at zero —
            # a failure the moment-recovery tolerances above would NOT catch.
            rng_os, state_os = init(MersenneTwister(0x7777), mf_problem)
            cstep_os = Reactant.@compile step_vi!(rng_os, mf_problem, state_os)
            adam_m(s) = Array(s.optimizer_state.tree.mean.state[1])  # Adam 1st moment, mean leaf
            m0 = adam_m(state_os)
            @test all(iszero, m0)
            cstep_os(rng_os, mf_problem, state_os)
            m1 = adam_m(state_os)
            cstep_os(rng_os, mf_problem, state_os)
            m2 = adam_m(state_os)
            @test any(!iszero, m1)   # momentum accumulated after the first step
            @test m1 != m2           # and keeps advancing on the second
        end
    end
end
