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
        @test state.iteration == 0
        @test size(state.residuals) == (6, 1)

        @test_throws ArgumentError update_nonlinear_residual(simple_lh, [0.0], [0.0])
        @test_throws ArgumentError VariationalProblem(
            simple_lh, [0.0]; divergence = ForwardKL(), optimizer = NewtonCG()
        )
        @test_throws ArgumentError VariationalProblem(
            simple_lh, [0.0]; divergence = ReverseKL(), optimizer = :adam
        )

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
        @test step_state.iteration == 1
        @test size(step_state.residuals, 1) == 8
        @test length(posterior(mgvi_problem, step_state).samples.keys) == 4

        # fit returns a VariationalPosterior
        mgvi_post = fit(mgvi_problem, 3; rng = MersenneTwister(5))
        @test mgvi_post isa VariationalPosterior
        @test mean(mgvi_post) ≈ analytic_mean atol = 0.2 rtol = 0.0
        @test length(mgvi_post.samples) == 8
        @test size(posterior_samples(mgvi_post.samples)) == (8, 1)

        # the explicit loop reuses one VIState and matches fit bit-for-bit
        rng, state = init(MersenneTwister(5), mgvi_problem)
        for _ in 1:3
            step_vi!(rng, mgvi_problem, state)
        end
        @test state.iteration == 3
        @test mean(posterior(mgvi_problem, state)) ≈ mean(mgvi_post) atol = 1.0e-12 rtol = 1.0e-12

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
        @test mean(geovi_post) ≈ analytic_mean atol = 0.2 rtol = 0.0

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
        @test mean(adam_post) ≈ analytic_mean atol = 1.0e-2 rtol = 0.0

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
            MGVIFamily(), ForwardKL(), lh, xi0, nothing
        )
    end

    @testset "VI phases (sample! / transform! / update!)" begin
        D, M = 40, 20
        setup = _linear_gaussian_setup(MersenneTwister(0x5eed); D = D, M = M, σ² = 0.25)
        lh = compose(GaussianLikelihood(setup.data; precision = setup.precision), ξ -> setup.A * ξ)
        solver = ConjugateGradient(rtol = 1.0e-10, maxiter = 200)
        outer = NewtonCG(maxiter = 20, xtol = 1.0e-9, cg_rtol = 1.0e-10, cg_maxiter = 200)
        est = MCEstimator(n_samples = 32, mirrored = true)

        # White-noise buffers preallocated at init (one row per base draw).
        mgvi = VariationalProblem(lh, zeros(D); family = MGVIFamily(solver = solver), estimator = est, optimizer = outer)
        _, st = init(MersenneTwister(1), mgvi)
        @test size(st.metric_white) == (16, M)   # n_base = 32 / 2 (mirrored)
        @test size(st.prior_white) == (16, D)

        geovi = VariationalProblem(
            lh, zeros(D);
            family = GeoVIFamily(solver = solver, curve = NewtonCG(cg_rtol = 1.0e-10, cg_maxiter = 200)),
            estimator = est, optimizer = outer,
        )

        # `sample!` is the only stochastic phase; `transform!` is a deterministic
        # function of the stored noise + mean, so two transforms at the same mean
        # give the same residuals (this is what enables recompute-Fisher reuse).
        rng, s = init(MersenneTwister(7), geovi)
        GeoVI.sample!(rng, geovi, s)
        GeoVI.transform!(geovi, s)
        r1 = copy(s.residuals)
        GeoVI.transform!(geovi, s)        # same noise, same mean → identical
        @test s.residuals ≈ r1
        GeoVI.sample!(rng, geovi, s)      # fresh noise → generally different
        GeoVI.transform!(geovi, s)
        @test !(s.residuals ≈ r1)

        # geoVI converges via the recompute-Fisher refinement: draw the noise once,
        # then re-`transform!` (recompute the Fisher at the moved mean) + `update!`.
        rng2, s2 = init(MersenneTwister(9), geovi)
        GeoVI.sample!(rng2, geovi, s2)
        for _ in 1:8
            GeoVI.transform!(geovi, s2)
            GeoVI.update!(geovi, s2)
        end
        @test mean(posterior(geovi, s2)) ≈ setup.μ_post atol = 0.15 rtol = 0.0

        # `step_vi!` (a fresh sample!+transform!+update! per call) also converges.
        post = fit(geovi, 8; rng = MersenneTwister(9))
        @test mean(post) ≈ setup.μ_post atol = 0.15 rtol = 0.0
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
            @test mean(post) ≈ setup.μ_post atol = 0.1 rtol = 0.0

            draws = posterior_samples(post.samples)
            @test size(draws) == (n_samples, D)
            @test vec(mean(draws; dims = 1)) ≈ setup.μ_post atol = 0.15 rtol = 0.0

            centered = draws .- mean(draws; dims = 1)
            tr_emp = sum(abs2, centered) / n_samples
            @test tr_emp ≈ tr(setup.Σ_post) rtol = 0.25

            # `rand` from the fitted distribution recovers the posterior moments.
            # Independent (non-mirrored) draws carry Monte-Carlo noise, so the
            # sample-mean norm is bounded by the per-dimension MC error rather
            # than a tight elementwise tolerance.
            n_rand = 512
            rdraws = rand(MersenneTwister(99), post, n_rand)
            mc_err = sqrt(tr(setup.Σ_post) / n_rand)
            @test norm(vec(mean(rdraws; dims = 1)) .- mean(post)) < 4 * mc_err
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
        @test mean(coupled_post) ≈ setup.μ_post atol = 0.1 rtol = 0.0

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
        @test mean(delta_post) ≈ mean(coupled_post) atol = 1.0e-12 rtol = 1.0e-12
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

            rng, state = init(MersenneTwister(0xfeed), problem)
            @test rng isa Reactant.ReactantRNG
            for _ in 1:8
                step_vi!(rng, problem, state)
            end
            @test state.iteration == 8
            @test nameof(typeof(state.cache)) == :ReactantVIStepCache

            post = posterior(problem, state)
            position_host = Array(mean(post))
            @test position_host ≈ Float32.(setup.μ_post) atol = 0.2 rtol = 0.0

            draws_host = Array(posterior_samples(post.samples))
            @test size(draws_host) == (n_samples, D)
            @test vec(mean(draws_host; dims = 1)) ≈ Float32.(setup.μ_post) atol = 0.25 rtol = 0.0

            centered = draws_host .- mean(draws_host; dims = 1)
            tr_emp = sum(abs2, centered) / n_samples
            @test tr_emp ≈ tr(setup.Σ_post) rtol = 0.35

            # AutoReactant requires a ReactantRNG; a plain RNG must error.
            _, bypass_state = init(MersenneTwister(0xfeed), problem)
            @test_throws ArgumentError step_vi!(MersenneTwister(0), problem, bypass_state)
        end
    end
end
