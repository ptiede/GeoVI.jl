const HAS_REACTANT = Base.find_package("Reactant") !== nothing

if HAS_REACTANT
    using Reactant
end

using GeoVI
using Test
using LinearAlgebra
using Optimisers
using Random

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
        lh = GaussianLikelihood(data; precision=precision)

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
        mat_lh = GaussianLikelihood(data; precision=mat_precision, sqrt_precision=mat_sqrt)
        @test fishermetric(mat_lh, y, v) ≈ mat_precision * v

        @test_throws ArgumentError GaussianLikelihood(data; precision=x -> 2 .* x)
    end

    @testset "likelihood composition" begin
        data = [1.0, -2.0]
        precision = [3.0, 5.0]
        base = GaussianLikelihood(data; precision=precision)

        A = [1.0 2.0; -1.0 0.5]
        forward(x) = A * x
        pushforward(x, v) = A * v
        pullback(x, η) = A' * η

        x = [0.4, -0.7]
        v = [1.2, -0.5]
        η = [0.3, 2.0]
        y = forward(x)

        manual = compose(base, forward; pushforward=pushforward, pullback=pullback)

        @test logdensity(manual, x) ≈ logdensity(base, y)
        @test normalized_residual(manual, x) ≈ normalized_residual(base, y)
        @test transformation(manual, x) ≈ transformation(base, y)
        @test rightsqrtmetric(manual, x, v) ≈ rightsqrtmetric(base, y, A * v)
        @test leftsqrtmetric(manual, x, η) ≈ A' * leftsqrtmetric(base, y, η)
        @test fishermetric(manual, x, v) ≈ A' * (precision .* (A * v))

        linearize(x) = (
            value=forward(x),
            pushforward=v -> A * v,
            pullback=η -> A' * η,
        )
        bundled = compose(base, forward; linearize=linearize)
        @test rightsqrtmetric(bundled, x, v) ≈ rightsqrtmetric(base, y, A * v)
        @test leftsqrtmetric(bundled, x, η) ≈ A' * leftsqrtmetric(base, y, η)
        @test fishermetric(bundled, x, v) ≈ A' * (precision .* (A * v))

        struct ToyLinearization{T}
            value::T
            jacobian::Matrix{Float64}
        end
        GeoVI.pushforward(lin::ToyLinearization, v::AbstractArray) = lin.jacobian * v
        GeoVI.pullback(lin::ToyLinearization, η::AbstractArray) = lin.jacobian' * η
        method_based = compose(base, forward; linearize=x -> ToyLinearization(forward(x), A))
        @test rightsqrtmetric(method_based, x, v) ≈ rightsqrtmetric(base, y, A * v)
        @test leftsqrtmetric(method_based, x, η) ≈ A' * leftsqrtmetric(base, y, η)
        @test fishermetric(method_based, x, v) ≈ A' * (precision .* (A * v))

        automatic = compose(base, forward)
        @test logdensity(automatic, x) ≈ logdensity(base, y)
        @test normalized_residual(automatic, x) ≈ normalized_residual(base, y)
        @test transformation(automatic, x) ≈ transformation(base, y)
        @test rightsqrtmetric(automatic, x, v) ≈ rightsqrtmetric(base, y, A * v) atol = 1e-6 rtol = 1e-6
        @test leftsqrtmetric(automatic, x, η) ≈ A' * leftsqrtmetric(base, y, η) atol = 1e-6 rtol = 1e-6
        @test fishermetric(automatic, x, v) ≈ A' * (precision .* (A * v)) atol = 1e-5 rtol = 1e-5

        noauto = compose(base, forward; adtype=GeoVI.ADTypes.NoAutoDiff())
        @test_throws ArgumentError rightsqrtmetric(noauto, x, v)
        @test_throws ArgumentError leftsqrtmetric(noauto, x, η)
        @test_throws ArgumentError compose(base, forward; linearize=linearize, pushforward=pushforward)
    end

    @testset "linearization interface" begin
        A = [2.0 -1.0; 0.5 3.0]
        forward(x) = A * x
        x = [0.3, -0.8]
        v = [1.1, -0.4]
        η = [0.7, -1.5]

        finite_diff = GeoVI._automatic_linearize(GeoVI.ADTypes.AutoFiniteDiff(), forward, x)
        @test finite_diff.value ≈ forward(x)
        @test GeoVI.pushforward(finite_diff, v) ≈ A * v atol = 1e-6 rtol = 1e-6
        @test GeoVI.pullback(finite_diff, η) ≈ A' * η atol = 1e-6 rtol = 1e-6
    end

    @testset "exponential-family likelihoods" begin
        v = [0.3, -0.5]
        eps = 1e-6

        poisson = PoissonLikelihood([2.0, 4.0]; weight=[1.5, 0.5])
        ηp = log.([3.0, 5.0])
        λ = exp.(ηp)
        @test logdensity(poisson, ηp) ≈ -sum([1.5, 0.5] .* (λ .- [2.0, 4.0] .* ηp))
        @test normalized_residual(poisson, ηp) ≈ sqrt.([1.5, 0.5]) .* ([2.0, 4.0] .- λ) ./ sqrt.(λ)
        @test leftsqrtmetric(poisson, ηp, v) ≈ sqrt.([1.5, 0.5] .* λ) .* v
        @test rightsqrtmetric(poisson, ηp, v) ≈ sqrt.([1.5, 0.5] .* λ) .* v
        @test fishermetric(poisson, ηp, v) ≈ ([1.5, 0.5] .* λ) .* v
        @test (
            transformation(poisson, ηp .+ eps .* v) .- transformation(poisson, ηp .- eps .* v)
        ) ./ (2 * eps) ≈ rightsqrtmetric(poisson, ηp, v) atol = 1e-6 rtol = 1e-6
        @test -leftsqrtmetric(poisson, ηp, normalized_residual(poisson, ηp)) ≈
            [1.5, 0.5] .* (λ .- [2.0, 4.0])

        bernoulli = BernoulliLikelihood([1.0, 0.0]; weight=[2.0, 0.75])
        ηb = [0.3, -0.4]
        p = 1 ./(1 .+ exp.(-ηb))
        @test logdensity(bernoulli, ηb) ≈
            -sum([2.0, 0.75] .* (log1p.(exp.(ηb)) .- [1.0, 0.0] .* ηb))
        @test fishermetric(bernoulli, ηb, v) ≈ ([2.0, 0.75] .* p .* (1 .- p)) .* v
        @test (
            transformation(bernoulli, ηb .+ eps .* v) .- transformation(bernoulli, ηb .- eps .* v)
        ) ./ (2 * eps) ≈ rightsqrtmetric(bernoulli, ηb, v) atol = 1e-6 rtol = 1e-6
        @test -leftsqrtmetric(bernoulli, ηb, normalized_residual(bernoulli, ηb)) ≈
            [2.0, 0.75] .* (p .- [1.0, 0.0])

        binomial = BinomialLikelihood([3.0, 1.0]; trials=[5.0, 2.0], weight=[1.0, 0.5])
        ηn = [0.2, -0.1]
        q = 1 ./(1 .+ exp.(-ηn))
        μ = [5.0, 2.0] .* q
        @test logdensity(binomial, ηn) ≈
            -sum([1.0, 0.5] .* ([5.0, 2.0] .* log1p.(exp.(ηn)) .- [3.0, 1.0] .* ηn))
        @test fishermetric(binomial, ηn, v) ≈
            ([1.0, 0.5] .* [5.0, 2.0] .* q .* (1 .- q)) .* v
        @test (
            transformation(binomial, ηn .+ eps .* v) .- transformation(binomial, ηn .- eps .* v)
        ) ./ (2 * eps) ≈ rightsqrtmetric(binomial, ηn, v) atol = 1e-6 rtol = 1e-6
        @test -leftsqrtmetric(binomial, ηn, normalized_residual(binomial, ηn)) ≈
            [1.0, 0.5] .* (μ .- [3.0, 1.0])

        @test_throws ArgumentError BernoulliLikelihood([0.0, 0.5])
        @test_throws ArgumentError BinomialLikelihood([2.0]; trials=[1.0])
    end

    @testset "samples" begin
        position = [10.0, 20.0]
        residuals = [1.0 2.0; 3.0 4.0]
        samples = Samples(position, residuals; keys=[:a, :b])

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
        cfg = VIConfig(n_iterations=4, n_samples=6, mirrored=true)
        @test cfg.adtype isa GeoVI.ADTypes.AutoFiniteDiff
        @test cfg.n_iterations == 4
        @test cfg.n_samples == 6
        @test cfg.draw_linear == (;)
        @test cfg.optimizer_options == (;)
        @test GeoVI._infer_adtype(cfg.adtype, [1.0, 2.0]) isa GeoVI.ADTypes.AutoFiniteDiff
        @test GeoVI._value_and_gradient(cfg.adtype, x -> sum(abs2, x), [1.0, 2.0])[2] ≈ [2.0, 4.0] atol = 1e-5

        cfg_alias = VIConfig(optimizer_options=(; maxiter=10), n_samples=4, adtype=nothing)
        @test cfg_alias.adtype isa GeoVI.ADTypes.AutoFiniteDiff
        @test cfg_alias.optimizer_options == (; maxiter=10)

        state = initialize_vi(:rng; config=cfg)
        @test state.iteration == 0
        @test state.rng == :rng
        @test state.sample_state === nothing

        @test_throws ArgumentError VIConfig(n_iterations=1, n_samples=3, mirrored=true)

        simple_lh = GaussianLikelihood([0.0]; precision=[1.0])
        problem = VariationalProblem(
            simple_lh,
            [0.0];
            family=MGVIFamily(),
            divergence=ReverseKL(),
            optimizer=NewtonCG(),
            config=cfg,
        )
        @test problem.adtype isa GeoVI.ADTypes.AutoFiniteDiff
        @test problem.n_base_draws == 3
        @test problem.optimizer_options.maxiter == 20
        problem_state = initialize_vi(problem, MersenneTwister(2))
        @test problem_state.iteration == 0
        @test problem_state.rng isa MersenneTwister

        @test_throws ArgumentError update_nonlinear_residual(simple_lh, [0.0], [0.0])
        @test_throws ArgumentError fit(
            simple_lh,
            [0.0];
            family=GeoVIFamily(),
            divergence=ForwardKL(),
            optimizer=NewtonCG(),
            config=VIConfig(n_iterations=1),
            rng=MersenneTwister(1),
        )
        @test_throws ArgumentError fit(
            simple_lh,
            [0.0];
            family=GeoVIFamily(),
            divergence=ReverseKL(),
            optimizer=:adam,
            config=VIConfig(n_iterations=1),
            rng=MersenneTwister(1),
        )
        @test_throws ArgumentError fit(
            simple_lh,
            [0.0];
            family=GeoVIFamily(),
            divergence=ReverseKL(),
            optimizer=NewtonCG(),
            config=VIConfig(n_iterations=1, adtype=GeoVI.ADTypes.NoAutoDiff()),
            rng=MersenneTwister(1),
        )
    end

    @testset "MGVI linear residuals" begin
        precision = [3.0, 5.0]
        base = GaussianLikelihood([0.0, 0.0]; precision=precision)

        A = [1.0 2.0; -1.0 0.5]
        forward(x) = A * x
        pushforward(x, v) = A * v
        pullback(x, η) = A' * η

        lh = compose(base, forward; pushforward=pushforward, pullback=pullback)
        xi = [0.2, -0.1]

        posterior_metric = I + A' * Diagonal(precision) * A

        rng_metric = MersenneTwister(11)
        metric_draw = draw_metric_sample(lh, xi, rng_metric)

        rng_residual = MersenneTwister(11)
        residual_draw = draw_linear_residual(
            lh,
            xi,
            rng_residual;
            cg_rtol=1e-12,
            cg_maxiter=10,
        )
        @test residual_draw.info.converged
        @test residual_draw.info.iterations > 0
        @test residual_draw.residual ≈ posterior_metric \ metric_draw.metric atol = 1e-10 rtol = 1e-10

        @test_throws ErrorException draw_linear_residual(
            lh,
            xi,
            MersenneTwister(11);
            cg_maxiter=0,
        )

        stalled_draw = draw_linear_residual(
            lh,
            xi,
            MersenneTwister(11);
            cg_maxiter=0,
            throw_on_failure=false,
        )
        @test !stalled_draw.info.converged
        @test stalled_draw.info.iterations == 0
        @test size(stalled_draw.residual) == size(xi)

        n_draws = 4_000
        draws = Matrix{eltype(xi)}(undef, 2, n_draws)
        rng = MersenneTwister(23)
        for i in 1:n_draws
            draw = draw_linear_residual(lh, xi, rng; cg_rtol=1e-10, cg_maxiter=10)
            @test draw.info.converged
            draws[:, i] = draw.residual
        end

        mean_draw = vec(sum(draws; dims=2) ./ n_draws)
        centered = draws .- reshape(mean_draw, :, 1)
        empirical_cov = centered * centered' / (n_draws - 1)
        analytic_cov = inv(Matrix(posterior_metric))

        @test norm(mean_draw) < 0.06
        @test empirical_cov ≈ analytic_cov atol = 0.035 rtol = 0.15
    end

    @testset "geoVI nonlinear residuals" begin
        precision = [3.0, 5.0]
        base = GaussianLikelihood([0.0, 0.0]; precision=precision)

        A = [1.0 2.0; -1.0 0.5]
        forward(x) = A * x
        pushforward(x, v) = A * v
        pullback(x, η) = A' * η

        lh = compose(base, forward; pushforward=pushforward, pullback=pullback)
        xi = [0.2, -0.1]

        rng = MersenneTwister(11)
        metric_sample = draw_metric_sample(lh, xi, rng)
        linear_draw = draw_linear_residual(
            lh,
            xi,
            metric_sample;
            cg_rtol=1e-12,
            cg_maxiter=10,
        )

        @test linear_draw.info.converged

        update = update_nonlinear_residual(
            lh,
            xi,
            linear_draw;
            optimizer_options=(; xtol=1e-10, cg_rtol=1e-12, cg_maxiter=10),
        )

        @test update.result.converged
        @test update.residual ≈ linear_draw.residual atol = 1e-12 rtol = 1e-12
        @test norm(update.result.gradient) < 1e-12
        @test update.result.value < 1e-24

        mirrored_draw = draw_residual(
            lh,
            xi,
            MersenneTwister(11);
            draw_linear_kwargs=(; cg_rtol=1e-12, cg_maxiter=10),
            optimizer_options=(; xtol=1e-10, cg_rtol=1e-12, cg_maxiter=10),
        )

        @test size(mirrored_draw.residuals) == (2, length(xi))
        @test mirrored_draw.linear.info.converged
        @test mirrored_draw.positive.result.converged
        @test mirrored_draw.negative.result.converged
        @test mirrored_draw.residuals[1, :] ≈ linear_draw.residual atol = 1e-12 rtol = 1e-12
        @test mirrored_draw.residuals[2, :] ≈ -linear_draw.residual atol = 1e-12 rtol = 1e-12
        @test norm(mirrored_draw.residuals[1, :] + mirrored_draw.residuals[2, :]) < 1e-12

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
            metric_sample=metric_sample,
            optimizer=Optimisers.Adam(0.05),
            optimizer_options=(; maxiter=400, miniter=50, xtol=1e-10, absdelta=1e-12),
        )

        @test adam_update.result.converged
        @test adam_update.result.optimizer_state !== nothing
        @test adam_update.result.value < adam_initial_value
        @test norm(adam_update.result.gradient) < norm(adam_initial_gradient)
        @test adam_update.residual ≈ linear_draw.residual atol = 1e-3 rtol = 1e-3

        toy_base = GaussianLikelihood([0.0]; precision=[4.0])
        toy_forward(x) = x .+ 0.25 .* x .^ 3
        toy_jac(x) = 1 .+ 0.75 .* x .^ 2
        toy_pushforward(x, v) = toy_jac(x) .* v
        toy_pullback(x, η) = toy_jac(x) .* η

        toy_lh = compose(
            toy_base,
            toy_forward;
            pushforward=toy_pushforward,
            pullback=toy_pullback,
        )
        toy_xi = [0.35]

        toy_rng = MersenneTwister(7)
        toy_metric_sample = draw_metric_sample(toy_lh, toy_xi, toy_rng)
        toy_linear_draw = draw_linear_residual(
            toy_lh,
            toy_xi,
            toy_metric_sample;
            cg_rtol=1e-12,
            cg_maxiter=20,
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
            optimizer_options=(; maxiter=20, xtol=1e-10, cg_rtol=1e-12, cg_maxiter=20),
        )

        @test toy_update.result.converged
        @test toy_update.result.value < initial_value
        @test norm(toy_update.result.gradient) < norm(initial_gradient)
        @test abs(toy_update.residual[1] - toy_linear_draw.residual[1]) > 1e-4

        skipped_update = update_nonlinear_residual(
            toy_lh,
            toy_xi,
            toy_linear_draw;
            optimizer_options=(; maxiter=0),
        )

        @test skipped_update.residual == toy_linear_draw.residual
        @test skipped_update.result.skipped
        @test skipped_update.result.iterations == 0
    end

    @testset "outer VI loop" begin
        lh = GaussianLikelihood([2.0]; precision=[4.0])
        xi0 = [0.0]
        analytic_mean = [1.6]

        mgvi_cfg = VIConfig(
            n_iterations=3,
            n_samples=8,
            mirrored=true,
            draw_linear=(; cg_rtol=1e-12, cg_maxiter=10),
            optimizer_options=(; maxiter=12, xtol=1e-10, cg_rtol=1e-12, cg_maxiter=10, fd_eps=1e-6),
        )

        mgvi_problem = VariationalProblem(
            lh,
            xi0;
            family=MGVIFamily(),
            divergence=ReverseKL(),
            optimizer=NewtonCG(),
            config=mgvi_cfg,
        )
        step_state = initialize_vi(mgvi_problem, MersenneTwister(5))
        step_samples, step_state = step_vi(mgvi_problem, xi0, step_state)
        @test step_state.iteration == 1
        @test step_state.sample_state.family isa MGVIFamily
        @test step_state.minimization_state.converged
        @test length(step_samples) == 8
        @test length(step_samples.keys) == 4

        mgvi_problem_samples, mgvi_problem_state = fit(mgvi_problem; rng=MersenneTwister(5))
        @test mgvi_problem_state.iteration == mgvi_cfg.n_iterations
        @test mgvi_problem_samples.position ≈ analytic_mean atol = 0.2 rtol = 0.0

        mgvi_samples, mgvi_state = fit(
            lh,
            xi0,
            MGVIFamily(),
            ReverseKL(),
            NewtonCG();
            config=mgvi_cfg,
            rng=MersenneTwister(5),
        )
        @test mgvi_state.iteration == mgvi_cfg.n_iterations
        @test mgvi_state.sample_state.family isa MGVIFamily
        @test mgvi_state.minimization_state.converged
        @test length(mgvi_samples) == mgvi_cfg.n_samples
        @test size(posterior_samples(mgvi_samples)) == (mgvi_cfg.n_samples, 1)
        @test mgvi_samples.position ≈ analytic_mean atol = 0.2 rtol = 0.0

        geovi_cfg = VIConfig(
            n_iterations=3,
            n_samples=8,
            mirrored=true,
            draw_linear=(; cg_rtol=1e-12, cg_maxiter=10),
            nonlinear_update=(; maxiter=4, xtol=1e-10, cg_rtol=1e-12, cg_maxiter=10),
            optimizer_options=(; maxiter=12, xtol=1e-10, cg_rtol=1e-12, cg_maxiter=10, fd_eps=1e-6),
        )

        geovi_samples, geovi_state = fit(
            lh,
            xi0;
            family=GeoVIFamily(),
            divergence=ReverseKL(),
            optimizer=NewtonCG(),
            config=geovi_cfg,
            rng=MersenneTwister(5),
        )
        @test geovi_state.iteration == geovi_cfg.n_iterations
        @test geovi_state.sample_state.family isa GeoVIFamily
        @test geovi_state.minimization_state.converged
        @test length(geovi_samples) == geovi_cfg.n_samples
        @test geovi_samples.position ≈ analytic_mean atol = 0.2 rtol = 0.0

        alias_samples, alias_state = fit(
            lh,
            xi0;
            family=MGVIFamily(),
            divergence=ReverseKL(),
            optimizer=NewtonCG(),
            config=mgvi_cfg,
            rng=MersenneTwister(5),
        )
        @test alias_state.iteration == mgvi_cfg.n_iterations
        @test alias_samples.position ≈ mgvi_samples.position atol = 1e-12 rtol = 1e-12

        kw_samples, kw_state = fit(
            MersenneTwister(9),
            lh,
            xi0;
            family=MGVIFamily(),
            divergence=ReverseKL(),
            optimizer=NewtonCG(),
            config=VIConfig(
                n_iterations=1,
                n_samples=4,
                mirrored=true,
                draw_linear=(; cg_rtol=1e-12, cg_maxiter=10),
                optimizer_options=(; maxiter=8, xtol=1e-10, cg_rtol=1e-12, cg_maxiter=10, fd_eps=1e-6),
            ),
        )
        @test kw_state.iteration == 1
        @test length(kw_samples) == 4

        adam = Optimisers.Adam(0.05)
        adam_cfg = VIConfig(
            n_iterations=1,
            n_samples=0,
            mirrored=true,
            optimizer_options=(; maxiter=400, miniter=50, xtol=1e-10, absdelta=1e-12, fd_eps=1e-6),
        )
        adam_samples, adam_state = fit(
            lh,
            xi0,
            MGVIFamily(),
            ReverseKL(),
            adam;
            config=adam_cfg,
            rng=MersenneTwister(11),
        )
        @test adam_state.iteration == 1
        @test adam_state.minimization_state.converged
        @test adam_state.minimization_state.optimizer_state !== nothing
        @test adam_samples.position ≈ analytic_mean atol = 1e-2 rtol = 0.0

        adam_step_cfg = VIConfig(
            n_iterations=0,
            n_samples=0,
            mirrored=true,
            optimizer_options=(; maxiter=1, xtol=0.0, fd_eps=1e-6),
        )
        adam_problem = VariationalProblem(
            lh,
            xi0;
            family=MGVIFamily(),
            divergence=ReverseKL(),
            optimizer=adam,
            config=adam_step_cfg,
        )
        adam_step_state0 = initialize_vi(adam_problem, MersenneTwister(12))
        adam_step1, adam_step_state1 = step_vi(adam_problem, xi0, adam_step_state0)
        adam_step2, adam_step_state2 = step_vi(adam_problem, adam_step1, adam_step_state1)
        fresh_second = GeoVI._update_position(
            adam_problem,
            Samples(adam_step1.position, nothing; keys=nothing),
            nothing,
        )
        @test adam_step_state1.minimization_state.optimizer_state !== nothing
        @test adam_step_state2.minimization_state.optimizer_state !== nothing
        @test abs(adam_step2.position[1] - fresh_second.x[1]) > 1e-8
    end

    @testset "Reactant extension" begin
        if !HAS_REACTANT
            @info "Skipping Reactant tests because `Reactant` is not available in the active environment."
        else
            data = Reactant.to_rarray(Float32[2.0])
            precision = Reactant.to_rarray(Float32[4.0])
            xi0 = Reactant.to_rarray(Float32[0.0])
            lh = GaussianLikelihood(data; precision=precision)
            optimizer = Optimisers.Adam(0.05f0)

            cfg = VIConfig(
                n_iterations=0,
                n_samples=2,
                mirrored=true,
                adtype=GeoVI.ADTypes.AutoFiniteDiff(),
                draw_linear=(; cg_rtol=1f-5, cg_maxiter=10),
                optimizer_options=(; maxiter=1, xtol=0.0, fd_eps=1f-4),
            )

            problem = VariationalProblem(
                lh,
                xi0;
                family=MGVIFamily(),
                divergence=ReverseKL(),
                optimizer=optimizer,
                config=cfg,
            )
            state0 = initialize_vi(problem, Reactant.ReactantRNG())
            samples1, state1 = step_vi(problem, xi0, state0)
            samples2, state2 = step_vi(problem, samples1, state1)

            @test state1.iteration == 1
            @test state2.iteration == 2
            @test length(samples1) == cfg.n_samples
            @test state1.cache !== nothing
            @test nameof(typeof(state1.cache)) == :ReactantVIStepCache
            @test state2.cache === state1.cache
            @test state1.minimization_state.optimizer_state !== nothing
            @test state2.minimization_state.optimizer_state !== nothing
        end
    end
end
