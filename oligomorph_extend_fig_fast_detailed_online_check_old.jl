using Base.Threads
using DifferentialEquations
using DiffEqCallbacks
using SpecialFunctions
using Statistics
using Printf

cd(@__DIR__)

# Output directory: override by setting the OUTPUT_DIR environment variable.
#   e.g.  julia oligomorph_extend_fig_fast_detailed_online_check_old.jl  → saves to ./output/oligo_data/
#   e.g.  OUTPUT_DIR=/path/to/dir julia oligomorph_...jl
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR", joinpath(@__DIR__, "output"))
mkpath(joinpath(OUTPUT_DIR, "oligo_data"))

const TWOπ = 2π

# -------------------------------
# basic helpers
# -------------------------------
@inline phase(t, θ) = t - θ

@inline function mod01(x)
    return mod(x, 1.0)
end

@inline function circular_distance01_periodic(θ1, θ2)
    return abs(mod(θ1 - θ2 + 0.5, 1.0) - 0.5)
end

@inline function signed_circ_diff01(a, b)
    d = a - b
    d -= floor(d + 0.5)   # wrap into (-0.5, 0.5]
    return d
end

# -------------------------------
# von Mises kernel
# -------------------------------
@inline function φ_vonmises(t, θ, κ, invI0)
    return exp(κ * cospi(2 * (t - θ))) * invI0
end

@inline function dφ_vonmises(t, θ, κ, invI0)
    φ = φ_vonmises(t, θ, κ, invI0)
    return TWOπ * κ * sinpi(2 * (t - θ)) * φ
end

@inline function d2φ_vonmises(t, θ, κ, invI0)
    x = t - θ
    c = cospi(2x)
    s = sinpi(2x)
    φ = exp(κ * c) * invI0
    return (TWOπ^2) * (κ^2 * s^2 - κ * c) * φ
end

# -------------------------------
# unpack helper
# -------------------------------
@inline function unpack_state(u)
    S  = u[1]
    J  = u[2]
    R  = u[3]
    M  = Int((length(u) - 3) ÷ 3)
    f  = @view u[4 : 3 + M]
    z  = @view u[4 + M : 3 + 2M]
    v  = @view u[4 + 2M : 3 + 3M]
    return S, J, R, f, z, v, M
end

# -------------------------------
# ODE
# p = (d, γ, c, β0, δ, α, κ, vm, invI0)
# -------------------------------
function define_oligo_M!(du, u, p, t)
    S, J, R, f, z, v, M = unpack_state(u)
    d, γ, c, β0, δ, α, κ, vm, invI0 = p

    β = β0 * (1 + δ * cos(TWOπ * t - π))

    sumF  = 0.0
    meanρ = 0.0
    curv  = 0.0

    @inbounds for m in 1:M
        θ   = z[m]
        φm  = φ_vonmises(t, θ, κ, invI0)
        ρm  = β * S * φm - (d + γ + α)
        d2  = β * S * d2φ_vonmises(t, θ, κ, invI0)

        sumF  += β * S * φm * f[m]
        meanρ += f[m] * ρm
        curv  += f[m] * v[m] * d2
    end

    du[1] = d * (S + J + R) - d * S + c * R - sumF * J - 0.5 * curv * J
    du[2] = (sumF - (d + γ)) * J + 0.5 * curv * J
    du[3] = γ * J - (d + c) * R

    @inbounds for m in 1:M
        θ   = z[m]
        φm  = φ_vonmises(t, θ, κ, invI0)
        ρm  = β * S * φm - (d + γ + α)
        dφ  = β * S * dφ_vonmises(t, θ, κ, invI0)
        d2  = β * S * d2φ_vonmises(t, θ, κ, invI0)

        du[3 + m]       = f[m] * (ρm - meanρ)
        du[3 + M + m]   = v[m] * dφ
        du[3 + 2M + m]  = v[m]^2 * d2 + 2vm
    end
end

# -------------------------------
# initial state
# -------------------------------
function make_init_state(M; S0=0.99, J0=0.01, R0=0.0,
                         f_init=fill(1.0 / M, M),
                         z_init=[i / M for i in 1:M],
                         v_init=fill(1e-4, M))
    u0 = Vector{Float64}(undef, 3 + 3M)
    u0[1] = S0
    u0[2] = J0
    u0[3] = R0
    u0[4 : 3 + M] .= f_init
    u0[4 + M : 3 + 2M] .= z_init
    u0[4 + 2M : 3 + 3M] .= v_init
    return u0
end


# -------------------------------
# count morphs
# -------------------------------
function count_morphs(z̄::AbstractVector{<:Real},
                      f̄::AbstractVector{<:Real},
                      v̄::AbstractVector{<:Real};
                      fcut=0.05, Dc_threshold=2.0)

    M = length(z̄)
    idx = findall(m -> f̄[m] ≥ fcut, 1:M)
    isempty(idx) && return 1

    Ne = length(idx)
    adj = falses(Ne, Ne)

    @inbounds for i in 1:Ne, j in i+1:Ne
        mi, mj = idx[i], idx[j]
        Δθ    = circular_distance01_periodic(z̄[mi], z̄[mj])
        σi    = sqrt(max(v̄[mi], 0.0))
        σj    = sqrt(max(v̄[mj], 0.0))
        σpool = sqrt(0.5 * (σi^2 + σj^2))
        Dc    = σpool > 0 ? Δθ / σpool : 0.0
        if Dc <= Dc_threshold
            adj[i, j] = true
            adj[j, i] = true
        end
    end

    visited = falses(Ne)
    comp = 0
    for i in 1:Ne
        if !visited[i]
            comp += 1
            stack = [i]
            while !isempty(stack)
                k = pop!(stack)
                if !visited[k]
                    visited[k] = true
                    for m in 1:Ne
                        if adj[k, m] && !visited[m]
                            push!(stack, m)
                        end
                    end
                end
            end
        end
    end

    return max(comp, 1)
end

# -------------------------------
# online tail collector
# 100-step window averages only
# -------------------------------
mutable struct TailCollectorExact
    M::Int
    window_steps::Int
    max_windows::Int
    prefix_skip::Int              # 先頭で捨てるサンプル数（末尾基準窓に合わせるため）

    sample_count::Int             # 記録した総サンプル数（t=0 を含む）
    step_in_window::Int

    sum_z::Vector{Float64}
    sum_f::Vector{Float64}
    sum_v::Vector{Float64}
    sum_zf::Float64               # mean(sum(z .* f)) 用

    zbar_hist::Vector{Float64}
    morph_hist::Vector{Float64}

    nhist::Int
    head::Int

    fcut::Float64
    Dc_threshold::Float64
end

function TailCollectorExact(M, total_saved_samples;
                            window_steps=100,
                            max_windows=300001,
                            fcut=0.05,
                            Dc_threshold=2.0)

    prefix_skip = mod(total_saved_samples, window_steps)

    return TailCollectorExact(
        M,
        window_steps,
        max_windows,
        prefix_skip,
        0,
        0,
        zeros(M),
        zeros(M),
        zeros(M),
        0.0,
        fill(NaN, max_windows),
        fill(NaN, max_windows),
        0,
        0,
        fcut,
        Dc_threshold
    )
end

@inline function reset_window!(tc::TailCollectorExact)
    tc.step_in_window = 0
    fill!(tc.sum_z, 0.0)
    fill!(tc.sum_f, 0.0)
    fill!(tc.sum_v, 0.0)
    tc.sum_zf = 0.0
    return nothing
end

@inline function push_summary!(tc::TailCollectorExact, zbar::Real, morph::Real)
    tc.head = (tc.head % tc.max_windows) + 1
    tc.zbar_hist[tc.head] = Float64(zbar)
    tc.morph_hist[tc.head] = Float64(morph)
    tc.nhist = min(tc.nhist + 1, tc.max_windows)
    return nothing
end

function record_step!(tc::TailCollectorExact, u)
    tc.sample_count += 1

    # offline 版で末尾基準窓に入らない先頭の端数サンプルを捨てる
    if tc.sample_count <= tc.prefix_skip
        return nothing
    end

    _, _, _, f, z, v, M = unpack_state(u)

    zf_now = 0.0
    @inbounds for m in 1:M
        tc.sum_z[m] += z[m]
        tc.sum_f[m] += f[m]
        tc.sum_v[m] += v[m]
        zf_now += z[m] * f[m]
    end
    tc.sum_zf += zf_now
    tc.step_in_window += 1

    if tc.step_in_window == tc.window_steps
        invw = 1.0 / tc.window_steps
        meanz = tc.sum_z .* invw
        meanf = tc.sum_f .* invw
        meanv = tc.sum_v .* invw

        morph = count_morphs(meanz, meanf, meanv;
                             fcut=tc.fcut, Dc_threshold=tc.Dc_threshold)

        # offline版と同じ: mean_t [ sum_m z_m(t) f_m(t) ]
        zbar = tc.sum_zf * invw

        push_summary!(tc, zbar, morph)
        reset_window!(tc)
    end

    return nothing
end

function get_histories(tc::TailCollectorExact)
    n = tc.nhist
    n == 0 && return Float64[], Float64[]

    zbar = Vector{Float64}(undef, n)
    morph = Vector{Float64}(undef, n)

    start = tc.head - n + 1
    for i in 1:n
        idx = mod(start + i - 2, tc.max_windows) + 1
        zbar[i] = tc.zbar_hist[idx]
        morph[i] = tc.morph_hist[idx]
    end
    return zbar, morph
end

function finalize_tail_metrics(tc::TailCollectorExact)
    zbar, morph = get_histories(tc)
    isempty(zbar) && return NaN, NaN

    # offline版と同じ: 円周補正なしの単純差分
    sumdtmpbar = 0.0
    @inbounds for i in 2:length(zbar)
        sumdtmpbar += zbar[i] - zbar[i - 1]
    end

    meanmorph = mean(morph)
    return meanmorph, sumdtmpbar
end

# -------------------------------
# solve without storing whole trajectory
# -------------------------------
function solve_and_collect_tail_exact(pars, u0, tspan;
                                      sample_dt=0.01,
                                      window_steps=100,
                                      max_windows=300001,
                                      fcut=0.05,
                                      Dc_threshold=2.0)

    invI0 = 1.0 / besseli(0.0, pars.a)
    params = (pars.d, pars.γ, pars.c, pars.β0, pars.δ, pars.α, pars.a, pars.vm, invI0)

    prob = ODEProblem(define_oligo_M!, copy(u0), tspan, params)

    M = Int((length(u0) - 3) ÷ 3)

    total_saved_samples = Int(round((tspan[2] - tspan[1]) / sample_dt)) + 1

    tc = TailCollectorExact(M, total_saved_samples;
        window_steps=window_steps,
        max_windows=max_windows,
        fcut=fcut,
        Dc_threshold=Dc_threshold
    )

    # offline版の save_start=true に対応して t=t0 のサンプルを先に記録
    record_step!(tc, copy(u0))

    cb_collect = FunctionCallingCallback(
        (u, t, integrator) -> begin
            record_step!(tc, u)
            nothing
        end;
        funcat = (tspan[1] + sample_dt):sample_dt:tspan[2],
        func_start = false
    )

    sol = solve(
        prob, RK4();
        dt=0.01,
        adaptive=true,
        callback=cb_collect,
        save_everystep=false,
        save_start=false,
        save_end=true,
        dense=false,
        maxiters=10^8
    )

    meanmorph, movesum = finalize_tail_metrics(tc)
    uend = copy(sol.u[end])

    return meanmorph, movesum, uend, sol.retcode, tc.nhist, tc
end

# -------------------------------
# main sweep
# -------------------------------
function main_M_online_exact(u0, sfn; d=0.1, c=0.5, fcut=0.05, Dc_threshold=2.0)
    n_years = 400000
    tspan   = (0.0, n_years)

    beta0_values = collect(1.2:0.3:10.2)
    delta_values = reverse(collect(0:0.005:0.4))

    nb = length(beta0_values)
    nd = length(delta_values)

    moved_mat = fill(NaN, nb, nd)
    morph_mat = fill(NaN, nb, nd)

    total_jobs = nb * nd
    done_jobs = Threads.Atomic{Int}(0)

    @threads for ib in eachindex(beta0_values)
        beta0 = beta0_values[ib]
        local_init = copy(u0)

        for (jd, δ) in enumerate(delta_values)
            @printf("[thread %d] start: β0=%.3f, δ=%.3f\n", threadid(), beta0, δ)
            flush(stdout)

            pars = (
                d = d,
                γ = 1.0,
                c = c,
                β0 = beta0,
                δ = δ,
                α = 0.0,
                a = 1.0,
                vm = 0.1 * 0.5 * (1 / 200)^2 / (TWOπ^2)
            )

            countmorph = NaN
            movesum = NaN
            retcode = :FAILED
            nhist = 0

            try
                countmorph, movesum, uend, retcode, nhist, tc = solve_and_collect_tail_exact(
                    pars, local_init, tspan;
                    sample_dt=0.01,
                    window_steps=100,
                    max_windows=300001,
                    fcut=fcut,
                    Dc_threshold=Dc_threshold
                )

                # offline版 main_M と揃えるなら continuation はしない
                # local_init .= uend

            catch err
                @warn "failed at β0=$(beta0), δ=$(δ). Writing NaN." exception=(err, catch_backtrace())
            end

            morph_mat[ib, jd] = countmorph
            moved_mat[ib, jd] = movesum

            finished = Threads.atomic_add!(done_jobs, 1) + 1
            @printf("[thread %d] done %d/%d: β0=%.3f, δ=%.3f, morph=%.3f, move=%.6f, nhist=%d, ret=%s\n",
                    threadid(), finished, total_jobs, beta0, δ, countmorph, movesum, nhist, string(retcode))
            flush(stdout)
        end
    end

    out_dir = joinpath(OUTPUT_DIR, "oligo_data")
    mkpath(out_dir)

    open(joinpath(out_dir, "$(sfn)_moved.csv"), "w") do io
        println(io, "beta\tdelta\tmoved")
        for ib in eachindex(beta0_values), jd in eachindex(delta_values)
            println(io, "$(beta0_values[ib])\t$(delta_values[jd])\t$(moved_mat[ib, jd])")
        end
    end

    open(joinpath(out_dir, "$(sfn)_morph.csv"), "w") do io
        println(io, "beta\tdelta\tmorph")
        for ib in eachindex(beta0_values), jd in eachindex(delta_values)
            println(io, "$(beta0_values[ib])\t$(delta_values[jd])\t$(morph_mat[ib, jd])")
        end
    end

    return moved_mat, morph_mat
end


function diagnose_one_run_exact(u0; beta0=7.0, δ=0.11, n_years=400000,
                                d=0.1, c=0.5,
                                sample_dt=0.01, fcut=0.05, Dc_threshold=2.0)

    tspan = (0.0, n_years)

    pars = (
        d = d,
        γ = 1.0,
        c = c,
        β0 = beta0,
        δ = δ,
        α = 0.0,
        a = 1.0,
        vm = 0.1 * 0.5 * (1 / 200)^2 / (TWOπ^2)
    )

    meanmorph, movesum, uend, retcode, nhist, tc = solve_and_collect_tail_exact(
        pars, u0, tspan;
        sample_dt=sample_dt,
        window_steps=100,
        max_windows=300001,
        fcut=fcut,
        Dc_threshold=Dc_threshold
    )

    println("========== summary ==========")
    println("beta0 = ", beta0)
    println("delta = ", δ)
    println("retcode = ", retcode)
    println("nhist = ", nhist)
    println("mean morph = ", meanmorph)
    println("move sum = ", movesum)

    S, J, R, f, z, v, M = unpack_state(uend)
    println("S = ", S, ", J = ", J, ", R = ", R)
    println("sum(f) = ", sum(f))
    println("min(f) = ", minimum(f), ", max(f) = ", maximum(f))
    println("min(v) = ", minimum(v), ", max(v) = ", maximum(v))
    println("f = ", collect(f))
    println("z = ", collect(z))
    println("v = ", collect(v))

    return meanmorph, movesum, uend, retcode, nhist, tc
end

# -------------------------------
# run
# -------------------------------
M = 5
u0 = make_init_state(
    M;
    S0=0.99,
    J0=0.01,
    R0=0.0,
    f_init=fill(1.0 / M, M),
    z_init=[i / M for i in 1:M],
    v_init=fill(0.0001 / (TWOπ^2), M)
)

# baseline (d=0.1, c=0.5)
main_M_online_exact(u0, "traj_for_figure_M2_fast_large_var_detailed2_online_old";
               fcut=0.0, Dc_threshold=2.0)

# c sensitivity
main_M_online_exact(u0, "traj_for_figure_M2_fast_large_var_detailed2_online_old_c=0.1";
               c=0.1, fcut=0.0, Dc_threshold=2.0)
main_M_online_exact(u0, "traj_for_figure_M2_fast_large_var_detailed2_online_old_c=1";
               c=1.0, fcut=0.0, Dc_threshold=2.0)

# d sensitivity
main_M_online_exact(u0, "traj_for_figure_M2_fast_large_var_detailed2_online_old_d=0.05";
               d=0.05, fcut=0.0, Dc_threshold=2.0)
main_M_online_exact(u0, "traj_for_figure_M2_fast_large_var_detailed2_online_old_d=0.2";
               d=0.2, fcut=0.0, Dc_threshold=2.0)