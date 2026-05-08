### figs_bifurcation.jl ###

using DifferentialEquations
using QuadGK
using SpecialFunctions
using Plots
using DelimitedFiles
using Interpolations
using CSV
using DataFrames

using Base.Threads

cd(@__DIR__)

# Output directory: override by setting the OUTPUT_DIR environment variable.
#   e.g.  julia figs_bifurcation.jl         → saves to ./output/
#   e.g.  OUTPUT_DIR=/path/to/dir julia figs_bifurcation.jl
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR", joinpath(@__DIR__, "output"))
mkpath(OUTPUT_DIR)


# Time transformation (1-periodic)
transform(t) = mod(t,1) #2π * mod(t, 1) - π

# -------------------------------
# infection function (von Mises)
# -------------------------------
function Func_vonmises(t, θ, κ)
    x = transform(t) - θ #mod(t - θ + π, 2π) - π
    return (1 / (besseli(0, κ))) * exp(κ * cos(2π * x))
end

# -------------------------------
# define ODE system
# -------------------------------
function define_sys_vonmises(du, u, p, t)
    S, J, R = u
    λ, r, d, γ, c, β0, δ, ω, α, κ, μ, θ = p

    φ = Func_vonmises(transform(t), θ, κ)
    β = β0 * (1 + δ * cos(2π * t - π))

    du[1] = d * (S+J+R) - d * S + c * R - β * S * φ * J
    du[2] = β * S * φ * J - (d + γ + α) * J
    du[3] = γ * J - d * R - c * R
end

function define_sys_two_species(du, u, p, t)
    S, J1, J2, R = u
    λ, r, d, γ, c, β0, δ, ω, α, κ, μ, θ1, θ2 = p

    φ1 = Func_vonmises(transform(t), θ1, κ)
    φ2 = Func_vonmises(transform(t), θ2, κ)
    β = β0 * (1 + δ * cos(2π * t - π))

    du[1] = d * (S+J1+J2+R) - d * S + c * R - β * S * φ1 * J1 - β * S * φ2 * J2
    du[2] = β * S * φ1 * J1 - (d + γ + α) * J1
    du[3] = β * S * φ2 * J2 - (d + γ + α) * J2
    du[4] = γ * (J1+J2) - d * R - c * R
end

# -------------------------------
# solve resident system
# -------------------------------
function solve_system_vonmises(θ, pars, tspan)
    u0 = [1.0 - 0.1, 0.1, 0.0]  # initial values: S, J, R
    params = [pars.λ, pars.r, pars.d, pars.γ, pars.c, pars.β0, pars.δ, pars.ω, pars.α, pars.a, pars.μ, θ]
    prob = ODEProblem(define_sys_vonmises, u0, tspan, params)
    solve(prob, RK4(), dt=0.01)
end

function solve_system_two_species(θ1, θ2, pars, tspan)
    u0 = [1.0 - 0.1-0.1, 0.1, 0.1, 0.0]  # initial values: S, J, R
    params = [pars.λ, pars.r, pars.d, pars.γ, pars.c, pars.β0, pars.δ, pars.ω, pars.α, pars.a, pars.μ, θ1, θ2]
    prob = ODEProblem(define_sys_two_species, u0, tspan, params)
    solve(prob, RK4(), dt=0.01)
end

# -------------------------------
# invasion fitness function
# -------------------------------
function integrate_resident_density_vonmises(sol, θm, pars,tend)
    result, _ = quadgk(t -> begin
        S_t = sol(t + tend -1)[1]
        J_t = sol(t + tend -1)[2]
        φ = Func_vonmises(transform(t), θm, pars.a)
        β = pars.β0 * (1 + pars.δ * cos(2π * t - π))
        return β * S_t * φ - (pars.d + pars.γ + pars.α)
    end, 0, 1)
    return result
end

function integrate_resident_density_two_species(sol, θ1m, θ2m, pars, tend)
    f1(t) = begin
        S_t = sol(t + tend - 1)[1]
        φ1 = Func_vonmises(transform(t), θ1m, pars.a)
        β = pars.β0 * (1 + pars.δ * cos(2π * t - π))
        return β * S_t * φ1 - (pars.d + pars.γ + pars.α)
    end

    f2(t) = begin
        S_t = sol(t + tend - 1)[1]
        φ2 = Func_vonmises(transform(t), θ2m, pars.a)
        β = pars.β0 * (1 + pars.δ * cos(2π * t - π))
        return β * S_t * φ2 - (pars.d + pars.γ + pars.α)
    end

    val1, err1 = quadgk(f1, 0, 1)
    val2, err2 = quadgk(f2, 0, 1)

    return val1, val2
end

# -------------------------------
# bifurcation scan
# -------------------------------
using Interpolations

function scan_rho_vs_theta(pars, δ_vals; θ_range=range(0, 1; length=300), ε=0.005, Tmax=500.0)
    result = []

    wrap(x) = mod(x,1) 

    θ_array = collect(θ_range)
    N = length(θ_array)

    for δv in δ_vals
        pars_δ = (; pars..., δ=δv)
        ρ_vals = Float64[]

        for θ in θ_array
            sol = solve_system_vonmises(θ, pars_δ, (0.0, Tmax))
            ρ = integrate_resident_density_vonmises(sol, wrap(θ + ε), pars_δ, Tmax)
            push!(ρ_vals, ρ)
        end

        # 周期境界を含めたゼロ交差チェック
        for i in 1:N
            i_next = i == N ? 1 : i + 1  # periodic

            θ1, θ2 = θ_array[i], θ_array[i_next]
            ρ1, ρ2 = ρ_vals[i], ρ_vals[i_next]

            if sign(ρ1) != sign(ρ2)
                # ゼロ点位置を線形補間
                θ_zero = θ1 - ρ1 * (θ2 - θ1) / (ρ2 - ρ1)

                # 安定性判定：正→負なら安定、負→正なら不安定
                stability = (ρ1 > 0 && ρ2 < 0) ? :stable : :unstable

                push!(result, (δv, θ_zero, stability))
                println("[INFO] δ = $δv, θ ≈ $(θ_zero), stability = $(stability)")
            end
        end
        println("End δ = $δv")
    end
    return result
end

function scan_rho_vs_theta(pars, δ_vals; θ_range=range(0,1;length=300),
    ε=0.01, Tmax=500.0, tol=0, refine=false)
    wrap(x) = mod(x,1)
    θs = collect(θ_range); N = length(θs)
    out = Tuple[]

    for δv in δ_vals
        pars_δ = (; pars..., δ=δv)
        g = Vector{Float64}(undef, N)

        # g(θ) ≈ [ρ(θ+ε;θ) − ρ(θ−ε;θ)]/(2ε)
        for i in 1:N
            θ = θs[i]
            sol = solve_system_vonmises(θ, pars_δ, (0.0, Tmax))
            ρp = integrate_resident_density_vonmises(sol, wrap(θ + ε), pars_δ, Tmax)
            ρm = integrate_resident_density_vonmises(sol, wrap(θ - ε), pars_δ, Tmax)
            g[i] = (ρp - ρm) / (2ε)
        end

        for i in 1:N
            j = (i == N ? 1 : i+1)
            θ1, θ2 = θs[i], θs[j]
            g1, g2 = g[i], g[j]

            # 近傍でほぼゼロ
            if abs(g1) < tol
                push!(out, (δv, θ1, :candidate)); continue
            elseif abs(g2) < tol
                push!(out, (δv, θ2, :candidate)); continue
            end

            if sign(g1) != sign(g2)
                # 円周補正付き内挿
                θ1i, θ2i = θ1, θ2
                if θ2i < θ1i
                    θ2i += 1
                end
                θ0 = θ1i - g1 * (θ2i - θ1i) / (g2 - g1)
                θ0 = wrap(θ0)

                stability = (g1 > 0 && g2 < 0) ? :stable : :unstable

                if refine
                    # 近傍2点で1回だけ再評価して secant で微修正（任意）
                    δθ = 2*(1/N)
                    θa, θb = wrap(θ0 - δθ), wrap(θ0 + δθ)
                    sola = solve_system_vonmises(θa, pars_δ, (0.0, Tmax))
                    solb = solve_system_vonmises(θb, pars_δ, (0.0, Tmax))
                    ga = (integrate_resident_density_vonmises(sola, wrap(θa+ε), pars_δ, Tmax) -
                    integrate_resident_density_vonmises(sola, wrap(θa-ε), pars_δ, Tmax)) / (2ε)
                    gb = (integrate_resident_density_vonmises(solb, wrap(θb+ε), pars_δ, Tmax) -
                    integrate_resident_density_vonmises(solb, wrap(θb-ε), pars_δ, Tmax)) / (2ε)
                    if sign(ga) != sign(gb)
                        # 円周距離で secant
                        Δ = mod(θb - θa + 1, 1)
                        θ0 = wrap(θa - ga * Δ / (gb - ga))
                    end
                end

                push!(out, (δv, θ0, stability))
            end
        end
        println("End δ = $δv")
    end
    return out
end




function scan_rho_vs_theta_with_ES(
    pars, δ_vals;
    θ_range = range(0, 1; length=300),
    ε = 0.005,                 # 選択勾配の近似用
    h = 0.005,                 # 2階微分の近似用（通常 ε と同程度でOK）
    Tmax = 500.0,
    verbose = true,
    κ2_tol = 1e-4             # 曲率の「ほぼゼロ」判定しきい値
)
    result = []

    # 周期角演算ヘルパ
    wrap(x) = mod(x,1) #mod(x + π, 2π) - π

    θ_array = collect(θ_range)
    N = length(θ_array)

    for δv in δ_vals
        pars_δ = (; pars..., δ = δv)
        ρ_vals = Float64[]

        # g(θ) ≈ ρ(θ+ε; θ) を前計算（住民 = θ, 変異 = θ+ε）
        for θ in θ_array
            sol = solve_system_vonmises(θ, pars_δ, (0.0, Tmax))
            ρ = integrate_resident_density_vonmises(sol, wrap(θ + ε), pars_δ, Tmax)
            push!(ρ_vals, ρ)
        end

        # 周期境界を含めたゼロ交差検出（g(θ) の符号変化）
        for i in 1:N
            i_next = (i == N ? 1 : i + 1)
            θ1, θ2 = θ_array[i], θ_array[i_next]
            ρ1, ρ2 = ρ_vals[i], ρ_vals[i_next]

            if sign(ρ1) != sign(ρ2)
                # まず線形補間で θ* を粗く推定（g(θ)=0 の近似点）
                θ_star = θ1 - ρ1 * (θ2 - θ1) / (ρ2 - ρ1)
                θ_star = wrap(θ_star)

                # 収束安定性（+→− が収束安定）
                conv = (ρ1 > 0 && ρ2 < 0) ? :convergent : :divergent

                # ---- 進化的安定性の判定 ----
                # 住民を θ_star に固定してエコロジー解を計算
                sol_star = solve_system_vonmises(θ_star, pars_δ, (0.0, Tmax))

                # ρ0 = ρ(θm=θ_star; resident θ_star) は理論上0だが、数値誤差用に計算
                ρ0 = integrate_resident_density_vonmises(sol_star, θ_star, pars_δ, Tmax)

                # 変異体側で ±h ずらす
                θp = wrap(θ_star + h)
                θm = wrap(θ_star - h)

                ρ_plus  = integrate_resident_density_vonmises(sol_star, θp, pars_δ, Tmax)
                ρ_minus = integrate_resident_density_vonmises(sol_star, θm, pars_δ, Tmax)

                κ2 = (ρ_plus - 2ρ0 + ρ_minus) / (h^2)

                evol =
                    if κ2 < -κ2_tol
                        :ESS
                    elseif κ2 > κ2_tol
                        :branching     # 収束安定かつ branching なら進化的分岐点
                    else
                        :neutral       # 平坦/不確定
                    end

                # 備考（典型的な組み合わせのガイド）
                note =
                    if conv == :convergent && evol == :ESS
                        "CSS（収束安定かつ進化的安定）"
                    elseif conv == :convergent && evol == :branching
                        "進化的分岐点（convergence stable, evolutionarily unstable）"
                    elseif conv == :divergent && evol == :ESS
                        "進化的に安定だが収束不安定（Garden-of-Eden型）"
                    else
                        ""
                    end

                push!(result, (δ = δv, θ_star = θ_star, conv = conv, evol = evol, κ2 = κ2, note = note))

                if verbose
                    println("[INFO] δ = $δv, θ* ≈ $(θ_star), conv = $(conv), evol = $(evol), κ2 ≈ $(κ2)")
                    if !isempty(note); println("       → $note"); end
                end
            end
        end

        if verbose
            println("End δ = $δv")
        end
    end

    return result
end

function scan_ES(
    pars, beta_vals, δ_vals;
    θ_range = range(0, 1; length=300),
    ε = 0.05,                 # 選択勾配の近似用
    h = 0.05,                 # 2階微分の近似用（通常 ε と同程度でOK）
    Tmax = 500.0,
    verbose = true,
    κ2_tol = 1e-3             # 曲率の「ほぼゼロ」判定しきい値
)
    pattern = []
    # 周期角演算ヘルパ
    wrap(x) = mod(x,1) #mod(x + π, 2π) - π

    θ_array = collect(θ_range)
    N = length(θ_array)

    for betav in beta_vals
        for δv in δ_vals
            pars_δ = (; pars..., δ = δv, β0 = betav)
            ρ_vals = Float64[]

            # g(θ) ≈ ρ(θ+ε; θ) を前計算（住民 = θ, 変異 = θ+ε）
            for θ in θ_array
                sol = solve_system_vonmises(θ, pars_δ, (0.0, Tmax))
                ρ = integrate_resident_density_vonmises(sol, wrap(θ + ε), pars_δ, Tmax)
                push!(ρ_vals, ρ)
            end

            result = []

            # 周期境界を含めたゼロ交差検出（g(θ) の符号変化）
            for i in 1:N
                i_next = (i == N ? 1 : i + 1)
                θ1, θ2 = θ_array[i], θ_array[i_next]
                ρ1, ρ2 = ρ_vals[i], ρ_vals[i_next]

                if sign(ρ1) != sign(ρ2)
                    # まず線形補間で θ* を粗く推定（g(θ)=0 の近似点）
                    θ_star = θ1 - ρ1 * (θ2 - θ1) / (ρ2 - ρ1)
                    θ_star = wrap(θ_star)

                    # 収束安定性（+→− が収束安定）
                    conv = (ρ1 > 0 && ρ2 < 0) ? :convergent : :divergent

                    # ---- 進化的安定性の判定 ----
                    # 住民を θ_star に固定してエコロジー解を計算
                    sol_star = solve_system_vonmises(θ_star, pars_δ, (0.0, Tmax))

                    # ρ0 = ρ(θm=θ_star; resident θ_star) は理論上0だが、数値誤差用に計算
                    ρ0 = integrate_resident_density_vonmises(sol_star, θ_star, pars_δ, Tmax)

                    # 変異体側で ±h ずらす
                    θp = wrap(θ_star + h)
                    θm = wrap(θ_star - h)

                    ρ_plus  = integrate_resident_density_vonmises(sol_star, θp, pars_δ, Tmax)
                    ρ_minus = integrate_resident_density_vonmises(sol_star, θm, pars_δ, Tmax)

                    κ2 = (ρ_plus - 2ρ0 + ρ_minus) / (h^2)

                    evol =
                        if κ2 < -κ2_tol
                            :ESS
                        elseif κ2 > κ2_tol
                            :branching     # 収束安定かつ branching なら進化的分岐点
                        else
                            :neutral       # 平坦/不確定
                        end

                    # 備考（典型的な組み合わせのガイド）
                    res =
                        if conv == :convergent && evol == :ESS
                            1 #"CSS（収束安定かつ進化的安定）"
                        elseif conv == :convergent && evol == :branching
                            2 #"進化的分岐点（convergence stable, evolutionarily unstable）"
                        elseif conv == :divergent && evol == :ESS
                            0 #"進化的に安定だが収束不安定（Garden-of-Eden型）"
                        else
                            -1 #""
                        end

                    push!(result, res)

                    if verbose
                        println("[INFO] δ = $δv, θ* ≈ $(θ_star), conv = $(conv), evol = $(evol), κ2 ≈ $(κ2)")
                    end
                end
            end

            if 2 in result
                push!(pattern, [betav,δv,2])
            elseif 1 in result
                push!(pattern, [betav,δv,1])
            else
                push!(pattern,[betav,δv,0])
            end

            if verbose
                println("End δ = $δv")
            end
        end
    end

    return pattern
end

using Base.Threads

function scan_ES_outer_threads(
    pars, beta_vals, δ_vals;
    θ_range = range(0, 1; length=300),
    ε = 0.01,
    h = 0.01,
    Tmax = 500.0,
    verbose = false,
    κ2_tol = 1e-3
)
    wrap(x) = mod(x,1) #mod(x + π, 2π) - π
    θ_array = collect(θ_range)
    N = length(θ_array)

    # （任意）BLAS の多重スレッドを止めたい場合はコメントアウト解除
    # using LinearAlgebra; BLAS.set_num_threads(1)

    # (β0, δ) の全組
    pairs = [(betav, δv) for betav in beta_vals for δv in δ_vals]

    Tβ = eltype(beta_vals)
    Tδ = eltype(δ_vals)
    patterns = Vector{Tuple{Tβ,Tδ,Int}}(undef, length(pairs))
    print_lock = ReentrantLock()

    @threads :dynamic for k in eachindex(pairs)
        betav, δv = pairs[k]
        pars_δ = (; pars..., δ = δv, β0 = betav)

        # ---- θ ループは逐次 ----
        ρ_vals = Vector{Float64}(undef, N)
        for i in 1:N
            θ = θ_array[i]
            sol = solve_system_vonmises(θ, pars_δ, (0.0, Tmax))
            ρ_vals[i] = integrate_resident_density_vonmises(sol, wrap(θ + ε), pars_δ, Tmax)
        end

        # ゼロ交差 → 収束安定 & 進化的安定の判定
        result_codes = Int[]
        for i in 1:N
            i_next = (i == N ? 1 : i + 1)
            ρ1 = ρ_vals[i]; ρ2 = ρ_vals[i_next]
            if sign(ρ1) != sign(ρ2)
                θ1 = θ_array[i]; θ2 = θ_array[i_next]
                θ_star = wrap(θ1 - ρ1 * (θ2 - θ1) / (ρ2 - ρ1))

                conv = (ρ1 > 0 && ρ2 < 0)  # true → convergent

                sol_star = solve_system_vonmises(θ_star, pars_δ, (0.0, Tmax))
                ρ0      = integrate_resident_density_vonmises(sol_star, θ_star,       pars_δ, Tmax)
                ρ_plus  = integrate_resident_density_vonmises(sol_star, wrap(θ_star+h), pars_δ, Tmax)
                ρ_minus = integrate_resident_density_vonmises(sol_star, wrap(θ_star-h), pars_δ, Tmax)
                κ2 = (ρ_plus - 2ρ0 + ρ_minus) / (h^2)

                # evol code: 1=ESS, 2=branching, -1=neutral
                evol_code = κ2 < -κ2_tol ? 1 : κ2 > κ2_tol ? 2 : -1

                # res codeの規約: 2=branching(収束安定), 1=CSS, 0=ESSだが収束不安定, -1=その他
                code =
                    conv && (evol_code == 2) ? 2 :
                    conv && (evol_code == 1) ? 1 :
                   (!conv) && (evol_code == 1) ? 0 : -1

                push!(result_codes, code)

                # if verbose
                #     lock(print_lock) do
                #         println("[T$(threadid())] β0=$betav, δ=$δv, θ*≈$θ_star, conv=$(conv ? :convergent : :divergent), κ2≈$κ2 → code=$code")
                #     end
                # end
            end
        end

        # ペアごとの最終パターン
        final_code = (2 in result_codes) ? 2 : (1 in result_codes ? 1 : 0)
        patterns[k] = (betav, δv, final_code)

        if 1 == 1
            lock(print_lock) do
                println("[T$(threadid())] finished β0=$betav, δ=$δv → pattern=$final_code")
            end
        end
    end

    return patterns  # [(β0, δ, code), ...] の一次元ベクタ（β0 外側優先の順）
end



function rho_data(pars, δv; θ_range=range(0, 1; length=300), ε=0.05)
    result = []

    θ_array = collect(θ_range)
    N = length(θ_array)

    pars_δ = (; pars..., δ=δv)
    ρ_vals = Float64[]

    for θ in θ_array
        sol = solve_system_vonmises(θ, pars_δ, (0.0, 500.0))
        ρ = integrate_resident_density_vonmises(sol, θ + ε, pars_δ, 500)
        push!(result,(θ,ρ))
    end
    println("End δ = $δv")

    return result
end

function rho_matrix(pars, δv; θ_range=range(0, 1; length=300))
    result = []

    θ_array = collect(θ_range)
    N = length(θ_array)

    pars_δ = (; pars..., δ=δv)
    ρ_vals = Float64[]

    for θ in θ_array
        sol = solve_system_vonmises(θ, pars_δ, (0.0, 500.0))
        for θm in θ_array
            ρ = integrate_resident_density_vonmises(sol, θm, pars_δ, 500)
            push!(result,(θ,θm,ρ))
        end
        println("End θ = $θ")
    end

    return result
end

function rho_list(θv, pars, δv, betav; θ_range=range(0, 1; length=300))
    result = []

    θ_array = collect(θ_range)
    N = length(θ_array)

    pars_δ = (; pars..., δ=δv, β0=betav)
    ρ_vals = Float64[]

    sol = solve_system_vonmises(θv, pars_δ, (0.0, 500.0))
    for θm in θ_array
        ρ = integrate_resident_density_vonmises(sol, θm, pars_δ, 500)
        push!(result,(θm,ρ))
    end

    return result
end

function rho_list_multi(θ1v, θ2v, pars, δv, betav; θ_range=range(0, 1; length=300))
    result = []

    θ_array = collect(θ_range)
    N = length(θ_array)

    pars_δ = (; pars..., δ=δv, β0=betav)
    ρ_vals = Float64[]

    sol = solve_system_two_species(θ1v, θ2v,  pars_δ, (0.0, 10000.0))
    for θm in θ_array
        ρ = integrate_resident_density_vonmises(sol, θm, pars_δ, 10000.0)
        push!(result,(θm,ρ))
    end

    return result
end

# -------------------------------
# write data to TSV file
# -------------------------------
function save_bifurcation_data(data, filename)
    open(filename, "w") do io
        println(io, "δ\tθ\tstability")
        for (δ, θ, s) in data
            println(io, "$(δ)\t$(θ)\t$(s)")
        end
    end
    println("[✓] Saved data to $(filename)")
end

function save_bifurcation_data_with_ES(data, filename)
    open(filename, "w") do io
        println(io, "δ\tθ\tstability\tstability")
        for (δ, θ, c, s) in data
            println(io, "$(δ)\t$(θ)\t$(c)\t$(s)")
        end
    end
    println("[✓] Saved data to $(filename)")
end

function save_pattern_with_ES(data, filename)
    open(filename, "w") do io
        println(io, "beta\tδ\tstability")
        for (δ, θ, c) in data
            println(io, "$(δ)\t$(θ)\t$(c)")
        end
    end
    println("[✓] Saved data to $(filename)")
end

function save_rho_data(data, filename)
    open(filename, "w") do io
        println(io, "θ\trho")
        for (θ, s) in data
            println(io, "$(θ)\t$(s)")
        end
    end
    println("[✓] Saved data to $(filename)")
end

function save_rho_matrix(data, filename)
    open(filename, "w") do io
        println(io, "θ\tθ2\trho")
        for (θ, θm, s) in data
            println(io, "$(θ)\t$(θm)\t$(s)")
        end
    end
    println("[✓] Saved data to $(filename)")
end





# ---------- helpers: last-year window ----------
function last_period_window(sol; Tperiod=1.0)
    t2 = sol.t[end]
    t1 = t2 - Tperiod
    return t1, t2
end

# ---------- analytic dφ/dθ for von Mises ----------

function Dphi_vonmises(τ, θ, κ)
    x = τ - θ #mod(τ - θ + π, 2π) - π
    φ = (1 / (besseli(0, κ))) * exp(κ * cos(2π*x))
    return 2π * κ * sin(2π*x) * φ
end

# ---------- selection gradient for a given resident sol ----------

function selection_gradient_components_for_resident_sol(sol_resident, θ, pars)
    t1, t2 = last_period_window(sol_resident; Tperiod=1.0)
    β0, δ, κ = pars.β0, pars.δ, pars.a

    f_A0(t) = begin
        S = sol_resident(t)[1]
        dφ = Dphi_vonmises(transform(t), θ, κ)
        β0 * S * dφ
    end
    f_A1(t) = begin
        S = sol_resident(t)[1]
        dφ = Dphi_vonmises(transform(t), θ, κ)
        β0 * δ * cos(2π * t - π) * S * dφ
    end

    A0 = QuadGK.quadgk(f_A0, t1, t2)[1]
    A1 = QuadGK.quadgk(f_A1, t1, t2)[1]
    G  = A0 + A1
    return G, A0, A1
end

# ---------- scan over all θ (resident run per θ) ----------

function selection_gradient_scan(pars;
        θ_grid=range(0, 1; length=360),
        tspan=(0.0, 600.0))

    N = length(θ_grid)
    Gs  = zeros(Float64, N)
    A0s = zeros(Float64, N)
    A1s = zeros(Float64, N)

    @threads for i in 1:N
        θ = θ_grid[i]
        sol = solve_system_vonmises(θ, pars, tspan)
        G, A0, A1 = selection_gradient_components_for_resident_sol(sol, θ, pars)
        Gs[i]  = G
        A0s[i] = A0
        A1s[i] = A1
    end

    return collect(θ_grid), Gs, A0s, A1s
end

# ---------- save ----------

function save_selection_gradient(filename, θ_grid, Gs, A0s, A1s)
    df = DataFrame(θ=θ_grid, grad=Gs, A0=A0s, A1=A1s)
    CSV.write(filename, df; delim='\t')
    println("[✓] Saved $(filename)")
end


# ===== 共通ユーティリティ =====

# 実時間での β(t)
beta(t, pars) = pars.β0 * (1 + pars.δ * cos(2π * t - π))

# 直近の「整数」周期で切ると位相ズレを避けやすい
function last_integer_period(sol; Tperiod=1.0)
    t2 = floor(sol.t[end] / Tperiod) * Tperiod
    t1 = t2 - Tperiod
    return t1, t2
end

# von Mises の d²φ/dθ²（解析式）
# u = 2π(τ-θ)
# ∂²φ/∂θ² = (2π)^2 [ κ^2 sin^2(u) - κ cos(u) ] φ
function D2phi_vonmises(τ, θ, κ)
    u  = 2π * (τ - θ)
    φ  = (1 / besseli(0, κ)) * exp(κ * cos(u))
    return (2π)^2 * (κ^2 * sin(u)^2 - κ * cos(u)) * φ
end

# 住民解に対する“選択曲率”（進化的安定性を判定）
# κ2(θ) = ∫ β(t) S_res(t) ∂²θ φ(t;θ) dt （最後の1周期）
function selection_curvature_for_resident_sol(sol_resident, θ, pars)
    t1, t2 = last_integer_period(sol_resident; Tperiod=1.0)
    f(t) = begin
        S = sol_resident(t)[1]
        beta(t, pars) * S * D2phi_vonmises(t, θ, pars.a)
    end
    QuadGK.quadgk(f, t1, t2)[1]
end

# ===== δ を掃引して CSS / repeller / branching を検出 =====

"""
detect_evo_points_over_delta(pars; δ_vals, θ_grid, tspan, tol_g, tol_k2)

- 各 δ について θ ∈ [0,1) 上で選択勾配 G(θ) を計算（解析式の積分）
- G=0 の点（周期境界を考慮したゼロ交差）を検出
- その点で住民解を解き直し，曲率 κ2 = ∂²ρ/∂θ_m²|_{θ_m=θ} を解析式で算出
- 収束安定（+→−）× 進化的安定（κ2<0）で CSS／repeller／branching を分類

戻り値: Vector{NamedTuple}
  (δ, θ_star, type, conv, evol, κ2)
"""
function detect_evo_points_over_delta(pars;
        δ_vals = 0.0:0.01:0.3,
        θ_grid = range(0, 1; length=300),
        tspan  = (0.0, 600.0),
        tol_g  = 1e-8,
        tol_k2 = 1e-6,
        verbose = true)

    wrap(x) = mod(x, 1)
    θs = collect(θ_grid)
    N  = length(θs)

    out = NamedTuple[]

    for δv in δ_vals
        pars_δ = (; pars..., δ = δv)

        # --- G(θ) を格子上で前計算（あなたの selection_gradient_* を使用） ---
        G = Vector{Float64}(undef, N)
        Threads.@threads for i in 1:N
            θ = θs[i]
            sol = solve_system_vonmises(θ, pars_δ, tspan)
            Gi, _, _ = selection_gradient_components_for_resident_sol(sol, θ, pars_δ)
            G[i] = Gi
        end

        # --- 周期境界を含めたゼロ交差検出 ---
        for i in 1:N
            j  = (i == N ? 1 : i + 1)
            θ1, θ2 = θs[i], θs[j]
            g1, g2 = G[i],  G[j]

            # グリッド点がそのままゼロっぽいとき
            if abs(g1) < tol_g || abs(g2) < tol_g || sign(g1) != sign(g2)
                # 円周跨ぎの線形内挿: 1→0 区間は θ2 に +1 してから内挿→wrap
                θ1i, θ2i = θ1, θ2
                g1i, g2i = g1, g2
                if θ2 < θ1
                    θ2i += 1
                end

                # ゼロが狙えるときだけ内挿
                if sign(g1i) != sign(g2i)
                    θ_star = θ1i - g1i * (θ2i - θ1i) / (g2i - g1i)
                    θ_star = wrap(θ_star)
                else
                    θ_star = (abs(g1i) < abs(g2i)) ? θ1 : θ2
                end

                # 収束安定性: +→− を stable，−→+ を unstable
                conv = (g1 > 0 && g2 < 0) ? :convergent :
                       (g1 < 0 && g2 > 0) ? :divergent  : :unknown

                # --- θ* で住民解を解き直して κ2 を解析式で算出 ---
                sol_star = solve_system_vonmises(θ_star, pars_δ, tspan)
                κ2 = selection_curvature_for_resident_sol(sol_star, θ_star, pars_δ)

                # 進化的安定性（ESS or branching）
                evol = κ2 < -tol_k2 ? :ESS :
                       κ2 >  tol_k2 ? :branching : :neutral

                # ラベル付け
                typ =
                    if conv == :convergent && evol == :ESS
                        :CSS
                    elseif conv == :convergent && evol == :branching
                        :branching_point
                    elseif conv == :divergent
                        :repeller
                    else
                        :other
                    end

                push!(out, (; δ = δv, θ_star, type = typ, conv, evol, κ2))

                if verbose
                    println("[δ=$(round(δv,digits=4))] θ*≈$(round(θ_star,digits=5)) ",
                            "conv=$(conv), evol=$(evol), κ2=$(round(κ2,digits=6)) → $(typ)")
                end
            end
        end
    end

    return out
end

# 結果保存（TSV）
function save_evo_points(data, filename)
    df = DataFrame(data)  # Vector{NamedTuple} をそのまま DataFrame 化
    select!(df, :δ, :θ_star, :type, :conv, :evol, :κ2)  # 列順をそろえる（任意）
    CSV.write(filename, df; delim='\t')
    println("[✓] Saved $(filename)")
end


# --- 解析用：d²φ/dθ²（von Mises）---
# u = 2π(τ-θ)
# ∂²φ/∂θ² = (2π)^2 [ κ^2 sin^2(u) - κ cos(u) ] φ
function D2phi_vonmises(τ, θ, κ)
    u  = 2π * (τ - θ)
    φ  = (1 / besseli(0, κ)) * exp(κ * cos(u))
    return (2π)^2 * (κ^2 * sin(u)^2 - κ * cos(u)) * φ
end

# 実時間の β(t)
beta(t, pars) = pars.β0 * (1 + pars.δ * cos(2π * t - π))

# 曲率 κ2(θ) = ∫ β(t) S_res(t) ∂²θ φ(t;θ) dt（最後の1周期）
function selection_curvature_for_resident_sol(sol_resident, θ, pars)
    t1, t2 = last_period_window(sol_resident; Tperiod=1.0)
    f(t) = begin
        S = sol_resident(t)[1]
        beta(t, pars) * S * D2phi_vonmises(t, θ, pars.a)
    end
    QuadGK.quadgk(f, t1, t2)[1]
end

"""
scan_ES_over_beta_delta(pars; beta_vals, δ_vals, θ_grid, tspan, tol_g, tol_k2, verbose)

各 (β0, δ) について θ∈[0,1) 上で
- 解析的選択勾配 G(θ) を計算（あなたの selection_gradient_components_for_resident_sol を使用）
- 周期境界を考慮して G=0 の点を検出（線形内挿）
- その θ* で住民解を解き直し、解析式で κ2 を計算
- 収束安定（+→−）× 進化的安定（κ2<0）で CSS/branching/その他 を分類
戻り値: Vector{Tuple{Float64,Float64,Int}}  （β0, δ, code）
"""
function scan_ES_over_beta_delta(pars;
        beta_vals::AbstractVector,
        δ_vals::AbstractVector,
        θ_grid = range(0, 1; length=480),
        tspan  = (0.0, 600.0),
        tol_g  = 1e-8,
        tol_k2 = 1e-6,
        verbose=false)

    wrap(x) = mod(x,1)
    θs = collect(θ_grid)
    N  = length(θs)

    pairs = [(β, δ) for β in beta_vals for δ in δ_vals]
    results = Vector{Tuple{Float64,Float64,Int}}(undef, length(pairs))

    Threads.@threads :dynamic for k in eachindex(pairs)
        βv, δv = pairs[k]
        pars_bd = (; pars..., β0 = βv, δ = δv)

        # --- G(θ) を前計算 ---
        G = Vector{Float64}(undef, N)
        for i in 1:N
            θ = θs[i]
            sol = solve_system_vonmises(θ, pars_bd, tspan)
            Gi, _, _ = selection_gradient_components_for_resident_sol(sol, θ, pars_bd)
            G[i] = Gi
        end

        # --- ゼロ交差検出 ＆ 各 θ* で κ2 を評価 ---
        found_css = false
        found_branch = false

        for i in 1:N
            j  = (i == N ? 1 : i + 1)
            θ1, θ2 = θs[i], θs[j]
            g1, g2 = G[i],  G[j]

            # ゼロ近傍 or 符号反転
            crosses = (abs(g1) < tol_g) || (abs(g2) < tol_g) || (sign(g1) != sign(g2))
            if !crosses; continue; end

            # 円周補正付き線形内挿
            θ1i, θ2i = θ1, θ2
            g1i, g2i = g1,  g2
            if θ2i < θ1i
                θ2i += 1
            end
            θ_star = if sign(g1i) != sign(g2i)
                wrap(θ1i - g1i * (θ2i - θ1i) / (g2i - g1i))
            else
                (abs(g1i) < abs(g2i)) ? θ1 : θ2
            end

            # 収束安定性: +→− が収束安定
            conv = (g1 > 0 && g2 < 0) ? :convergent :
                   (g1 < 0 && g2 > 0) ? :divergent  : :unknown

            # θ* で κ2 を評価
            sol_star = solve_system_vonmises(θ_star, pars_bd, tspan)
            κ2 = selection_curvature_for_resident_sol(sol_star, θ_star, pars_bd)
            evol = κ2 < -tol_k2 ? :ESS : κ2 > tol_k2 ? :branching : :neutral

            if conv == :convergent && evol == :branching
                found_branch = true
            elseif conv == :convergent && evol == :ESS
                found_css = true
            end

            if verbose
                println("[β0=$(βv), δ=$(δv)] θ*≈$(round(θ_star,digits=5))  ",
                        "G:$(sign(g1))→$(sign(g2))  κ2=$(round(κ2,digits=6))  ",
                        "→ conv=$(conv), evol=$(evol)")
            end
        end

        code = found_branch ? 2 : found_css ? 1 : 0
        results[k] = (βv, δv, code)
        if verbose
            println("[β0=$(βv), δ=$(δv)] → code = $code")
        end
    end

    return results  # [(β0, δ, code), ...]
end

# 行形式で保存（TSV）
function save_patterns_tsv(patterns::Vector{<:Tuple}, filename::AbstractString; delim='\t')
    β = [p[1] for p in patterns]
    δ = [p[2] for p in patterns]
    code = [p[3] for p in patterns]
    df = DataFrame(beta0 = β, delta = δ, code = code)
    CSV.write(filename, df; delim=delim)
    println("[✓] Saved $(filename)  (rows=$(nrow(df)))")
end

# 行列形式で保存（行=beta0, 列=delta）
# ※ scan_ES_over_beta_delta 内の順序（β 外側 × δ 内側）を前提にしています
function save_patterns_matrix(patterns::Vector{<:Tuple},
                              beta_vals::AbstractVector,
                              δ_vals::AbstractVector,
                              filename::AbstractString)
    nb = length(beta_vals)
    nd = length(δ_vals)
    @assert length(patterns) == nb * nd "patterns の長さが beta×delta と一致しません"

    codes = [p[3] for p in patterns]
    M = reshape(codes, nd, nb)'  # rows=beta, cols=delta

    open(filename, "w") do io
        # ヘッダ
        print(io, "beta0\\delta")
        for δ in δ_vals
            print(io, '\t', δ)
        end
        println(io)
        # 本体
        for i in 1:nb
            print(io, beta_vals[i])
            for j in 1:nd
                print(io, '\t', M[i, j])
            end
            println(io)
        end
    end
    println("[✓] Saved $(filename)  (matrix $(nb)×$(nd))")
end

# （オプション）DataFrame にすぐしたい場合
function patterns_dataframe(patterns::Vector{<:Tuple})
    DataFrame(beta0 = [p[1] for p in patterns],
              delta = [p[2] for p in patterns],
              code  = [p[3] for p in patterns])
end


# ================== NEW: selection gradient at a specific θ ==================

"""
selection_gradient_at_theta(pars, θ; tspan=(0.0,600.0))

Resident を θ に固定して解き、最後の1周期で
G(θ) = ∫ β(t) S_res(t) ∂φ/∂θ (t; θ) dt を返す。
"""
function selection_gradient_at_theta(pars, θ; tspan=(0.0, 600.0))
    sol = solve_system_vonmises(θ, pars, tspan)
    G, _, _ = selection_gradient_components_for_resident_sol(sol, θ, pars)
    return G
end

# ================== NEW: δ を動かし、指定ルールの θ で勾配を記録 ==================

"""
scan_special_gradients(pars; δ_vals, tspan=(0.0,600.0))

各 δ について:
  - β_min = β0*(1-δ) > γ + d なら θ=0 と θ=1 で G を評価（理論上同じだが両方保存）
  - そうでなく、β(t)=γ+d となる θ が存在すれば（|y|≤1）、
      cos(2πθ - π) = y を満たす 2 解 θ*1, θ*2 で G を評価して保存
戻り値: Vector{NamedTuple} with keys (:δ, :θ, :grad, :note)
"""
function scan_special_gradients(pars; δ_vals, tspan=(0.0,600.0))
    out = NamedTuple[]
    β0, γ, d = pars.β0, pars.γ, pars.d

    wrap(x) = mod(x, 1)

    for δ in δ_vals
        # 範囲チェック
        βmin = β0 * (1 - δ)
        βmax = β0 * (1 + δ)

        if βmin > (γ + d)
            # 規定：θ=0 と θ=1（=0 と同値だが両方残す）
            θs = [0.0, 1.0]
            for θ in θs
                Gθ = selection_gradient_at_theta((; pars..., δ=δ), wrap(θ); tspan=tspan)
                push!(out, (; δ=δ, θ=wrap(θ), grad=Gθ, note="βmin>γ+d ⇒ θ=0/1"))
            end
        else
            # β(θ) = γ + d となる θ を解く： cos(2πθ - π) = y
            # y = ((γ+d)/β0 - 1)/δ
            if δ == 0
                # 季節性なし。β(t)=β0 が閾値と等しい/等しくないで分岐
                if isapprox(β0, γ + d; atol=1e-12, rtol=1e-12)
                    # 任意の θ で等しいが、代表として θ=0 を保存
                    Gθ = selection_gradient_at_theta((; pars..., δ=δ), 0.0; tspan=tspan)
                    push!(out, (; δ=δ, θ=0.0, grad=Gθ, note="δ=0 & β0=γ+d ⇒ θ=0"))
                else
                    # 解なし（β は θ に依らず一定で閾値と不一致）
                    push!(out, (; δ=δ, θ=NaN, grad=NaN, note="δ=0 & no solution for β=γ+d"))
                end
            else
                y = ((γ + d) / β0 - 1.0) / δ
                if abs(y) <= 1.0
                    # 2 解：u = 2πθ - π; u = ±acos(y) + 2πk
                    acosy = acos(y)
                    u1 =  acosy
                    u2 = -acosy
                    # θ = (u + π) / (2π)  を [0,1) に wrap
                    θ1 = wrap((u1 + π) / (2π))
                    θ2 = wrap((u2 + π) / (2π))

                    for (θ, tag) in ((θ1, "root1"), (θ2, "root2"))
                        Gθ = selection_gradient_at_theta((; pars..., δ=δ), θ; tspan=tspan)
                        push!(out, (; δ=δ, θ=θ, grad=Gθ, note="β(θ)=γ+d @ $tag"))
                    end
                else
                    # 閾値が β(t) の振幅範囲外 → 解なし
                    push!(out, (; δ=δ, θ=NaN, grad=NaN, note="no θ with β=γ+d"))
                end
            end
        end
    end

    return out
end

# ================== NEW: 保存関数（TSV） ==================

function save_special_gradients(data, filename::AbstractString)
    df = DataFrame(data)

    # 欲しい列の順序（Symbol）
    cols = [:δ, :θ, :grad]

    # 列名を Symbol で取得して比較
    existing_syms = names(df, Symbol)
    keep = [c for c in cols if c in existing_syms]

    # もし何も一致しなければ、そのまま全部出す（安全網）
    if isempty(keep)
        @warn "No expected columns found; saving all columns as-is" cols=cols existing=existing_syms
    else
        select!(df, keep)
    end

    CSV.write(filename, df; delim='\t')
    println("[✓] Saved $(filename)  (rows=$(nrow(df)), cols=$(ncol(df)))")
end



# -------------------------------
# example usage
# -------------------------------
# Define parameters as NamedTuple
pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 2.0,
    δ = 0.0,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)



#δ_vals = 0.:0.01:0.4
#beta_vals = 2:0.1:10#10:0.1:60
#scan_data = scan_ES_outer_threads(pars, beta_vals, δ_vals)
#save_pattern_with_ES(scan_data, "ES_data.tsv")

# Run scan and plot
#δ_vals = 0.0:0.01:1.0
#δ_vals = 0.0:0.001:0.2
#bifurcation_data = scan_rho_vs_theta(pars, δ_vals; Tmax=500)
#save_bifurcation_data(bifurcation_data, "bifurcation_data_figure.tsv")

#bifurcation_data = scan_rho_vs_theta_with_ES(pars, δ_vals; Tmax=500)
#save_bifurcation_data_with_ES(bifurcation_data, "bifurcation_data_figure_with_ES.tsv")

beta_vals = 1.2:0.05:10
δ_vals    = 0.0:0.002:0.4

patterns = scan_ES_over_beta_delta(pars; beta_vals=beta_vals, δ_vals=δ_vals, θ_grid=range(0,1;length=100), tspan=(0.0,500.0), tol_g=1e-8, tol_k2=1e-6, verbose=true)

save_patterns_tsv(patterns, joinpath(OUTPUT_DIR, "ES_pattern_long.tsv"))



pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 2.0,
    δ = 0.01,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)

θ_grid = range(0, 1; length=300)

θs, Gs, A0s, A1s = selection_gradient_scan(pars; θ_grid=θ_grid, tspan=(0.0, 500.0))

save_selection_gradient(joinpath(OUTPUT_DIR, "selection_gradient_0.01.tsv"), θs, Gs, A0s, A1s)

pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 2.0,
    δ = 0.1,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)

θ_grid = range(0, 1; length=300)

θs, Gs, A0s, A1s = selection_gradient_scan(pars; θ_grid=θ_grid, tspan=(0.0, 500.0))

save_selection_gradient(joinpath(OUTPUT_DIR, "selection_gradient_0.1.tsv"), θs, Gs, A0s, A1s)

pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 1.1,
    δ = 0.00005,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)

θ_grid = range(0, 1; length=300)

θs, Gs, A0s, A1s = selection_gradient_scan(pars; θ_grid=θ_grid, tspan=(0.0, 500.0))

save_selection_gradient(joinpath(OUTPUT_DIR, "selection_gradient_extinction.tsv"), θs, Gs, A0s, A1s)

pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 1.1,
    δ = 0.1,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)

# # δ を 0 〜 0.8 で掃引
# δ_vals = 0.0:0.000001:0.001

# # 計算
# res = scan_special_gradients(pars; δ_vals=δ_vals, tspan=(0.0, 600.0))

# println("length(res) = ", length(res))
# println(first(res, min(3, length(res))))  # 先頭数件を表示

# # 保存
# save_special_gradients(res, "special_gradients.tsv")


pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 1.11,
    δ = 0.1,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)

# # δ を 0 〜 0.8 で掃引
# δ_vals = 0.0:0.000001:0.001

# # 計算
# res = scan_special_gradients(pars; δ_vals=δ_vals, tspan=(0.0, 600.0))

# println("length(res) = ", length(res))
# println(first(res, min(3, length(res))))  # 先頭数件を表示

# # 保存
# save_special_gradients(res, "special_gradients2.tsv")


pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 2.0,
    δ = 0.01,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)

δ_vals = 0.0:0.001:0.2
pts = detect_evo_points_over_delta(pars; δ_vals=δ_vals,
                                  θ_grid=range(0,1;length=300),
                                  tspan=(0.0, 500.0),
                                  tol_g=1e-8, tol_k2=1e-6,
                                  verbose=true)
save_evo_points(pts, joinpath(OUTPUT_DIR, "evo_points.tsv"))

pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 9.35,
    δ = 0.01,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)

# δ_vals = 0.0:0.01:0.4
# pts = detect_evo_points_over_delta(pars; δ_vals=δ_vals,
#                                   θ_grid=range(0,1;length=300),
#                                   tspan=(0.0, 500.0),
#                                   tol_g=1e-8, tol_k2=1e-6,
#                                   verbose=true)
# save_evo_points(pts, "evo_points_branching.tsv")


### Invasion fitness ###
pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 2,
    δ = 0.01,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)
rholist = rho_list(0.5, pars, 0.0,2; θ_range=range(0, 1; length=200))
save_rho_data(rholist, joinpath(OUTPUT_DIR, "rho_list_figure_0_2.tsv"))

rholist = rho_list(0.5, pars, 0.025,2; θ_range=range(0, 1; length=200))
save_rho_data(rholist, joinpath(OUTPUT_DIR, "rho_list_figure_0.025_2.tsv"))

rholist = rho_list(0.395254761661653, pars, 0.1,2; θ_range=range(0, 1; length=200))
save_rho_data(rholist, joinpath(OUTPUT_DIR, "rho_list_figure_0.1_2.tsv"))



"""
pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 30.0,
    δ = 0.0,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)
δ_vals = 0.0:0.003:0.3
#bifurcation_data = scan_rho_vs_theta_with_ES(pars, δ_vals)
#save_bifurcation_data_with_ES(bifurcation_data, "bifurcation_data_figure_with_ES_30.tsv")

δ_vals = 0.15:0.001:0.25
#bifurcation_data = scan_rho_vs_theta_with_ES(pars, δ_vals)
#save_bifurcation_data_with_ES(bifurcation_data, "bifurcation_data_figure_with_ES_30_detail.tsv")

pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 40.0,
    δ = 0.0,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)
δ_vals = 0.0:0.003:0.3
#bifurcation_data = scan_rho_vs_theta_with_ES(pars, δ_vals)
#save_bifurcation_data_with_ES(bifurcation_data, "bifurcation_data_figure_with_ES_40.tsv")

δ_vals = 0.25:0.0005:0.3
#bifurcation_data = scan_rho_vs_theta_with_ES(pars, δ_vals)
#save_bifurcation_data_with_ES(bifurcation_data, "bifurcation_data_figure_with_ES_40_detail.tsv")


#rhodata = rho_data(pars, 0.06; θ_range=range(-π, π; length=300), ε=0.05)
#save_rho_data(rhodata, "rho_data_figure_0.06.tsv")


#rhodata = rho_data(pars, 0.01; θ_range=range(-π, π; length=300), ε=0.05)
#save_rho_data(rhodata, "rho_data_figure_0.01.tsv")

#θ_grid = range(-π, π; length=300)

#θs, Gs, A0s, A1s = selection_gradient_scan(pars; θ_grid=θ_grid, tspan=(0.0, 500.0))

#save_selection_gradient("selection_gradient_0.06.tsv", θs, Gs, A0s, A1s)


#rhomatrix = rho_matrix(pars, 0.0; θ_range=range(-π, π; length=200))
#save_rho_matrix(rhomatrix, "rho_matrix_figure_0.tsv")

# rholist = rho_list(0.0, pars, 0.0,10; θ_range=range(-π, π; length=200))
# save_rho_data(rholist, "rho_list_figure_0_10.tsv")

# rholist = rho_list(-0.56, pars, 0.06,10; θ_range=range(-π, π; length=200))
# save_rho_data(rholist, "rho_list_figure_0.06_10.tsv")

# rholist = rho_list(0.0, pars, 0.0,20; θ_range=range(-π, π; length=200))
# save_rho_data(rholist, "rho_list_figure_0_20.tsv")
# rholist = rho_list(0.0, pars, 0.0,30; θ_range=range(-π, π; length=200))
# save_rho_data(rholist, "rho_list_figure_0_30.tsv")

# rholist = rho_list_multi(-π/2, π/2, pars, 0.0, 30; θ_range=range(-π, π; length=200))
# save_rho_data(rholist, "rho_list_figure_multi_0_30.tsv")

# rholist = rho_list_multi(-π/2, π/2, pars, 0.0, 30; θ_range=range(-π, π; length=200))
# save_rho_data(rholist, "rho_list_figure_multi_0_30.tsv")

# rholist = rho_list_multi(-1.911, 0.09684, pars, 0.15, 30; θ_range=range(-π, π; length=200))
# save_rho_data(rholist, "rho_list_figure_multi_0.15_30.tsv")

# rholist = rho_list_multi(-1.886, 0.466, pars, 0.25, 50; θ_range=range(-π, π; length=200))
# save_rho_data(rholist, "rho_list_figure_multi_0.25_50.tsv")
"""



