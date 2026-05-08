### bifurcation_diagram.jl ###

using DifferentialEquations
using QuadGK
using SpecialFunctions
using Plots
using DelimitedFiles
using Interpolations
using CSV
using DataFrames

cd(@__DIR__)

# Output directory: override by setting the OUTPUT_DIR environment variable.
#   e.g.  julia bifurcation_diagram.jl         → saves to ./output/
#   e.g.  OUTPUT_DIR=/path/to/dir julia bifurcation_diagram.jl
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
    u0 = [1.0 - 0.1-0.05, 0.1, 0.05, 0.0]  # initial values: S, J, R
    params = [pars.λ, pars.r, pars.d, pars.γ, pars.c, pars.β0, pars.δ, pars.ω, pars.α, pars.a, pars.μ, θ1, θ2]
    prob = ODEProblem(define_sys_two_species, u0, tspan, params)
    solve(prob, RK4(), dt=0.01)
end

# -------------------------------
# invasion fitness function
# -------------------------------
function integrate_resident_density_vonmises(sol, θm, pars)
    result, _ = quadgk(t -> begin
        S_t = sol(t + 199)[1]
        J_t = sol(t + 199)[2]
        φ = Func_vonmises(t, θm, pars.a)
        β = pars.β0 * (1 + pars.δ * cos(2π * t - π))
        return β * S_t * φ - (pars.d + pars.γ + pars.α)
    end, 0, 1)
    return result
end

function integrate_resident_density_two_species(sol, θ1m, θ2m, pars)
    f1(t) = begin
        S_t = sol(t + 199)[1]
        φ1 = Func_vonmises(t, θ1m, pars.a)
        β = pars.β0 * (1 + pars.δ * cos(2π * t - π))
        return β * S_t * φ1 - (pars.d + pars.γ + pars.α)
    end

    f2(t) = begin
        S_t = sol(t + 199)[1]
        φ2 = Func_vonmises(t, θ2m, pars.a)
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

function scan_rho_vs_theta(pars, δ_vals; θ_range=range(0, 1; length=300), ε=0.05)
    result = []

    θ_array = collect(θ_range)
    N = length(θ_array)

    for δv in δ_vals
        pars_δ = (; pars..., δ=δv)
        ρ_vals = Float64[]

        for θ in θ_array
            sol = solve_system_vonmises(θ, pars_δ, (0.0, 200.0))
            ρ = integrate_resident_density_vonmises(sol, θ + ε, pars_δ)
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

function plot_rho_vs_theta(pars, δ; θ_range=range(0, 1; length=300),savepath=nothing)
    ρ_vals = Float64[]
    ρ_valsm = Float64[]
    ρ_valsp = Float64[]

    # δだけ上書き
    pars_δ = (; pars..., δ=δ)

    for θm in θ_range
        sol = solve_system_vonmises(θm, pars_δ, (0.0, 200.0))
        ρm1 = integrate_resident_density_vonmises(sol, θm - 0.01, pars_δ)
        ρ0  = integrate_resident_density_vonmises(sol, θm, pars_δ)
        ρp1 = integrate_resident_density_vonmises(sol, θm + 0.01, pars_δ)
        push!(ρ_vals, ρ0)
        push!(ρ_valsm, ρm1)
        push!(ρ_valsp, ρp1)
    end

    plot(θ_range, ρ_vals, label="ρ(θm)", xlabel="θm", ylabel="Invasion fitness", title="ρ(θm) at δ = $(δ)")
    plot!(θ_range, ρ_valsm, label="ρ(θm)_plus", xlabel="θm", ylabel="Invasion fitness", title="ρ(θm) at δ = $(δ)")
    plot!(θ_range, ρ_valsp, label="ρ(θm)_minus", xlabel="θm", ylabel="Invasion fitness", title="ρ(θm) at δ = $(δ)")
    hline!([0.0], linestyle=:dash, color=:black, label="ρ=0")
    savefig(savepath)
    println("[✓] Saved plot to $(savepath)")
end

function scan_ess(pars; θ_range=range(0, 1; length=300), ε=0.001)
    θs = collect(θ_range)
    out = Tuple{Float64,Float64,Symbol}[]
    for θ in θs
        sol = solve_system_vonmises(θ, pars, (0.0, 300.0))
        ρp = integrate_resident_density_vonmises(sol, mod(θ+ε,1.0), pars)
        ρm = integrate_resident_density_vonmises(sol, mod(θ-ε,1.0), pars)
        # 勾配の符号反転をチェック
        if sign(ρm) != sign(ρp)
            # 安定性：右が負・左が正（内側へ押し戻す）なら stable
            stab = (ρm > 0 && ρp < 0) ? :stable : :unstable
            push!(out, (pars.δ, θ, stab))
            println((pars.δ, θ, stab, ρm, ρp))
        end
    end
    return out
end

function scan_ess(pars; θ_range=range(0, 1; length=300), ε=0.001)
    θs = collect(θ_range)
    out = Tuple{Float64,Float64,Symbol}[]
    for θ in θs
        sol = solve_system_vonmises(θ, pars, (0.0, 300.0))
        ρp = integrate_resident_density_vonmises(sol, mod(θ+ε,1.0), pars)
        ρm = integrate_resident_density_vonmises(sol, mod(θ-ε,1.0), pars)
        # 勾配の符号反転をチェック
        if sign(ρm) != sign(ρp)
            # 安定性：右が負・左が正（内側へ押し戻す）なら stable
            stab = (ρm < 0 && ρp > 0) ? :stable : :unstable
            push!(out, (pars.δ, θ, stab))
            println((pars.δ, θ, stab, ρm, ρp))
        end
    end
    return out
end

function has_stable_ess(pars; θ_range=range(0,1;length=300), ε=0.01)
    any(s == :stable for (_,_,s) in scan_ess(pars; θ_range=θ_range, ε=ε))
end

# NamedTuple の上書きはこれで安全
setparam(pars, name::Symbol, val) = merge(pars, (; name => val))

function detect_critical_delta_paramscan(pars_base, param_name::Symbol, param_vals;
                                         δ_range=(0.0, 1.0), tol=1e-3,
                                         θ_range=range(0, 1; length=200), ε=0.001)
    critical_points = []
    for pval in param_vals
        pars = setparam(pars_base, param_name, pval)
        δ_low, δ_high = δ_range
        δ_c = nothing
        while δ_high - δ_low > tol
            δ_mid = (δ_low + δ_high)/2
            pars_mid = setparam(pars, :δ, δ_mid)
            if has_stable_ess(pars_mid; θ_range=θ_range, ε=ε)
                δ_high = δ_mid; δ_c = δ_mid
            else
                δ_low  = δ_mid
            end
            println(has_stable_ess(pars_mid; θ_range=θ_range, ε=ε))
        end
        if δ_c !== nothing
            push!(critical_points, (pval, δ_c))
            println("[✓] $param_name = $pval → δ_c ≈ $δ_c")
        else
            println("[×] $param_name = $pval → no stable ESS found")
        end
    end
    return critical_points
end


# function has_stable_ess(pars, δ; θ_range=range(0, 1; length=300), ε=0.05)
#     pars_δ = (; pars..., δ=δ)
#     bif_data = scan_rho_vs_theta(pars_δ, [δ]; θ_range=θ_range, ε=ε)
#     return any(s == :stable for (_, _, s) in bif_data)
# end

# function detect_critical(pars; δ_range=(0.0, 1.0), tol=1e-5)
#     critical_points = []

#     δ_low, δ_high = δ_range
#     δ_c = nothing

#     while δ_high - δ_low > tol
#         δ_mid = (δ_low + δ_high) / 2
#         if has_stable_ess(pars, δ_mid)
#             δ_high = δ_mid
#             δ_c = δ_mid
#         else
#             δ_low = δ_mid
#         end
#     end

#     if δ_c !== nothing
#         push!(critical_points, δ_c)
#         println("[✓] δ_c ≈ $δ_c")
#     else
#         println("[×] no stable ESS found in range")
#     end

#     return critical_points
# end

# function detect_critical_delta_fast(pars_base; a_vals=0.5:0.5:5.0, δ_range=(0.0, 1.0), tol=1e-3)
#     critical_points = []

#     for a in a_vals
#         pars = (; pars_base..., a=a)

#         δ_low, δ_high = δ_range
#         δ_c = nothing

#         while δ_high - δ_low > tol
#             δ_mid = (δ_low + δ_high) / 2
#             if has_stable_ess(pars, δ_mid)
#                 δ_high = δ_mid
#                 δ_c = δ_mid
#             else
#                 δ_low = δ_mid
#             end
#         end

#         if δ_c !== nothing
#             push!(critical_points, (a, δ_c))
#             println("[✓] a = $a → δ_c ≈ $δ_c")
#         else
#             println("[×] a = $a → no stable ESS found in range")
#         end
#     end

#     return critical_points
# end

# function detect_critical_delta_paramscan(pars_base, param_name::Symbol, param_vals;
#                                          δ_range=(0.0, 1.0), tol=1e-3,
#                                          θ_range=range(0, 1; length=200), ε=0.01)
#     critical_points = []

#     for pval in param_vals
#         # 修正：NamedTuple 上書き構文
#         pars = (; pars_base..., param_name => pval)

#         # 2分探索で δ_c を求める
#         δ_low, δ_high = δ_range
#         δ_c = nothing

#         while δ_high - δ_low > tol
#             δ_mid = (δ_low + δ_high) / 2
#             if has_stable_ess(pars, δ_mid; θ_range=θ_range, ε=ε)
#                 δ_high = δ_mid
#                 δ_c = δ_mid
#             else
#                 δ_low = δ_mid
#             end
#         end

#         if δ_c !== nothing
#             push!(critical_points, (pval, δ_c))
#             println("[✓] $param_name = $pval → δ_c ≈ $δ_c")
#         else
#             println("[×] $param_name = $pval → no stable ESS found")
#         end
#     end

#     return critical_points
# end


function export_rho_theta_matrix(pars, δ; θ_vals=range(0, 1; length=100),
                                              θm_vals=range(0, 1; length=100),
                                              tspan=(0.0, 200.0),
                                              filename="rho_matrix.tsv")

    pars_δ = (; pars..., δ=δ)
    open(filename, "w") do io
        println(io, "θ\tθm\tρ")  # ヘッダー

        for θ in θ_vals
            sol = solve_system_vonmises(θ, pars_δ, tspan)
            for θm in θm_vals
                ρ = integrate_resident_density_vonmises(sol, θm, pars_δ)
                println(io, "$(θ)\t$(θm)\t$(ρ)")
            end
        end
    end
    println("[✓] Exported ρ(θₘ; θ) matrix to $filename")
end

function export_TEP_matrix(pars, δ; θ_vals=range(-π, π; length=100),
                                              θm_vals=range(-π, π; length=100),
                                              tspan=(0.0, 200.0),
                                              filename="rho_matrix.tsv")

    pars_δ = (; pars..., δ=δ)
    open(filename, "w") do io
        println(io, "θ\tθm\tρ")  # ヘッダー

        for θ in θ_vals
            sol = solve_system_vonmises(θ, pars_δ, tspan)
            for θm in θm_vals
                if θ == θm
                    tmp = 0
                else
                    ρ = integrate_resident_density_vonmises(sol, θm, pars_δ)
                    sol2 = solve_system_vonmises(θm, pars_δ, tspan)
                    ρ2 = integrate_resident_density_vonmises(sol2, θ, pars_δ)
                    tmp = (ρ > 0 && ρ2 > 0) ? 1 : 0
                end
                println(io, "$(θ)\t$(θm)\t$(tmp)")
            end
        end
    end
    println("[✓] Exported ρ(θₘ; θ) matrix to $filename")
end

function calculate_coESS_matrix(pars, δ; θ_vals=range(0, 1; length=100),
                                              θm_vals=range(0, 1; length=100),
                                              tspan=(0.0, 200.0), epsilon=0.00001,
                                              filename1="rho_matrix.tsv",filename2="rho_matrix.tsv")

    pars_δ = (; pars..., δ=δ)
    io = open(filename1, "w")
    io2 = open(filename2, "w")
    println(io, "θ\tθm\tρ")  # ヘッダー
    println(io2, "θ\tθm\tρ")  # ヘッダー

    for θ1 in θ_vals
        for θ2 in θm_vals
            sol = solve_system_two_species(θ1, θ2, pars_δ, tspan)
            θ1m = θ1 + epsilon
            θ2m = θ2 + epsilon
            θ1m == θ1m > π ? θ1m - 2*π : θ1m 
            θ2m == θ2m > π ? θ2m - 2*π : θ2m 
            ρ1,ρ2 = integrate_resident_density_two_species(sol, θ1m, θ2, pars_δ)
            println(io, "$(θ1)\t$(θ2)\t$(ρ1)\t$(ρ2)")
            ρ1,ρ2 = integrate_resident_density_two_species(sol, θ1, θ2m, pars_δ)
            println(io2, "$(θ1)\t$(θ2)\t$(ρ1)\t$(ρ2)")
        end
    end

    close(io)
    close(io2)
    
    println("[✓] Exported ρ(θₘ; θ) matrix to $filename1")
    println("[✓] Exported ρ(θₘ; θ) matrix to $filename2")
end

# -------------------------------
# bifurcation plot
# -------------------------------
function plot_bifurcation(data; savepath=nothing)
    stable_pts = [(δ, θ) for (δ, θ, s) in data if s == :stable]
    unstable_pts = [(δ, θ) for (δ, θ, s) in data if s == :unstable]

    scatter(first.(stable_pts), last.(stable_pts); label="Stable ESS", color=:blue, markersize=3, xlims=(0,1), ylims=(-π,π))
    scatter!(first.(unstable_pts), last.(unstable_pts); label="Unstable", color=:red, markersize=3)
    xlabel!("δ")
    ylabel!("θ (ESS candidates)")
    title!("ESS Bifurcation Diagram")
    if !isnothing(savepath)
        savefig(savepath)
        println("[✓] Saved plot to $(savepath)")
    end
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

function save_critical_points(critical_points, filename::String)
    df = DataFrame(param = [x[1] for x in critical_points],
                   delta_c = [x[2] for x in critical_points])
    CSV.write(filename, df)
    println("Saved results to $filename")
end

using Plots

function plot_critical_delta(critical_points; xlabel_str="a (κ)", ylabel_str="critical δ", title_str="Critical δ vs a", savepath=nothing)
    a_vals = first.(critical_points)
    δ_vals = last.(critical_points)

    scatter(a_vals, δ_vals;
        xlabel=xlabel_str,
        ylabel=ylabel_str,
        title=title_str,
        marker=:circle,
        linewidth=2,
        legend=false)

    if savepath !== nothing
        savefig(savepath)
        println("[✓] Saved plot to $(savepath)")
    end
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

critical_pts = detect_critical_delta_paramscan(pars, :β0, 1.2:0.1:10.2; tol=1e-4,ε=0.01)
#plot_critical_delta(critical_pts; xlabel_str="β0", ylabel_str="critical δ", title_str="δ_c vs β0", savepath="critical_plot_beta_deep.png")
save_critical_points(critical_pts, joinpath(OUTPUT_DIR, "critical_delta_vs_beta.csv"))

# Run scan and plot
#δ_vals = 0.0:0.01:1.0
#δ_vals = 0.0:0.001:0.1
#bifurcation_data = scan_rho_vs_theta(pars, δ_vals)
#save_bifurcation_data(bifurcation_data, "bifurcation_data.tsv")
#plot_bifurcation(bifurcation_data; savepath="bifurcation_plot.png")

#critical_data = detect_critical_delta_fast(pars; a_vals=0.5:0.5:5.0, δ_range=(0.0, 1.0), tol=1e-3)
#plot_critical_delta(critical_data; xlabel_str="a (κ)", ylabel_str="critical δ", title_str="Critical δ vs a", savepath="critical_plot.png")

#critical_pts = detect_critical_delta_paramscan(pars, :a, 0.5:0.5:5.0)
#plot_critical_delta(critical_pts; xlabel_str="a", ylabel_str="critical δ", title_str="δ_c vs a", savepath="critical_plot_a.png")

#critical_pts = detect_critical_delta_paramscan(pars, :γ, 0.1:0.1:2.0)
#plot_critical_delta(critical_pts; xlabel_str="γ", ylabel_str="critical δ", title_str="δ_c vs γ", savepath="critical_plot_γ.png")

#critical_pts = detect_critical_delta_paramscan(pars, :c, 0.1:0.1:2.0)
#plot_critical_delta(critical_pts; xlabel_str="c", ylabel_str="critical δ", title_str="δ_c vs c", savepath="critical_plot_c.png")

#critical_pts = detect_critical_delta_paramscan(pars, :β0, 8:2:50)
#plot_critical_delta(critical_pts; xlabel_str="β0", ylabel_str="critical δ", title_str="δ_c vs β0", savepath="critical_plot_beta.png")



#critical_pts = detect_critical_delta_paramscan(pars, :d, 0.1:0.1:2.0)
#plot_critical_delta(critical_pts; xlabel_str="d", ylabel_str="critical δ", title_str="δ_c vs d", savepath="critical_plot_d.png")


#export_rho_theta_matrix(pars, 0.0; θ_vals=range(-π, π; length=200),θm_vals=range(-π, π; length=200),tspan=(0.0, 200.0),filename="rho_matrix_delta=0.tsv")


#export_rho_theta_matrix(pars, 0.05; θ_vals=range(-π, π; length=200),θm_vals=range(-π, π; length=200),tspan=(0.0, 200.0),filename="rho_matrix_delta=0.05.tsv")
                                              
#plot_rho_vs_theta(pars, 0.5; savepath="confirmation_plot.png")

#detect_critical(pars)


pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 20.0,
    δ = 1.0,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 10,     # κ in von Mises
    μ = 0.01
)

#export_rho_theta_matrix(pars, 0.0; θ_vals=range(-π, π; length=200),θm_vals=range(-π, π; length=200),tspan=(0.0, 200.0),filename="rho_matrix_delta=0_20.tsv")


#export_rho_theta_matrix(pars, 0.5; θ_vals=range(-π, π; length=200),θm_vals=range(-π, π; length=200),tspan=(0.0, 200.0),filename="rho_matrix_delta=0.5_20.tsv")

#export_TEP_matrix(pars, 0.5; θ_vals=range(-π, π; length=200),θm_vals=range(-π, π; length=200),tspan=(0.0, 200.0),filename="TEP_matrix_delta=1.0_20_a=10.tsv")
#export_TEP_matrix(pars, 0.5; θ_vals=range(-π, π; length=200),θm_vals=range(-π, π; length=200),tspan=(0.0, 200.0),filename="TEP_matrix_delta=1.0_20_a=1.tsv")

pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 10.0,
    δ = 0.0,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)

# # Run scan and plot
# δ_vals = 0.0:0.01:1.0
# #δ_vals = 0.0:0.001:0.1
# bifurcation_data = scan_rho_vs_theta(pars, δ_vals)
# save_bifurcation_data(bifurcation_data, "bifurcation_data_tmp.tsv")
# plot_bifurcation(bifurcation_data; savepath="bifurcation_plot_tmp.png")


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
#export_TEP_matrix(pars, 0.1; θ_vals=range(-π, π; length=200),θm_vals=range(-π, π; length=200),tspan=(0.0, 200.0),filename="TEP_matrix_delta=0.1_30_a=1.tsv")
#export_TEP_matrix(pars, 0.15; θ_vals=range(-π, π; length=200),θm_vals=range(-π, π; length=200),tspan=(0.0, 200.0),filename="TEP_matrix_delta=0.15_30_a=1.tsv")

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
#export_TEP_matrix(pars, 0.15; θ_vals=range(-π, π; length=200),θm_vals=range(-π, π; length=200),tspan=(0.0, 200.0),filename="TEP_matrix_delta=0.15_40_a=1.tsv")
#export_TEP_matrix(pars, 0.2; θ_vals=range(-π, π; length=200),θm_vals=range(-π, π; length=200),tspan=(0.0, 200.0),filename="TEP_matrix_delta=0.2_40_a=1.tsv")

#calculate_coESS_matrix(pars, 0.22; θ_vals=range(-π, π; length=300),θm_vals=range(-π, π; length=300),tspan=(0.0, 200.0),filename1="coESS_matrix1_delta=0.22_40_a=1.tsv",filename2="coESS_matrix2_delta=0.22_40_a=1.tsv")
#calculate_coESS_matrix(pars, 0.2; θ_vals=range(-π, π; length=300),θm_vals=range(-π, π; length=300),tspan=(0.0, 200.0),filename1="coESS_matrix1_delta=0.2_40_a=1.tsv",filename2="coESS_matrix2_delta=0.2_40_a=1.tsv")
#calculate_coESS_matrix(pars, 0.15; θ_vals=range(-π, π; length=300),θm_vals=range(-π, π; length=300),tspan=(0.0, 200.0),filename1="coESS_matrix1_delta=0.15_40_a=1.tsv",filename2="coESS_matrix2_delta=0.15_40_a=1.tsv")
#calculate_coESS_matrix(pars, 0; θ_vals=range(-π, π; length=300),θm_vals=range(-π, π; length=300),tspan=(0.0, 200.0),filename1="coESS_matrix1_delta=0_40_a=1.tsv",filename2="coESS_matrix2_delta=0_40_a=1.tsv")

pars = (
    λ = 0.,
    r = 0.0,
    d = 0.1,
    γ = 1,
    c = 0.5,
    β0 = 15.0,
    δ = 0.0,     # will be replaced
    ω = 1.0,
    α = 0.,
    a = 1,     # κ in von Mises
    μ = 0.01
)

#calculate_coESS_matrix(pars, 0; θ_vals=range(-π, π; length=300),θm_vals=range(-π, π; length=300),tspan=(0.0, 200.0),filename1="coESS_matrix1_delta=0_15_a=1.tsv",filename2="coESS_matrix2_delta=0_15_a=1.tsv")
#calculate_coESS_matrix(pars, 0.1; θ_vals=range(-π, π; length=300),θm_vals=range(-π, π; length=300),tspan=(0.0, 200.0),filename1="coESS_matrix1_delta=0.1_15_a=1.tsv",filename2="coESS_matrix2_delta=0.1_15_a=1.tsv")
