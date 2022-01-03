
# This file contains codes to solve the basic aiyagri model without labor supply 
# The code has been optimized for Julia

#= Steps:

1. Define the parameters in the household instance
2. Specify the utility functions
3. Import the relevant function
4. Calulate eqbm for different values of r
5. Calculate eqbm r
6. Plot the policy function and  asset distribution 
=#

# This code will run the will run Aiyagari model taking into account code optimizations

## Loading Packages
using Plots, ForwardDiff, Interpolations, LinearAlgebra, Roots, SparseArrays, NLsolve, Parameters, Dierckx

## Calling our Packages
include("Tauchen.jl")
include("get_grids.jl")
include("aiyagari_functions.jl")

## Defining the household instance 

Household = @with_kw (
           
 β = 0.98,
 μ=1.5,
 frish=2.0,
 δ = 0.075,
 #ψ = 0.0,
 θ = 0.3,

 τy = 0.0,
 ϕg = 0.0,
 ϕb = 0.0,

 Na = 100, #no of points in asset grid
 NaL = 100, # no of grid points used for Q; L for large
 Ne = 5,  # no of points in AR(1) shock
 amin = 0,
 amax = 40,

 ϵ = exp.(Tauchen(0.0, 0.6, 0.3, Ne)[1]), 
 Pϵ = Tauchen(0.0, 0.6, 0.3, Ne)[2],
 a_grid = get_grids(amin, amax, Na, 2),
 a_gridL = get_grids(amin, amax, NaL, 2),
 Amat = [a_grid[i] for i in 1:Na, j in 1:Ne],
 AmatL = [a_gridL[i] for i in 1:NaL, j in 1:Ne],
 ϵmat = [ϵ[j] for i in 1:Na, j in 1:Ne],
 ϵmatL = [ϵ[j] for i in 1:NaL, j in 1:Ne]
   
)

hh = Household()  


## Defining the necessary functions

Uc(c, μ) = c.^(-μ) 
Uc_inv(y, μ) = (y).^(-1/μ) 
n(c, ϵ, w, τy, ψ, frish, μ) = @. 1.0  #((1-τy)*w*ϵ*Uc(c, μ)/ψ)^(1/frish)  # This is from the MRS



## Finding all the roots at once
function f!(F, x)  # x= r, 0.0, 0.0, A
    F[1], F[2] = get_aggregates(hh, x[1], 0.0, 0.0, x[2])[[2, 4]]
end

r_eqmb2, A_est =  nlsolve(f!, [0.015, 2.0]).zero  
#   0.01892726141118009, 0.5554533938091648
# Not bad this is fast !!


## Getting final Policies
a_new, c_new, n_new = pol_EGM(hh, r_eqmb2, 0.0, 0.0, A_est; tol=1e-8, maxiter=100_00)

## Getting Plots
get_a_plots(20, hh, a_new)  # plot for asset policy 
#savefig("figs\\endoL_A_dis.png")
get_c_plots(20, hh, c_new)  # plot for consumption policy 
get_n_plots(20, hh, n_new)  # plot for labor policy 

get_wealth_dis_plot(100, hh, a_new)  # plot for the asset distribution at eqbm r and w





#---- METHOD 2 ---#: Graphical : We'll plot the asset demand and supply curve and look at the intersection to get ebm residual

N = get_aggregates(hh, 0.015, 0.0, 0.0, 1.0)[1] + 0.28 # N is independent of r 
r_vals = collect(LinRange(0.001, 0.0198, 20))

roi(x) =@. hh.θ*(x/N)^(hh.θ-1) - hh.δ
@time k_vals = [get_aggregates(hh, r_vals[i], 0.0, 0.0, 1.0)[end] for i in 1:length(r_vals)]

demand = @. roi(k_vals)
labels =  ["demand for capital" "supply of capital"]
plot(k_vals, [demand r_vals], label = labels)
plot!(xlabel = "capital", ylabel = "interest rate")
## Plot looks good
savefig("figs\\basic_aiya_r.png")



