# This code will run the will run Aiyagari model taking into account code optimizations

## Loading Packages
using Plots
using ForwardDiff
using Interpolations
using LinearAlgebra
using Roots
using SparseArrays
using NLsolve
using Parameters

 

## Calling our Packages
include("Tauchen.jl")
include("get_grids.jl")

## Defining the household instance 

Household = @with_kw (
           
 β = 0.98,
 μ=2.0,
 frish=2.0,
 δ = 0.075,
 #ψ = 150.0,
 θ = 0.3,

 τy = 0.4,
 ϕg = 0.2,
 ϕb = 1.0,

 Na = 100, #no of points in asset grid
 Ne = 5,  # no of points in AR(1) shock
 amin = 0,
 amax = 40,

 ϵ = exp.(Tauchen(0.0, 0.6, 0.3, Ne)[1]), 
 Pϵ = Tauchen(0.0, 0.6, 0.3, Ne)[2],
 a_grid = get_grids(amin, amax, Na, 2),
 Amat = [a_grid[i] for i in 1:Na, j in 1:Ne],
 ϵmat = [ϵ[j] for i in 1:Na, j in 1:Ne],
   
 #s_grid = vec(collect(Iterators.product(a_grid, ϵ))), # the vec is as mentioned in the notes s = (a, ϵ)
 #Ns = length(s_grid)
)

hh = Household()  


## Defining the necessary functions

Uc(c, μ) = c.^(-μ) 
Uc_inv(y, μ) = (y).^(-1/μ) 
n(c, ϵ, w, τy, ψ, frish, μ) = @. ((1-τy)*w*ϵ*Uc(c, μ)/ψ)^(1/frish)  # This is from the MRS

#c0 = repeat([0.02], 100, 5)
function get_cmin(ϵmat, τy, w, ψ, frish, μ, T, Na, Ne)
    cmin = zeros(Na, Ne)
    for i in 1:Na, j in 1:Ne
        cmin[i, j] = find_zero(c-> (1-τy)*w*ϵmat[1, 1]*n(c, ϵmat[1, 1], w, τy, ψ, frish, μ) + T -c, 1.0)
    end
    return cmin
end


function get_endo_c(β, r, τy, μ, Pϵ, c_old)  # this function has been optimized for speed
    B = β*(1+r*(1-τy))*Uc(c_old, μ)*Pϵ'  
    return Uc_inv(B, μ) 
end


get_endo_a(Amat, ϵmat, r, τy, ψ, w, frish, μ, T, ct) = @. 1/(1+(1-τy)*r)*(ct + Amat - (1-τy)*w*ϵmat*n(ct, ϵmat, w, τy, ψ, frish, μ) - T)


function get_cbind(Amat, ϵmat, r, τy, w, ψ, frish, μ, T, Na, Ne)
    cbind = zeros(Na, Ne)
    for i in 1:Na, j in 1:Ne
        cbind[i, j] = find_zero(c-> (1+(1-τy)*r)*Amat[i, j] + (1-τy)*w*ϵmat[i, j]*n(c, ϵmat[i, j], w, τy, ψ, frish, μ) + T -c, 1.0)
    end
    return cbind
end


function get_c_new(r, w, β, Pϵ, τy, ψ, frish, μ, T, c_old, cbind, Amat, ϵmat, Ne)
    
    ct = get_endo_c(β, r, τy, μ, Pϵ, c_old)
    at = get_endo_a(Amat, ϵmat, r, τy, ψ, w, frish, μ, T, ct)

    #c_new = zeros(Na, Ne)
    cnonbind = similar(ct)

    for j in 1:Ne
        cnonbind[:, j] = LinearInterpolation(at[:, j], ct[:, j], extrapolation_bc=Line()).(Amat[:, j])
    end

    
    for j = 1:Ne
        ct[:,j] = (Amat[:,j] .> at[1,j]).*cnonbind[:,j] .+ (Amat[:,j] .<= at[1,j]).*cbind[:,j]
    end

    return ct
end


function pol_EGM(hh, r, T, ψ, A; tol=1e-8, maxiter=100_00)

    @unpack β, μ, frish, δ, θ, τy, Pϵ, Amat, ϵmat, Na, Ne = hh

    w = (1-θ)*A*(((r+δ)/(A*θ))^(-θ/(1-θ)))

    cbind = get_cbind(Amat, ϵmat, r, τy, w, ψ, frish, μ, T, Na, Ne)
    #c_old = get_cmin(ϵmat, τy, w, ψ, frish, μ, T, Na, Ne)
    #c_old = @. r*Amat+w*ϵmat+T
    c_old = copy(cbind)

    iter = 1
    normdiff = 100.0

    while iter < maxiter && normdiff > tol  # this process can be parallelized --- learn this
        c_new = get_c_new(r, w, β, Pϵ, τy, ψ, frish, μ, T, c_old, cbind, Amat, ϵmat, Ne)
    
        normdiff = opnorm(c_new - c_old)
        iter = iter + 1
        c_old = copy(c_new)
    
    end
  
    a_new = @. (1+(1-τy)*r)*Amat + (1-τy)*w*ϵmat*n(c_old, ϵmat, w, τy, ψ, frish, μ) + T - c_old
    n_new = @. n(c_old, ϵmat, w, τy, ψ, frish, μ)
    
    return a_new, c_old, n_new
end

# Everything is correct till this point of time.
# EGM is fast. Convergence take less than 0.3 seconds on i3-7gen 2.4Ghz processor

# We'll define the tranistion matrix and finally the employ the root finding 

## Building the transition matrix

function Qtran(a_grid, a_policy, Pϵ, Na, Ne)
    # for now suppose that a_policy is a matrix with size conforming to a_grid
    # we'll later change this so that a_grid and a_pol can be different

    Q = spzeros(Na*Ne, Na*Ne)

    for j in 1:Ne
        sj = (j-1)*Na

        for i in 1:Na
            k = searchsortedlast(a_grid[:,j], a_policy[i, j]) # return the index of the last value in a_grid less than or equal to a'(a_i, e_j)

            if (0 < k && k <Na) # this is to adjust for first and last grid points
                k = k
            elseif k == Na
                k = Na-1
            else
                k = 1
            end

            #(0 < k && k <Na) ? k = k : (k==Na) ? k = Na-1: k = 1  

            wk = (a_policy[i, j] - a_grid[k, j])/(a_grid[k+1, j] - a_grid[k, j])
            wk = min(max(wk, 0.0), 1.0)

            for m in 1:Ne
                tm = (m-1)*Na
                Q[k+tm, i+sj] = (1-wk)*Pϵ[j, m]
                Q[k+1+tm, i+sj] = wk*Pϵ[j, m]
            end

        end

    end

    return Q

end


# Function to calculate the stationary distribution

function get_st_dis(Q, Na, Ne)
    Ns = Na*Ne

    iter = 1
    normdiff = 100
    λ_old = repeat([1/Ns], Ns, 1)

    while normdiff > 1e-16
       λ_new = Q*λ_old

       normdiff = opnorm(λ_new - λ_old)
       iter = iter + 1

       λ_old = copy(λ_new)

    end
    return λ_old    
end


## Getting the aggregates which will be calirated to meet the targets !!

function get_aggregates(hh, r, T, ψ, A)

    println("The input values are r = $r, T=$T, ψ= $ψ, A=$A")
    # As the root finder will run it'll keep printing the values that its seaching over
    # Its nice to keep track of the progress

    @unpack β, μ, frish, δ, θ, τy, Pϵ, Amat, ϵmat, Na, Ne, ϕg, ϕb = hh

    w = (1-θ)*A*(((r+δ)/(A*θ))^(-θ/(1-θ)))

    a_new, n_new = pol_EGM(hh, r, T, ψ, A)[[1, 3]] 

    Q = Qtran(Amat, a_new, Pϵ, Na, Ne)
    λ_st = get_st_dis(Q, Na, Ne)

    A_new = sum(vec(a_new).*λ_st)
    N_new = sum(vec(n_new).*vec(ϵmat).*λ_st)

    Kdd = N_new*((r+δ)/θ)^(-1/(1-θ))

    Ydd = A*(Kdd^θ)*(N_new^(1-θ))

    G = ϕg*Ydd
    B = ϕb*Ydd

    GBC_resi = τy*(w*N_new + r*A_new) - (G + r*B + T)
    Asset_mkt_resi = Kdd + B - A_new

    println("The Y is $Ydd, N is $N_new, govt buget residual is $GBC_resi, asset market residual is $Asset_mkt_resi")

    return N_new -0.28, Ydd -1.0, GBC_resi, Asset_mkt_resi
end


## Checking our functions
@time get_aggregates(hh, r, T, ψ, A)  #0.3 sec to run 2 iteration on i3-7gen 2.4Ghz


## Finding all the roots at once
function f!(F, x)  # x= r, T, ψ, A
    # Eqn 1:4 calibrates for N = 0.28, Ydd = 1.0, govt_budget_balance, asset market clearing
    F[1], F[2], F[3], F[4] = get_aggregates(hh, x[1], x[2], x[3], x[4])
end

r_eqmb2, T_est, ψ_est, A_est =  nlsolve(f!, [0.012 ; 0.05; 30; 5.0]).zero  
# 0.019281575981053848,  0.1903396724914803, 58.19947194593684, 2.1747107556105156
# Not bad this is fast !!


## Getting some Plots
a_new, c_new, n_new = pol_EGM(hh, r_eqmb2, T_est, ψ_est, A_est; tol=1e-8, maxiter=100_00)

#Asset policy
plot(hh.Amat[1:20, 1], a_new[1:20,1], label="low_shock", title="Asset Policy Aiyagari with endo L", xlabel="Current Assets", ylabel="Savings")
plot!(hh.Amat[1:20, 3], a_new[1:20,3], label="medium_shock", legend=:topleft)
plot!(hh.Amat[1:20, 5], a_new[1:20,5], label="high_shock")
savefig("figs\\endoL_A_p.png")

# Consumption Policy
plot(hh.Amat[1:20, 1], c_new[1:20,1], label="low_shock", title="Consumption Policy Aiyagari with endo L", xlabel="Current Assets", ylabel="Consumption")
plot!(hh.Amat[1:20, 3], c_new[1:20,3], label="medium_shock", legend=:bottomright)
plot!(hh.Amat[1:20, 5], c_new[1:20,5], label="high_shock")
savefig("figs\\endoL_C_p.png")


# Labor Policy
plot(hh.Amat[1:20, 1], n_new[1:20,1], label="low_shock", title="Labor Policy Aiyagari with endo L", xlabel="Current Assets", ylabel="Labor Supply")
plot!(hh.Amat[1:20, 3], n_new[1:20,3], label="medium_shock", legend=:bottomright)
plot!(hh.Amat[1:20, 5], n_new[1:20,5], label="high_shock")
savefig("figs\\endoL_L_p.png")

# Getting the asset distribution
Q = Qtran(hh.Amat, a_new, hh.Pϵ, hh.Na, hh.Ne)
λ_st = reshape(get_st_dis(Q, hh.Na, hh.Ne), hh.Na, hh.Ne)
λ_a = sum(λ_st, dims=2)  #obtaining the row sums


plot(hh.Amat[1:80, 1], λ_a[1:80], label="Density", title="Asset Disbn in Aiyagari endo L", xlabel="Asset Level")
savefig("figs\\endoL_A_dis.png")

## Since estimates should be independent of no of grid points obtain smooth graphs by adding more points

## Everyhting works well in this version of the code

