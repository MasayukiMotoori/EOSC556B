using Downloads, LinearAlgebra, FFTW, Random
using CSV, DataFrames, DataFramesMeta
using Statistics, StatsPlots
using ForwardDiff
using Printf
using Plots

struct SourceLocalization
    c::Matrix{Float64} # 2x`m` matrix with the beacon positions
    d::Vector{Float64} # vector of distances (noisy)
    x::Vector{Float64} # true position (you're not meant to know this!)
end


function r(sl,x)
    rx = zeros(size(sl.c,2))
    for i in 1:size(sl.c,2)
        rx[i] = dot(x - sl.c[:,i], x - sl.c[:,i])  - sl.d[i]^2
    end
    return rx
end

function J(sl,x)
    Jx = zeros(size(sl.c)[2], length(x))
    for i in 1:size(sl.c)[2]
        Jx[i, :] .= 2 * x - 2 * sl.c[:, i]
    end
    return Jx
end

function Hr(sl,x,rx)
    Hrx = zeros(length(x), length(x))
    Hrx[1,1] = 2 * dot(ones(size(sl.c)[2]),rx)
    Hrx[2,2] = 2 * dot(ones(size(sl.c)[2]),rx)
    return Hrx
end

function gauss_newton(sl,x0, maxits=100, ϵ=1e-6, μ=1e-4)
    x = x0
    k = 0
    itn_err = Float64[]
    x1tr = Float64[]
    x2tr = Float64[]
    rx = r(sl, x)
    Jx = J(sl, x)    
    push!(x1tr,x0[1])
    push!(x2tr,x0[2])
#    push!(itn_err, norm(Jx' * rx)) # Initialize with initial error
    push!(itn_err,0.5*dot(rx, rx))   # Initialize with initial error
    for i in 1:maxits
        rx = r(sl,x)
        Jx = J(sl,x)
        Hrx = Hr(sl,x,rx)
        dx = -(Jx'*Jx + Hrx)\(Jx'*rx)
        α = 1.0
        β = 0.5
        directional_derivative = (Jx'*rx)'*dx
        f_old = 0.5*dot(rx,rx)
        f_new = 0.5*dot(r(sl,x + α*dx),r(sl,x + α*dx))
        while f_new >= f_old + α *μ* directional_derivative
            α *= β
            f_new = 0.5*dot(r(sl,x + α*dx),r(sl,x + α*dx))
            if norm(α) < ϵ
                break
            end    
        end
        if norm(dx) < ϵ
            break
        end
        x += α*dx 
        k += 1
        push!(itn_err,0.5*dot(rx, rx))   # Initialize with initial error
#        push!(itn_err,norm(dx)) 
        push!(x1tr,x[1])
        push!(x2tr,x[2])
    end
    xtr= zeros(2,k+1)
    xtr[1,:] = vec(x1tr)
    xtr[2,:] = vec(x2tr ) 
    return x, k, itn_err, xtr
end

function gradient_descent(sl, x0, maxits=100, ϵ=1e-6, μ=1e-4)
    x = x0
    k = 0
    itn_err = Float64[]
    x1tr = Float64[]
    x2tr = Float64[]
    rx = r(sl, x)
    Jx = J(sl, x)    
    push!(x1tr,x0[1])
    push!(x2tr,x0[2])
#    push!(itn_err,norm(Jx' * rx))   # Initialize with initial error
    push!(itn_err,0.5*dot(rx, rx))   # Initialize with initial error
    for i in 1:maxits
        rx = r(sl, x)
        Jx = J(sl, x)
        α = 1
        β = 0.5
        dx = -Jx' * rx
        directional_derivative = (Jx' * rx)' * dx
        f_old = 0.5 * dot(rx, rx)
        f_new = 0.5 * dot(r(sl, x + α * dx), r(sl, x + α * dx))
        while f_new >= f_old + α * μ * directional_derivative
            α *= β
            f_new = 0.5 * dot(r(sl, x + α * dx), r(sl, x + α * dx))
            if norm(α) < ϵ
                break
            end    
        end
        if norm(dx) < ϵ
            break
        end
        x += α * dx 
        k += 1
#        push!(itn_err, norm(dx)) 
        push!(itn_err,0.5*dot(rx, rx))   # Initialize with initial error
        push!(x1tr,x[1])
        push!(x2tr,x[2])
    end
    xtr= zeros(2,k+1)
    xtr[1,:] = vec(x1tr)
    xtr[2,:] = vec(x2tr )   
    return x, k, itn_err, xtr
end

function generate_data(m, η=0.1)
    c = 2.0.*randn(2, m)  # beacon positions
    x = randn(2)     # true (unkown) position
    d = [norm(x - ci) + η*randn() for ci in eachcol(c)]
    return SourceLocalization(c , d, x)
end

function localize_source(sl,optimization,x0)
    xest, k, itn_err, xtr = optimization(sl, x0) 
    return xest, k, itn_err, xtr
end

function plotmap(sl)
    #    scatter!(xguess[1,:], xguess[2,:], label="guess", c="orange", ms=5)
        scatter!(sl.c[1,:], sl.c[2,:], color=:green,label="beacons", shape=:square, ms=3)
        for i in 1:size(sl.c, 2)
            θ = LinRange(0, 2π, 100)  # Create points along the circumference
            cx = sl.c[1, i] .+ sl.d[i] * cos.(θ)  # x-coordinates of the circle points
            cy = sl.c[2, i] .+ sl.d[i] * sin.(θ)  # y-coordinates of the circle points
            plot!(cx, cy, label="Dist from b_$i",color=:black)
        end
        scatter!(sl.x[1,:], sl.x[2,:], color=:red,label="true position", shape=:cross, ms=10)
    end
    
    function get_rgrid(sl, xar, nx1=200,nx2=200, mirgin=0.1)
        x1range = [minimum(xar[1,:]), maximum(xar[1,:])]
        x1range += mirgin.*[-1,1]
        x2range = [minimum(xar[2,:]), maximum(xar[2,:])]
        x2range += mirgin.*[-1,1]
        x1grid, x2grid, rgrid = zeros(nx1+1), zeros(nx2+1), zeros(nx1+1,nx2+1)
    
        for ix1 in 1:nx1+1
            x1grid[ix1] = (nx1 - ix1-1)/nx1*x1range[1] + (ix1-1)/nx1*x1range[2]
        end
    
        for ix2 in 1:nx2+1
            x2grid[ix2] = (nx2 - ix2 -1)/nx2*x2range[1] + (ix2-1)/nx2*x2range[2]
        end
    
        for ix1 in 1:nx1+1
            for ix2 in 1:nx2+1
                rx = r(sl, [x1grid[ix1],x2grid[ix2]])
                Jx = J(sl, [x1grid[ix1],x2grid[ix2]])
    #            rgrid[ix1,ix2] = norm(Jx' * rx)
                rgrid[ix1,ix2] = 0.5*dot(rx, rx)
            end
        end
        return x1grid, x2grid, rgrid 
    end
 
m = 4
η= 0
sl = generate_data(m,η) # Assuming generate_data returns a tuple with appropriate types
x0 = 2*randn(2)
xest1, k1, itn_err1, xtr1 = localize_source(sl, gradient_descent,x0)
xest2, k2, itn_err2, xtr2 = localize_source(sl, gauss_newton,x0)

xar = hcat( sl.c, sl.x)
nx1,nx2, mirgin = 200,200,2
x1grid, x2grid, rgrid = get_rgrid(sl,xar,nx1,nx2,mirgin)
num_points = 20
tv = range(log10(minimum(rgrid)), stop=log10(maximum(rgrid)), length=num_points)
tl = ["$(@sprintf("%.2e", 10^i))" for i in tv]
contourf(x1grid, x2grid, log10.(rgrid), colorbar_ticks=(tv,tl), color=:blues, lw=1)  # Change color to Plasma
plotmap(sl)
plot!(leg=:outertopleft, frame=:box, title="Error value Distribution", xlabel="x1",ylabel="x2",aspect_ratio=:equal, size=(800, 600), dpi=300)
#savefig("Hw4_extra_rgrid.png")