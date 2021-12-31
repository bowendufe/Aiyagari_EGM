# module to get grids

function get_grids(amin, amax, np, power)

    grid = LinRange(amin^(1/power), amax^(1/power), np)
    return grid.^power
end