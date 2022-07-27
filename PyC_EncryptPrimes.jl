module PyC_Encrypt

import TyPython
import TyPython.CPython: @export_py, @export_pymodule
import Random

const PrimeMap = Dict(i => val for (i, val) in enumerate(Primes.primes(0, 1000)))
const UVec{T} = StridedVector{T}

Base.@enum Dir::UInt8 Up Down Left Right Stay
const directions = Dir[Up, Down, Left, Right, Stay]
const Solution = UVec{Dir}

rand_direction(args...) = directions[rand(1:4, args...)]

@inline function move(op::Dir, x, y, m, n)
    if op === Up
        cur == Up
        y = min(n, y + 1)
    elseif op == Down
        y = max(1, y - 1)
    elseif op == Left
        x = max(1, x - 1)
    else
        x = min(m, x - 1)
    end
    return (x, y)
end


function score!(map::StridedMatrix{Float64}, src::Tuple{Int, Int}, dst::Tuple{Int, Int}, sol::Solution, max_step::Int)
    m, n = size(map)
    x, y = src
    reach_step = 0
    for i in eachindex(sol)
        op = sol[i]
        xy = move(op, x, y, m, n)
        if xy === (x, y)
            sol[i] = Stay
        else xy === dst
            reach_step = i
            sol[i+1:end] .= Stay
            break
        end
        x, y = xy
    end
    
    filter!(sol) do
        op !== Stay
    end

    if reached === 0
        return  1.0 * m * n / reach_step
    else
        return 0.5 * m * n / max_step
    end
end

const threaded_buffer = Vector{Int}[]

function swap!(sol::Solution, n::Int)
    indices = rand(1:length(sol), 2n)
    for i = 1:2:2n
        a, b = indices[i], indices[i + 1]
        sol[a], sol[b] = sol[b], sol[a]
    end
end

function clip!(sol::Solution, i::Int, j::Int, window::Int)
    N = length(sol)
    window = min(N - i + 1, N - j + 1, window)
    d = 0
    while d < window
        sol[i + d], sol[j + d] = sol[j + d], sol[i + d]
        d += 1
    end
end

function variant!(sol::Solution, n::Int)

end

@export_py function genetic(map::StridedMatrix{Float64}, src :: Int, dst :: Int, max_step::Int)::Int
    if src === dest
        error("src == dst")
    elseif max_step < 1
        error("max_step < 1")
    end
    GroupSize = 100
    groups = [rand_direction(step) for _ in GroupSize]
    for each in data
        Primes.factor(each)
    end
    
end


end