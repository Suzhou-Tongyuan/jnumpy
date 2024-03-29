module Utils
import TyPython: DevOnly
import IOCapture
DevOnly.@devonly import MacroTools

function capture_out(f)
    capture = IOCapture.capture() do
        f()
    end
    capture.output
end

@generated function unroll_do!(f, ::Val{N}, args...) where N
    block = Expr(:block)
    for i = 1:N
        push!(block.args, :(f($i, args...)))
    end
    block
end

DevOnly.@staticinclude("Utils.DevOnly.jl")

end
