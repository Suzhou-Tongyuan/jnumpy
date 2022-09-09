module Utils
import IOCapture
import MacroTools

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

macro suppress_error(ex)
    MacroTools.@q begin
        old_stderr = stderr
        rd, wr = redirect_stderr()
        try
            $__source__
            $(esc(ex))
        finally
            redirect_stderr(old_stderr)
            close(wr)
        end
    end
end

end