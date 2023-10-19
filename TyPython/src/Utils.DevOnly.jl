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
