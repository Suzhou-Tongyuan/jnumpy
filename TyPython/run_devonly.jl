# remove *.compiled.jl in src/
import TOML
ProjName = TOML.parsefile("Project.toml")["name"]

julia_compiler = "julia"
directory = "src"

for (root, dirs, files) in walkdir(directory)
    for file in files
        if endswith(file, ".log")
            rm(joinpath(root, file))
        end
        if endswith(file, ".compiled.jl")
            rm(joinpath(root, file))
        end
    end
end

run(
    Cmd(
        `$julia_compiler --compile=min -O0 --project=. -e "using $ProjName"`;
        env = Dict("JULIA_DEVONLY_COMPILE" => "1")
    )
)
