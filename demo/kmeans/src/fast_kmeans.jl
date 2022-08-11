module fast_kmeans

using TyPython
using TyPython.CPython
using MKL
import ParallelKMeans

@export_py function _fast_kmeans(x::StridedArray, n::Int)::Tuple{StridedArray, StridedArray}
    r = ParallelKMeans.kmeans(ParallelKMeans.Hamerly(), x, n)
    return (r.assignments, r.centers)
end

function init()
    @export_pymodule _fast_kmeans begin
        _fast_kmeans = Pyfunc(_fast_kmeans)
    end
end

precompile(init, ())

end