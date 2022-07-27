module PyC_KMeans
using TyPython
using TyPython.CPython
using MKL
import ParallelKMeans
CPython.init()

@export_py function kmeans_impl(x::StridedMatrix{Float64}, n_clusters::Int)::Tuple{StridedVector{Int}, StridedMatrix{Float64}}
    model = ParallelKMeans.kmeans(ParallelKMeans.Hamerly(), x, n_clusters)
    (model.assignments, model.centers)
end

function init()
    @export_pymodule _fast_kmeans begin
        kmeans = Pyfunc(kmeans_impl)
    end
end

function __init__()
    init()
end

end