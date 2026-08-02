"""
    compute_glcm_gpu(disc::CuArray{Int}, 
                    gray_levels::CuArray{Int}, 
                    gpu_data::GPUData)::Array{Float64}

    Compute the Gray Level Co-occurrence Matrix (GLCM) on the GPU.

    # Arguments
    - `disc`: Discretized image stored on the GPU.
    - `gray_levels`: Gray levels.
    - `gpu_data`: GPU data container containing:
        - `gpu_data.img`: Original image stored on the GPU.
        - `gpu_data.mask`: ROI mask stored on the GPU.
        - `gpu_data.mask_indices`: Linear indices of valid ROI voxels.

    # Returns
    - `G`: Symmetric GLCM matrices on the CPU.

    # Notes
    - GLCM accumulation is performed on the GPU using atomic operations.
    - Symmetrization is performed on the CPU after GPU computation.
"""

function compute_glcm_gpu(disc::CuArray{Int},
    gray_levels::CuArray{Int},
    gpu_data::GPUData)::Array{Float64}
    dim = ndims(disc)
    if dim == 2
        dirs_x = CuArray([1, 0, 1, 1])
        dirs_y = CuArray([0, 1, 1, -1])
        dirs_z = CuArray([0, 0, 0, 0])
    else
        dirs_x = CuArray([1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, -1])
        dirs_y = CuArray([0, 1, 0, 1, -1, 0, 0, 1, 1, 1, 1, -1, 1])
        dirs_z = CuArray([0, 0, 1, 0, 0, 1, -1, 1, -1, 1, -1, 1, 1])
    end

    Ng = length(gray_levels)
    min_gl, max_gl = Int.(extrema(gray_levels))
    lut = CUDA.zeros(Int, max_gl - min_gl + 1)

    @cuda threads = CUDA_THREADS blocks = cld(Ng, CUDA_THREADS) lut_kernel!(gray_levels, lut, min_gl, Ng)
    mapped_disc = CUDA.zeros(Int, size(disc))
    Nx, Ny = size(mapped_disc)
    Nz = (dim == 3) ? size(mapped_disc, 3) : 1
    @cuda threads = CUDA_THREADS blocks = cld(length(disc), CUDA_THREADS) mapped_disc_kernel!(disc, mapped_disc, gpu_data.mask, length(disc), lut, min_gl)

    G_d = CUDA.zeros(Float64, Ng, Ng, length(dirs_x))

    n = length(gpu_data.mask_indices)
    blocks_x = cld(n, CUDA_BLOCK_WIDTH_2D)
    blocks_y = cld(length(dirs_x), CUDA_BLOCK_HEIGHT_2D)
    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = (blocks_x, blocks_y) glcm_kernel!(G_d, gpu_data.mask, gpu_data.mask_indices, mapped_disc, dirs_x, dirs_y, dirs_z, length(dirs_x), Nx, Ny, Nz, n)
    G_all = Array(G_d)

    for d in axes(G_all, 3)
        sym_sum = @view G_all[:, :, d]
        sym_sum .+= sym_sum'
    end
    return permutedims(G_all, (3, 1, 2))
end

"""
    compute_glrlm_gpu(
        mask::CuArray{Bool},
        mask_indices::CuArray{Int},
        discretized_img::CuArray{Int}
    )::Array{Float64}

    Computes the Gray Level Run Length Matrix (GLRLM) on the GPU.

    # Arguments
    - `mask`: ROI mask stored on the GPU.
    - `mask_indices`: Linear indices of valid ROI voxels.
    - `discretized_img`: Discretized image stored on the GPU.

    # Returns
    - `Array{Float64}` containing the GLRLM 
"""
function compute_glrlm_gpu(mask::CuArray{Bool}, mask_indices::CuArray{Int}, discretized_img::CuArray{Int})::Array{Float64}
    dim = ndims(discretized_img)

    if dim == 2
        angles_x = CuArray([1, 0, 1, 1, -1, 0, -1, -1])
        angles_y = CuArray([0, 1, 1, -1, 0, -1, -1, 1])
        angles_z = CuArray([0, 0, 0, 0, 0, 0, 0, 0])
    else
        angles_x = CuArray([1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1])
        angles_y = CuArray([0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1])
        angles_z = CuArray([0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
    end

    masked_img = apply_mask(discretized_img, mask_indices)
    gray_levels = unique_gpu(masked_img)
    num_gl = length(gray_levels)
    min_gl, max_gl = Int.(extrema(gray_levels))
    gl_lut = CUDA.zeros(Int, max_gl - min_gl + 1)

    Nx, Ny = size(discretized_img)
    Nz = (dim == 3) ? size(discretized_img, 3) : 1
    num_indices = length(mask_indices)

    @cuda threads = CUDA_THREADS blocks = cld(num_gl, CUDA_THREADS) lut_kernel!(gray_levels, gl_lut, min_gl, num_gl)

    max_run_length_possible = maximum(size(discretized_img))

    num_angles = length(angles_x)

    P_glrlm = CUDA.zeros(Float64, num_gl, max_run_length_possible, num_angles)

    actual_max_run = CUDA.ones(Int, 1)

    blocks_x = cld(num_indices, CUDA_BLOCK_WIDTH_2D)
    blocks_y = cld(num_angles, CUDA_BLOCK_HEIGHT_2D)
    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = (blocks_x, blocks_y) glrlm_kernel!(discretized_img, mask, mask_indices, gl_lut, P_glrlm, actual_max_run, Nx, Ny, Nz, angles_x, angles_y, angles_z, num_angles, num_indices, num_gl, min_gl, max_run_length_possible)
    CUDA.synchronize()

    actual_max = Array(actual_max_run)[1]
    return Array(P_glrlm[:, 1:actual_max, :])
end

function compute_gldm_gpu(
    discretized_img::CuArray{Int},
    mask::CuArray{Bool},
    mask_cpu::BitArray,
    mask_indices::CuArray{Int},
    gldm_a::Int)::Tuple{CuArray{Int},CuArray{Int}}

    masked_img = apply_mask(discretized_img, mask_indices)

    gray_levels = unique_gpu(masked_img)
    num_gl = length(gray_levels)
    min_gl, max_gl = Int.(extrema(gray_levels))
    gl_lut = CUDA.zeros(Int, max_gl - min_gl + 1)

    @cuda threads = CUDA_THREADS blocks = cld(num_gl, CUDA_THREADS) lut_kernel!(
        gray_levels, gl_lut, min_gl, num_gl)

    n_dims = ndims(discretized_img)
    sz = size(discretized_img)

    Nx, Ny = sz
    Nz = (n_dims == 3) ? sz[3] : 1

    if n_dims == 2
        offsets_x = CuArray([-1, -1, -1, 0, 0, 1, 1, 1])
        offsets_y = CuArray([-1, 0, 1, -1, 1, -1, 0, 1])
        offsets_z = CuArray([0, 0, 0, 0, 0, 0, 0, 0])
    else
        offsets_x = CuArray([-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        offsets_y = CuArray([-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1])
        offsets_z = CuArray([-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1])
    end
    num_offsets = length(offsets_x)

    num_indices = length(mask_indices)
    is_interior = CUDA.zeros(Int, num_indices)
    is_border = CUDA.ones(Int, num_indices)

    @cuda threads = CUDA_THREADS blocks = cld(num_indices, CUDA_THREADS) classify_mask_indices!(mask_indices, is_interior, is_border, Nx, Ny, Nz, num_indices)

    interior_mask = CUDA.zeros(Int, sum(is_interior))
    border_mask = CUDA.zeros(Int, sum(is_border))

    interior_idx = cumsum(is_interior)
    border_idx = cumsum(is_border)

    @cuda threads = CUDA_THREADS blocks = cld(num_indices, CUDA_THREADS) assign_border_interior!(mask_indices, interior_mask, border_mask, interior_idx, border_idx, is_interior, is_border, num_indices)

    n_int = length(interior_mask)
    n_bord = length(border_mask)

    max_dependence = 3^n_dims
    P_gldm = CUDA.zeros(Int, num_gl, max_dependence)

    dep_interior = CUDA.ones(Int, n_int)

    bx = cld(n_int, CUDA_BLOCK_WIDTH_2D)
    by = cld(num_offsets, CUDA_BLOCK_HEIGHT_2D)
    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = (bx, by) gldm_interior_dependence!(discretized_img, mask, interior_mask, dep_interior, offsets_x, offsets_y, offsets_z, Nx, Ny, Nz, n_int, num_offsets, gldm_a)

    @cuda threads = CUDA_THREADS blocks = cld(n_int, CUDA_THREADS) gldm_histogram_scatter!(discretized_img, interior_mask, gl_lut, dep_interior, min_gl, P_gldm, n_int)

    dep_border = CUDA.ones(Int, n_bord)

    bx = cld(n_bord, CUDA_BLOCK_WIDTH_2D)
    by = cld(num_offsets, CUDA_BLOCK_HEIGHT_2D)
    @cuda threads = (CUDA_BLOCK_WIDTH_2D, CUDA_BLOCK_HEIGHT_2D) blocks = (bx, by) gldm_border_dependence!(discretized_img, mask, border_mask, dep_border, offsets_x, offsets_y, offsets_z, Nx, Ny, Nz, n_bord, num_offsets, gldm_a)

    @cuda threads = CUDA_THREADS blocks = cld(n_bord, CUDA_THREADS) gldm_histogram_scatter!(discretized_img, border_mask, gl_lut, dep_border, min_gl, P_gldm, n_bord)

    col_has_data = vec(any(!iszero, P_gldm; dims=1))
    col_has_data_cpu = Array(col_has_data)
    last_col = findlast(col_has_data_cpu)
    last_col = last_col === nothing ? 0 : last_col
    P_gldm = P_gldm[:, 1:last_col]

    return P_gldm, gray_levels
end

function marching_cubes_surface_gpu(mask::CuArray{Bool,3},
    spacing::CuArray{Float64},
    isolevel::Float64=0.5)::Vector{Triangle3D}

    include("src/utils/utils_gpu/shape_3D_features_lookup_tables_gpu.jl")

    mask_length = length(mask)
    mask_size = size(mask)
    (Nx, Ny, Nz) = (mask_size[1] - 1, mask_size[2] - 1, mask_size[3] - 1)

    cube_indices = CUDA.zeros(Int, Nx, Ny, Nz)
    # how many triangles every voxel generates
    triangle_count = CUDA.zeros(Int, Nx, Ny, Nz)

    blocks_x = cld(Nx, CUDA_BLOCK_WIDTH_3D)
    blocks_y = cld(Ny, CUDA_BLOCK_HEIGHT_3D)
    blocks_z = cld(Nz, CUDA_BLOCK_DEPTH_3D)
    @cuda threads = (CUDA_BLOCK_WIDTH_3D, CUDA_BLOCK_HEIGHT_3D, CUDA_BLOCK_DEPTH_3D) blocks = (blocks_x, blocks_y, blocks_z) calculate_cubeindex!(mask, cube_indices, Nx, Ny, Nz, mask_length, isolevel)

    @cuda threads = (CUDA_BLOCK_WIDTH_3D, CUDA_BLOCK_HEIGHT_3D, CUDA_BLOCK_DEPTH_3D) blocks = (blocks_x, blocks_y, blocks_z) count_triangles!(cube_indices, triangle_count, casesClassic_gpu, Nx, Ny, Nz)

    counts = vec(triangle_count)
    triangles_idx = cumsum(counts) .- counts
    num_of_triangles = sum(triangle_count)
    triangles = CuArray{Triangle3D}(undef, num_of_triangles)

    @cuda threads = (CUDA_BLOCK_WIDTH_3D, CUDA_BLOCK_HEIGHT_3D, CUDA_BLOCK_DEPTH_3D) blocks = (blocks_x, blocks_y, blocks_z) generate_triangles!(mask, triangles, triangle_count, triangles_idx, cube_indices, spacing, casesClassic_gpu, Nx, Ny, Nz, num_of_triangles, isolevel)

    return Array(triangles)

end