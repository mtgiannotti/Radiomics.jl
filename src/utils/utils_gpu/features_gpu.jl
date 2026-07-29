"""
    compute_glcm_gpu(disc::CuArray, gray_levels::CuArray, gpu_data::GPUData)

    Compute the Gray Level Co-occurrence Matrix (GLCM) on the GPU.

    # Arguments
    - `disc::CuArray`: Discretized image stored on the GPU.
    - `gray_levels::CuArray`: Gray levels.
    - `gpu_data::GPUData`: GPU data container containing:
        - `gpu_data.img`: Original image stored on the GPU.
        - `gpu_data.mask`: ROI mask stored on the GPU.
        - `gpu_data.mask_indices`: Linear indices of valid ROI voxels.

    # Returns
    - `G::Array`: Symmetric GLCM matrices on the CPU.

    # Notes
    - GLCM accumulation is performed on the GPU using atomic operations.
    - Symmetrization is performed on the CPU after GPU computation.
"""

function compute_glcm_gpu(disc::CuArray,
    gray_levels::CuArray,
    gpu_data::GPUData)
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

function compute_glrlm_gpu(mask, mask_indices, discretized_img)
    dim = ndims(discretized_img)

    if dim == 2
        angles_x = [1, 0, 1, 1, -1, 0, -1, -1]
        angles_y = [0, 1, 1, -1, 0, -1, -1, 1]
        angles_z = [0, 0, 0, 0, 0, 0, 0, 0]
    else
        angles_x = [1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1]
        angles_y = [0, 0, 0, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1]
        angles_z = [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1]
    end

    masked_img = apply_mask(discretized_img, mask_indices)
    gray_levels = unique_gpu(masked_img)
    num_gl = length(gray_levels)

    Nx, Ny = size(discretized_img)
    Nz = (dim == 3) ? size(discretized_img, 3) : 1
    img_length = length(discretized_img)

    gl_map = GPUDict(CUDA.zeros(Int, num_gl), CUDA.zeros(Int, num_gl))
    @cuda threads = CUDA_THREADS blocks = cld(num_gl, CUDA_THREADS) assign_gl_map_values!(gl_map.values, gray_levels, num_gl)

    max_run_length_possible = maximum(size(discretized_img))

    num_angles = length(angles_x)
    P_glrlm = CUDA.zeros(Float64, num_gl, max_run_length_possible, num_angles)
    return zeros(1, 1, 1) #placeholder
end