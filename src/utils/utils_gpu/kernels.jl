""" 
    IMPORTANT: 
    - better information about how these kernels are called and executed can be found in the documentation of their respective caller functions
    - all CUDA.jl kernels must return `nothing`
"""

"""
    findall_kernel!(mask::CuArray,
                    idx::CuArray,
                    valid_idx::CuArray,
                    mask_length::CuArray)
    
    Extracts all valid ROI indices 

    # Arguments:
    - `mask::CuArray`: The binary mask defining the region of interest stored on the GPU
    - `idx::CuArray`: The vector containing the position where each thread will write if the mask is true
    - `valid_idx::CuArray`: The vector where all valid ROI indices are stored
    - `mask_length::CuArray`: The length of `mask`

    # Caller functions:
    - `init_gpu` in `utils/utils_gpu/utils.jl`
"""

function findall_kernel!(mask, idx, valid_idx, mask_length)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > mask_length
        return nothing
    end

    if mask[i]
        valid_idx[idx[i]] = i
    end

    return nothing
end

"""
    assign_uniques!(img::CuArray,
                    is_boundary::CuArray,
                    idx::CuArray,
                    uniques::CuArray)

    Extracts all unique values inside an array 

    # Arguments
    - `img::CuArray`: The input image stored on the GPU 
    - `is_boundary::CuArray`: The binary array where each element indicates whether the corrisponding position is a boundary (1) or not (0)
    - `idx`: The array where each element indicates the position where every thread will write if the corrisponding position is a boundary 
    - `uniques`: The array where unique values will be stored 

    # Caller functions:
    - `unique_gpu` in `utils/utils_gpu/utils.jl`
"""

function assign_uniques!(img, is_boundary, idx, uniques)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > length(is_boundary)
        return nothing
    end

    if is_boundary[i] != 0
        uniques[idx[i]] = img[i]
    end
    return nothing
end

"""
    set_boundaries!(x::CuArray,
                    is_boundary::CuArray)

    Finds boundaries inside a sorted array. Example:
    x = [1, 1, 1, 2, 3, 3, 6, 6, 7]
    elements in position 1, 4, 5, 7, 8 are boundaries 

    # Arguments
    - `x`: Input array
    - `is_boundary`: A binary array containing boundary flags for the corresponding position

    # Caller functions:
    - `unique_gpu.jl` in `utils/utils_gpu/utils.jl`
"""

function set_boundaries!(x, is_boundary)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > length(x)
        return nothing
    end

    if i == 1
        is_boundary[i] = 1
    else
        is_boundary[i] = Int(x[i] != x[i-1])
    end

    return nothing
end

"""
    assign!(img::CuArray, 
            mask_indices::CuArray,
            roi::CuArray,
            n::CuArray)

    Extracts the intensity of all voxels belonging to the ROI 

    # Arguments
    - `img::CuArray`: The input image stored on the GPU
    - `mask_indices::CuArray`: The array containing all valid ROI indices
    - `roi::CuArray`: The array where the intensity of the voxels belonging to the ROI are stored
    - `n::CuArray`: The length of `mask_indices`

    # Caller functions:
    - `apply_mask` in `utils/utils_gpu/utils.jl`
"""

function assign!(img, mask_indices, roi, n)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > n
        return nothing
    end

    roi[i] = img[mask_indices[i]]

    return nothing
end


"""
    Kernels called in 'discretize_image_gpu':
    - bin_nbins_kernel!
    - bin_width_kernel!
"""

function bin_nbins_kernel!(img_f32, mask_indices, inv_bin_width, n_bins, vmin, disc, n_of_indices)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > n_of_indices
        return nothing
    end

    v = img_f32[mask_indices[i]]
    b = CUDA.min(Int(floor((v - vmin) * inv_bin_width)) + 1, n_bins)
    disc[mask_indices[i]] = b

    return nothing
end

function bin_width_kernel!(img_f32, mask_indices, inv_bin_width, bin_offset, disc, n_of_indices)
    i = threadIdx().x + (blockIdx().x - 1) + blockDim().x
    if i > n_of_indices
        return nothing
    end

    v = img_f32[mask_indices[i]]
    b = Int(floor(v * inv_bin_width)) - bin_offset + 1
    disc[mask_indices[i]] = b

    return nothing
end

"""
    GLCM kernels:
    - lut_kernel!
    - mapped_disc_kernel!
    - glcm_kernel!
"""

function lut_kernel!(gray_levels, lut, min_gl, Ng)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > Ng
        return nothing
    end

    lut[Int(gray_levels[i])-min_gl+1] = i
    return nothing
end

function mapped_disc_kernel!(disc, mapped_disc, mask, N, lut, min_gl)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > N
        return nothing
    end

    @inbounds if mask[i]
        mapped_disc[i] = lut[disc[i]-min_gl+1]
    end

    return nothing
end

function glcm_kernel!(G, mask, mapped_disc, dirs_x, dirs_y, dirs_z, dirs_length, Nx, Ny, Nz)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x # maps threads to mask
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y # maps threads to directions

    if i > Nx * Ny * Nz || j > dirs_length
        return nothing
    end

    # here we map a 1D index into 3D coordinates
    z = fld(i - 1, Nx * Ny) + 1     # depth index
    r = (i - 1) % (Nx * Ny)         # index inside 2d plane of size Nx * Ny
    y = fld(r, Nx) + 1              # row index from 1 to Ny
    x = (r % Nx) + 1                # column index from 1 to Nx

    if !mask[x, y, z]
        return nothing
    end

    dx = dirs_x[j]
    dy = dirs_y[j]
    dz = dirs_z[j]

    nx = x + dx
    ny = y + dy
    nz = z + dz

    if nx < 1 || nx > Nx || ny < 1 || ny > Ny || nz < 1 || nz > Nz
        return
    end

    if !mask[nx, ny, nz]
        return nothing
    end

    i_disc = mapped_disc[x, y, z]
    j_disc = mapped_disc[nx, ny, nz]

    # only perform one sum, symmetrization is applied on the CPU because it's faster this way -> fewer threads wait for synchronization due to race condition
    CUDA.@atomic G[i_disc, j_disc, j] += 1.0f0

    return nothing
end