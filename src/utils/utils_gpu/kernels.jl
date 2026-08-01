""" 
    IMPORTANT: 
    - better information about how these kernels are called and executed can be found in the documentation of their respective caller functions
    - all CUDA.jl kernels must return `nothing`
"""

"""
    findall_kernel!(mask::CuDeviceArray{Bool},
                    idx::CuDeviceArray{Int},
                    valid_idx::CuDeviceArray{Int32},
                    mask_length::Int)
    
    Extracts all valid ROI indices 

    # Arguments:
    - `mask::CuArray`: The binary mask defining the region of interest stored on the GPU
    - `idx::CuArray`: The vector containing the position where each thread will write if the mask is true
    - `valid_idx::CuArray`: The vector where all valid ROI indices are stored
    - `mask_length::Int`: The length of `mask`

    # Caller functions:
    - `init_gpu` in `utils/utils_gpu/utils.jl`
"""

function findall_kernel!(mask::CuDeviceArray{Bool},
    idx::CuDeviceArray{Int},
    valid_idx::CuDeviceArray{Int32},
    mask_length::Int)

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
    assign_uniques!(img::CuDeviceArray{Int},
                    is_boundary::CuDeviceArray{Int32},
                    idx::CuDeviceArray{Int},
                    uniques::CuDeviceArray{Int})

    Extracts all unique values inside an array 

    # Arguments
    - `img::CuDeviceArray`: The input image stored on the GPU 
    - `is_boundary::CuDeviceArray`: The binary array where each element indicates whether the corrisponding position is a boundary (1) or not (0)
    - `idx::CuDeviceArray`: The array where each element indicates the position where every thread will write if the corrisponding position is a boundary 
    - `uniques::CuDeviceArray`: The array where unique values will be stored 

    # Caller functions:
    - `unique_gpu` in `utils/utils_gpu/utils.jl`
"""

function assign_uniques!(img::CuDeviceArray{Int},
    is_boundary::CuDeviceArray{Int32},
    idx::CuDeviceArray{Int},
    uniques::CuDeviceArray{Int})

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
    set_boundaries!(xx::CuDeviceArray{Int},
                    is_boundary::CuDeviceArray{Int32})

    Finds boundaries inside a sorted array. Example:
    x = [1, 1, 1, 2, 3, 3, 6, 6, 7]
    elements in position 1, 4, 5, 7, 8 are boundaries 

    # Arguments
    - `x`: Input array
    - `is_boundary`: A binary array containing boundary flags for the corresponding position

    # Caller functions:
    - `unique_gpu.jl` in `utils/utils_gpu/utils.jl`
"""

function set_boundaries!(x::CuDeviceArray{Int},
    is_boundary::CuDeviceArray{Int32})

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
    assign!(img::CuDeviceArray{Int},
            mask_indices::CuDeviceArray{Int},
            roi::CuDeviceArray{Int},
            n::Int)

    Extracts the intensity of all voxels belonging to the ROI 

    # Arguments
    - `img::CuDeviceArray`: The input image stored on the GPU
    - `mask_indices::CuDeviceArray`: The array containing all valid ROI indices
    - `roi::CuArray`: The array where the intensity of the voxels belonging to the ROI are stored
    - `n::Int`: The length of `mask_indices`

    # Caller functions:
    - `apply_mask` in `utils/utils_gpu/utils.jl`
"""

function assign!(img::CuDeviceArray{Int},
    mask_indices::CuDeviceArray{Int},
    roi::CuDeviceArray{Int},
    n::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > n
        return nothing
    end

    roi[i] = img[mask_indices[i]]

    return nothing
end


"""
    bin_nbins_kernel!(img::CuDeviceArray{Float64}, 
                    mask_indices::CuDeviceArray{Int},
                    inv_bin_width::Float64, 
                    n_bins::Int,
                    vmin::Float64, 
                    disc::CuDeviceArray{Int},
                    n_of_indices::Int)

    Each CUDA thread processes one voxel inside the ROI.


    # Arguments
    - `img::CuDeviceArray`: Input image stored on the GPU.
    - `mask_indices::CuDeviceArray`: Indices of voxels inside the ROI.
    - `inv_bin_width::Float64`: Inverse of the bin width.
    - `n_bins::Int`: Number of gray-level bins.
    - `vmin::Float64`: Minimum image intensity.
    - `disc::CuDeviceArray`: Discretized image stored on the GPU.
    - `n_of_indices::Int`: Number of ROI voxels.

    # Returns
    Returns `nothing`. The discretized image `disc` is modified directly on the GPU.
"""
function bin_nbins_kernel!(img::CuDeviceArray{Float64},
    mask_indices::CuDeviceArray{Int},
    inv_bin_width::Float64,
    n_bins::Int,
    vmin::Float64,
    disc::CuDeviceArray{Int},
    n_of_indices::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > n_of_indices
        return nothing
    end

    v = img[mask_indices[i]]
    b = CUDA.min(Int(floor((v - vmin) * inv_bin_width)) + 1, n_bins)
    disc[mask_indices[i]] = b

    return nothing
end


"""
    bin_width_kernel!(img::CuDeviceArray, 
                    mask_indices::CuDeviceArray{Int},
                    inv_bin_width::Float64, 
                    bin_offset::Int,
                    disc::CuDeviceArray{Int}, 
                    n_of_indices::Int)

    Each CUDA thread processes one voxel inside the ROI.

    # Arguments
    - `img::CuDeviceArray`: Input image stored on the GPU.
    - `mask_indices::CuDeviceArray`: Indices of voxels inside the ROI.
    - `inv_bin_width::Float64`: Inverse of the bin width.
    - `bin_offset::Int`: Offset.
    - `disc::CuDeviceArray`: Discretized image stored on the GPU.
    - `n_of_indices::Int`: Number of ROI voxels.

    # Returns
    Returns `nothing`. The discretized image `disc` is modified directly on the GPU.
"""
function bin_width_kernel!(img::CuDeviceArray{Float64},
    mask_indices::CuDeviceArray{Int},
    inv_bin_width::Float64,
    bin_offset::Int,
    disc::CuDeviceArray{Int},
    n_of_indices::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > n_of_indices
        return nothing
    end

    v = img[mask_indices[i]]
    b = Int(floor(v * inv_bin_width)) - bin_offset + 1
    disc[mask_indices[i]] = b

    return nothing
end


"""
    lut_kernel!(gray_levels::CuDeviceArray{Int}, 
                lut::CuDeviceArray{Int}, 
                min_gl::Int, 
                Ng::Int)

    CUDA kernel for constructing a gray level look up table (LUT).

    Each CUDA thread processes one gray level and assigns its corresponding
    index in the LUT

    # Arguments
    - `gray_levels`: Sorted array of unique gray levels present in the ROI
    - `lut`: Lookup table stored on the GPU.
    - `min_gl`: Minimum gray level value.
    - `Ng`: Number of gray levels.

    # Returns
    Returns `nothing`. The LUT is modified directly on the GPU.
"""
function lut_kernel!(gray_levels::CuDeviceArray{Int},
    lut::CuDeviceArray{Int},
    min_gl::Int,
    Ng::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > Ng
        return nothing
    end

    lut[Int(gray_levels[i])-min_gl+1] = i
    return nothing
end


"""
    mapped_disc_kernel!(disc::CuDeviceArray{Int}, 
                        mapped_disc::CuDeviceArray{Int},
                        mask::CuDeviceArray{Bool}, 
                        N::Int,
                        lut::CuDeviceArray{Int}, 
                        min_gl::Int)

    CUDA kernel for mapping discretized image gray levels to compact indices.

    Each CUDA thread processes one voxel.

    # Arguments
    - `disc::CuDeviceArray`: Discretized image.
    - `mapped_disc::CuDeviceArray`: Output array containing mapped indices.
    - `mask::CuDeviceArray`: ROI mask indicating valid voxels.
    - `N::Int`: Total number of voxels.
    - `lut::CuDeviceArray`: Gray level lookup table.
    - `min_gl::Int`: Minimum gray level value.

    # Returns
    Returns `nothing`. The mapped discretized image is modified directly on the GPU
"""
function mapped_disc_kernel!(disc::CuDeviceArray{Int},
    mapped_disc::CuDeviceArray{Int},
    mask::CuDeviceArray{Bool},
    N::Int,
    lut::CuDeviceArray{Int},
    min_gl::Int)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > N
        return nothing
    end

    @inbounds if mask[i]
        mapped_disc[i] = lut[disc[i]-min_gl+1]
    end

    return nothing
end

"""
    glcm_kernel!(G::CuDeviceArray{Float64}, 
                mask::CuDeviceArray{ool}, 
                mask_indices::CuDeviceArray{Int},
                mapped_disc::CuDeviceArray{Int}, 
                dirs_x::CuDeviceArray{Int}, 
                dirs_y::CuDeviceArray{Int},
                dirs_z::CuDeviceArray{Int}, 
                dirs_length::Int, 
                Nx::Int, 
                Ny::Int, 
                Nz::Int,
                num_valid::Int)

    CUDA kernel for computing the GLCM matrix.

    Each CUDA thread processes one voxel/direction pair.

    Symmetrization of the GLCM is performed on the CPU.

    # Arguments
    - `G::CuDeviceArray`: Output GLCM matrix stored on the GPU.
    - `mask::CuDeviceArray`: Binary ROI mask stored on the GPU.
    - `mask_indices::CuDeviceArray`: Indices of valid voxels inside the ROI.
    - `mapped_disc::CuDeviceArray`: Discretized image.
    - `dirs_x::CuDeviceArray`: x components of the directions.
    - `dirs_y::CuDeviceArray`: y components of the directions.
    - `dirs_z::CuDeviceArray`: z components of the directions.
    - `dirs_length::Int`: Number of directions.
    - `Nx::Int`: Image width
    - `Ny::Int`: Image height
    - `Nz::Int`: Image depth
    - `num_valid::Int`: Number of valid voxels in the ROI

    # Returns
    Returns `nothing`. The GLCM matrix `G` is modified directly on the GPU
"""
function glcm_kernel!(G::CuDeviceArray{Float64},
    mask::CuDeviceArray{Bool},
    mask_indices::CuDeviceArray{Int},
    mapped_disc::CuDeviceArray{Int},
    dirs_x::CuDeviceArray{Int},
    dirs_y::CuDeviceArray{Int},
    dirs_z::CuDeviceArray{Int},
    dirs_length::Int,
    Nx::Int,
    Ny::Int,
    Nz::Int,
    num_valid::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x # maps threads to mask
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y # maps threads to directions

    if i > num_valid || j > dirs_length
        return nothing
    end

    lin_idx = mask_indices[i]

    # here we map a 1D index into 3D or 2D coordinates
    z = 1
    if Nz > 1
        z = fld(lin_idx - 1, Nx * Ny) + 1 # depth index
    end
    r = (lin_idx - 1) % (Nx * Ny)   # index inside 2d plane of size Nx * Ny
    y = fld(r, Nx) + 1              # row index from 1 to Ny
    x = (r % Nx) + 1                # column index from 1 to Nx

    dx = dirs_x[j]
    dy = dirs_y[j]
    dz = Nz > 1 ? dirs_z[j] : 0

    nx = x + dx
    ny = y + dy
    nz = z + dz

    if nx < 1 || nx > Nx || ny < 1 || ny > Ny
        return
    end

    if Nz > 1
        if nz < 1 || nz > Nz
            return nothing
        end
    end

    if !mask[nx, ny, nz]
        return nothing
    end

    i_disc = mapped_disc[x, y, z]
    j_disc = mapped_disc[nx, ny, nz]

    # only perform one sum, symmetrization is applied on the CPU because it's faster this way -> fewer threads wait for synchronization due to race condition
    CUDA.@atomic G[i_disc, j_disc, j] += 1.0

    return nothing
end

"""
    glrlm_kernel!(img::CuArray{Int}, 
                mask::CuArray{Bool}, 
                mask_indices::CuArray{Int},
                gl_lut::CuArray{Int}, 
                P_glrlm::CuArray{Int}, 
                actual_max_run::CuArray{Int},
                Nx::Int, 
                Ny::Int, 
                Nz::Int,
                angles_x::CuArray{Int}, 
                angles_y::CuArray{Int}, 
                angles_z::CuArray{Int},
                num_angles::Int, 
                num_indices::Int,
                num_gl::Int, 
                min_gl::Int, 
                max_run_length::Int
    )

    CUDA kernel for computing the Gray Level Run Length Matrix (GLRLM).

    Each CUDA thread processes one voxel/direction pair.

    # Arguments
    - `img`: Discretized image stored on the GPU.
    - `mask`: Binary ROI mask stored on the GPU.
    - `mask_indices`: Indices of valid voxels inside the ROI.
    - `gl_lut`: Lookup table mapping gray levels to GLRLM indices.
    - `P_glrlm`: Output GLRLM matrix stored on the GPU.
    - `actual_max_run`: Single element array storing the maximum detected run length.
    - `Nx`: Image width.
    - `Ny`: Image height.
    - `Nz`: Image depth.
    - `angles_x`: x components of the run directions.
    - `angles_y`: y components of the run directions.
    - `angles_z`: z components of the run directions.
    - `num_angles`: Number of directions.
    - `num_indices`: Number of valid voxels in the ROI.
    - `num_gl`: Number of unique gray levels.
    - `min_gl`: Minimum gray level in the discretized image.
    - `max_run_length`: Maximum run length.

    # Returns
    Returns `nothing`. The GLRLM matrix `P_glrlm` and the maximum run length
    `actual_max_run` are modified directly on the GPU.
"""

function glrlm_kernel!(
    img::CuDeviceArray{Int},
    mask::CuDeviceArray{Bool},
    mask_indices::CuDeviceArray{Int},
    gl_lut::CuDeviceArray{Int},
    P_glrlm::CuDeviceArray{Float64},
    actual_max_run::CuDeviceArray{Int},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    angles_x::CuDeviceArray{Int},
    angles_y::CuDeviceArray{Int},
    angles_z::CuDeviceArray{Int},
    num_angles::Int,
    num_indices::Int,
    num_gl::Int,
    min_gl::Int,
    max_run_length::Int
)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > num_indices || j > num_angles
        return
    end

    dx = angles_x[j]
    dy = angles_y[j]
    dz = angles_z[j]

    idx = mask_indices[i]

    gl = img[idx]
    gl_idx = gl_lut[gl-min_gl+1]

    z = 1
    r = idx - 1

    if Nz > 1
        z = fld(r, Nx * Ny) + 1
        r = r % (Nx * Ny)
    end

    y = fld(r, Nx) + 1
    x = (r % Nx) + 1

    prev_x = x - dx
    prev_y = y - dy
    prev_z = z - dz

    if prev_x >= 1 && prev_x <= Nx &&
       prev_y >= 1 && prev_y <= Ny &&
       (Nz == 1 || (prev_z >= 1 && prev_z <= Nz))

        prev_idx = prev_x +
                   (prev_y - 1) * Nx +
                   (prev_z - 1) * Nx * Ny

        if mask[prev_idx] && img[prev_idx] == gl
            return
        end
    end

    run_length = 1

    next_x = x + dx
    next_y = y + dy
    next_z = z + dz

    while next_x >= 1 && next_x <= Nx &&
              next_y >= 1 && next_y <= Ny &&
              (Nz == 1 || (next_z >= 1 && next_z <= Nz))

        next_idx = next_x +
                   (next_y - 1) * Nx +
                   (next_z - 1) * Nx * Ny

        if !(mask[next_idx] && img[next_idx] == gl)
            break
        end

        run_length += 1

        next_x += dx
        next_y += dy
        next_z += dz
    end

    if run_length <= max_run_length
        bin = gl_idx +
              (run_length - 1) * num_gl +
              (j - 1) * num_gl * max_run_length

        CUDA.atomic_add!(
            pointer(P_glrlm, bin),
            Float64(1)
        )
        CUDA.atomic_max!(
            pointer(actual_max_run, 1),
            Int(run_length)
        )
    end

    return
end

function classify_mask_indices!(
    mask_indices::CuDeviceArray{Int},
    is_interior::CuDeviceArray{Int},
    is_border::CuDeviceArray{Int},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    num_indices::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > num_indices
        return nothing
    end

    idx = mask_indices[i]

    z = 1
    r = idx - 1
    if Nz > 1
        z = fld(r, Nx * Ny) + 1
        r = r % (Nx * Ny)
    end
    y = fld(r, Nx) + 1
    x = (r % Nx) + 1

    interior = 0

    if Nz <= 1
        if (1 < x < Nx) && (1 < y < Ny)
            interior = 1
        end
    else
        if (1 < x < Nx) && (1 < y < Ny) && (1 < z < Nz)
            interior = 1
        end
    end

    is_interior[i] = interior
    is_border[i] = 1 - interior

    return nothing
end

function assign_border_interior!(mask_indices::CuDeviceArray{Int},
    interior_mask::CuDeviceArray{Int},
    border_mask::CuDeviceArray{Int},
    interior_idx::CuDeviceArray{Int},
    border_idx::CuDeviceArray{Int},
    is_interior::CuDeviceArray{Int},
    is_border::CuDeviceArray{Int},
    num_indices::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > num_indices
        return nothing
    end

    if is_interior[i] == 1
        interior_mask[interior_idx[i]] = mask_indices[i]
    else
        border_mask[border_idx[i]] = mask_indices[i]
    end

    return nothing
end

@inline function decode_xyz(idx::Int, Nx::Int, Ny::Int, Nz::Int)
    z = 1
    r = idx - 1
    if Nz > 1
        z = fld(r, Nx * Ny) + 1
        r = r % (Nx * Ny)
    end
    y = fld(r, Nx) + 1
    x = (r % Nx) + 1
    return x, y, z
end

@inline function encode_xyz(x::Int, y::Int, z::Int, Nx::Int, Ny::Int)
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

function gldm_interior_dependence!(
    discretized_img::CuDeviceArray{Int},
    mask::CuDeviceArray{Bool},
    interior_mask::CuDeviceArray{Int},
    dependence_count::CuDeviceArray{Int},
    offsets_x::CuDeviceArray{Int},
    offsets_y::CuDeviceArray{Int},
    offsets_z::CuDeviceArray{Int},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    num_interior::Int,
    num_offsets::Int,
    gldm_a::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    o = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > num_interior || o > num_offsets
        return nothing
    end

    idx = interior_mask[i]
    x, y, z = decode_xyz(idx, Nx, Ny, Nz)

    nx = x + offsets_x[o]
    ny = y + offsets_y[o]
    nz = z + offsets_z[o]
    nidx = encode_xyz(nx, ny, nz, Nx, Ny)

    gl = discretized_img[idx]
    if mask[nidx] && abs(gl - discretized_img[nidx]) <= gldm_a
        CUDA.@atomic dependence_count[i] += 1
    end
    return nothing
end

function gldm_border_dependence!(
    discretized_img::CuDeviceArray{Int},
    mask::CuDeviceArray{Bool},
    border_mask::CuDeviceArray{Int},
    dependence_count::CuDeviceArray{Int},
    offsets_x::CuDeviceArray{Int},
    offsets_y::CuDeviceArray{Int},
    offsets_z::CuDeviceArray{Int},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    num_border::Int,
    num_offsets::Int,
    gldm_a::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    o = threadIdx().y + (blockIdx().y - 1) * blockDim().y

    if i > num_border || o > num_offsets
        return nothing
    end

    idx = border_mask[i]
    x, y, z = decode_xyz(idx, Nx, Ny, Nz)

    nx = x + offsets_x[o]
    ny = y + offsets_y[o]
    nz = z + offsets_z[o]

    if (1 <= nx <= Nx) && (1 <= ny <= Ny) && (1 <= nz <= Nz)
        nidx = encode_xyz(nx, ny, nz, Nx, Ny)
        gl = discretized_img[idx]
        if mask[nidx] && abs(gl - discretized_img[nidx]) <= gldm_a
            CUDA.@atomic dependence_count[i] += 1
        end
    end
    return nothing
end

function gldm_histogram_scatter!(
    discretized_img::CuDeviceArray{Int},
    idx_list::CuDeviceArray{Int},
    gl_lut::CuDeviceArray{Int},
    dependence_count::CuDeviceArray{Int},
    min_gl::Int,
    P_gldm::CuDeviceArray{Int},
    n::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > n
        return nothing
    end

    idx = idx_list[i]
    gl = discretized_img[idx]
    gl_idx = gl_lut[gl-min_gl+1]

    CUDA.@atomic P_gldm[gl_idx, dependence_count[i]] += 1

    return nothing
end