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
    assign_uniques!(img::CuDeviceArray{T},
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

function assign_uniques!(img::CuDeviceArray{T},
    is_boundary::CuDeviceArray{Int32},
    idx::CuDeviceArray{Int},
    uniques::CuDeviceArray{T}) where T

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
    set_boundaries!(xx::CuDeviceArray{T},
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

function set_boundaries!(x::CuDeviceArray{T},
    is_boundary::CuDeviceArray{Int32}) where T

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

"""
    classify_mask_indices!(mask_indices::CuDeviceArray{Int},
                          is_interior::CuDeviceArray{Int},
                          is_border::CuDeviceArray{Int},
                          Nx::Int,
                          Ny::Int,
                          Nz::Int,
                          num_indices::Int)

    Classifies ROI voxels into interior and border voxels.

    Each thread processes one voxel index from `mask_indices`

    # Arguments
    - `mask_indices`: Linear indices of voxels belonging to the ROI.
    - `is_interior`: Binary output array indicating interior voxels.
    - `is_border`: Binary output array indicating border voxels.
    - `Nx`: Image width.
    - `Ny`: Image height.
    - `Nz`: Image depth.
    - `num_indices`: Number of ROI voxels.

    # Returns
    Returns `nothing`. The classification arrays are modified directly on the GPU.
"""
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

"""
    assign_border_interior!(mask_indices::CuDeviceArray{Int},
                            interior_mask::CuDeviceArray{Int},
                            border_mask::CuDeviceArray{Int},
                            interior_idx::CuDeviceArray{Int},
                            border_idx::CuDeviceArray{Int},
                            is_interior::CuDeviceArray{Int},
                            is_border::CuDeviceArray{Int},
                            num_indices::Int)

    Separates ROI voxel indices into interior and border lists.

    Each thread writes a voxel index into either the interior or border output
    array

    # Arguments
    - `mask_indices`: ROI voxel indices.
    - `interior_mask`: Output array containing interior voxel indices.
    - `border_mask`: Output array containing border voxel indices.
    - `interior_idx`: Write positions for interior voxels.
    - `border_idx`: Write positions for border voxels.
    - `is_interior`: Interior classification flags.
    - `is_border`: Border classification flags.
    - `num_indices`: Number of ROI voxels.

    # Returns
    Returns `nothing`. Output arrays are modified directly on the GPU.
"""
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

"""
    decode_xyz(idx::Int, Nx::Int, Ny::Int, Nz::Int)

    Converts a linear voxel index into 3D coordinates.

    # Arguments
    - `idx`: Linear voxel index.
    - `Nx`: Image width.
    - `Ny`: Image height.
    - `Nz`: Image depth.

    # Returns
    Returns `(x, y, z)` coordinates corresponding to the voxel position.
"""
@inline function decode_xyz(idx::Int, Nx::Int, Ny::Int, Nz::Int)::Tuple{Int,Int,Int}
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

"""
    encode_xyz(x::Int, y::Int, z::Int, Nx::Int, Ny::Int)

    Converts 3D voxel coordinates into a linear index.

    # Arguments
    - `x`: X coordinate.
    - `y`: Y coordinate.
    - `z`: Z coordinate.
    - `Nx`: Image width.
    - `Ny`: Image height.

    # Returns
    Returns the linear index corresponding to `(x,y,z)`.
"""
@inline function encode_xyz(x::Int, y::Int, z::Int, Nx::Int, Ny::Int)::Int
    return x + (y - 1) * Nx + (z - 1) * Nx * Ny
end

"""
    gldm_interior_dependence!(discretized_img::CuDeviceArray{Int},
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

    Computes gray level dependence counts for interior ROI voxels.

    Each thread evaluates one voxel/offset pair

    # Arguments
    - `discretized_img`: Discretized image stored on the GPU.
    - `mask:`: Binary ROI mask.
    - `interior_mask`: Interior voxel indices.
    - `dependence_count`: Output dependence counts.
    - `offsets_x`, `offsets_y`, `offsets_z`: offsets.
    - `Nx`, `Ny`, `Nz`: Image dimensions.
    - `num_interior`: Number of interior voxels.
    - `num_offsets`: Number of neighborhood offsets.
    - `gldm_a:`: Maximum allowed gray-level difference.

    # Returns
    Returns `nothing`. Dependence counts are updated directly on the GPU.
"""
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

"""
    gldm_border_dependence!(discretized_img::CuDeviceArray{Int},
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

    Computes gray level dependence counts for border ROI voxels.

    This kernel is equivalent to `gldm_interior_dependence!` 

    # Arguments
    - `discretized_img`: Discretized image stored on the GPU.
    - `mask:`: Binary ROI mask.
    - `border_mask`: Border voxel indices.
    - `dependence_count`: Output dependence counts.
    - `offsets_x`, `offsets_y`, `offsets_z`: offsets.
    - `Nx`, `Ny`, `Nz`: Image dimensions.
    - `num_border`: Number of border voxels.
    - `num_offsets`: Number of neighborhood offsets.
    - `gldm_a:`: Maximum allowed gray-level difference.

    # Returns
    Returns `nothing`. Dependence counts are modified directly on the GPU.
"""
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

"""
    gldm_histogram_scatter!(discretized_img::CuDeviceArray{Int},
                            idx_list::CuDeviceArray{Int},
                            gl_lut::CuDeviceArray{Int},
                            dependence_count::CuDeviceArray{Int},
                            min_gl::Int,
                            P_gldm::CuDeviceArray{Int},
                            n::Int)

    Builds the GLDM histogram from voxel dependence counts.

    Each thread maps a voxel gray level and its dependence count into the
    corresponding GLDM histogram bin

    # Arguments
    - `discretized_img`: Discretized image.
    - `idx_list`: List of voxel indices.
    - `gl_lut`: Gray-level lookup table.
    - `dependence_count`: Computed dependence values.
    - `min_gl`: Minimum gray level.
    - `P_gldm`: Output GLDM matrix.
    - `n`: Number of voxels.

    # Returns
    Returns `nothing`. The GLDM matrix is updated directly on the GPU.
"""
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

"""
    calculate_cubeindex!(mask::CuDeviceArray{Bool},
                         cubeindex::CuDeviceArray{Int},
                         Nx::Int,
                         Ny::Int,
                         Nz::Int,
                         mask_length::Int,
                         isolevel::Float64)

    Computes Marching Cubes cube indices for a binary volume.

    Each thread evaluates one voxel

    # Arguments
    - `mask`: Binary volume mask.
    - `cubeindex`: Output cube configuration indices.
    - `Nx`, `Ny`, `Nz`: Volume dimensions.
    - `mask_length`: Total number of voxels.
    - `isolevel`: Threshold used for classification.

    # Returns
    Returns `nothing`. Cube indices are stored directly on the GPU.
"""
function calculate_cubeindex!(mask::CuDeviceArray{Bool},
    cubeindex::CuDeviceArray{Int},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    mask_length::Int,
    isolevel::Float64)
    x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    z = threadIdx().z + (blockIdx().z - 1) * blockDim().z

    if x > Nx || y > Ny || z > Nz
        return nothing
    end

    v0 = Float64(mask[x, y, z])
    v1 = Float64(mask[x+1, y, z])
    v2 = Float64(mask[x+1, y+1, z])
    v3 = Float64(mask[x, y+1, z])
    v4 = Float64(mask[x, y, z+1])
    v5 = Float64(mask[x+1, y, z+1])
    v6 = Float64(mask[x+1, y+1, z+1])
    v7 = Float64(mask[x, y+1, z+1])

    cubeindex[x, y, z] = 0
    if v0 > isolevel
        cubeindex[x, y, z] |= 1
    end
    if v1 > isolevel
        cubeindex[x, y, z] |= 2
    end
    if v2 > isolevel
        cubeindex[x, y, z] |= 4
    end
    if v3 > isolevel
        cubeindex[x, y, z] |= 8
    end
    if v4 > isolevel
        cubeindex[x, y, z] |= 16
    end
    if v5 > isolevel
        cubeindex[x, y, z] |= 32
    end
    if v6 > isolevel
        cubeindex[x, y, z] |= 64
    end
    if v7 > isolevel
        cubeindex[x, y, z] |= 128
    end

    return nothing
end

"""
    count_triangles!(cube_indices::CuDeviceArray{Int},
                     triangle_count::CuDeviceArray{Int},
                     casesClassic::CuDeviceArray,
                     Nx::Int,
                     Ny::Int,
                     Nz::Int)

    Counts the number of triangles generated

    # Arguments
    - `cube_indices`: Cube indices.
    - `triangle_count`: Output triangle counts.
    - `casesClassic`: Marching Cubes lookup table.
    - `Nx`, `Ny`, `Nz`: Volume dimensions.

    # Returns
    Returns `nothing`. Triangle counts are written directly on the GPU.
"""
function count_triangles!(cube_indices::CuDeviceArray{Int},
    triangle_count::CuDeviceArray{Int},
    casesClassic::CuDeviceArray,
    Nx::Int,
    Ny::Int,
    Nz::Int)
    x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    z = threadIdx().z + (blockIdx().z - 1) * blockDim().z

    if x > Nx || y > Ny || z > Nz
        return nothing
    end

    triangle_sum = 0
    for k in 1:16
        val = casesClassic[cube_indices[x, y, z]+1, k]
        val != -1 ? triangle_sum += 1 : break
    end

    triangle_count[x, y, z] = Int(triangle_sum/3)

    return nothing

end

"""
    generate_triangles!(mask::CuDeviceArray{Bool},
                        triangles::CuDeviceArray{Triangle3D},
                        triangles_count::CuDeviceArray{Int},
                        triangles_idx::CuDeviceArray{Int},
                        cube_indices::CuDeviceArray{Int},
                        spacing::CuDeviceArray{Float64},
                        casesClassic::CuDeviceArray,
                        nx::Int,
                        ny::Int,
                        nz::Int,
                        num_triangles::Int,
                        isolevel::Float64)

    Generates Marching Cubes triangles from cube configurations.

    Each CUDA thread processes one cube, computes the vertices,
    and writes the resulting triangles into the output array.

    # Arguments
    - `mask`: Binary input volume.
    - `triangles`: Output triangle array.
    - `triangles_count`: Number of triangles per cube.
    - `triangles_idx`: Prefix sum offsets for triangle placement.
    - `cube_indices`: Marching Cubes cube configurations.
    - `spacing`: Physical voxel spacing.
    - `casesClassic`: Marching Cubes lookup table.
    - `nx`, `ny`, `nz`: Volume dimensions.
    - `num_triangles`: Total number of output triangles.
    - `isolevel`: Threshold.

    # Returns
    Returns `nothing`. Output triangle array is written directly on the GPU.
"""
function generate_triangles!(mask::CuDeviceArray{Bool},
    triangles::CuDeviceArray{Triangle3D},
    triangles_count::CuDeviceArray{Int},
    triangles_idx::CuDeviceArray{Int},
    cube_indices::CuDeviceArray{Int},
    spacing::CuDeviceArray{Float64},
    casesClassic::CuDeviceArray,
    nx::Int,
    ny::Int,
    nz::Int,
    num_triangles::Int,
    isolevel::Float64)

    x = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    y = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    z = threadIdx().z + (blockIdx().z - 1) * blockDim().z

    if x > nx || y > ny || z > nz
        return nothing
    end

    lin_idx = encode_xyz(x, y, z, nx, ny)

    v0 = Float64(mask[x, y, z])
    v1 = Float64(mask[x+1, y, z])
    v2 = Float64(mask[x+1, y+1, z])
    v3 = Float64(mask[x, y+1, z])
    v4 = Float64(mask[x, y, z+1])
    v5 = Float64(mask[x+1, y, z+1])
    v6 = Float64(mask[x+1, y+1, z+1])
    v7 = Float64(mask[x, y+1, z+1])

    sx, sy, sz = spacing[1], spacing[2], spacing[3]
    x0, x1 = (x - 1) * sx, x * sx
    y0, y1 = (y - 1) * sy, y * sy
    z0, z1 = (z - 1) * sz, z * sz

    p0 = (x0, y0, z0)
    p1 = (x1, y0, z0)
    p2 = (x1, y1, z0)
    p3 = (x0, y1, z0)
    p4 = (x0, y0, z1)
    p5 = (x1, y0, z1)
    p6 = (x1, y1, z1)
    p7 = (x0, y1, z1)

    cidx = cube_indices[x, y, z] + 1

    i = 1
    triangle_number = 0
    while i + 2 <= 16
        e1 = casesClassic[cidx, i]
        e1 == -1 && break
        e2 = casesClassic[cidx, i+1]
        e3 = casesClassic[cidx, i+2]

        a = get_vert_on_edge(e1, p0, p1, p2, p3, p4, p5, p6, p7, v0, v1, v2, v3, v4, v5, v6, v7, isolevel)
        b = get_vert_on_edge(e2, p0, p1, p2, p3, p4, p5, p6, p7, v0, v1, v2, v3, v4, v5, v6, v7, isolevel)
        c = get_vert_on_edge(e3, p0, p1, p2, p3, p4, p5, p6, p7, v0, v1, v2, v3, v4, v5, v6, v7, isolevel)

        triangles[triangles_idx[lin_idx]+triangle_number+1] = (a, b, c)

        triangle_number += 1
        i += 3
    end

    return nothing
end

"""
    all_verts_kernel!(triangles::CuDeviceArray{Triangle3D},
                      all_verts::CuDeviceArray{Point3D},
                      num_triangles::Int)

    # Arguments
    - `triangles::CuDeviceArray`: Input triangle list.
    - `all_verts::CuDeviceArray`: Output vertex array.
    - `num_triangles::Int`: Number of triangles.

    # Returns
    Returns `nothing`. Vertices are stored directly on the GPU.
"""

function all_verts_kernel!(
    triangles::CuDeviceArray{Triangle3D},
    all_verts::CuDeviceArray{Point3D},
    num_triangles::Int)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > num_triangles
        return nothing
    end

    a, b, c = triangles[i]
    k = 3i - 2

    all_verts[k] = a
    all_verts[k+1] = b
    all_verts[k+2] = c

    return nothing
end

"""
    diam2d_kernel!(verts::CuDeviceArray{Point3D},
                   d_slice::CuDeviceArray{Float64,1},
                   d_row::CuDeviceArray{Float64,1},
                   d_column::CuDeviceArray{Float64,1},
                   num_verts::Int)

    Computes maximum distances between vertices

    # Arguments
    - `verts::CuDeviceArray`: 
    - `d_slice::CuDeviceArray`: 
    - `d_row::CuDeviceArray`: 
    - `d_column::CuDeviceArray`:
    - `num_verts::Int`: Number of vertices.

    # Returns
    Returns `nothing`. Distance values are updated atomically on the GPU.
"""
function diam2d_kernel!(verts::CuDeviceArray{Point3D},
    d_slice::CuDeviceArray{Float64,1},
    d_row::CuDeviceArray{Float64,1},
    d_column::CuDeviceArray{Float64,1},
    num_verts)

    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if i > num_verts
        return nothing
    end

    a = verts[i]
    for j in (i+1):num_verts
        b = verts[j]
        dx = a[1] - b[1]
        dy = a[2] - b[2]
        dz = a[3] - b[3]
        dist2 = dx * dx + dy * dy + dz * dz

        if a[3] == b[3]
            CUDA.@atomic d_slice[1] = max(d_slice[1], dist2)
        end

        if a[2] == b[2]
            CUDA.@atomic d_row[1] = max(d_row[1], dist2)
        end

        if a[1] == b[1]
            CUDA.@atomic d_column[1] = max(d_column[1], dist2)
        end

    end

    return nothing
end