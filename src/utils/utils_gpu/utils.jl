"""
    init_gpu(img_host::CuArray,
             mask_host::CuArray,
             verbose:Bool)

    Verifies that the CUDA.jl library and CUDA driver are installed and configured properly,
    and that the hardware is CUDA compatible.

    # Arguments
    - `img_host::AbstractArray`: The input image (2D or 3D array) stored on the CPU
    - `mask_host::AbstractArray`: The binary mask defining the region of interest (same shape as `img_host`) stored on the CPU
    - `verbose::Bool`: If `true`, prints a compatibility confirmation message

    # Returns 
    If compatible:
    - `img_device::CuArray`: The input image stored on the GPU as `CuArray`
    - `mask_device::CuArray`: The binary mask stored on the GPU as `CuArray` 
    - `mask_indices::CuArray`: The vector of valid ROI indices 
    - `true`: flag indicating that GPU execution can proceed normally
    If not compatible:
    - `img_device`, `mask_device` and `mask_indices` are returned as `nothing`
    - `false`: Flag indicating that GPU execution cannot proceed
"""

function init_gpu(img_host::AbstractArray,
    mask_host::AbstractArray,
    verbose::Bool)
    compatible, errors = can_use_cuda()
    if compatible
        if verbose
            println("current hardware is CUDA compatible. Allocating resources to the GPU...")
        end

        img_device = CuArray(img_host)
        mask_device = CuArray(mask_host)
        mask_indices = findall_gpu(mask_device)

        return img_device, mask_device, mask_indices, true
    else
        error_msg = errors * "Falling back to the CPU"
        @warn error_msg
        return nothing, nothing, nothing, false
    end
end

"""
    can_use_cuda()

    Checks for the correct installation and configuration of CUDA.jl and the CUDA driver 
    
    # Arguments

    # Returns
    - `compatible::Bool`: Flag indicating whether the current system is CUDA compatible
    - `errors::String`: Error messages describing any detected issues
"""

function can_use_cuda()
    errors = ""
    compatible = true

    if !CUDACore.functional()
        compatible = false
        errors += "- CUDA.jl does not appear to be functional and has not been installed or configured properly\n"
    end

    if !CUDACore.has_cuda()
        compatible = false
        errors += "- The CUDA driver has not been installed or the system does not have a CUDA compatible GPU. Please refer to https://developer.nvidia.com/cuda-downloads"
    end

    return compatible, errors
end

"""
    findall_gpu(mask_device::CuArray)

    Scans the mask and extracts all valid ROI indices 

    # Arguments
    - `mask_device::CuArray`: The binary mask defining the region of interest stored on the GPU

    # Returns 
    - `valid_idx::CuArray`: The vector containing all valid ROI indices

    # Implementation
    The function reshapes `mask_device` into a 1D vector called `vec_mask`. Since GPU memory allocation can't be dynamic, the function needs to know allocation size beforehand.
    In this specific case, the function needs to allocate an array containing all valid ROI indices. To know how big this array will be, since we're dealing with a binary mask we sum all the elements in `mask_device`,
    the resulting sum will represent the number of useful voxels in `num_of_useful_voxels`. We use this value to allocate `valid_idx` which is the `CuArray` containing all valid ROI indices and has length `num_of_useful_voxels`.

    Since dynamic allocation is not allowed on the GPU, it's not possible to push an element into an array, to get around this we calculate the prefix sum of the mask so that every thread knows where it needs to write inside `valid_idx`
    and then call the kernel `findall_kernel!` to finally extract all indices.
    
    - Prefix sum and indexing example:
    The prefix sum of an array is an array where each element is the sum of all previous elements including itself
    vec_mask = [1, 0, 1, 1, 0]
    prefix_sum = [1, 1, 2, 3, 3]
    each element in `prefix_sum` is the position inside `valid_idx` where each thread will write if the mask in that specific position is true
"""

function findall_gpu(mask_device)
    vec_mask = vec(mask_device)
    num_of_useful_voxels = CUDA.sum(mask_device)
    valid_idx = CUDA.zeros(Int32, num_of_useful_voxels)
    prefix_sum = cumsum(Int32.(vec_mask))
    mask_length = length(vec_mask)

    @cuda threads = 256 blocks = cld(mask_length, 256) findall_kernel!(vec_mask, prefix_sum, valid_idx, mask_length)

    CUDA.synchronize()

    return valid_idx
end

"""
    apply_mask(img::CuArray,
               mask_indices::CuArray)
    Performs boolean indexing on the GPU and returns a 1D vector containing all the elements inside the ROI. CPU counterpart: roi = img[mask]

    # Arguments
    - `img::CuArray`: The input image stored on the GPU as a `CuArray`
    - `mask_indices::CuArray`: A vector of valid ROI indices 

    # Returns 
    - `roi::CuArray`: 1D vector containing all the elements inside the ROI
"""

function apply_mask(img, mask_indices)
    n = length(mask_indices)
    roi = CuArray{eltype(img)}(undef, n)

    @cuda threads = 256 blocks = cld(n, 256) assign!(img, mask_indices, roi, n)

    return roi
end

"""
    discretize_image_gpu(img::CuArray{<:Real},
                         mask_indices::CuArray,
                         n_bins::Union{Int,Nothing}=nothing,
                         bin_width::Union{<:Real,Nothing}=nothing)
    
    - Refer to utils/utils_cpu/utils.jl

    - TODO: use Float64 instead of Float32 for better accuracy
"""

function discretize_image_gpu(img::CuArray{<:Real},
    mask::CuArray,
    mask_indices::CuArray;
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{<:Real,Nothing}=nothing,
    vmin::Union{Float64,Nothing}=nothing,
    vmax::Union{Float64,Nothing}=nothing)

    if CUDA.sum(mask) == 0
        return zeros(Int32, size(img_f32)), 0, Int[], 0.0f0
    end

    if isnothing(vmin) || isnothing(vmax)
        vals = view(img, mask)
        vmin = minimum(vals)
        vmax = maximum(vals)
    end

    disc = CUDA.zeros(Int, size(img))

    if !isnothing(n_bins) && !isnothing(bin_width)
        error("Specify either n_bins or bin_width, not both.")
    elseif isnothing(n_bins) && isnothing(bin_width)
        bin_width = 25.0
    end

    n_of_indices = length(mask_indices)

    if !isnothing(n_bins)
        bin_width_used = (vmax - vmin) / Float64(n_bins)
        if bin_width_used ≈ 0.0
            bin_width_used = 1.0
        end

        inv_bin_width = 1.0 / bin_width_used

        threads = 256
        blocks = cld(n_of_indices, threads)
        @cuda threads = threads blocks = blocks bin_nbins_kernel!(img, mask_indices, inv_bin_width, n_bins, vmin, disc, n_of_indices)
    else
        bin_width_used = bin_width
        inv_bin_width = 1.0f0 / bin_width_used
        bin_offset = Int(floor(vmin * inv_bin_width))

        threads = 256
        blocks = cld(n_of_indices, threads)
        @cuda threads = threads blocks = blocks bin_width_kernel!(img, mask_indices, inv_bin_width, bin_offset, disc, n_of_indices)
    end
    gray_levels = unique_gpu(apply_mask(disc, mask_indices))
    n_bins_actual = length(gray_levels)

    return disc, n_bins_actual, gray_levels, bin_width_used
end

function unique_gpu(img)
    img = sort(img)
    n = length(img)
    is_boundary = CUDA.zeros(Int32, n)

    @cuda threads = 256 blocks = cld(n, 256) set_boundaries!(img, is_boundary)

    idx = CUDA.cumsum(is_boundary)
    num_of_uniques = Int(CUDA.sum(is_boundary))

    uniques = CUDA.zeros(eltype(img), num_of_uniques)

    @cuda threads = 256 blocks = cld(n, 256) assign_uniques!(img, is_boundary, idx, uniques)
    return uniques
end