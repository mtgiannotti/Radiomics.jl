function init_gpu(img_host, mask_host, verbose)
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
        return undef, undef, undef, false
    end
end

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

function findall_gpu(mask_device)
    vec_mask = vec(mask_device)
    num_of_useful_voxels = CUDA.sum(mask_device)
    valid_idx = CUDA.zeros(Int32, num_of_useful_voxels)
    prefix_sum = cumsum(Int32.(vec_mask))
    mask_length = length(vec_mask)

    @cuda threads = 256 blocks = cld(mask_length, 256) findall_kernel!(vec_mask, prefix_sum, valid_idx, mask_length)

    return valid_idx
end

function apply_mask(img, mask_indices)
    n = length(mask_indices)
    roi = CuArray{eltype(img)}(undef, n)

    @cuda threads = 256 blocks = cld(n, 256) assign!(img, mask_indices, roi, n)

    return roi
end

function discretize_image_gpu(img::CuArray{<:Real},
    mask::CuArray,
    mask_indices::CuArray;
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{<:Real,Nothing}=nothing)

    img_f32 = convert(CuArray{Float32}, img)
    bin_width_f32 = isnothing(bin_width) ? nothing : Float32(bin_width)

    if CUDA.sum(mask) == 0
        return zeros(Int32, size(img_f32)), 0, Int[], 0.0f0
    end

    vals = view(img_f32, mask)
    vmin = minimum(vals)
    vmax = maximum(vals)

    if !isnothing(n_bins) && !isnothing(bin_width_f32)
        error("Specify either n_bins or bin_width, not both.")

    elseif isnothing(n_bins) && isnothing(bin_width_f32)
        bin_width_f32 = 25.0f0
    end

    disc = CUDA.zeros(Int, size(img_f32))

    n_of_indices = length(mask_indices)

    if !isnothing(n_bins)
        bin_width_used = (vmax - vmin) / Float32(n_bins)
        if bin_width_used ≈ 0.0f0
            bin_width_used = 1.0f0
        end

        inv_bin_width = 1.0f0 / bin_width_used

        threads = 256
        blocks = cld(n_of_indices, threads)
        @cuda threads = threads blocks = blocks bin_nbins_kernel!(img_f32, mask_indices, inv_bin_width, n_bins, vmin, disc, n_of_indices)
    else
        bin_width_used = bin_width_f32
        inv_bin_width = 1.0f0 / bin_width_used
        bin_offset = Int(floor(vmin * inv_bin_width))

        threads = 256
        blocks = cld(n_of_indices, threads)
        @cuda threads = threads blocks = blocks bin_width_kernel!(img_f32, mask_indices, inv_bin_width, bin_offset, disc, n_of_indices)
    end
    gray_levels = unique_gpu(apply_mask(disc, mask_indices))
    n_bins_actual = length(gray_levels)

    return disc, n_bins_actual, Int64.(gray_levels), bin_width_used
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