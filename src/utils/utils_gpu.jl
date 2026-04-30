"""

    GPU kernels and functions that call them and GPU compatible functions from utils.jl go here

"""

function bin_nbins_kernel!(img_f32, mask, inv_bin_width, n_bins, vmin, disc)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > length(mask)
        return nothing
    end

    if mask[i]
        v = img_f32[i]
        b = min(Int(floor((v - vmin) * inv_bin_width)) + 1, n_bins)
        disc[i] = b
    end

    return nothing
end

function bin_width_kernel!(img_f32, mask, inv_bin_width, bin_offset, disc)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > length(mask)
        return nothing
    end

    if mask[i]
        v = img_f32[i]
        b = Int(floor(v * inv_bin_width)) - bin_offset + 1
        disc[i] = b
    end

    return nothing
end

function assign!(img, mask, idx, roi)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i > length(mask)
        return nothing
    end

    if mask[i] != 0
        roi[idx[i]] = img[i]
    end

    return nothing
end

function apply_mask(img, mask)
    n = sum(mask)
    roi = CuArray{eltype(img)}(undef, n)
    indexes = cumsum(vec(mask))

    threads = 256
    blocks = cld(length(mask), threads)
    @cuda threads = threads blocks = blocks assign!(img, mask, indexes, roi)

    return roi
end

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

function discretize_image_gpu(img::CuArray{<:Real},
    mask::CuArray;
    n_bins::Union{Int,Nothing}=nothing,
    bin_width::Union{<:Real,Nothing}=nothing)

    img_f32 = convert(CuArray{Float32}, img)
    bin_width_f32 = isnothing(bin_width) ? nothing : Float32(bin_width)

    if CUDA.sum(mask) == 0
        return zeros(Int, size(img_f32)), 0, Int[], 0.0f0
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

    if !isnothing(n_bins)
        bin_width_used = (vmax - vmin) / Float32(n_bins)
        if bin_width_used ≈ 0.0f0
            bin_width_used = 1.0f0
        end

        inv_bin_width = 1.0f0 / bin_width_used

        threads = 256
        blocks = cld(length(mask), threads)
        @cuda threads = threads blocks = blocks bin_nbins_kernel!(img_f32, mask, inv_bin_width, n_bins, vmin, disc)
    else
        bin_width_used = bin_width_f32
        inv_bin_width = 1.0f0 / bin_width_used
        bin_offset = Int(floor(vmin * inv_bin_width))

        threads = 256
        blocks = cld(length(mask), threads)
        @cuda threads = threads blocks = blocks bin_width_kernel!(img_f32, mask, inv_bin_width, bin_offset, disc)
    end
    #gray_levels = sort(unique(apply_mask(disc, mask)))
    gray_levels = Array(unique_gpu(apply_mask(disc, mask)))
    n_bins_actual = length(gray_levels)

    return Array(disc), n_bins_actual, Int64.(gray_levels), bin_width_used
end

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

function unique_gpu(img)
    img = sort(img)
    n = length(img)
    is_boundary = CUDA.zeros(Int, n)

    @cuda threads = 256 blocks = cld(n, 256) set_boundaries!(img, is_boundary)

    idx = CUDA.cumsum(is_boundary)
    num_of_uniques = Int(CUDA.sum(is_boundary))

    uniques = CUDA.zeros(eltype(img), num_of_uniques)

    CUDA.@sync begin
        @cuda threads = 256 blocks = cld(n, 256) assign_uniques!(img, is_boundary, idx, uniques)
    end
    return uniques
end