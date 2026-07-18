function compute_glcm_gpu(disc, gray_levels, gpu_data)
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

    @cuda threads = 256 blocks = cld(Ng, 256) lut_kernel!(gray_levels, lut, min_gl, Ng)
    mapped_disc = CUDA.zeros(Int, size(disc))
    Nx, Ny = size(mapped_disc)
    Nz = (dim == 3) ? size(mapped_disc, 3) : 1
    @cuda threads = 256 blocks = cld(length(disc), 256) mapped_disc_kernel!(disc, mapped_disc, gpu_data.mask, length(disc), lut, min_gl)

    G_d = CUDA.zeros(Float64, Ng, Ng, length(dirs_x))

    threads = (16, 16)
    n = length(gpu_data.mask_indices)
    blocks_x = cld(n, threads[1])
    blocks_y = cld(length(dirs_x), threads[2])
    @cuda threads = threads blocks = (blocks_x, blocks_y) glcm_kernel!(G_d, gpu_data.mask, gpu_data.mask_indices, mapped_disc, dirs_x, dirs_y, dirs_z, length(dirs_x), Nx, Ny, Nz, n)
    G_all = Array(G_d)

    for d in axes(G_all, 3)
        sym_sum = @view G_all[:, :, d]
        sym_sum .+= sym_sum'
    end
    return permutedims(G_all, (3, 1, 2))
end