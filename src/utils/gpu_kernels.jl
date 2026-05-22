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

function glcm_kernel!(G, mask, mapped_disc, dirs_x, dirs_y, dirs_z, mask_length, dirs_length, Nx, Ny, Nz)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x # maps threads to mask
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y # maps threads to directions

    if i > Nx * Ny * Nz || j > dirs_length
        return nothing
    end

    z = fld(i - 1, Nx * Ny) + 1
    r = (i - 1) % (Nx * Ny)

    y = fld(r, Nx) + 1
    x = (r % Nx) + 1

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

    CUDA.@atomic G[j, i_disc, j_disc] += 1.0f0
    CUDA.@atomic G[j, j_disc, i_disc] += 1.0f0

    return nothing
end