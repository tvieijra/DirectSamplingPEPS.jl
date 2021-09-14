function get_stats(loc_energies::Array{T}, energy::Float64, psi_sqs::Array{T}, gradients::Array{T,7}) where T
    norm_peps = sum(psi_sqs)
    psi_times_loc_energies = psi_sqs .* loc_energies

    var = sum(psi_sqs .* (loc_energies .* conj(loc_energies)))/norm_peps

    energy_err = sum((loc_energies .* conj(loc_energies)) .* ((psi_sqs/norm_peps) .^ 2))

    grad_norm = norm(gradients)

    return [energy, sqrt(real(energy_err)), real(var), grad_norm, real(norm_peps)]
end
