using DirectSamplingPEPS

let
  T = Float64
  Lx = 4
  Ly = 4
  physdim = 2
  bdim = 3
  edgedim = 1
  samplecut = bdim
  contractcut = 3*bdim
  marginalcut = bdim 
  contractconv = true
  rotsymm = false
  positive = true

  lr = 0.05
  clipnorm=40
  diagshift = 0.01
  tol = 1e-5
  numit = 1000
  numsamples = 1000
  sign = false
  adam = false
 
  J = 1
  sign_rule = false
 

  stats = zeros(Float64, numit, 5)

  peps = init_PEPS(Lx, Ly, physdim, bdim, edgedim, positive, T)
  marginals = get_marginals(peps, marginalcut)
  normalize_peps_stochastically!(peps, samplecut, marginals)

  num_threads = Threads.nthreads()
  samples_per_thread = numsamples ÷ num_threads

  psi_sqs = zeros(T, num_threads, samples_per_thread)
  samples = zeros(Int64, num_threads, samples_per_thread, peps.Lx, peps.Ly)
  loc_energies = zeros(T, num_threads, samples_per_thread)
  loc_gradients = zeros(T, num_threads, samples_per_thread, peps.Ly, peps.Lx, peps.phys_dim, peps.b_dim, peps.b_dim, peps.b_dim, peps.b_dim)

  if adam
    if clipnorm > 0
      opt = Adam{T}(step_size = lr, m = zeros(T, size(loc_gradients)[3:end]...), v = zeros(T, size(loc_gradients)[3:end]...), use_clipping=true, clip_norm=clipnorm)
    else
      opt = Adam{T}(step_size = lr, m = zeros(T, size(loc_gradients)[3:end]...), v = zeros(T, size(loc_gradients)[3:end]...))
    end
  else
    if clipnorm > 0
      opt = SGD(step_size = lr, use_clipping=true, clip_norm=clipnorm)
    else
      opt = SGD(step_size = lr)
    end 
  end

  marginals = get_marginals(peps, marginalcut)
  begin_samples = zeros(Int64, (num_threads,1,Lx,Ly))
  begin_psis = zeros(T, (num_threads,1))
  get_samples!(peps, begin_psis, begin_samples, samplecut, true, false, marginals)  
  begin_samples = reshape(begin_samples, (num_threads,Lx,Ly))

  for k in 1:numit
    fill!(psi_sqs, 0)
    fill!(samples, 0)
    fill!(loc_energies, 0)
    fill!(loc_gradients, 0)

    get_samples_mcmc!(peps, psi_sqs, samples, samplecut, 100, 1, begin_samples, true, transition_single_flip!)
    energy, norm_peps = get_energy!(peps, x->get_heisenberg_mels(x,J,sign_rule,false), psi_sqs, samples, loc_energies, contractcut, rotsymm, true)
    get_local_log_gradients(peps, samples, norm_peps, loc_energies, loc_gradients, contractcut, rotsymm)
    update_vec, gradients = stochastic_reconfiguration(loc_gradients, psi_sqs, loc_energies, diagshift, tol)
    update!(opt, peps, update_vec)
    stats[k,:] = get_stats(loc_energies, energy, psi_sqs, gradients)
    println("iteration number: "*string(k))
    println("\t" * "energy: " * string(stats[k,1]))
    println("\t" * "energy error: " * string(stats[k,2]))
    println("\t" * "hamiltonian variance: " * string(stats[k,3]))
    println("\t" * "gradient norm: " * string(stats[k,4]))
    println("\t" * "PEPS norm: " * string(stats[k,5]))
    GC.gc()
  end
end

