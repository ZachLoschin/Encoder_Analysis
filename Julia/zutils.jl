
using Random
using Distributions
using Plots
using StatsBase
using CSV
using DataFrames
using LinearAlgebra
using StateSpaceDynamics
using Optim
using Statistics
const SSD = StateSpaceDynamics
using Random


# Helper function to split matrix into chunks
function split_into_trials(mat::Matrix, trial_length::Int)
    n_trials = size(mat, 1) ÷ trial_length
    [mat[((i-1)*trial_length + 1):(i*trial_length), :] for i in 1:n_trials]
end


function train_test_split(X::Vector{<:Matrix{<:Real}}, 
                          y::Vector{<:Matrix{<:Real}}, 
                          split_ratio::Float64=0.8)
    n = length(X)  # Number of samples
    perm = randperm(n)  # Shuffle indices
    split_idx = round(Int, split_ratio * n)  # Compute split point

    train_idx = perm[1:split_idx]
    test_idx = perm[split_idx+1:end]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test
end


function class_probs_inner(model::HiddenMarkovModel, Y::Matrix{<:Real}, X::Union{Matrix{<:Real},Nothing}=nothing;)
    data = X === nothing ? (Y,) : (X, Y)
    # transpose data so that correct dimensions are passed to EmissionModels.jl, a bit hacky but works for now.
    transpose_data = Matrix.(transpose.(data))
    num_obs = size(transpose_data[1], 1)
    # initialize forward backward storage
    FB_storage = SSD.initialize_forward_backward(model, num_obs)

    # Get class probabilities using Estep
    SSD.estep!(model, transpose_data, FB_storage)

    return exp.(FB_storage.γ)
end


function class_probs(
    model::HiddenMarkovModel,
    Y_trials::Vector{<:Matrix{<:Real}},
    X_trials::Union{Vector{<:Matrix{<:Real}}, Nothing} = nothing
)
    n_trials = length(Y_trials)
    # Preallocate storage for class probabilities
    all_class_probs = Vector{Matrix{Float64}}(undef, n_trials)
    # Loop through each trial and compute class probabilities
    for i in 1:n_trials
        Y = Y_trials[i]
        X = X_trials === nothing ? nothing : X_trials[i]
        all_class_probs[i] = class_probs_inner(model, Y, X)
    end

    return all_class_probs
end

function fit!(
    model::HiddenMarkovModel,
    Y::Vector{<:Matrix{<:Real}},
    X::Union{Vector{<:Matrix{<:Real}},Nothing},
    max_iters::Int=100,
    tol::Float64=1e-6,
)
    println("modified fit")
    lls = [-Inf]
    data = X === nothing ? (Y,) : (X, Y)

    # Initialize log_likelihood
    log_likelihood = -Inf

    # Transform each matrix in each tuple to the correct orientation
    transposed_matrices = map(data_tuple -> Matrix.(transpose.(data_tuple)), data)
    zipped_matrices = collect(zip(transposed_matrices...))
    total_obs = sum(size(trial_mat[1], 1) for trial_mat in zipped_matrices)

    # Initialize a vector of ForwardBackward storage and an aggregate storage
    FB_storage_vec = [SSD.initialize_forward_backward(model, size(trial_tuple[1], 1)) for trial_tuple in zipped_matrices]
    Aggregate_FB_storage = SSD.initialize_forward_backward(model, total_obs)

    # Store aggregate storage history
    aggregate_storage_history = []
    A_storage = []
    π_storage = []
    Σ_storage = []
    β_storage = []

    for iter in 1:max_iters
        if iter % 1 ==0
            println("Iter:",iter)
        end
        # Broadcast estep!() to all storage structs
        SSD.estep!.(Ref(model), zipped_matrices, FB_storage_vec)
        

        # Collect storage structs into one struct for M-step
        SSD.aggregate_forward_backward!(Aggregate_FB_storage, FB_storage_vec)

        # Store a deep copy of Aggregate_FB_storage to preserve past states
        push!(aggregate_storage_history, deepcopy(Aggregate_FB_storage))
        push!(A_storage, deepcopy(model.A))
        push!(π_storage, deepcopy(model.πₖ))
        push!(Σ_storage, hcat(model.B[1].Σ, model.B[2].Σ))
        push!(β_storage, hcat(model.B[1].β, model.B[2].β))

        # Calculate log_likelihood
        log_likelihood_current = sum([SSD.logsumexp(FB_vec.α[:, end]) for FB_vec in FB_storage_vec])  #  / size(FB_vec.α, 2)
        push!(lls, log_likelihood_current)
        

        # Check for convergence
        if abs(log_likelihood_current - log_likelihood) < tol
            break
        else
            log_likelihood = log_likelihood_current
        end

        # Get data trial tuples stacked for mstep!()
        stacked_data = SSD.stack_tuples(zipped_matrices)
        stacked_data_test = SSD.stack_tuples(zipped_matrices_test)

        # M-step
        mstep!(model, FB_storage_vec, Aggregate_FB_storage, stacked_data, Aggregate_FB_storage_test, stacked_data_test)
    end

    return lls, FB_storage_vec, aggregate_storage_history, A_storage, π_storage, Σ_storage, β_storage
end

function mstep!(model::HiddenMarkovModel, FB_storage_vec::Vector{SSD.ForwardBackward{Float64}}, Aggregate_FB_storage::SSD.ForwardBackward, data, Aggregate_FB_storage_test, data_test)
    # update initial state distribution
    SSD.update_initial_state_distribution!(model, FB_storage_vec)
    # update transition matrix
    SSD.update_transition_matrix!(model, FB_storage_vec)
    # update regression models
    # update_emissions!(model, Aggregate_FB_storage, data, Aggregate_FB_storage_test, data_test)
    SSD.update_emissions!(model, Aggregate_FB_storage, data)
end

# function update_transition_matrix!(
#     model::HiddenMarkovModel, FB_storage_vec::SSD.Vector{ForwardBackward{Float64}}, α::Float64 = 1.0
# )
#     K = model.K
#     A_new = zeros(K, K)

#     for j in 1:K
#         for k in 1:K
#             num = exp(SSD.logsumexp(vcat([FB_trial.ξ[j, k, 2:end] for FB_trial in FB_storage_vec]...)))
#             denom = exp(SSD.logsumexp(vcat([FB_trial.ξ[j, :, 2:end]' for FB_trial in FB_storage_vec]...)))  
#             A_new[j, k] = num / denom
#         end
#     end

#     # Apply prior favoring staying in the same state
#     for j in 1:K
#         A_new[j, j] += α
#     end

#     # Renormalize the transition matrix
#     model.A .= A_new ./ sum(A_new, dims=2)
# end


function update_emissions!(model::SSD.HiddenMarkovModel, FB_storage::SSD.ForwardBackward, data, FB_storage_test, data_test)
    # update regression models
    w = exp.(permutedims(FB_storage.γ))
    w_test = exp.(permutedims(FB_storage_test.γ))
    # check threading speed here
    for k in 1:(model.K)
        fit_reg_sweep!(model.B[k], data..., w[:, k], data_test..., w_test[:,k])
    end
end

function fit_reg_sweep!(
    model::RegressionEmission,
    X::Matrix{<:Real},
    y::Matrix{<:Real},
    w::Vector{Float64},
    X_test::Matrix{<:Real},
    y_test::Matrix{<:Real},
    w_test::Vector{Float64}
)
    lambdas = [0.0, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8] # Log-spaced lambda values
    best_lambda = nothing
    best_r2 = -Inf  # R^2 can be negative if model is worse than baseline
    best_model = deepcopy(model)  # To store the best model

    for (i, λ) in enumerate(lambdas)
        model.λ = λ
        
        # Fit on training data
        opt_problem = SSD.create_optimization(model, X, y, w)
        f(β) = SSD.objective(opt_problem, β)
        g!(G, β) = SSD.objective_gradient!(G, opt_problem, β)

        opts = Optim.Options(;
            x_abstol=1e-8, x_reltol=1e-8,
            f_abstol=1e-8, f_reltol=1e-8,
            g_abstol=1e-8, g_reltol=1e-8,
        )

        result = optimize(f, g!, vec(model.β), LBFGS(), opts)

        # Update model parameters
        model.β = SSD.vec_to_matrix(result.minimizer, opt_problem.β_shape)

        # Compute R^2 on validation set
        y_pred = X_test * model.β  # Direct matrix multiplication
        r2 = compute_weighted_r2(y_pred, y_test, w_test)

        # Store best model based on R^2
        if r2 > best_r2
            best_r2 = r2
            best_lambda = λ
            best_model = deepcopy(model)  # Save best model
        end
    end

    # Set model to best found values
    model.λ = best_lambda
    model.β = best_model.β

    println("Best λ found: ", best_lambda, " with R^2: ", best_r2)

    return model
end

# Compute Weighted R^2
function compute_weighted_r2(y_pred::Matrix{<:Real}, 
                             y_true::Matrix{<:Real}, 
                             w::Vector{Float64})
    # Ensure w is a column vector for broadcasting
    w = reshape(w, :, 1)

    # Weighted residual sum of squares
    ss_res = sum(w .* ((y_true .- y_pred) .^ 2))
    
    # Weighted total sum of squares
    ss_tot = sum(w .* ((y_true .- mean(y_true, dims=1)) .^ 2))

    # Compute weighted R^2
    return 1 - (ss_res / ss_tot)
end



function kernelize_X(Xvec::Vector{Matrix{Float64}}, kernel_size::Int)
    # Prepare an output vector of matrices with the same length as Xvec.
    new_Xvec = Vector{Matrix}(undef, length(Xvec))
    
    for (i, X) in enumerate(Xvec)
        T, K = size(X)
        # We can only form full windows; if T < kernel_size, we cannot form any window.
        new_T = T - kernel_size + 1
        if new_T < 1
            error("Matrix number $(i) does not have enough time points to form a window of size $(kernel_size).")
        end
        
        # The new matrix will have new_T rows and (kernel_size * K) columns.
        newX = Array{eltype(X)}(undef, new_T, kernel_size * K)
        # For each window starting at time index t,
        # extract rows t through t+kernel_size-1 and concatenate them.
        for t in 1:new_T
            window = X[t : t + kernel_size - 1, :]  # This is a kernel_size x K matrix.
            # We want to create a 1 x (kernel_size*K) vector by concatenating
            # the rows in time order. One easy way is to take the transpose of the window
            # (so that time becomes the column index) and then vectorize.
            newX[t, :] = vec(permutedims(window))'
        end
        
        new_Xvec[i] = newX
    end
    return new_Xvec
end



function chop_Y(Yvec::Vector{Matrix{Float64}}, kernel_size::Int)
    # Prepare an output vector of matrices with the same length as Yvec.
    new_Yvec = Vector{Matrix{Float64}}(undef, length(Yvec))
    
    for (i, Y) in enumerate(Yvec)
        T, P = size(Y)
        new_T = T - kernel_size + 1
        if new_T < 1
            error("Matrix number $(i) does not have enough time points to form a window of size $(kernel_size).")
        end
        
        # The new Y matrix will take the row at the end of each kernel window.
        # That is, newY[t, :] = Y[t + kernel_size - 1, :].
        # Since t goes from 1 to new_T, we can simply take the slice:
        new_Y = Y[kernel_size:end, :]
        
        new_Yvec[i] = new_Y
    end
    
    return new_Yvec
end

function label_data(model, Y, X)
    data = X === nothing ? (Y,) : (X, Y)
    
    # Transform each matrix in each tuple to the correct orientation
    transposed_matrices = map(data_tuple -> Matrix.(transpose.(data_tuple)), data)
    zipped_matrices = collect(zip(transposed_matrices...))
    total_obs = sum(size(trial_mat[1], 1) for trial_mat in zipped_matrices)

    FB_storage_vec = [SSD.initialize_forward_backward(model, size(trial_tuple[1],1)) for trial_tuple in zipped_matrices]
    Aggregate_FB_storage = SSD.initialize_forward_backward(model, total_obs)
    SSD.estep!.(Ref(model), zipped_matrices, FB_storage_vec)
    return FB_storage_vec
end

function FB_heatmap(FB_storage)
    # Number of trials
    num_trials = length(FB_storage)
    num_timepoints = 100
    # Initialize the matrix
    heatmap_data = zeros(num_trials, num_timepoints)
    # Populate the matrix
    for trial in 1:num_trials
        heatmap_data[trial, :] = exp.(FB_storage[trial].γ[1, 1:num_timepoints])
    end
    # Create the heatmap
    heatmap(heatmap_data, color=:viridis, xlabel="Timepoints", ylabel="Trials", clabel="Value", yflip=true)
end

function compute_param_diffs(aggregate_storage_history, Σ_storage, π_storage, A_storage, β_storage)
    num_iters = length(aggregate_storage_history)

    # Initialize storage for parameter changes
    sigma_changes = Float64[]
    pi_changes = Float64[]
    A_changes = Float64[]
    overall_changes = Float64[]
    β_changes = []

    for i in 2:num_iters
        # Extract current and previous parameter values
        Σ_prev, Σ_curr = Σ_storage[i-1][:], Σ_storage[i][:]  # Flatten matrices
        π_prev, π_curr = π_storage[i-1][:], π_storage[i][:]  # Flatten vectors
        A_prev, A_curr = A_storage[i-1][:], A_storage[i][:]  # Flatten matrices
        β_prev, β_curr = β_storage[i-1][:], β_storage[i][:]  # Flatten matrices

        # Compute element-wise absolute differences and sum
        sigma_change = sum(abs.(Σ_curr .- Σ_prev))
        pi_change = sum(abs.(π_curr .- π_prev))
        A_change = sum(abs.(A_curr .- A_prev))
        β_change = sum(abs.(β_curr .- β_prev))

        # Compute overall change as the sum of all individual changes
        overall_change = sigma_change + pi_change + A_change + β_change

        # Store the computed changes
        push!(sigma_changes, sigma_change)
        push!(pi_changes, pi_change)
        push!(A_changes, A_change)
        push!(overall_changes, overall_change)
        push!(β_changes, β_change)
    end

    return sigma_changes, pi_changes, A_changes, overall_changes, β_changes
end


function compute_param_diffs(Σ_storage, π_storage, A_storage, β_storage)
    num_iters = length(Σ_storage)

    # Initialize storage for parameter changes
    sigma_changes = Float64[]
    pi_changes = Float64[]
    A_changes = Float64[]
    overall_changes = Float64[]
    β_changes = []

    for i in 2:num_iters
        # Extract current and previous parameter values
        Σ_prev, Σ_curr = Σ_storage[i-1][:], Σ_storage[i][:]  # Flatten matrices
        π_prev, π_curr = π_storage[i-1][:], π_storage[i][:]  # Flatten vectors
        A_prev, A_curr = A_storage[i-1][:], A_storage[i][:]  # Flatten matrices
        β_prev, β_curr = β_storage[i-1][:], β_storage[i][:]  # Flatten matrices

        # Compute element-wise absolute differences and sum
        sigma_change = sum(abs.(Σ_curr .- Σ_prev))
        pi_change = sum(abs.(π_curr .- π_prev))
        A_change = sum(abs.(A_curr .- A_prev))
        β_change = sum(abs.(β_curr .- β_prev))

        # Compute overall change as the sum of all individual changes
        overall_change = sigma_change + pi_change + A_change + β_change

        # Store the computed changes
        push!(sigma_changes, sigma_change)
        push!(pi_changes, pi_change)
        push!(A_changes, A_change)
        push!(overall_changes, overall_change)
        push!(β_changes, β_change)
    end

    return sigma_changes, pi_changes, A_changes, overall_changes, β_changes
end

function load_data_encoder(path, cond)
    # Helper function to chunk matrix into 600-timepoint segments
    function chunk_matrix(mat, chunk_size)
        num_chunks = size(mat, 1) ÷ chunk_size
        [mat[(i-1)*chunk_size + 1 : i*chunk_size, :] for i in 1:num_chunks]
    end

    # Construct file paths
    probe1_path = path * "Probe1_" * cond * "_Uncut.csv"
    probe2_path = path * "Probe2_" * cond * "_Uncut.csv"

    PCA_P1_path = path * "PCA_Probe1_" * cond * "_Uncut.csv"
    PCA_P2_path = path * "PCA_Probe2_" * cond * "_Uncut.csv"

    SVD_path = path * "SVD_Feats_" * cond * "_Uncut.csv"
    KP_path = path * "Keypoint_Feats_" * cond * "_Uncut.csv"

    # Load the data into matrices
    probe1_mat = Matrix(CSV.read(probe1_path, DataFrame; header=false))
    probe2_mat = Matrix(CSV.read(probe2_path, DataFrame; header=false))

    PCA_P1_mat = Matrix(CSV.read(PCA_P1_path, DataFrame; header=false))
    PCA_P2_mat = Matrix(CSV.read(PCA_P2_path, DataFrame; header=false))

    SVD_mat = Matrix(CSV.read(SVD_path, DataFrame; header=false))
    KP_mat = Matrix(CSV.read(KP_path, DataFrame; header=false))

    # Chunk size
    chunk_size = 600

    # Chunk all matrices
    probe1_chunks = chunk_matrix(probe1_mat, chunk_size)
    probe2_chunks = chunk_matrix(probe2_mat, chunk_size)

    PCA_P1_chunks = chunk_matrix(PCA_P1_mat, chunk_size)
    PCA_P2_chunks = chunk_matrix(PCA_P2_mat, chunk_size)

    SVD_chunks = chunk_matrix(SVD_mat, chunk_size)
    KP_chunks = chunk_matrix(KP_mat, chunk_size)

    return probe1_chunks, probe2_chunks, PCA_P1_chunks, PCA_P2_chunks, SVD_chunks, KP_chunks
end

function load_data(path, cond, TW)
    
    # Neural probe data paths -> check metadata.txt or excel sheet for probe locations
    probe1_path = path*"Probe1_"*cond*".csv"
    # probe1_path_PCA = path*"Probe1_"*cond*"_PC"*".csv"
    probe2_path = path*"Probe2_"*cond*".csv"

    # Kinematic feats paths -> tongue path has tongue length, jaw_feats has jaw pos and jaw vel 
    kin_path = path*"Jaw_"*cond*".csv"
    tongue_path = path*"Tongue_"*cond*".csv"
    # PCA_feats_path = path*"PCA_feats_"*cond*".csv"
    # PCA_feats_path_uncut = path*"PCA_feats_uncut_"*cond*".csv"
    Jaw_feats_path = path*"Jaw_feats_"*cond*".csv"
    
    # Trial information paths -> trial lengths and first lick contact paths
    trial_lpath = path*"Trial_End_"*cond*".csv"
    first_lick_path = path*"First_Contact_"*cond*".csv"
    
    
    # Load the data
    probe1_df = CSV.read(probe1_path, DataFrame, header=false)
    # probe1_df_PCA = CSV.read(probe1_path_PCA, DataFrame, header=false)
    probe2_df = CSV.read(probe2_path, DataFrame, header=false)


    kin_df = CSV.read(kin_path, DataFrame, header=false)
    tongue_df = CSV.read(tongue_path, DataFrame, header=false)
    # PCA_feats = CSV.read(PCA_feats_path, DataFrame, header=false)
    # PCA_feats_uncut = CSV.read(PCA_feats_path_uncut, DataFrame, header=false)
    Jaw_feats = CSV.read(Jaw_feats_path, DataFrame, header=false)
    
    lastL_df = CSV.read(trial_lpath, DataFrame, header=false)
    firstL_df = CSV.read(first_lick_path, DataFrame, header=false)
    
    
    # Convert to matrices for easier use
    probe1_matrix = Matrix(probe1_df)
    # probe1_matrix_PCA = Matrix(probe1_df_PCA)
    probe2_matrix = Matrix(probe2_df)
    
    
    kin_data = kin_df[:, 1]
    tongue_data = tongue_df[:, 1]
    # feats_matrix = Matrix(PCA_feats)
    # feats_matrix_uncut = Matrix(PCA_feats_uncut)
    Jaw_feats_matrix = Matrix(Jaw_feats)
    
    lastL = Vector(lastL_df[1, :])
    firstL = Vector(firstL_df[1, :])
    
    # Adjust licks to sampling rate from MATLAB code
    firstL_adjusted = round.(Int, firstL .* 100)
    trial_lengths = lastL  # round.(Int, lastL .* 100);
    
    # Initialize storage
    kin_trials = Vector{Matrix{Float64}}(undef, length(lastL))
    tongue_trials = Vector{Vector{Float64}}(undef, length(lastL))
    X_probe1 = Vector{Matrix{Float64}}(undef, length(lastL))
    # X_probe1_PCA = Vector{Matrix{Float64}}(undef, length(lastL))
    X_probe2 = Vector{Matrix{Float64}}(undef, length(lastL))
    # Y = Vector{Matrix{Float64}}(undef, length(lastL))
    # Y_uncut = Vector{Matrix{Float64}}(undef, length(lastL))
    Jaw_Y = Vector{Matrix{Float64}}(undef, length(lastL))
    
    # Start slicing and populating the data
    start_idx = 1

    for i in 1:length(lastL)
        start_idx = (i-1)*600 + 1
        end_idx = start_idx + lastL[i] - 1

        Jaw_feat_trial = Jaw_feats_matrix[start_idx:end_idx, :]
        Jaw_Y[i] = Jaw_feat_trial

    end

    start_idx = 1
    for i in 1:length(lastL)
        end_idx = start_idx + lastL[i] - 1
        # Slice and store tongue data
        tongue_trials[i] = tongue_data[start_idx:end_idx]
        
        # Slice and store neural data for the trial
        trial_data_P1 = probe1_matrix[start_idx:end_idx, :]
        X_probe1[i] = trial_data_P1

        trial_data_P2 = probe2_matrix[start_idx:end_idx, :]
        X_probe2[i] = trial_data_P2

        # Update the start index for the next trial
        start_idx = end_idx + 1
    end
    
    return X_probe1, X_probe2, kin_trials, tongue_trials, trial_lengths, firstL_adjusted, Jaw_Y
end


function load_data_uncut(path, cond, TW, trial_length)

    # Neural probe data paths -> check metadata.txt or excel sheet for probe locations
    probe1_path = path*"Probe1_uncut_"*cond*".csv"
    probe2_path = path*"Probe2_uncut_"*cond*".csv"
    # PCA_feats_path = path*"PCA_feats_uncut_"*cond*".csv"
    tongue_path = path*"Tongue_uncut_"*cond*".csv"
    Jaw_feats_path = path*"Jaw_feats_"*cond*".csv"

    # Read in uncut data
    data_df_uncut_p1 = CSV.read(probe1_path, DataFrame, header=false)
    data_df_uncut_p2 = CSV.read(probe2_path, DataFrame, header=false)
    tongue_df_uncut = CSV.read(tongue_path, DataFrame, header=false)
    # PCA_feats_uncut = CSV.read(PCA_feats_path, DataFrame, header=false)
    Jaw_feats = CSV.read(Jaw_feats_path, DataFrame, header=false)

    num_trials = size(tongue_df_uncut)[2]

    # Process data_df_uncut and PCA_feats_uncut into trials
    data_matrix_uncut_P1 = Matrix(data_df_uncut_p1)
    data_matrix_uncut_P2 = Matrix(data_df_uncut_p2)
    # feats_matrix_uncut = Matrix(PCA_feats_uncut)
    Jaw_feats_matrix_uncut = Matrix(Jaw_feats)

    X_uncut_p1 = Vector{Matrix{Float64}}(undef, num_trials)
    X_uncut_p2 = Vector{Matrix{Float64}}(undef, num_trials)
    Y_uncut = Vector{Matrix{Float64}}(undef, num_trials)
    Jaw_Y_uncut = Vector{Matrix{Float64}}(undef, num_trials)
    tongue_trials_uncut = Vector{Vector{Float64}}(undef, num_trials)

    start_idx = 1
    for i in 1:num_trials
        end_idx = start_idx + trial_length - 1
        
        tongue_trials_uncut[i] = tongue_df_uncut[:,i]

        # Slice and store neural data for the trial
        trial_data = data_matrix_uncut_P1[start_idx:end_idx, :]
        X_uncut_p1[i] = trial_data

        trial_data = data_matrix_uncut_P2[start_idx:end_idx, :]
        X_uncut_p2[i] = trial_data

        # Slice and store kin feats
        # feat_trial = feats_matrix_uncut[start_idx:end_idx, :]
        # Y_uncut[i] = feat_trial

        Jaw_feat_trial = Jaw_feats_matrix_uncut[start_idx:end_idx, :]
        Jaw_Y_uncut[i] = Jaw_feat_trial
        
        # Update the start index for the next trial
        start_idx = end_idx + 1
    end

    return X_uncut_p1, X_uncut_p2, tongue_trials_uncut, Jaw_Y_uncut
end

function kernelize_features(X_train::Vector{Matrix{T}}, lags::Int=4) where T
    processed_features = Vector{Matrix{T}}(undef, length(X_train))

    for (i, X) in enumerate(X_train)
        num_timepoints, num_features = size(X)

        # Define the valid time range where we have full context
        start_idx = lags + 1
        end_idx = num_timepoints - lags

        # Initialize new feature matrix
        num_new_timepoints = end_idx - start_idx + 1
        lagged_features = Matrix{T}(undef, num_new_timepoints, num_features * (2 * lags + 1))

        for (j, t) in enumerate(start_idx:end_idx)
            # Collect past, current, and future feature values
            feature_vector = vcat((X[t + lag, :] for lag in -lags:lags)...)
            lagged_features[j, :] = feature_vector
        end

        # Remove additional 94 timepoints from the start
        processed_features[i] = lagged_features
    end
    
    return processed_features
end


function kernelize_past_features(X_train::Vector{Matrix{T}}, lags::Int=4) where T
    processed_features = Vector{Matrix{T}}(undef, length(X_train))

    for (i, X) in enumerate(X_train)
        num_timepoints, num_features = size(X)

        # Define the valid time range where we have full past context
        start_idx = lags + 1
        end_idx = num_timepoints

        # Initialize new feature matrix
        num_new_timepoints = end_idx - start_idx + 1
        lagged_features = Matrix{T}(undef, num_new_timepoints, num_features * (lags + 1))

        for (j, t) in enumerate(start_idx:end_idx)
            # Collect past and current feature values
            feature_vector = vcat((X[t - lag, :] for lag in 0:lags)...) 
            lagged_features[j, :] = feature_vector
        end

        # Remove additional 94 timepoints from the start
        processed_features[i] = lagged_features
    end
    
    return processed_features
end

function trim_Y_train_past(Y_train::Vector{Matrix{T}}, lags::Int=4) where T
    processed_Y = Vector{Matrix{T}}(undef, length(Y_train))

    for (i, Y) in enumerate(Y_train)
        num_timepoints, num_features = size(Y)

        # Define valid time range (same as X kernelization)
        start_idx = lags + 1
        end_idx = num_timepoints

        # Extract the matching time range
        processed_Y[i] = Y[start_idx:end_idx, :]
    end

    return processed_Y
end


function trim_Y_train_past(Y_train::Vector{Vector{T}}, lags::Int=4) where T
    processed_Y = Vector{Vector{T}}(undef, length(Y_train))

    for (i, Y) in enumerate(Y_train)
        num_timepoints = length(Y)

        # Define valid time range (same as X kernelization)
        start_idx = lags + 1
        end_idx = num_timepoints

        # Extract the matching time range
        processed_Y[i] = Y[start_idx:end_idx]
    end

    return processed_Y
end


function trim_Y_train(Y_train::Vector{Matrix{T}}, lags::Int=4) where T
    processed_Y = Vector{Matrix{T}}(undef, length(Y_train))

    for (i, Y) in enumerate(Y_train)
        num_timepoints, num_features = size(Y)

        # Define valid time range (same as X kernelization)
        start_idx = lags + 1
        end_idx = num_timepoints - lags

        # Extract the matching time range
        processed_Y[i] = Y[start_idx:end_idx, :]
    end

    return processed_Y
end

function trim_Y_train(Y_train::Vector{Vector{T}}, lags::Int=4) where T
    processed_Y = Vector{Vector{T}}(undef, length(Y_train))

    for (i, Y) in enumerate(Y_train)
        num_timepoints = length(Y)

        # Define valid time range (same as X kernelization)
        start_idx = lags + 1
        end_idx = num_timepoints - lags

        # Extract the matching time range
        processed_Y[i] = Y[start_idx:end_idx]
    end

    return processed_Y
end
# Function to create gamma vectors
function construct_gamma(Y::Vector{Matrix{Float64}}, time=10, factor=1)
    γ = Vector{Matrix{Float64}}(undef, length(Y))

    for i in 1:length(γ)
        trial_length = size(Y[i])[1]
        gamma_matrix = zeros(Float64, trial_length, 2)

        End_idx = trial_length

        
        gamma_matrix[1:time, 1] .= 0.0
        gamma_matrix[1:time, 2] .= 0.0

        gamma_matrix[101:time, 1] .= 1.0
        gamma_matrix[101:time, 2] .= 0.0

        # Middle part with no weights  (area we want to learn transition)
        gamma_matrix[time+1:End_idx-time*factor-1, 1] .= 0.0
        gamma_matrix[time+1:End_idx-time*factor-1, 2] .= 0.0

        # Second GLM weights (from trial end and back)
        gamma_matrix[End_idx-time*factor:end, 1] .= 0.0
        gamma_matrix[End_idx-time*factor:end, 2] .= 1.0

        # Construct trial dependent gamma_matrix
        γ[i] = gamma_matrix

    end

    return γ
end


# Define a heatmap function for convenience
function plot_heatmap(data, title_str)
    heatmap(data[:,1:300], color=:seismic, xlabel="Time", ylabel="Trial", title=title_str)
end

# Function to create gamma vectors
function construct_FC_gamma(Y::Vector{Vector{Float64}}, firstcontact::Vector{Int})
    γ = Vector{Matrix{Float64}}(undef, length(Y))

    for i in 1:length(γ)
        trial_length = size(Y[i])[1]
        gamma_matrix = zeros(Float64, trial_length, 2)

        End_idx = trial_length

        time = firstcontact[i]        

        # First GLM weights (first contact + some)
        gamma_matrix[time-40:time+10, 1] .= 1.0
        gamma_matrix[time-40:time+10, 2] .= 0.0

        # Second GLM weights (from trial end and back)
        gamma_matrix[End_idx-10:end, 2] .= 1.0

        # # Construct trial dependent gamma_matrix
        γ[i] = gamma_matrix

    end

    return γ
end

# Function to save model parameters to individual CSV files
function save_model_to_csv(m, ll_hist, folder_path)

    # Ensure the folder exists, create it if not
    isdir(folder_path) || mkpath(folder_path)

    # Save each vector/matrix as a separate CSV file in the specified folder
    CSV.write(joinpath(folder_path, "vector1_Beta1.csv"), DataFrame(m.B[1].regression.β, :auto), writeheader=false)
    CSV.write(joinpath(folder_path, "vector2_Beta2.csv"), DataFrame(m.B[2].regression.β, :auto), writeheader=false)
    CSV.write(joinpath(folder_path, "vector3_Sigma1.csv"), DataFrame(m.B[1].regression.Σ, :auto), writeheader=false)
    CSV.write(joinpath(folder_path, "vector4_Sigma2.csv"), DataFrame(m.B[2].regression.Σ, :auto), writeheader=false)
    CSV.write(joinpath(folder_path, "vector5_A.csv"), DataFrame(m.A, :auto), writeheader=false)
    CSV.write(joinpath(folder_path, "vector6_Pi.csv"), DataFrame(πₖ = m.πₖ), writeheader=false)
end


function z_score(x::Vector{})
    return (x.-minimum(x)) ./ (maximum(x).-minimum(x))
end

# Function to load model parameters from individual CSV files
function load_model_from_csv(folder_path)

    # Read each CSV file and store in variables
    vector1_β1 = CSV.read(joinpath(folder_path, "vector1_Beta1.csv"), DataFrame, header=false) |> Matrix
    vector2_β2 = CSV.read(joinpath(folder_path, "vector2_Beta2.csv"), DataFrame, header=false) |> Matrix
    vector3_Σ1 = CSV.read(joinpath(folder_path, "vector3_Sigma1.csv"), DataFrame, header=false) |> Matrix
    vector4_Σ2 = CSV.read(joinpath(folder_path, "vector4_Sigma2.csv"), DataFrame, header=false) |> Matrix
    vector5_A = CSV.read(joinpath(folder_path, "vector5_A.csv"), DataFrame, header=false) |> Matrix
    vector6_πₖ = CSV.read(joinpath(folder_path, "vector6_Pi.csv"), DataFrame, header=false) |> Matrix

    println(size(vector1_β1))

    model = SwitchingGaussianRegression(; num_features=size(vector1_β1)[1]-1, num_targets=size(vector1_β1)[2], K=2, λ=1.0)
    model.B[1].regression.β = vector1_β1
    model.B[2].regression.β = vector2_β2
    model.B[1].regression.Σ = vector3_Σ1
    model.B[2].regression.Σ = vector4_Σ2
    model.A = vector5_A

    π_vec = [vector6_πₖ[1]; vector6_πₖ[2]]
    model.πₖ = π_vec

    return model
end


function average_neural_plot(R1_data, R4_data)
    data = R1_data
    # Number of trials
    num_trials = size(R1_data, 1)
    # Initialize an array to store trial averages
    trialaves = []
    # Compute trial averages
    for trial in 1:num_trials
        trialave = mean(data[trial], dims=2)  # Compute mean across dimension 2
        push!(trialaves, trialave)  # Store the result
    end
    # Stack all trial averages into a matrix for averaging
    trialaves_matrix = hcat(trialaves...)
    # Compute the overall average across trials
    overall_average = mean(trialaves_matrix, dims=2)
    # Plot the overall average
    plot(overall_average, xlabel="Timepoints", ylabel="Average Value", title="Trial-Averaged Neural Data Plot", label="R1")




    # Number of trials
    data = R4_data
    num_trials = size(R4_data, 1)
    # Initialize an array to store trial averages
    trialaves = []
    # Compute trial averages
    for trial in 1:num_trials
        trialave = mean(data[trial], dims=2)  # Compute mean across dimension 2
        push!(trialaves, trialave)  # Store the result
    end
    # Stack all trial averages into a matrix for averaging
    trialaves_matrix = hcat(trialaves...)
    # Compute the overall average across trials
    overall_average = mean(trialaves_matrix, dims=2)
    # Plot the overall average
    plot!(overall_average, xlabel="Timepoints", ylabel="Average Value", label="R4")


end



function fit_switching_encoder!(
    model::SSD.HiddenMarkovModel,
    Y::Vector{<:Matrix{<:Real}},
    X::Union{Vector{<:Matrix{<:Real}},Nothing}=nothing;
    max_iters::Int=100,
    tol::Float64=1e-6,
)
    println("Fitting w EM")
    lls = [-Inf]
    data = X === nothing ? (Y,) : (X, Y)

    # Initialize log_likelihood
    log_likelihood = -Inf

    # Transform each matrix in each tuple to the correct orientation
    transposed_matrices = map(data_tuple -> Matrix.(transpose.(data_tuple)), data)
    zipped_matrices = collect(zip(transposed_matrices...))
    total_obs = sum(size(trial_mat[1], 1) for trial_mat in zipped_matrices)

    # initialize a vector of ForwardBackward storage and an aggregate storage
    FB_storage_vec = [SSD.initialize_forward_backward(model, size(trial_tuple[1],1)) for trial_tuple in zipped_matrices]
    Aggregate_FB_storage = SSD.initialize_forward_backward(model, total_obs)
    
    for iter in 1:max_iters
        println(iter)
        # broadcast estep!() to all storage structs
        output = SSD.estep!.(Ref(model), zipped_matrices, FB_storage_vec)

        # collect storage stucts into one struct for m step
        SSD.aggregate_forward_backward!(Aggregate_FB_storage, FB_storage_vec)

        # Calculate log_likelihood
        log_likelihood_current = SSD.logsumexp(Aggregate_FB_storage.α[:, end])
        push!(lls, log_likelihood_current)

        # Check for convergence
        if abs(log_likelihood_current - log_likelihood) < tol
            finish!(p)
            break
        else
            log_likelihood = log_likelihood_current
        end

        # Get data trial tuples stacked for mstep!()
        stacked_data = SSD.stack_tuples(zipped_matrices)

        # M_step
        SSD.mstep!(model, FB_storage_vec, Aggregate_FB_storage, stacked_data)
    end

    return lls
end