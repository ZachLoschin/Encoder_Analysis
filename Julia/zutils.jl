
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

function plot_sliding_window_r2(R1_mean, R1_std, R4_mean, R4_std; window_size=10)
    max_points = 400
    num_windows = div(max_points, window_size)  # 30 windows for 300 points

    # Generate time axis from -1s to 2s
    time_axis = range(-1, stop=2, length=num_windows)

    # Define custom x-ticks
    xticks = -1:0.5:3

    p = plot(
        time_axis, R1_mean[1:num_windows];
        ribbon=R1_std[1:num_windows],
        label="R1",
        lw=2,
        color=:blue,
        xlabel="Time (s)",
        ylabel="R²",
        title="Sliding Window R² (Mean ± Std)",
        legend=:topright,
        fillalpha=0.2,
        xticks=xticks,
        ylim=(-2, 1)
    )

    plot!(
        time_axis, R4_mean[1:num_windows];
        ribbon=R4_std[1:num_windows],
        label="R4",
        lw=2,
        color=:red,
        fillalpha=0.2,
    )

    return p
end

function sliding_window_r2(X, Y, results_folder::String, condition::String)
    # Load best model parameters
    validation_df, best_lambda, best_train2, best_valr2, best_testr2, best_beta = load_results_from_csv(results_folder)

    # Kernelize and trim data
    lags = 4
    X_ready = kernelize_past_features(X, lags)
    Y_ready = trim_Y_train_past(Y, lags)

    # Setup model with best lambda and beta
    input_dim = size(X_ready[1], 2)
    output_dim = size(Y_ready[1], 2)
    model = SSD.GaussianRegressionEmission(
        input_dim=input_dim,
        output_dim=output_dim,
        include_intercept=true,
        λ=best_lambda,
    )
    model.β .= best_beta

    # Sliding window parameters
    window_size = 10
    trial_len = size(X_ready[1], 1)
    num_windows = fld(trial_len, window_size)

    r2_per_window = Float64[]  # To store mean R² for each window
    r2_per_window_std = Float64[]  # To store std of R² for each window

    for w in 1:num_windows
        idx_start = (w - 1) * window_size + 1
        idx_end = w * window_size

        r2_values_per_trial = Float64[]

        for i in 1:length(X_ready)
            X_trial = X_ready[i]
            Y_trial = Y_ready[i]

            # Skip trials that are too short
            if size(X_trial, 1) < idx_end
                continue
            end

            X_win = X_trial[idx_start:idx_end, :]
            Y_win = Y_trial[idx_start:idx_end, :]

            x_with_intercept = hcat(ones(size(X_win, 1)), X_win)
            Y_pred = x_with_intercept * model.β

            push!(r2_values_per_trial, r2_score(Y_win, Y_pred))
        end

        push!(r2_per_window, mean(r2_values_per_trial))
        push!(r2_per_window_std, std(r2_values_per_trial))
    end

    # Save results to CSV
    result_filename = joinpath(results_folder, "$(condition)_Sliding_Window.csv")
    results_df = DataFrame(Window=1:num_windows, Mean_R2=r2_per_window, Std_R2=r2_per_window_std)
    CSV.write(result_filename, results_df)

    println("R² results saved to: $result_filename")

    return r2_per_window, r2_per_window_std
end



# Define the function to perform the fitting loop
function fit_and_evaluate_dis(X_r1, X_r4, Y_r1, Y_r4, LRCs, λ_values, save_folder::String)
    # Check if the save folder exists, if not, create it
    if !isdir(save_folder)
        mkdir(save_folder)
    end

    # Chop the engaged state periods to GC to FC time
    X_cut = Vector{Matrix{Float64}}(undef, length(X))
    Y_cut = Vector{Matrix{Float64}}(undef, length(Y))

    for i in 1:length(X)
        len = LRCs[i]  # get the desired length
        X_cut[i] = X[i][len-10:len, :]
        Y_cut[i] = Y[i][len-10:len, :]
    end

    # Kernelize the X data and chop the Y data
    lags = 4
    X_ready = kernelize_past_features(X_cut, lags)
    Y_ready = trim_Y_train_past(Y_cut, lags)

    # Train / test/ val split
    X_train, Y_train, X_val, Y_val, X_test, Y_test = stratified_train_val_test_split(X_r1, Y_r1, X_r4, Y_r4)
    # X_train, Y_train, X_temp, Y_temp = train_test_split(X_ready, Y_ready, 0.8)
    # X_val, Y_val, X_test, Y_test = train_test_split(X_temp, Y_temp, 0.5)

    # Prepare data for training and validation
    X_train_cat = vcat(X_train...)
    y_train_cat = vcat(Y_train...)
    X_test_cat = vcat(X_test...)
    y_test_cat = vcat(Y_test...)
    x_val_cat = vcat(X_val...)
    y_val_cat = vcat(Y_val...)
    X_val_with_intercept = hcat(ones(size(x_val_cat, 1)), x_val_cat)  # Add ones to X_val

    # Initialize results list
    results = []  

    for λ in λ_values
        println("Evaluating λ: ", λ)

        # Initialize the model
        encoder_model = SSD.GaussianRegressionEmission(
            input_dim = size(X_train[1])[2],
            output_dim = size(Y_train[1])[2],
            include_intercept = true,
            λ = λ
        )

        # Fit the model
        SSD.fit!(encoder_model, X_train_cat, y_train_cat)

        # Predict on validation set
        y_val_pred = X_val_with_intercept * encoder_model.β

        # Compute R²
        r2_val = r2_score(y_val_cat, y_val_pred)

        # Save the result: store λ, r², and model parameters
        push!(results, (λ = λ, r2 = r2_val, β = copy(encoder_model.β)))
    end

    for result in results
        println("λ: ", result.λ, " - R²: ", result.r2)
    end

    # Find the best λ based on validation R²
    r2_values = [result.r2 for result in results]
    best_index = argmax(r2_values)
    best_beta = results[best_index].β
    best_lambda = results[best_index].λ

    # Initialize the model using the best betas from the validation set
    best_encoder_model = SSD.GaussianRegressionEmission(
        input_dim = size(X_train[1])[2],  # Include intercept dimension
        output_dim = size(Y_train[1])[2],
        include_intercept = true,
        λ = best_lambda  # Use the best lambda found during the sweep
    )

    # Set the best betas to the model
    best_encoder_model.β .= best_beta

    # Prepare the test data with intercept (add ones column)
    x_test_with_intercept = hcat(ones(size(X_test_cat, 1)), X_test_cat)

    # Predict on the test set
    y_test_pred = x_test_with_intercept * best_encoder_model.β  # Make predictions using best betas

    # Compute R² on the test set
    r2_test = r2_score(y_test_cat, y_test_pred)

    println("R² on test set: ", r2_test)

    # Save the results to CSV files
    save_results_to_csv(results, best_index, best_lambda, best_beta, r2_test, save_folder)

end


function fit_and_evaluate(X_r1, X_r4, Y_r1, Y_r4, FCs, λ_values, save_folder::String)
    # Check if the save folder exists, if not, create it
    if !isdir(save_folder)
        mkdir(save_folder)
    end

    # Chop to GC-to-FC time and kernelize all R1 and R4 data
    lags = 4
    function preprocess(X, Y, FCs)
        X_cut = Vector{Matrix{Float64}}(undef, length(X))
        Y_cut = Vector{Matrix{Float64}}(undef, length(Y))
        for i in 1:length(X)
            len = FCs[i]
            X_cut[i] = X[i][97:len, :]
            Y_cut[i] = Y[i][97:len, :]
        end
        X_ready = kernelize_past_features(X_cut, lags)
        Y_ready = trim_Y_train_past(Y_cut, lags)
        return X_ready, Y_ready
    end

    X_r1_ready, Y_r1_ready = preprocess(X_r1, Y_r1, FCs[1:length(X_r1)])
    X_r4_ready, Y_r4_ready = preprocess(X_r4, Y_r4, FCs[length(X_r1)+1:end])

    # Stratified train/val/test split
    X_train, Y_train, X_val, Y_val, X_test, Y_test = stratified_train_val_test_split(
        X_r1_ready, Y_r1_ready, X_r4_ready, Y_r4_ready)

    # Prepare data for training and validation
    X_train_cat = vcat(X_train...)
    y_train_cat = vcat(Y_train...)
    X_test_cat = vcat(X_test...)
    y_test_cat = vcat(Y_test...)
    x_val_cat = vcat(X_val...)
    y_val_cat = vcat(Y_val...)
    X_val_with_intercept = hcat(ones(size(x_val_cat, 1)), x_val_cat)

    # Initialize results list
    results = []  

    for λ in λ_values
        println("Evaluating λ: ", λ)

        # Initialize and fit the model
        encoder_model = SSD.GaussianRegressionEmission(
            input_dim = size(X_train[1], 2),
            output_dim = size(Y_train[1], 2),
            include_intercept = true,
            λ = λ
        )
        SSD.fit!(encoder_model, X_train_cat, y_train_cat)

        # Predict on validation set
        y_val_pred = X_val_with_intercept * encoder_model.β
        r2_val = r2_score(y_val_cat, y_val_pred)

        # Predict on training set
        x_train_with_intercept = hcat(ones(size(X_train_cat, 1)), X_train_cat)
        y_train_pred = x_train_with_intercept * encoder_model.β
        r2_train = r2_score(y_train_cat, y_train_pred)

        # Store result
        push!(results, (λ = λ, r2_train = r2_train, r2_val = r2_val, β = copy(encoder_model.β)))
    end

    # Pick best λ and set up best model
    r2_values = [result.r2_val for result in results]
    best_index = argmax(r2_values)
    best_beta = results[best_index].β
    best_lambda = results[best_index].λ

    best_encoder_model = SSD.GaussianRegressionEmission(
        input_dim = size(X_train[1], 2),
        output_dim = size(Y_train[1], 2),
        include_intercept = true,
        λ = best_lambda
    )
    best_encoder_model.β .= best_beta

    # Predict and evaluate on test set
    x_test_with_intercept = hcat(ones(size(X_test_cat, 1)), X_test_cat)
    y_test_pred = x_test_with_intercept * best_encoder_model.β
    r2_test = r2_score(y_test_cat, y_test_pred)

    println("R² on test set: ", r2_test)

    # Save results
    save_results_to_csv(results, best_index, best_lambda, best_beta, r2_test, save_folder)
end


function r2_score(y_true, y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true, dims=1)).^2)
    return 1 - (ss_res / ss_tot)
end

function load_results_from_csv(folder_path::String)
    # Load the validation results
    validation_file = joinpath(folder_path, "validation_results.csv")
    validation_df = CSV.read(validation_file, DataFrame)
    println("Loaded validation results from $validation_file")

    # Load the best validation results (lambda and R²s)
    best_results_file = joinpath(folder_path, "best_validation_results.csv")
    best_results_df = CSV.read(best_results_file, DataFrame)
    best_lambda = best_results_df.Best_Lambda[1]                # Best lambda
    best_r2_train = best_results_df.R2_Train_Best[1]            # Best training R²
    best_r2_val = best_results_df.R2_Validation_Best[1]         # Best validation R²
    best_r2_test = best_results_df.R2_Test_Best[1]              # Best test R²
    println("Loaded best validation and test results from $best_results_file")

    # Load the best betas and reshape them based on saved shape
    best_betas_file = joinpath(folder_path, "best_betas.csv")
    best_betas_df = CSV.read(best_betas_file, DataFrame)
    
    # Extract the flattened betas and the shape information
    flattened_betas = best_betas_df.flattened_betas
    rows = best_betas_df.rows[1]
    cols = best_betas_df.cols[1]

    # Reshape the flattened betas to their original dimensions
    best_beta = reshape(flattened_betas, rows, cols)
    println("Loaded and reshaped best betas from $best_betas_file")

    # Return all relevant results
    return validation_df, best_lambda, best_r2_train, best_r2_val, best_r2_test, best_beta
end


function stratified_train_val_test_split(X_r1, y_r1, X_r4, y_r4;
    train_ratio=0.7,
    val_ratio=0.15,
    seed=10)

    Random.seed!(seed)

    # Helper to split one trial type
    function split_type(X, y, train_ratio, val_ratio)
        n = length(X)
        perm = randperm(n)
        n_train = round(Int, train_ratio * n)
        n_val = round(Int, val_ratio * n)

        train_idx = perm[1:n_train]
        val_idx   = perm[n_train+1:n_train+n_val]
        test_idx  = perm[n_train+n_val+1:end]

        return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]
    end

    # Split both R1 and R4
    Xr1_train, yr1_train, Xr1_val, yr1_val, Xr1_test, yr1_test = split_type(X_r1, y_r1, train_ratio, val_ratio)
    Xr4_train, yr4_train, Xr4_val, yr4_val, Xr4_test, yr4_test = split_type(X_r4, y_r4, train_ratio, val_ratio)

    # Concatenate R1 and R4 to make final sets
    X_train = vcat(Xr1_train, Xr4_train)
    Y_train = vcat(yr1_train, yr4_train)

    X_val = vcat(Xr1_val, Xr4_val)
    Y_val = vcat(yr1_val, yr4_val)

    X_test = vcat(Xr1_test, Xr4_test)
    Y_test = vcat(yr1_test, yr4_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test
end




function save_results_to_csv(results, best_index, best_lambda, best_beta, r2_test, folder_path::String)
    # Check if the folder exists; if not, create it
    if !isdir(folder_path)
        mkpath(folder_path)
    end

    # Create a DataFrame with λ, training R², and validation R²
    validation_df = DataFrame(
        λ = [result.λ for result in results],
        R2_Train = [result.r2_train for result in results],
        R2_Validation = [result.r2_val for result in results]
    )

    # Save the training/validation results to a CSV file
    validation_file = joinpath(folder_path, "validation_results.csv")
    CSV.write(validation_file, validation_df)
    println("Validation results saved to $validation_file")

    # Create a DataFrame for the best lambda and its R²s
    best_results_df = DataFrame(
        Best_Lambda = [best_lambda],
        R2_Train_Best = [results[best_index].r2_train],
        R2_Validation_Best = [results[best_index].r2_val],
        R2_Test_Best = [r2_test]
    )

    # Save the best lambda results to a CSV file
    best_results_file = joinpath(folder_path, "best_validation_results.csv")
    CSV.write(best_results_file, best_results_df)
    println("Best validation/test results saved to $best_results_file")

    # Flatten the best_beta matrix and store its shape
    best_betas_flat = vec(best_beta)
    beta_shape = size(best_beta)

    # Create a DataFrame for the flattened betas
    best_betas_df = DataFrame(
        flattened_betas = best_betas_flat,
        rows = beta_shape[1],  # Number of rows in best_beta
        cols = beta_shape[2]   # Number of columns in best_beta
    )

    # Save the best betas to a CSV file
    best_betas_file = joinpath(folder_path, "best_betas.csv")
    CSV.write(best_betas_file, best_betas_df)
    println("Best betas saved to $best_betas_file")
end



# Helper function to split matrix into chunks
function split_into_trials(mat::Matrix, trial_length::Int)
    n_trials = size(mat, 1) ÷ trial_length
    [mat[((i-1)*trial_length + 1):(i*trial_length), :] for i in 1:n_trials]
end


using Random  # Make sure to import the Random module

function train_test_split(X::Vector{<:Matrix{<:Real}}, 
                          y::Vector{<:Matrix{<:Real}}, 
                          split_ratio::Float64=0.8,
                          seed::Int=42)  # Default seed value for reproducibility
    n = length(X)  # Number of samples

    # Set the random seed to ensure the permutation is the same every time
    Random.seed!(seed)

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

function load_data_encoder(path, condition)
    # Helper function to chunk matrix into 600-timepoint segments
    function chunk_matrix(mat, chunk_size)
        num_chunks = size(mat, 1) ÷ chunk_size
        [mat[(i-1)*chunk_size + 1 : i*chunk_size, :] for i in 1:num_chunks]
    end

    # Construct file paths
    probe1_path = path * "Probe1_" * condition * "_Uncut.csv"
    probe2_path = path * "Probe2_" * condition * "_Uncut.csv"

    PCA_P1_path = path * "PCA_Probe1_" * condition * "_Uncut.csv"
    PCA_P2_path = path * "PCA_Probe2_" * condition * "_Uncut.csv"

    SVD_path = path * "SVD_Feats_" * condition * "_Uncut.csv"
    KP_path = path * "Keypoint_Feats_" * condition * "_Uncut.csv"

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

function load_data_encoder_cut(path, condition)
    # Construct file paths
    probe1_path = path * "Probe1_" * condition * "_Cut.csv"
    probe2_path = path * "Probe2_" * condition * "_Cut.csv"
    PCA_P1_path = path * "PCA_Probe1_" * condition * "_Cut.csv"
    PCA_P2_path = path * "PCA_Probe2_" * condition * "_Cut.csv"
    SVD_path = path * "SVD_Feats_" * condition * "_Cut.csv"
    KP_path = path * "Keypoint_Feats_" * condition * "_Cut.csv"
    FCs_path = path * "FCs_" * condition * ".csv"
    LRCs_path = path * "LRCs_" * condition * ".csv"
    Tongue_path = path * "Tongue_" * condition * ".csv"

    # Load the data into matrices
    probe1_mat = Matrix(CSV.read(probe1_path, DataFrame; header=false))
    probe2_mat = Matrix(CSV.read(probe2_path, DataFrame; header=false))
    PCA_P1_mat = Matrix(CSV.read(PCA_P1_path, DataFrame; header=false))
    PCA_P2_mat = Matrix(CSV.read(PCA_P2_path, DataFrame; header=false))
    SVD_mat = Matrix(CSV.read(SVD_path, DataFrame; header=false))  # already downsampled to 100Hz
    KP_mat = Matrix(CSV.read(KP_path, DataFrame; header=false))
    FCs_mat = Matrix(CSV.read(FCs_path, DataFrame; header=false))
    LRCs = vec(Matrix(CSV.read(LRCs_path, DataFrame; header=false)))  # 1D vector of rounded trial lengths
    Tongue_mat = Matrix(CSV.read(Tongue_path, DataFrame; header=false))

    # Chunking function
    function chunk_matrix(mat, lengths)
        chunks = Vector{Matrix{Float64}}()
        start_idx = 1
        for len in lengths
            stop_idx = start_idx + len - 1
            push!(chunks, mat[start_idx:stop_idx, :])
            start_idx = stop_idx + 1
        end
        return chunks
    end

    # Chunk all data at 100Hz
    probe1_chunks = chunk_matrix(probe1_mat, LRCs)
    probe2_chunks = chunk_matrix(probe2_mat, LRCs)
    PCA_P1_chunks = chunk_matrix(PCA_P1_mat, LRCs)
    PCA_P2_chunks = chunk_matrix(PCA_P2_mat, LRCs)
    KP_chunks = chunk_matrix(KP_mat, LRCs)
    SVD_chunks = chunk_matrix(SVD_mat, LRCs)

    return probe1_chunks, probe2_chunks, PCA_P1_chunks, PCA_P2_chunks,
           SVD_chunks, KP_chunks, FCs_mat, LRCs, Tongue_mat
end

function ave_vector(X)
    meanX = mean(hcat(X...), dims=2)
    return meanX
end

function average_PCs(PCA_Obj)
    PCA_stack = cat(PCA_Obj...; dims=3)
    trial_average = mean(PCA_stack; dims=3)
    trial_average = dropdims(trial_average; dims=3)
    return trial_average
end

function load_data(path, condition, TW)
    
    # Neural probe data paths -> check metadata.txt or excel sheet for probe locations
    probe1_path = path*"Probe1_"*condition*".csv"
    # probe1_path_PCA = path*"Probe1_"*condition*"_PC"*".csv"
    probe2_path = path*"Probe2_"*condition*".csv"

    # Kinematic feats paths -> tongue path has tongue length, jaw_feats has jaw pos and jaw vel 
    kin_path = path*"Jaw_"*condition*".csv"
    tongue_path = path*"Tongue_"*condition*".csv"
    # PCA_feats_path = path*"PCA_feats_"*condition*".csv"
    # PCA_feats_path_uncut = path*"PCA_feats_uncut_"*condition*".csv"
    Jaw_feats_path = path*"Jaw_feats_"*condition*".csv"
    
    # Trial information paths -> trial lengths and first lick contact paths
    trial_lpath = path*"Trial_End_"*condition*".csv"
    first_lick_path = path*"First_Contact_"*condition*".csv"
    
    
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


function load_data_uncut(path, condition, TW, trial_length)

    # Neural probe data paths -> check metadata.txt or excel sheet for probe locations
    probe1_path = path*"Probe1_uncut_"*condition*".csv"
    probe2_path = path*"Probe2_uncut_"*condition*".csv"
    # PCA_feats_path = path*"PCA_feats_uncut_"*condition*".csv"
    tongue_path = path*"Tongue_uncut_"*condition*".csv"
    Jaw_feats_path = path*"Jaw_feats_"*condition*".csv"

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

        # Secondition GLM weights (from trial end and back)
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

        # Secondition GLM weights (from trial end and back)
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