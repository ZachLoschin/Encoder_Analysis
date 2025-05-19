using Pkg
# Pkg.activate("C:\\Research\\Encoder_Modeling\\Encoder_Analysis")

using Random
using StateSpaceDynamics
using Distributions
using Plots
using StatsBase
using CSV
using DataFrames
using LinearAlgebra
using MultivariateStats
using Glob
using Dates
# include("C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Julia\\Zutils.jl")
# include(".\\Julia\\Zutils.jl")
include(".\\Zutils.jl")
# using StatsPlots

# For testing and debugging
Random.seed!(1234);

const SSD = StateSpaceDynamics

base_path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\R16\\"
session_folders = filter(isdir, glob("*", base_path))

for session_path in session_folders
    try
        session = splitpath(session_path)[end]
        session_save = replace(session, "-" => "_")
        println("session save: ", session_save)

        # Automatically detect probe number from folder name
        if endswith(session, "_P1")
            prb = 1
        elseif endswith(session, "_P2")
            prb = 2
        else
            println("Skipping folder without probe suffix: $session")
            continue
        end
        println("Processing session: $session with Probe $prb")
        session_path = session_path * "\\"
        session_path = replace(session_path, "-" => "_")

        println(session_path)
        if prb == 1
            println("Probe 1 Processing -> Check this!")
            Probe1_R1, Probe2_R1, PCA_P1_R1, PCA_P2_R1, KP_R1, Jaw_R1 = load_data_encoder_noSVD(session_path, "R1")
            Probe1_R4, Probe2_R4, PCA_P1_R4, PCA_P2_R4, KP_R4, Jaw_R4 = load_data_encoder_noSVD(session_path, "R16")

            Probe1_R1_Cut, Probe2_R1_Cut, PCA_P1_R1_Cut, PCA_P2_R1_Cut, KP_R1_Cut, FCs_R1, SCs_R1, LRCs_R1, Tongue_mat_R1, Jaw_R1_Cut = load_data_encoder_cut_noSVD(session_path, "R1")
            Probe1_R4_Cut, Probe2_R4_Cut, PCA_P1_R4_Cut, PCA_P2_R4_Cut, KP_R4_Cut, FCs_R4, SCs_R4, LRCs_R4, Tongue_mat_R4, Jaw_R4_Cut = load_data_encoder_cut_noSVD(session_path, "R16")
        else
            println("Probe 2 Processing -> Check this!")
            Probe11_R1, Probe1_R1, PCA_P11_R1, PCA_P1_R1, KP_R1, Jaw_R1 = load_data_encoder_noSVD(session_path, "R1")
            Probe11_R4, Probe1_R4, PCA_P11_R4, PCA_P1_R4, KP_R4, Jaw_R4 = load_data_encoder_noSVD(session_path, "R16")

            Probe11_R1_Cut, Probe1_R1_Cut, PCA_P11_R1_Cut, PCA_P1_R1_Cut, KP_R1_Cut, FCs_R1, SCs_R1, LRCs_R1, Tongue_mat_R1, Jaw_R1_Cut = load_data_encoder_cut_noSVD(session_path, "R1")
            Probe11_R4_Cut, Probe1_R4_Cut, PCA_P11_R4_Cut, PCA_P1_R4_Cut, KP_R4_Cut, FCs_R4, SCs_R4, LRCs_R4, Tongue_mat_R4, Jaw_R4_Cut = load_data_encoder_cut_noSVD(session_path, "R16")
        end

        # Assuming KP_R1 is a vector of matrices
        for i in 1:length(KP_R1)
            # Replace NaN values in each matrix with 0
            KP_R1[i] .= replace(KP_R1[i], NaN => 0.0)
        end

        for i in 1:length(KP_R4)
            # Replace NaN values in each matrix with 0
            KP_R4[i] .= replace(KP_R4[i], NaN => 0.0)
        end

        # Assuming KP_R1 is a vector of matrices
        for i in 1:length(KP_R1_Cut)
            # Replace NaN values in each matrix with 0
            KP_R1_Cut[i] .= replace(KP_R1_Cut[i], NaN => 0.0)
        end

        for i in 1:length(KP_R4_Cut)
            # Replace NaN values in each matrix with 0
            KP_R4_Cut[i] .= replace(KP_R4_Cut[i], NaN => 0.0)
        end

        """
        Prefit the encoder model
        """

        println("Prefitting Encoder")
        lags=4
        leads = 0
        start_time = 90
        dif = 100-lags;


        X_R1 = [X[start_time-lags+1:end,:] for X in KP_R1]
        X_R4 = [X[start_time-lags+1:end,:] for X in KP_R4]


        Y_R1 = [Y[start_time-lags+1:end, :] for Y in PCA_P1_R1]
        Y_R4 = [Y[start_time-lags+1:end, :] for Y in PCA_P1_R4]

        X_R1_kernel = kernelize_window_features(X_R1)
        X_R4_kernel = kernelize_window_features(X_R4)

        Y_R1_trimmed = kernelize_window_features(Y_R1)
        Y_R4_trimmed = kernelize_window_features(Y_R4)

        FCs_R4 = FCs_R4 .- start_time
        FCs_R1 = FCs_R1 .- start_time

        LRCs_R4 = LRCs_R4 .- start_time
        LRCs_R1 = LRCs_R1 .- start_time

        FCs = cat(FCs_R1, FCs_R4, dims=2)
        LRCs= cat(LRCs_R1, LRCs_R4, dims=1)

        X_R1 = [X_R1_kernel[i][(FCs_R1[i]-3):(FCs_R1[i]), :] for i in eachindex(X_R1_kernel)]
        X_R4 = [X_R4_kernel[i][(FCs_R4[i]-3):(FCs_R4[i]+10), :] for i in eachindex(X_R4_kernel)]

        Y_R1 = [Y_R1_trimmed[i][(FCs_R1[i]-3):(FCs_R1[i]), :] for i in eachindex(Y_R1_trimmed)]
        Y_R4 = [Y_R4_trimmed[i][(FCs_R4[i]-3):(FCs_R4[i]+10), :] for i in eachindex(Y_R4_trimmed)]

        X_eng = cat(X_R1, X_R4, dims=1)
        Y_eng = cat(Y_R1, Y_R4, dims=1)

        # Prefit engaged model
        X_eng = vcat(X_eng...)
        Y_eng = vcat(Y_eng...)

        β_eng, Σ_eng = weighted_ridge_regression(X_eng, Y_eng, 0.01)


        """
        Set up the switching encoder model
        """

        println("Setting up switching model")

        X_R1 = [X[start_time-lags:end,:] for X in KP_R1_Cut]
        X_R4 = [X[start_time-lags:end,:] for X in KP_R4_Cut]
        X = cat(X_R1, X_R4, dims=1)
        Y = cat(PCA_P1_R1_Cut, PCA_P1_R4_Cut, dims=1)
        Y = [y[start_time-lags:end, :] for y in Y]

        X_kern = kernelize_window_features(X)
        Y_trim = kernelize_window_features(Y)

        X_ready = permutedims.(X_kern)
        Y_ready = permutedims.(Y_trim)

        # Initialize the Gaussian HMM-GLM
        model = SwitchingGaussianRegression(;K=1, input_dim=size(X_ready[1])[1], output_dim=size(Y_ready[1])[1], include_intercept=true)

        model.B[1].β = β_eng
        model.B[1].Σ = Σ_eng

        lls = fit_custom!(model, Y_ready, X_ready, max_iters=100)

        plot(lls)
        title!("Training Log-Likelihood")
        xlabel!("EM Iteration")
        ylabel!("Log-Likelihood")


        """
        Plot the trial averaged inference
        """

        println("Calculating average inference")
        X_R1 = [X[start_time-lags:300,:] for X in KP_R1]
        X_R4 = [X[start_time-lags:300,:] for X in KP_R4]

        Y_R1 = [Y[start_time-lags:300,:] for Y in PCA_P1_R1]
        Y_R4 = [Y[start_time-lags:300, :] for Y in PCA_P1_R4]

        X_R1_kernel = kernelize_window_features(X_R1)
        X_R4_kernel = kernelize_window_features(X_R4)

        Y_R1_trimmed = kernelize_window_features(Y_R1)
        Y_R4_trimmed = kernelize_window_features(Y_R4)

        # Y_R1_trimmed = trim_Y_train_past(Y_R1)
        # Y_R4_trimmed = trim_Y_train_past(Y_R4)

        YY = permutedims.(Y_R1_trimmed)
        XX = permutedims.(X_R1_kernel)

        YY_R4 = permutedims.(Y_R4_trimmed)
        XX_R4 = permutedims.(X_R4_kernel)


        FB_R1 = label_data(model, YY, XX);
        FB_R4 = label_data(model, YY_R4, XX_R4);

        V1 = SSD.viterbi(model, YY, XX);
        V4 = SSD.viterbi(model, YY_R4, XX_R4);

        # Extract γ[1, :] for each K in OO
        γ_vectors_R1 = [FB_R1[K].γ[1, :] for K in eachindex(FB_R1)]
        γ_mean_R1 = mean(exp.(hcat(γ_vectors_R1...)), dims=2)

        γ_vectors_R4 = [FB_R4[K].γ[1, :] for K in eachindex(FB_R4)]
        γ_mean_R4 = mean(exp.(hcat(γ_vectors_R4...)), dims=2)

        plot(γ_mean_R1; label="R1")
        plot!(γ_mean_R4; label="R4")
        title!("State Inference")
        ylabel!("State Probability")
        xlabel!("Time")


        println("Calculating average inference")
        X_R1 = [X[start_time-lags:300,:] for X in KP_R1]
        X_R4 = [X[start_time-lags:300,:] for X in KP_R4]

        Y_R1 = [Y[start_time-lags:300,:] for Y in PCA_P1_R1]
        Y_R4 = [Y[start_time-lags:300, :] for Y in PCA_P1_R4]

        X_R1_kernel = kernelize_window_features(X_R1)
        X_R4_kernel = kernelize_window_features(X_R4)

        Y_R1_trimmed = kernelize_window_features(Y_R1)
        Y_R4_trimmed = kernelize_window_features(Y_R4)

        X_check = cat(X_R1_kernel, X_R4_kernel, dims=1)
        Y_check = cat(Y_R1_trimmed, Y_R4_trimmed, dims=1)

        X_reg = vcat(X_check...)
        Y_true = vcat(Y_check...)

        X_bias = hcat(ones(size(X_reg,1)), X_reg)
        Y_pred = X_bias * β_eng

        r2_scores = []

        for j = 1:10
            r2 = r2_score(Y_true[:,j], Y_pred[:,j])
            push!(r2_scores, r2)
        end

        println("HERHEHERHERHE")
        if !isdir(joinpath("Results_Window_R16\\" *session_save))
            println("Making dir")
            mkpath(joinpath("Results_Window_R16\\" *session_save))
        end

        r2_df = DataFrame(r2_scores', :auto)
        CSV.write(joinpath("Results_Window_R16\\" *session_save, "Single_State_PC_Encoding_NoInit.csv"), r2_df; header=false)

 catch err
        # Prepare error message as a string
        error_msg = IOBuffer()
        println(error_msg, "Error: ", err)
        println(error_msg, "Stacktrace:")
        for frame in stacktrace(catch_backtrace())
            println(error_msg, frame)
        end

        # # Save to file
        # error_file = joinpath("Results_Window", session_save, "KP2PC", "error_log.txt")
        # open(error_file, "w") do f
        #     write(f, String(take!(error_msg)))
        # end
    end
end
