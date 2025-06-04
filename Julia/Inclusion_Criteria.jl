
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

base_path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\R14_Not_Filtered\\"
session_folders = filter(isdir, glob("*", base_path))

all_r2scores = DataFrame(Session = String[], R2_PC1 = Float64[], R2_PC2 = Float64[])

for session_path in session_folders
    # try
        session = splitpath(session_path)[end]
        session_save = replace(session, "-" => "_")
        println("sessin save: ", session_save)

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
        if prb == 1
            println("Probe 1 Processing -> Check this!")
            Probe1_R1, Probe2_R1, PCA_P1_R1, PCA_P2_R1, KP_R1, Jaw_R1 = load_data_encoder_noSVD(session_path, "R1")
            Probe1_R4, Probe2_R4, PCA_P1_R4, PCA_P2_R4, KP_R4, Jaw_R4 = load_data_encoder_noSVD(session_path, "R4")

            Probe1_R1_Cut, Probe2_R1_Cut, PCA_P1_R1_Cut, PCA_P2_R1_Cut, KP_R1_Cut, FCs_R1, SCs_R1, LRCs_R1, Tongue_mat_R1, Jaw_R1_Cut = load_data_encoder_cut_noSVD(session_path, "R1")
            Probe1_R4_Cut, Probe2_R4_Cut, PCA_P1_R4_Cut, PCA_P2_R4_Cut, KP_R4_Cut, FCs_R4, SCs_R4, LRCs_R4, Tongue_mat_R4, Jaw_R4_Cut = load_data_encoder_cut_noSVD(session_path, "R4")
        else
            println("Probe 2 Processing -> Check this!")
            Probe11_R1, Probe1_R1, PCA_P11_R1, PCA_P1_R1, KP_R1, Jaw_R1 = load_data_encoder_noSVD(session_path, "R1")
            Probe11_R4, Probe1_R4, PCA_P11_R4, PCA_P1_R4, KP_R4, Jaw_R4 = load_data_encoder_noSVD(session_path, "R4")

            Probe11_R1_Cut, Probe1_R1_Cut, PCA_P11_R1_Cut, PCA_P1_R1_Cut, KP_R1_Cut, FCs_R1, SCs_R1, LRCs_R1, Tongue_mat_R1, Jaw_R1_Cut = load_data_encoder_cut_noSVD(session_path, "R1")
            Probe11_R4_Cut, Probe1_R4_Cut, PCA_P11_R4_Cut, PCA_P1_R4_Cut, KP_R4_Cut, FCs_R4, SCs_R4, LRCs_R4, Tongue_mat_R4, Jaw_R4_Cut = load_data_encoder_cut_noSVD(session_path, "R4")
        end


        if occursin("TD3d", session)
            println("TD3d shift fixed")
            """
            Cut the kinematics up (in the case of the TD3d shift)
            """

            KP_R1 = [el[51:end, :] for el in KP_R1];
            KP_R4 = [el[51:end, :] for el in KP_R4];
            KP_R1_Cut = [el[51:end, :] for el in KP_R1_Cut];
            KP_R4_Cut = [el[51:end, :] for el in KP_R4_Cut];


            """
            Cut from the end of neural data to conserve sizes
            """
            PCA_P1_R1 = [el[1:end-50, :] for el in PCA_P1_R1];
            PCA_P1_R4 = [el[1:end-50, :] for el in PCA_P1_R4];
            PCA_P1_R1_Cut = [el[1:end-50, :] for el in PCA_P1_R1_Cut];
            PCA_P1_R4_Cut = [el[1:end-50, :] for el in PCA_P1_R4_Cut];
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
        Fit the encoder model up to the FC to test how good PC encoding is on engaged state
        """

        println("Prefitting Encoders")
        lags=4
        leads = 0
        start_time = 90
        dif = 100-lags;

        # Get enough data to create kernel and start at GC still
        # X_R1 = [X[100-lags:end,:] for X in Jaw_R1]
        # X_R4 = [X[100-lags:end,:] for X in Jaw_R4]

        X_R1 = [X[start_time-lags+1:end,:] for X in KP_R1]
        X_R4 = [X[start_time-lags+1:end,:] for X in KP_R4]


        Y_R1 = [Y[start_time-lags+1:end, 1:2] for Y in PCA_P1_R1]
        Y_R4 = [Y[start_time-lags+1:end, 1:2] for Y in PCA_P1_R4]

        X_R1_kernel = kernelize_window_features(X_R1)
        X_R4_kernel = kernelize_window_features(X_R4)

        Y_R1_trimmed = kernelize_window_features(Y_R1)
        Y_R4_trimmed = kernelize_window_features(Y_R4)


        # Y_R1_trimmed = trim_Y_train_past(Y_R1)
        # Y_R4_trimmed = trim_Y_train_past(Y_R4)

        FCs_R4 = FCs_R4 .- start_time
        FCs_R1 = FCs_R1 .- start_time

        LRCs_R4 = LRCs_R4 .- start_time
        LRCs_R1 = LRCs_R1 .- start_time

        FCs = cat(FCs_R1, FCs_R4, dims=2)
        LRCs= cat(LRCs_R1, LRCs_R4, dims=1)

        X_R1 = [X_R1_kernel[i][11:(FCs_R1[i]), :] for i in eachindex(X_R1_kernel)]
        X_R4 = [X_R4_kernel[i][11:(FCs_R4[i]), :] for i in eachindex(X_R4_kernel)]

        Y_R1 = [Y_R1_trimmed[i][11:(FCs_R1[i]), :] for i in eachindex(Y_R1_trimmed)]
        Y_R4 = [Y_R4_trimmed[i][11:(FCs_R4[i]), :] for i in eachindex(Y_R4_trimmed)]

        X_eng_tr = cat(X_R1, X_R4, dims=1)
        Y_eng_tr = cat(Y_R1, Y_R4, dims=1)

        # Prefit engaged model
        X_eng = vcat(X_eng_tr...)
        Y_eng = vcat(Y_eng_tr...)

        β_eng, Σ_eng = weighted_ridge_regression(X_eng, Y_eng, 0.01)



        num_trials = length(X_eng_tr)
        _, O = size(Y_R4_trimmed[1])  # O = number of PCs (output dims)



        X_bias = hcat(ones(size(X_eng,1)), X_eng)
        Y_PRED = X_bias * β_eng
        r2score = r2_score(Y_eng[:,1], Y_PRED[:,1])
        r2score2 = r2_score(Y_eng[:,2], Y_PRED[:,2])

        println(r2score)
        println(r2score2)

        if !isdir(joinpath("Results_Window_R14\\" *session_save))
            mkpath(joinpath("Results_Window_R14\\" *session_save))
        end

        push!(all_r2scores, (Session = session, R2_PC1 = r2score, R2_PC2 = r2score2))



    # catch err
    #     # Prepare error message as a string
    #     error_msg = IOBuffer()
    #     println(error_msg, "Error: ", err)
    #     println(error_msg, "Stacktrace:")
    #     for frame in stacktrace(catch_backtrace())
    #         println(error_msg, frame)
    #     end

    #     # # Save to file
    #     # error_file = joinpath("Results_Window", session_save, "KP2PC", "error_log.txt")
    #     # open(error_file, "w") do f
    #     #     write(f, String(take!(error_msg)))
    #     # end
    # end
end

CSV.write("Results_Window_R14\\All_R2_Means.csv", all_r2scores)

