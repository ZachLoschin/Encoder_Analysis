
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

base_path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\R14\\"
session_folders = filter(isdir, glob("*", base_path))

for session_path in session_folders
    try
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


        """
        R1 PC trajectories
        """

        n_trials = length(PCA_P1_R1_Cut)
        n_cols = ceil(Int, sqrt(n_trials))
        n_rows = ceil(Int, n_trials / n_cols)

        plt = plot(layout = (n_rows, n_cols), size = (n_cols * 300, n_rows * 300))

        for trial in 1:n_trials
            plot!(plt[trial],
                PCA_P1_R1[trial][:,1],
                label = "",
                ylim = (-5, 20),
                xticks = false,
                title = "Trial $trial")
            plot!(plt[trial], PCA_P1_R1[trial][:,2], label = "")
        end

        # Make sure path exists
        save_dir = joinpath("Results_Window_R14\\" * session_save)
        if !isdir(save_dir)
            mkpath(save_dir)
        end

        savepath = joinpath(save_dir, "PC_Trajectories_R1.png")
        savefig(plt, savepath)


        """
        R4 PC trajectories
        """

        n_trials = length(PCA_P1_R4_Cut)
        n_cols = ceil(Int, sqrt(n_trials))
        n_rows = ceil(Int, n_trials / n_cols)

        plt = plot(layout = (n_rows, n_cols), size = (n_cols * 300, n_rows * 300))

        for trial in 1:n_trials
            plot!(plt[trial],
                PCA_P1_R4[trial][:,1],
                label = "",
                ylim = (-5, 20),
                xticks = false,
                title = "Trial $trial")
            plot!(plt[trial], PCA_P1_R4[trial][:,2], label = "")
        end
        display(plt)

        # Make sure path exists
        save_dir = joinpath("Results_Window_R14\\" * session_save)
        if !isdir(save_dir)
            mkpath(save_dir)
        end

        savepath = joinpath(save_dir, "PC_Trajectories_R4.png")
        savefig(plt, savepath)



    catch err
        # Prepare error message as a string
        error_msg = IOBuffer()
        println(error_msg, "Error: ", err)
        println(error_msg, "Stacktrace:")
        for frame in stacktrace(catch_backtrace())
            println(error_msg, frame)
        end
    end
end


