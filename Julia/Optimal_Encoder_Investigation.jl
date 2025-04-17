
using Pkg
Pkg.activate("C:\\Research\\Encoder_Modeling\\Encoder_Analysis")

using Random
using StateSpaceDynamics
using Distributions
using Plots
using StatsBase
using CSV
using DataFrames
using LinearAlgebra
using MultivariateStats

using Dates
include("C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Julia\\Zutils.jl")
# using StatsPlots

# For testing and debugging
Random.seed!(1234);

const SSD = StateSpaceDynamics

# path = "C:\\Users\\zachl\\OneDrive\\BU_YEAR1\\Research\\Tudor_Data\\Disengagement_Analysis_2025\\preprocessed_data\\TD13d_2024-11-13\\";  # Probe 2
path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\TD13d_2024-11-12\\";  # Probe 1

Probe1_R1, Probe2_R1, PCA_P1_R1, PCA_P2_R1, SVD_R1, KP_R1 = load_data_encoder(path, "R1");
Probe1_R4, Probe2_R4, PCA_P1_R4, PCA_P2_R4, SVD_R4, KP_R4 = load_data_encoder(path, "R4");

Probe1_R1_Cut, Probe2_R1_Cut, PCA_P1_R1_Cut, PCA_P2_R1_Cut, SVD_R1_Cut, KP_R1_Cut, FCs_R1, LRCs_R1, Tongue_mat_R1  = load_data_encoder_cut(path, "R1");
Probe1_R4_Cut, Probe2_R4_Cut, PCA_P1_R4_Cut, PCA_P2_R4_Cut, SVD_R4_Cut, KP_R4_Cut, FCs_R4, LRCs_R4, Tongue_mat_R4  = load_data_encoder_cut(path, "R4");



"""
Visualizations to sanity check data preprocessing and import
"""

# Check uncut neural data and features
P1_R1_Ave = ave_vector(Probe1_R1)
P1_R4_Ave = ave_vector(Probe1_R4)
plot(P1_R1_Ave, label="R1 Uncut Ave")
plot!(P1_R4_Ave, label="R4 Uncut Ave")
title!("Uncut Population Average")

# Stack trials into a 3D array: 600 (time) x 12 (PCs) x N_trials
PCA_P1_R1_Ave = average_PCs(PCA_P1_R1)
plot(PCA_P1_R1_Ave)
title!("R1 Neural PCs")

PCA_P1_R4_Ave = average_PCs(PCA_P1_R4)
plot(PCA_P1_R4_Ave)
title!("R4 Neural PCs")

# Check the SVD features
SVD_R1_Ave = average_PCs(SVD_R1)
plot(SVD_R1_Ave[:,1:5])
title!("R1 Motion SVDs")
plot(SVD_R1_Ave[:,51:55])
title!("R1 Movie SVDs")

SVD_R4_Ave = average_PCs(SVD_R4)
plot(SVD_R4_Ave[:,1:5])
title!("R4 Motion SVDs")
plot(SVD_R4_Ave[:,51:55])
title!("R4 Movie SVDs")



"""
Section for testing SVD features to nerual FRs
"""

SVD_R1_selected = [hcat(x[:, 1:20], x[:, 51:70]) for x in SVD_R1]
SVD_R4_selected = [hcat(x[:, 1:20], x[:, 51:70]) for x in SVD_R4]

X = cat(SVD_R1_selected, SVD_R4_selected, dims=1)
Y = cat(Probe1_R1, Probe1_R4, dims=1)
FCs = cat(FCs_R1, FCs_R4, dims=2)
LRCs= cat(LRCs_R1, LRCs_R4, dims=1)
# λ_values = [0.0001, 0.001, 0.01, 0.1, 0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]
λ_values = [0.01]
fit_and_evaluate(SVD_R1_selected, SVD_R4_selected, Probe1_R1, Probe1_R4, FCs, LRCs, λ_values, "Results\\TD13d_11_12\\SVD_To_Neural_FRs")

"""
Section for testing SVD features to nerual PCs
"""

X = cat(SVD_R1, SVD_R4, dims=1)
Y = cat(PCA_P1_R1, PCA_P1_R4, dims=1)
fit_and_evaluate(SVD_R1_selected, SVD_R4_selected, PCA_P1_R1, PCA_P1_R4 , FCs, LRCs, λ_values, "Results\\TD13d_11_12\\SVD_To_Neural_PCs")

"""
Section for testing KP Features to nerual FRs
"""

# Assuming KP_R1 is a vector of matrices
for i in 1:length(KP_R1)
    # Replace NaN values in each matrix with 0
    KP_R1[i] .= replace(KP_R1[i], NaN => 0.0)
end

for i in 1:length(KP_R4)
    # Replace NaN values in each matrix with 0
    KP_R4[i] .= replace(KP_R4[i], NaN => 0.0)
end


X = cat(KP_R1, KP_R4, dims=1)
Y = cat(Probe1_R1, Probe1_R4, dims=1)
fit_and_evaluate(KP_R1, KP_R4, Probe1_R1, Probe1_R4, FCs, LRCs, λ_values, "Results\\TD13d_11_12\\KP_To_Neural_FRs")


"""
Section for testing KP Features to nerual PCs
"""

X = cat(KP_R1, KP_R4, dims=1)
Y = cat(PCA_P1_R1, PCA_P1_R4, dims=1)
fit_and_evaluate(KP_R1, KP_R4, PCA_P1_R1, PCA_P1_R4, FCs, LRCs, λ_values, "Results\\TD13d_11_12\\KP_To_Neural_PCs")


"""
********************************* Sliding Window R2 Results SVD -> FRs
"""
results_folder = "Results/TD13d_11_12/SVD_To_Neural_FRs"
R1_mean, R1_std = sliding_window_r2(SVD_R1_selected, Probe1_R1, results_folder, "R1")
R4_mean, R4_std = sliding_window_r2(SVD_R4_selected, Probe1_R4, results_folder, "R4")

p = plot_sliding_window_r2(R1_mean, R1_std, R4_mean, R4_std)
savefig(p, joinpath(results_folder, "Sliding_Window_R2.png"))

"""
Sliding Window R2 Results SVD -> PCs
"""
results_folder = "Results/TD13d_11_12/SVD_To_Neural_PCs"
R1_mean, R1_std = sliding_window_r2(SVD_R1_selected, PCA_P1_R1, results_folder, "R1")
R4_mean, R4_std = sliding_window_r2(SVD_R4_selected, PCA_P1_R4, results_folder, "R4")

p = plot_sliding_window_r2(R1_mean, R1_std, R4_mean, R4_std)
savefig(p, joinpath(results_folder, "Sliding_Window_R2.png"))

"""
Sliding Window R2 Results KP-> PCs
"""
results_folder = "Results/TD13d_11_12/KP_To_Neural_PCs"
R1_mean, R1_std = sliding_window_r2(KP_R1, PCA_P1_R1, results_folder, "R1")
R4_mean, R4_std = sliding_window_r2(KP_R4, PCA_P1_R4, results_folder, "R4")

p = plot_sliding_window_r2(R1_mean, R1_std, R4_mean, R4_std)
savefig(p, joinpath(results_folder, "Sliding_Window_R2.png"))

"""
Sliding Window R2 Results KP-> FR
"""
results_folder = "Results/TD13d_11_12/KP_To_Neural_FRs"
R1_mean, R1_std = sliding_window_r2(KP_R1, Probe1_R1, results_folder, "R1")
R4_mean, R4_std = sliding_window_r2(KP_R4, Probe1_R4, results_folder, "R4")

p = plot_sliding_window_r2(R1_mean, R1_std, R4_mean, R4_std)
savefig(p, joinpath(results_folder, "Sliding_Window_R2.png"))







_, λ, r2_train, r2_val, r2_test, r2_fullcut, _ = load_results_from_csv("Results\\TD13d_11_12\\SVD_To_Neural_FRs")
plot_r2_summary(r2_train, r2_val, r2_test; best_r2_fullcut=r2_fullcut)




function plot_r2_summary(best_r2_train::Float64, best_r2_val::Float64, best_r2_test::Float64; best_r2_fullcut::Union{Nothing, Float64}=nothing)
    labels = ["Train", "Validation", "Test"]
    values = [best_r2_train, best_r2_val, best_r2_test]

    if best_r2_fullcut !== nothing
        push!(labels, "Full Cut")
        push!(values, best_r2_fullcut)
    end

    bar(labels, values;
        legend=false,
        ylabel="R²",
        title="Best R² Scores",
        ylim=(0,1),
        bar_width=0.6,
        color=:steelblue,
        framestyle=:box)
end




# """
# ******* DISENGAGED ENCODER FITTING BELOW *******
# """


# """
# Section for testing SVD features to nerual FRs
# """

# SVD_R1_selected = [hcat(x[:, 1:20], x[:, 51:70]) for x in SVD_R1]
# SVD_R4_selected = [hcat(x[:, 1:20], x[:, 51:70]) for x in SVD_R4]

# X = cat(SVD_R1_selected, SVD_R4_selected, dims=1)
# Y = cat(Probe1_R1, Probe1_R4, dims=1)
# LRCs= cat(LRCs_R1, LRCs_R4, dims=1)
# λ_values = [0.0001, 0.0, 1000.0]
# fit_and_evaluate_dis(X, Y, LRCs, λ_values, "Results\\TD13d_11_12\\Dis_SVD_To_Neural_FRs")


# results_folder = "Results\\TD13d_11_12\\Dis_SVD_To_Neural_FRs"
# R1_mean, R1_std = sliding_window_r2(SVD_R1_selected, Probe1_R1, results_folder, "R1")
# R4_mean, R4_std = sliding_window_r2(SVD_R4_selected, Probe1_R4, results_folder, "R4")

# p = plot_sliding_window_r2(R1_mean, R1_std, R4_mean, R4_std)
# savefig(p, joinpath(results_folder, "Sliding_Window_R2.png"))


# """
# Section for testing SVD features to nerual FRs
# """
# X = cat(SVD_R1_selected, SVD_R4_selected, dims=1)
# Y = cat(PCA_P1_R1, PCA_P1_R4, dims=1)
# fit_and_evaluate_dis(X, Y, LRCs, λ_values, "Results\\TD13d_11_12\\Dis_SVD_To_Neural_PCs")



# results_folder = "Results/TD13d_11_12/Dis_SVD_To_Neural_PCs"
# R1_mean, R1_std = sliding_window_r2(SVD_R1_selected, PCA_P1_R1, results_folder, "R1")
# R4_mean, R4_std = sliding_window_r2(SVD_R4_selected, PCA_P1_R4, results_folder, "R4")

# p = plot_sliding_window_r2(R1_mean, R1_std, R4_mean, R4_std)
# savefig(p, joinpath(results_folder, "Sliding_Window_R2.png"))



# """
# Section for testing KP Features to nerual FRs
# """

# # Assuming KP_R1 is a vector of matrices
# for i in 1:length(KP_R1)
#     # Replace NaN values in each matrix with 0
#     KP_R1[i] .= replace(KP_R1[i], NaN => 0.0)
# end

# for i in 1:length(KP_R4)
#     # Replace NaN values in each matrix with 0
#     KP_R4[i] .= replace(KP_R4[i], NaN => 0.0)
# end


# X = cat(KP_R1, KP_R4, dims=1)
# Y = cat(Probe1_R1, Probe1_R4, dims=1)
# fit_and_evaluate_dis(X, Y, LRCs, λ_values, "Results\\TD13d_11_12\\Dis_KP_To_Neural_FRs")



# """
# Sliding Window R2 Results KP-> PCs
# """

# results_folder = "Results/TD13d_11_12/Dis_KP_To_Neural_FRs"
# R1_mean, R1_std = sliding_window_r2(KP_R1, Probe1_R1, results_folder, "R1")
# R4_mean, R4_std = sliding_window_r2(KP_R4, Probe1_R4, results_folder, "R4")

# p = plot_sliding_window_r2(R1_mean, R1_std, R4_mean, R4_std)
# savefig(p, joinpath(results_folder, "Sliding_Window_R2.png"))


# """
# Section for testing KP Features to nerual PCs
# """

# X = cat(KP_R1, KP_R4, dims=1)
# Y = cat(PCA_P1_R1, PCA_P1_R4, dims=1)
# fit_and_evaluate_dis(X, Y, LRCs, λ_values, "Results\\TD13d_11_12\\Dis_KP_To_Neural_PCs")


# """
# Sliding Window R2 Results KP-> FR
# """
# results_folder = "Results/TD13d_11_12/Dis_KP_To_Neural_PCs"
# R1_mean, R1_std = sliding_window_r2(KP_R1, PCA_P1_R1, results_folder, "R1")
# R4_mean, R4_std = sliding_window_r2(KP_R4, PCA_P1_R4, results_folder, "R4")

# p = plot_sliding_window_r2(R1_mean, R1_std, R4_mean, R4_std)
# savefig(p, joinpath(results_folder, "Sliding_Window_R2.png"))










# """
# Testing ground
# """
# SVD_R1_selected = [x[:,100] for x in SVD_R1]
# SVD_R4_selected = [x[:,100] for x in SVD_R4]

# X = cat(SVD_R1_selected, SVD_R4_selected, dims=1)

# num_trials = length(SVD_R1_selected) + length(SVD_R4_selected)
# X_fake = [randn(600, 1) for _ in 1:num_trials]



# Y = cat(Probe1_R1, Probe1_R4, dims=1)
# FCs = cat(FCs_R1, FCs_R4, dims=2)
# λ_values = [0.0001, 0.0, 1000.0]
# fit_and_evaluate(X_fake, Y, FCs, λ_values, "Results\\TD13d_11_12\\SVD_To_Neural_FRs")
