
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

using Dates
# include("C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Julia\\Zutils.jl")
include(".\\Julia\\Zutils.jl")
# using StatsPlots

# For testing and debugging
Random.seed!(1234);

const SSD = StateSpaceDynamics

# path = "C:\\Users\\zachl\\OneDrive\\BU_YEAR1\\Research\\Tudor_Data\\Disengagement_Analysis_2025\\preprocessed_data\\TD13d_2024-11-13\\";  # Probe 2
path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\TD13d_2024-11-12\\";  # Probe 1
# path = "U:\\eng_research_economo2\\ZFL\\Disengagement_Encoder\\TD13d_2024-11-12\\"
Probe1_R1, Probe2_R1, PCA_P1_R1, PCA_P2_R1, SVD_R1, KP_R1 = load_data_encoder(path, "R1");
Probe1_R4, Probe2_R4, PCA_P1_R4, PCA_P2_R4, SVD_R4, KP_R4 = load_data_encoder(path, "R4");

Probe1_R1_Cut, Probe2_R1_Cut, PCA_P1_R1_Cut, PCA_P2_R1_Cut, SVD_R1_Cut, KP_R1_Cut, FCs_R1, LRCs_R1, Tongue_mat_R1  = load_data_encoder_cut(path, "R1");
Probe1_R4_Cut, Probe2_R4_Cut, PCA_P1_R4_Cut, PCA_P2_R4_Cut, SVD_R4_Cut, KP_R4_Cut, FCs_R4, LRCs_R4, Tongue_mat_R4  = load_data_encoder_cut(path, "R4");


# Load the data
_, λ_SVD_FRs, r2_train_SVD_FRs, r2_val_SVD_FRs, r2_test_SVD_FRs, r2_fullcut_SVD_FRs, best_β = load_results_from_csv("Results\\TD13d_11_12_FC_FIT\\SVD_Red_To_Neural_PCs")


# Setup the X and Y variables for switching model -> 97:end cuts out pregc period leaving enough for 4 point kernel
SVD_R1_selected = [hcat(x[97:end, 1:20], x[97:end, 51:70]) for x in SVD_R1_Cut]
SVD_R4_selected = [hcat(x[97:end, 1:20], x[97:end, 51:70]) for x in SVD_R4_Cut]
X = cat(SVD_R1_selected, SVD_R4_selected, dims=1)
Y = cat(PCA_P1_R1_Cut, PCA_P1_R4_Cut, dims=1)
Y = [y[97:end, :] for y in Y]

# Preprocess by cutting out pre GC times and Kernelizing
lags=4
X_kern = kernelize_past_features(X, lags)
Y_trim = trim_Y_train_past(Y, lags)

# Transpose for input to switching regression model
X_ready = permutedims.(X_kern)
Y_ready = permutedims.(Y_trim)
include(".\\Julia\\Zutils.jl")
# Initialize the Gaussian HMM-GLM
model = SwitchingGaussianRegression(;K=2, input_dim=size(X_ready[1])[1], output_dim=size(Y_ready[1])[1], include_intercept=true)

model.B[1].β = best_β

# Initialize the model with domain knowledge
model.A = [0.5 0.5; 0.5 0.5]
model.πₖ = [0.5; 0.5]

lls = fit_custom!(model, Y_ready, X_ready, max_iters=100)

plot(lls)
title!("Training Log-Likelihood")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")



"""
Plot the trial averaged inference
"""
# Get the uncut data labeling and averaging
X_R1 = [X[97:300,:] for X in SVD_R1]
X_R4 = [X[97:300,:] for X in SVD_R4]

Y_R1 = [Y[97:300,:] for Y in PCA_P1_R1]
Y_R4 = [Y[97:300,:] for Y in PCA_P1_R4]

X_R1_kernel = kernelize_past_features(X_R1, 4)
X_R4_kernel = kernelize_past_features(X_R4, 4)

Y_R1_trimmed = trim_Y_train_past(Y_R1, 4)
Y_R4_trimmed = trim_Y_train_past(Y_R4, 4)

YY = permutedims.(Y_R1_trimmed)
XX = permutedims.(X_R1_kernel)

FB_R1 = label_data(model, Y_R1_trimmed, X_R1_kernel);
FB_R4 = label_data(model, Y_R4_trimmed, X_R4_kernel);

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
