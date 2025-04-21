
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

"""
Note that the names of the imported data are swapped right no because this session requires probe 2 data to be analyzed. So below Probe 11 
is probe 1 and probe1 is acutally Probe2_R1. I will write a general function later that cleans this up.
"""


# path = "C:\\Users\\zachl\\OneDrive\\BU_YEAR1\\Research\\Tudor_Data\\Disengagement_Analysis_2025\\preprocessed_data\\TD13d_2024-11-13\\";  # Probe 2
path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\TD13d_2024-11-13\\";  # Probe 2
# path = "U:\\eng_research_economo2\\ZFL\\Disengagement_Encoder\\TD13d_2024-11-12\\"
Probe11_R2, Probe1_R1, PCA_P11_R1, PCA_P1_R1, SVD_R1, KP_R1 = load_data_encoder(path, "R1");
Probe11_R4, Probe1_R4, PCA_P11_R4, PCA_P1_R4, SVD_R4, KP_R4 = load_data_encoder(path, "R4");

Probe11_R1_Cut, Probe1_R1_Cut, PCA_P11_R1_Cut, PCA_P1_R1_Cut, SVD_R1_Cut, KP_R1_Cut, FCs_R1, LRCs_R1, Tongue_mat_R1  = load_data_encoder_cut(path, "R1");
Probe11_R4_Cut, Probe1_R4_Cut, PCA_P11_R4_Cut, PCA_P1_R4_Cut, SVD_R4_Cut, KP_R4_Cut, FCs_R4, LRCs_R4, Tongue_mat_R4  = load_data_encoder_cut(path, "R4");


# Load the data
_, λ_SVD_FRs, r2_train_SVD_FRs, r2_val_SVD_FRs, r2_test_SVD_FRs, r2_fullcut_SVD_FRs, best_β = load_results_from_csv("Results\\TD13d_11_13\\SVD_Red_To_Neural_FRs")

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

# Setup the X and Y variables for switching model -> 97:end cuts out pregc period leaving enough for 4 point kernel
SVD_R1_selected = [hcat(x[97:end, 1:20], x[97:end, 51:70]) for x in SVD_R1_Cut]
SVD_R4_selected = [hcat(x[97:end, 1:20], x[97:end, 51:70]) for x in SVD_R4_Cut]
X = cat(SVD_R1_selected, SVD_R4_selected, dims=1)
# X = cat(KP_R1_Cut, KP_R4_Cut, dims=1)
# X = [x[97:end, :] for x in X]
# Y = cat(PCA_P1_R1_Cut, PCA_P1_R4_Cut, dims=1)
Y = cat(Probe1_R1_Cut, Probe1_R4_Cut, dims=1)
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
model.B[1].λ = λ_SVD_FRs
model.B[2].λ = λ_SVD_FRs
# Initialize the model with domain knowledge
model.A = [0.2 0.8; 0.5 0.5]
model.πₖ = [0.5; 0.5]

lls = fit_custom!(model, Y_ready, X_ready, max_iters=100)

plot(lls)
title!("Training Log-Likelihood")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")



"""
Plot the trial averaged inference
"""

SVD_R1_selected = [hcat(x[97:end, 1:20], x[97:end, 51:70]) for x in SVD_R1]
SVD_R4_selected = [hcat(x[97:end, 1:20], x[97:end, 51:70]) for x in SVD_R4]

# Get the uncut data labeling and averaging
X_R1 = [X[97:200,:] for X in SVD_R1_selected]
X_R4 = [X[97:200,:] for X in SVD_R4_selected]

Y_R1 = [Y[97:200,:] for Y in Probe1_R1]
Y_R4 = [Y[97:200,:] for Y in Probe1_R4]

X_R1_kernel = kernelize_past_features(X_R1, 4)
X_R4_kernel = kernelize_past_features(X_R4, 4)

Y_R1_trimmed = trim_Y_train_past(Y_R1, 4)
Y_R4_trimmed = trim_Y_train_past(Y_R4, 4)

YY = permutedims.(Y_R1_trimmed)
XX = permutedims.(X_R1_kernel)

YY_R4 = permutedims.(Y_R4_trimmed)
XX_R4 = permutedims.(X_R4_kernel)




FB_R1 = label_data(model, YY, XX);
FB_R4 = label_data(model, YY_R4, XX_R4);

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

Tongue_R1 = Tongue_mat_R1[101:200, :];
Tongue_R4 = Tongue_mat_R4[101:200, :];

# Save the data to export to MATLAB figure making
R4_Tongue = permutedims(hcat(Tongue_R4...))
R1_Tongue = permutedims(hcat(Tongue_R1...))
R4_States = permutedims(hcat(γ_vectors_R4...))
R1_States = permutedims(hcat(γ_vectors_R1...))

R4_Tongue_df = DataFrame(permutedims(Tongue_R4), :auto)
R1_Tongue_df = DataFrame(permutedims(Tongue_R1), :auto)
R4_States_df = DataFrame(R4_States, :auto)
R1_States_df = DataFrame(R1_States, :auto)

# Write DataFrames to CSV without headers
CSV.write(joinpath("Results\\TD13d_11_13\\SVD_Red_To_Neural_FRs" , "R4_Tongue_Reg.csv"), R4_Tongue_df; header=false)
CSV.write(joinpath("Results\\TD13d_11_13\\SVD_Red_To_Neural_FRs"  , "R1_Tongue_Reg.csv"), R1_Tongue_df; header=false)
CSV.write(joinpath("Results\\TD13d_11_13\\SVD_Red_To_Neural_FRs"  , "R4_States_Reg.csv"), R4_States_df; header=false)
CSV.write(joinpath("Results\\TD13d_11_13\\SVD_Red_To_Neural_FRs"  , "R1_States_Reg.csv"), R1_States_df; header=false)
