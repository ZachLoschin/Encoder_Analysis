
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
path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\YH1_2023-05-08\\";  # Probe 1
# path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\TD10si_2024-07-09\\";  # Probe 1

# path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\TD1d_2023-02-22\\";  # Probe 2

prb = 1;

if prb == 1
    println("Probe 1 Processing -> Check this!")
    Probe1_R1, Probe2_R1, PCA_P1_R1, PCA_P2_R1, KP_R1, Jaw_R1 = load_data_encoder_noSVD(path, "R1");
    Probe1_R4, Probe2_R4, PCA_P1_R4, PCA_P2_R4, KP_R4, Jaw_R4 = load_data_encoder_noSVD(path, "R4");

    Probe1_R1_Cut, Probe2_R1_Cut, PCA_P1_R1_Cut, PCA_P2_R1_Cut, KP_R1_Cut, FCs_R1, LRCs_R1, Tongue_mat_R1, Jaw_R1_Cut  = load_data_encoder_cut_noSVD(path, "R1");
    Probe1_R4_Cut, Probe2_R4_Cut, PCA_P1_R4_Cut, PCA_P2_R4_Cut, KP_R4_Cut, FCs_R4, LRCs_R4, Tongue_mat_R4, Jaw_R4_Cut = load_data_encoder_cut_noSVD(path, "R4");
else
    println("Probe 2 Processing -> Check this!")
    Probe11_R1, Probe1_R1, PCA_P11_R1, PCA_P1_R1, KP_R1, Jaw_R1 = load_data_encoder_noSVD(path, "R1");
    Probe11_R4, Probe1_R4, PCA_P11_R4, PCA_P1_R4, KP_R4, Jaw_R4 = load_data_encoder_noSVD(path, "R4");

    Probe11_R1_Cut, Probe1_R1_Cut, PCA_P11_R1_Cut, PCA_P1_R1_Cut, KP_R1_Cut, FCs_R1, LRCs_R1, Tongue_mat_R1, Jaw_R1_Cut = load_data_encoder_cut_noSVD(path, "R1");
    Probe11_R4_Cut, Probe1_R4_Cut, PCA_P11_R4_Cut, PCA_P1_R4_Cut, KP_R4_Cut, FCs_R4, LRCs_R4, Tongue_mat_R4, Jaw_R4_Cut = load_data_encoder_cut_noSVD(path, "R4");
end


"""
Verify data
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

# Check cut data
P1_R1_Ave_Cut = [x[1:200,:] for x in Probe1_R1_Cut if size(x,1) >=200]
P1_R1_Ave_Cut = ave_vector(P1_R1_Ave_Cut)
P1_R4_Ave_Cut = [x[1:200,:] for x in Probe1_R4_Cut if size(x,1) >=200]
P1_R4_Ave_Cut = ave_vector(P1_R4_Ave_Cut)
plot(P1_R1_Ave_Cut, label="R1 Cut Ave")
plot!(P1_R4_Ave_Cut, label="R4 Cut Ave")

# Check cut neural PCs
PCA_P1_R1_Ave_Cut = [x[1:200,:] for x in PCA_P1_R1_Cut if size(x,1) >=200]
PCA11_Ave = average_PCs(PCA_P1_R1_Ave_Cut)
plot(PCA11_Ave)
PCA_P1_R4_Ave_Cut = [x[1:200,:] for x in PCA_P1_R4_Cut if size(x,1) >=200]
PCA14_Ave = average_PCs(PCA_P1_R4_Ave_Cut)
plot(PCA14_Ave)




# Load the data -> we didnt fit an encoder with lagged inputs.
# _, λ_SVD_FRs, r2_train_SVD_FRs, r2_val_SVD_FRs, r2_test_SVD_FRs, r2_fullcut_SVD_FRs, best_β = load_results_from_csv("Results\\TD13d_11_12_AR\\SVD_Red_To_Neural_PCs")

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
Prefit the encoder models
"""
lags=4
dif = 101-lags;



# SVD_R1_selected = [hcat(x[100-lags:end, 1], x[100-lags:end, 2]) for x in SVD_R1]
# SVD_R4_selected = [hcat(x[100-lags:end, 1], x[100-lags:end, 2]) for x in SVD_R4]

# # Get the uncut data labeling and averaging
# X_R1 = [X[1:end,:] for X in SVD_R1_selected]
# X_R4 = [X[1:end,:] for X in SVD_R4_selected]

X_R1 = [X[100-lags:end,:] for X in Jaw_R1]

Y_R1 = [Y[100-lags:end,:] for Y in PCA_P1_R1]

X_R1_kernel = kernelize_past_features(X_R1, lags)

Y_R1_trimmed = trim_Y_train_past(Y_R1, lags)

FCs = FCs_R1
LRCs= LRCs_R1


# # Add autoregressive features in Neural acitvity
# X_R1_Cut = [x[2:end, :] for x in X_R1_kernel];  # Remove frist row since first timepoints can't be predicted
# X_R4_Cut = [x[2:end, :] for x in X_R4_kernel];

# Y_R1_Cut = [y[1:end-1, :] for y in Y_R1_trimmed];  # Remove last row since there is no end+1 timepoint.
# Y_R4_Cut = [y[1:end-1, :] for y in Y_R4_trimmed];

# Y_R1_AR = [y[2:end, :] for y in Y_R1_trimmed];  # Cut first timepoint because it can't be predicted
# Y_R4_AR = [y[2:end, :] for y in Y_R4_trimmed]; 

# X_R1_AR = [hcat(X_R1_Cut[i], Y_R1_Cut[i]) for i in eachindex(X_R1_Cut)]
# X_R4_AR = [hcat(X_R4_Cut[i], Y_R4_Cut[i]) for i in eachindex(X_R4_Cut)]

# X_R1 = [X_R1_AR[i][1:(FCs_R1[i]-97), :] for i in eachindex(X_R1_AR)]
# X_R4 = [X_R4_AR[i][1:(FCs_R4[i]-97), :] for i in eachindex(X_R4_AR)]

# Y_R1 = [Y_R1_AR[i][1:(FCs_R1[i]-97), :] for i in eachindex(Y_R1_AR)]
# Y_R4 = [Y_R4_AR[i][1:(FCs_R4[i]-97), :] for i in eachindex(Y_R4_AR)]

X_R1 = [X_R1_kernel[i][1:(FCs_R1[i]-dif), :] for i in eachindex(X_R1_kernel)]

Y_R1 = [Y_R1_trimmed[i][1:(FCs_R1[i]-dif), :] for i in eachindex(Y_R1_trimmed)]


X_eng = X_R1
Y_eng = Y_R1


# X_R1 = [X_R1_AR[i][LRCs_R1[i]-107:(LRCs_R1[i]-dif), :] for i in eachindex(X_R1_AR)]
# X_R4 = [X_R4_AR[i][LRCs_R4[i]-107:(LRCs_R4[i]-dif), :]  for i in eachindex(X_R4_AR)]

# Y_R1 = [Y_R1_AR[i][LRCs_R1[i]-107:(LRCs_R1[i]-dif), :]  for i in eachindex(Y_R1_AR)]
# Y_R4 = [Y_R4_AR[i][LRCs_R4[i]-107:(LRCs_R4[i]-dif), :]  for i in eachindex(Y_R4_AR)]

# X_diseng = cat(X_R1, X_R4, dims=1)
# Y_diseng = cat(Y_R1, Y_R4, dims=1)


# Prefit engaged model
X_eng = vcat(X_eng...)
Y_eng = vcat(Y_eng...)

β_eng = weighted_ridge_regression(X_eng, Y_eng, 0.01)


# X_diseng = vcat(X_diseng...)
# Y_diseng = vcat(Y_diseng...)

# β_diseng = weighted_ridge_regression(X_diseng, Y_diseng, 0.01)



"""
Set up the switching encoder model
"""

# Setup the X and Y variables for switching model -> 97:end cuts out pregc period leaving enough for 4 point kernel
# SVD_R1_selected = [hcat(x[100-lags:end, 1:20], x[100-lags:end, 51:71]) for x in SVD_R1_Cut]
# SVD_R4_selected = [hcat(x[100-lags:end, 1:20], x[100-lags:end, 51:71]) for x in SVD_R4_Cut]

# SVD_R1_selected = [hcat(x[100-lags:end, 1], x[100-lags:end, 2]) for x in SVD_R1_Cut]
# SVD_R4_selected = [hcat(x[100-lags:end, 1], x[100-lags:end, 2]) for x in SVD_R4_Cut]

X_R1 = [X[100-lags:end,:] for X in Jaw_R1_Cut]
X = X_R1

# X = cat(SVD_R1_selected, SVD_R4_selected, dims=1)
# X = cat(KP_R1_Cut, KP_R4_Cut, dims=1)
# X = [x[97:end, :] for x in X]
Y = PCA_P1_R1_Cut
# Y = cat(Probe1_R1_Cut, Probe1_R4_Cut, dims=1)
Y = [y[100-lags:end, :] for y in Y]

# Preprocess by cutting out pre GC times and Kernelizing

X_kern = kernelize_past_features(X, lags)
Y_trim = trim_Y_train_past(Y, lags)


# # Add autoregressive features in Neural acitvity
# X_cut = [x[2:end, :] for x in X_kern];  # Remove frist row since first timepoints can't be predicted
# Y_cut = [y[1:end-1, :] for y in Y_trim];  # Remove last row since there is no end+1 timepoint.
# Y_AR = [y[2:end, :] for y in Y_trim];  # Cut first timepoint because it can't be predicted

# X_AR = [hcat(X_cut[i], Y_cut[i]) for i in eachindex(X_cut)]


# # Transpose for input to switching regression model
# X_ready = permutedims.(X_AR)
# Y_ready = permutedims.(Y_AR)

X_ready = permutedims.(X_kern)
Y_ready = permutedims.(Y_trim)
# Y_ready = [randn(size(y)) for y in Y_ready]


include(".\\Julia\\Zutils.jl")
# Initialize the Gaussian HMM-GLM
model = SwitchingGaussianRegression(;K=2, input_dim=size(X_ready[1])[1], output_dim=size(Y_ready[1])[1], include_intercept=true)

model.B[1].β = β_eng
# model.B[2].β = β_diseng
# model.B[2].β = β_diseng

# model.B[1].λ = 10000
# model.B[2].λ = 10000
# Initialize the model with domain knowledge
# model.A = [0.99 0.005 0.005; 0.005 0.99 0.005; 0.005 0.005 0.99];
# model.πₖ = [0.4; 0.3; 0.3]
model.A = [0.9999 0.0001; 0.0001 0.9999]
model.πₖ = [0.0001; 0.9999]


lls = fit_custom!(model, Y_ready, X_ready, max_iters=300)

plot(lls)
title!("Training Log-Likelihood")
xlabel!("EM Iteration")
ylabel!("Log-Likelihood")



"""
Plot the trial averaged inference
"""

# SVD_R1_selected = [hcat(x[100-lags:end, 1:20], x[100-lags:end, 51:71]) for x in SVD_R1]
# SVD_R4_selected = [hcat(x[100-lags:end, 1:20], x[100-lags:end, 51:71]) for x in SVD_R4]

# SVD_R1_selected = [hcat(x[100-lags:end, 1], x[100-lags:end, 2]) for x in SVD_R1]
# SVD_R4_selected = [hcat(x[100-lags:end, 1], x[100-lags:end, 2]) for x in SVD_R4]


# # Get the uncut data labeling and averaging
# X_R1 = [X[100-lags:300,:] for X in SVD_R1_selected]
# X_R4 = [X[100-lags:300,:] for X in SVD_R4_selected]

X_R1 = [X[100-lags:300,:] for X in Jaw_R1]

Y_R1 = [Y[100-lags:300,:] for Y in PCA_P1_R1]

X_R1_kernel = kernelize_past_features(X_R1, lags)

Y_R1_trimmed = trim_Y_train_past(Y_R1, lags)

# # Add autoregressive features in Neural acitvity
# X_R1_Cut = [x[2:end, :] for x in X_R1_kernel];  # Remove frist row since first timepoints can't be predicted
# X_R4_Cut = [x[2:end, :] for x in X_R4_kernel];

# Y_R1_Cut = [y[1:end-1, :] for y in Y_R1_trimmed];  # Remove last row since there is no end+1 timepoint.
# Y_R4_Cut = [y[1:end-1, :] for y in Y_R4_trimmed];

# Y_R1_AR = [y[2:end, :] for y in Y_R1_trimmed];  # Cut first timepoint because it can't be predicted
# Y_R4_AR = [y[2:end, :] for y in Y_R4_trimmed]; 

# X_R1_AR = [hcat(X_R1_Cut[i], Y_R1_Cut[i]) for i in eachindex(X_R1_Cut)]
# X_R4_AR = [hcat(X_R4_Cut[i], Y_R4_Cut[i]) for i in eachindex(X_R4_Cut)]

# YY = permutedims.(Y_R1_AR)
# XX = permutedims.(X_R1_AR)

# YY_R4 = permutedims.(Y_R4_AR)
# XX_R4 = permutedims.(X_R4_AR)


YY = permutedims.(Y_R1_trimmed)
XX = permutedims.(X_R1_kernel)



FB_R1 = label_data(model, YY, XX);

V1 = SSD.viterbi(model, YY, XX);


# Extract γ[1, :] for each K in OO
γ_vectors_R1 = [FB_R1[K].γ[1, :] for K in eachindex(FB_R1)]
γ_mean_R1 = mean(exp.(hcat(γ_vectors_R1...)), dims=2)


plot(γ_mean_R1; label="R1")
title!("State Inference")
ylabel!("State Probability")
xlabel!("Time")

Tongue_R1 = Tongue_mat_R1[dif:300, :];

# Save the data to export to MATLAB figure making
R1_Tongue = permutedims(hcat(Tongue_R1...))
R1_States = permutedims(hcat(γ_vectors_R1...))


R1_Vit = permutedims(hcat(V1...))

# Convert matrices to DataFrames, using :auto for column names (if you don't want specific column names)

Tongue_R1 = Tongue_R1[1:201, :];


R1_Tongue_df = DataFrame(permutedims(Tongue_R1), :auto)


"""
Look at the encoding accuracy
"""

# Here is the data
X_R1_kernel;

Y_R1_trimmed;


# Here are the states
R1_States;


# Get predictions at each time point from the correct emission model based on the state
trial = 40;

X_trial = X_R4_kernel[trial];
Y_trial = Y_R4_trimmed[trial];
T, D = size(X_trial)
_, O = size(Y_trial)  # O = output dimension

# Initialize prediction matrix
y_pred = zeros(T, O)

for i in 1:size(X_trial,1)
   # Find the state
   state = exp(R4_States[trial,i])

   N, D = size(X_trial)
   X_bias = hcat(ones(N), X_trial)  # Add intercept column
#    X_bias = X_trial

   if state == 1.0
    y_pred[i,:] = (reshape(X_bias[i, :], 1,:) * model.B[1].β)
   else
    y_pred[i,:] = (reshape(X_bias[i, :], 1,:) * model.B[2].β)
   end
end

plot(Y_trial[:,1], label="PC1")
plot!(y_pred[:,1], label="Pred PC1")





"""
Visualization of wtf is going on
"""
trial = 2
x = 1:length(R1_States[trial, :])

X_R1_trimmed = trim_Y_train_past(X_R1, lags)


plot(
    plot(x, 1 .- exp.(R1_States[trial, :]), label="State Inference", ylabel="State", legend=:topright, title="Single Trial Inference and Features"),
    plot(x, Tongue_R1[:, trial], label="Tongue", ylabel="Tongue", legend=:topright),
    plot(x, X_R1_trimmed[trial][:,1], label="Jaw Pos") |> p -> plot!(p, x, X_R1_trimmed[trial][:,2], label="Jaw Vel", ylabel="Jaw Feats"),
    plot(x, Y_R1_trimmed[trial][:,:], label=false),
    layout = @layout([a; b; c; d]),
    link = :x,
    size=(800,600),
)











"""
VITERBI STATES SAVED
"""


R1_States_Vit_df = DataFrame(R1_Vit, :auto)
R1_States_df = DataFrame(R1_States, :auto)


# Write DataFrames to CSV without headers

CSV.write(joinpath("Results_423\\TD10si_07_09\\Jaw2PC"  , "R1_Tongue_Reg.csv"), R1_Tongue_df; header=false)
CSV.write(joinpath("Results_423\\TD10si_07_09\\Jaw2PC"  , "R1_States_Reg.csv"), R1_States_df; header=false)
CSV.write(joinpath("Results_423\\TD10si_07_09\\Jaw2PC"  , "R1_States_Vit_Reg.csv"), R1_States_Vit_df; header=false)


