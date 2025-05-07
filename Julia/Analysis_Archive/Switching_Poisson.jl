
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
path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\TD13d_2024-11-13\\";  # Probe 2
# path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\TD15d_2024-11-26\\";  # Probe 1
# path = "U:\\eng_research_economo2\\ZFL\\Disengagement_Encoder\\TD13d_2024-11-12\\"

a = 2

if a==1
    println("Setup for Probe 1")
    Probe1_R1, Probe2_R1, PCA_P1_R1, PCA_P2_R1, SVD_R1, KP_R1, Jaw_R1, SC_P1_R1, SC_P2_R1 = load_data_encoder(path, "R1");
    Probe1_R4, Probe2_R4, PCA_P1_R4, PCA_P2_R4, SVD_R4, KP_R4, Jaw_R4, SC_P1_R4, SC_P2_R4 = load_data_encoder(path, "R4");

    Probe1_R1_Cut, Probe2_R1_Cut, PCA_P1_R1_Cut, PCA_P2_R1_Cut, SVD_R1_Cut, KP_R1_Cut, FCs_R1, LRCs_R1, Tongue_mat_R1, Jaw_R1_Cut, SC_P1_R1_Cut, SC_P2_R1_Cut  = load_data_encoder_cut(path, "R1");
    Probe1_R4_Cut, Probe2_R4_Cut, PCA_P1_R4_Cut, PCA_P2_R4_Cut, SVD_R4_Cut, KP_R4_Cut, FCs_R4, LRCs_R4, Tongue_mat_R4, Jaw_R4_Cut, SC_P1_R4_Cut, SC_P2_R4_Cut  = load_data_encoder_cut(path, "R4");
else
    println("Setup for Probe 2")
    Probe11_R1, Probe1_R1, PCA_P11_R1, PCA_P1_R1, SVD_R1, KP_R1, Jaw_R1, SC_P11_R1, SC_P1_R1 = load_data_encoder(path, "R1");
    Probe11_R4, Probe1_R4, PCA_P11_R4, PCA_P1_R4, SVD_R4, KP_R4, Jaw_R4, SC_P11_R4, SC_P1_R4 = load_data_encoder(path, "R4");

    Probe11_R1_Cut, Probe1_R1_Cut, PCA_P11_R1_Cut, PCA_P1_R1_Cut, SVD_R1_Cut, KP_R1_Cut, FCs_R1, LRCs_R1, Tongue_mat_R1, Jaw_R1_Cut, SC_P11_R1_Cut, SC_P1_R1_Cut  = load_data_encoder_cut(path, "R1");
    Probe11_R4_Cut, Probe1_R4_Cut, PCA_P11_R4_Cut, PCA_P1_R4_Cut, SVD_R4_Cut, KP_R4_Cut, FCs_R4, LRCs_R4, Tongue_mat_R4, Jaw_R4_Cut, SC_P11_R4_Cut, SC_P1_R4_Cut  = load_data_encoder_cut(path, "R4");
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

# Check cut SVD Features
SVD_R1_Ave = [x[1:200,:] for x in SVD_R1_Cut if size(x,1) >=200]
SVD_R1_Ave = average_PCs(SVD_R1_Ave)
plot(SVD_R1_Ave[:,1:5])
title!("R1 Motion SVDs")
plot(SVD_R1_Ave[:,51:55])
title!("R1 Movie SVDs")

SVD_R4_Ave = [x[1:200,:] for x in SVD_R4_Cut if size(x,1) >=200]
SVD_R4_Ave = average_PCs(SVD_R4_Ave)
plot(SVD_R4_Ave[:,1:5])
title!("R4 Motion SVDs")
plot(SVD_R4_Ave[:,51:55])
title!("R4 Movie SVDs")


"""
Visualize the spiking data
"""
# Stack into a 3D array: timepoints × neurons × trials
data_stack = stack(SC_P1_R1, dims=3)  # result is 600×46×152
neuron_avg = mean(data_stack, dims=3)  # still 600×46×1
neuron_avg_R1 = dropdims(neuron_avg, dims=3)

# Assuming neuron_avg is 46×600 (neurons × timepoints)
heatmap(permutedims(neuron_avg_R1),
    xlabel = "Time (samples)",
    ylabel = "Neuron",
    colorbar_title = "Activity",
    title = "Trial-Averaged Neural Binned Spikes R1")


# Stack into a 3D array: timepoints × neurons × trials
data_stack = stack(SC_P1_R4, dims=3)  # result is 600×46×152
neuron_avg = mean(data_stack, dims=3)  # still 600×46×1
neuron_avg_R4 = dropdims(neuron_avg, dims=3)

# Assuming neuron_avg is 46×600 (neurons × timepoints)
heatmap(permutedims(neuron_avg_R4),
    xlabel = "Time (samples)",
    ylabel = "Neuron",
    colorbar_title = "Activity",
    title = "Trial-Averaged Neural Binned Spikes R4")




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
X_R4 = [X[100-lags:end,:] for X in Jaw_R4]

# Y_R1 = [Y[100-lags:end,1:2] for Y in PCA_P1_R1]
# Y_R4 = [Y[100-lags:end,1:2] for Y in PCA_P1_R4]

Y_R1 = [Y[100-lags:end,:] for Y in SC_P1_R1]
Y_R4 = [Y[100-lags:end,:] for Y in SC_P1_R4]

X_R1_kernel = kernelize_past_features(X_R1, lags)
X_R4_kernel = kernelize_past_features(X_R4, lags)

Y_R1_trimmed = trim_Y_train_past(Y_R1, lags)
Y_R4_trimmed = trim_Y_train_past(Y_R4, lags)

FCs = cat(FCs_R1, FCs_R4, dims=2)
LRCs= cat(LRCs_R1, LRCs_R4, dims=1)


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
X_R4 = [X_R4_kernel[i][1:(FCs_R4[i]-dif), :] for i in eachindex(X_R4_kernel)]

Y_R1 = [Y_R1_trimmed[i][1:(FCs_R1[i]-dif), :] for i in eachindex(Y_R1_trimmed)]
Y_R4 = [Y_R4_trimmed[i][1:(FCs_R4[i]-dif), :] for i in eachindex(Y_R4_trimmed)]

X_eng = cat(X_R1, X_R4, dims=1)
Y_eng = cat(Y_R1, Y_R4, dims=1)


X_R1 = [X_R1_kernel[i][(LRCs_R1[i]-dif-5):(LRCs_R1[i]-dif), :] for i in eachindex(X_R1_kernel)]
X_R4 = [X_R4_kernel[i][(LRCs_R4[i]-dif-5):(LRCs_R4[i]-dif), :] for i in eachindex(X_R4_kernel)]

Y_R1 = [Y_R1_trimmed[i][(LRCs_R1[i]-dif-5):(LRCs_R1[i]-dif), :] for i in eachindex(Y_R1_trimmed)]
Y_R4 = [Y_R4_trimmed[i][(LRCs_R4[i]-dif-5):(LRCs_R4[i]-dif), :] for i in eachindex(Y_R4_trimmed)]

X_diseng = cat(X_R1, X_R4, dims=1)
Y_diseng = cat(Y_R1, Y_R4, dims=1)


# Prefit engaged model
X_eng = vcat(X_eng...)
Y_eng = vcat(Y_eng...)

m_eng = PoissonRegressionEmission(input_dim = size(X_eng,2), output_dim=size(Y_eng,2), include_intercept=true)

SSD.fit!(m_eng, X_eng, Y_eng)


X_diseng = vcat(X_diseng...)
Y_diseng = vcat(Y_diseng...)

m_diseng = PoissonRegressionEmission(input_dim = size(X_diseng,2), output_dim=size(Y_diseng,2), include_intercept=true)
SSD.fit!(m_diseng, X_diseng, Y_diseng)


"""
Set up the switching encoder model
"""

# Setup the X and Y variables for switching model -> 97:end cuts out pregc period leaving enough for 4 point kernel
# SVD_R1_selected = [hcat(x[100-lags:end, 1:20], x[100-lags:end, 51:71]) for x in SVD_R1_Cut]
# SVD_R4_selected = [hcat(x[100-lags:end, 1:20], x[100-lags:end, 51:71]) for x in SVD_R4_Cut]

# SVD_R1_selected = [hcat(x[100-lags:end, 1], x[100-lags:end, 2]) for x in SVD_R1_Cut]
# SVD_R4_selected = [hcat(x[100-lags:end, 1], x[100-lags:end, 2]) for x in SVD_R4_Cut]

X_R1 = [X[100-lags:end,:] for X in Jaw_R1_Cut]
X_R4 = [X[100-lags:end,:] for X in Jaw_R4_Cut]
X = cat(X_R1, X_R4, dims=1)

# X = cat(SVD_R1_selected, SVD_R4_selected, dims=1)
# X = cat(KP_R1_Cut, KP_R4_Cut, dims=1)
# X = [x[97:end, :] for x in X]
Y = cat(SC_P1_R1_Cut, SC_P1_R4_Cut, dims=1)

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
input_dim = size(X_ready[1])[1]
output_dim = size(Y_ready[1])[1]
model = SwitchingPoissonRegression(;K=2, input_dim=size(X_ready[1])[1], output_dim=size(Y_ready[1])[1], include_intercept=true)

model.B[1] = m_eng
model.B[2] = m_diseng
# model.B[2].β = randn(input_dim, output_dim)

# model.B[1].λ = 0.01
# model.B[2].λ = 0.01
# Initialize the model with domain knowledge
# model.A = [0.99 0.005 0.005; 0.005 0.99 0.005; 0.005 0.005 0.99];
# model.πₖ = [0.4; 0.3; 0.3]
model.A = [0.9999 0.0001; 0.0001 0.9999]
model.πₖ = [0.0001; 0.9999]


lls = SSD.fit!(model, Y_ready, X_ready, max_iters=100)

function SSD.emission_loglikelihoods!(model::HiddenMarkovModel, FB_storage::SSD.ForwardBackward, data...)
    log_likelihoods = FB_storage.loglikelihoods

    # Calculate observation wise likelihoods for all states
    for k in 1:model.K
        log_likelihoods[k, :] .= SSD.loglikelihood(model.B[k], data...)
    end
end

function SSD.loglikelihood(
    model::SSD.PoissonRegressionEmission,
    Φ::Matrix{<:Real},
    Y::Matrix{<:Real},
    w::Vector{Float64}=ones(size(Y, 1)),
)

    println(model.include_intercept)  
    # add intercept if specified
    println(size(Φ, 2))
    println(size(model.β,1))

    if model.include_intercept && size(Φ, 2) == size(model.β,1) - 1
        Φ = hcat(ones(size(Φ, 1)), Φ)
    end
    # calculate log likelihood
    η = clamp.(Φ * model.β, -30, 30)
    rate = exp.(η)

    obs_wise_loglikelihood = w .* (Y .* log.(rate) .- rate .- SSD.loggamma.(Int.(Y) .+ 1))
    

    # sum across independent feature log likelihoods if mulitple features
    if size(obs_wise_loglikelihood, 2) > 1
        obs_wise_loglikelihood = sum(obs_wise_loglikelihood, dims=2)
    end

    return obs_wise_loglikelihood
end



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
X_R4 = [X[100-lags:300,:] for X in Jaw_R4]

Y_R1 = [Y[100-lags:300,:] for Y in SC_P1_R1]
Y_R4 = [Y[100-lags:300,:] for Y in SC_P1_R4]

X_R1_kernel = kernelize_past_features(X_R1, lags)
X_R4_kernel = kernelize_past_features(X_R4, lags)

Y_R1_trimmed = trim_Y_train_past(Y_R1, lags)
Y_R4_trimmed = trim_Y_train_past(Y_R4, lags)

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

Tongue_R1 = Tongue_mat_R1[dif:300, :];
Tongue_R4 = Tongue_mat_R4[dif:300, :];

# Save the data to export to MATLAB figure making
R4_Tongue = permutedims(hcat(Tongue_R4...))
R1_Tongue = permutedims(hcat(Tongue_R1...))
R4_States = permutedims(hcat(γ_vectors_R4...))
R1_States = permutedims(hcat(γ_vectors_R1...))

R4_Vit = permutedims(hcat(V4...))
R1_Vit = permutedims(hcat(V1...))

# Convert matrices to DataFrames, using :auto for column names (if you don't want specific column names)
Tongue_R4 = Tongue_R4[1:201, :];
Tongue_R1 = Tongue_R1[1:201, :];

R4_Tongue_df = DataFrame(permutedims(Tongue_R4), :auto)
R1_Tongue_df = DataFrame(permutedims(Tongue_R1), :auto)


"""
Look at the encoding accuracy
"""

# Here is the data
X_R1_kernel;
X_R4_kernel;
Y_R1_trimmed;
Y_R4_trimmed;

# Here are the states
R1_States;
R4_States;

# # Get predictions at each time point from the correct emission model based on the state
# trial = 1;

# X_trial = X_R4_kernel[trial];
# Y_trial = Y_R4_trimmed[trial];
# T, D = size(X_trial)
# _, O = size(Y_trial)  # O = output dimension

# # Initialize prediction matrix
# y_pred = zeros(T, O)

# for i in 1:size(X_trial,1)
#    # Find the state
#    state = exp(R4_States[trial,i])

#    N, D = size(X_trial)
# #    X_bias = hcat(ones(N), X_trial)  # Add intercept column

#    if state == 1.0
#     y_pred[i,:] = (reshape(X_trial[i, :], 1,:) * model.B[1].β)
#    else
#     y_pred[i,:] = (reshape(X_trial[i, :], 1,:) * model.B[2].β)
#    end
# end

# plot(Y_trial[:,1], label="PC1")
# plot!(y_pred[:,1], label="Pred PC1")





"""
Visualization of wtf is going on
"""
trial = 5
x = 1:length(R4_States[trial, :])

X_R1_trimmed = trim_Y_train_past(X_R4, lags)


plot(
    plot(x, 1 .- exp.(R4_States[trial, :]), label="State Inference", ylabel="State", legend=:topright, title="Single Trial Inference and Features"),
    plot(x, Tongue_R4[:, trial], label="Tongue", ylabel="Tongue", legend=:topright),
    plot(x, X_R4_kernel[trial][:,1], label="Jaw Pos") |> p -> plot!(p, x, X_R4_kernel[trial][:,2], label="Jaw Vel", ylabel="Jaw Feats"),
    plot(x, Y_R4_trimmed[trial][:,:], label=false),
    layout = @layout([a; b; c; d]),
    link = :x,
    size=(800,600),
)











"""
VITERBI STATES SAVED
"""

R4_States_df = DataFrame(R4_Vit, :auto)
R1_States_df = DataFrame(R1_Vit, :auto)

# Write DataFrames to CSV without headers
CSV.write(joinpath("Results_Poisson\\TD13d_11_13\\Jaw2Spike" , "R4_Tongue_Reg.csv"), R4_Tongue_df; header=false)
CSV.write(joinpath("Results_Poisson\\TD13d_11_13\\Jaw2Spike"  , "R1_Tongue_Reg.csv"), R1_Tongue_df; header=false)
CSV.write(joinpath("Results_Poisson\\TD13d_11_13\\Jaw2Spike"  , "R4_States_Reg.csv"), R4_States_df; header=false)
CSV.write(joinpath("Results_Poisson\\TD13d_11_13\\Jaw2Spike"  , "R1_States_Reg.csv"), R1_States_df; header=false)


