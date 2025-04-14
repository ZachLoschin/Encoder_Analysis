
using Pkg
Pkg.activate("C:\\Users\\zachl\\OneDrive\\BU_YEAR1\\Research\\Tudor_Data\\Disengagement_Analysis_2025")

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
include("C:\\Users\\zachl\\OneDrive\\BU_YEAR1\\Research\\Tudor_Data\\Disengagement_Analysis_2025\\HMM_GLM_Fitting\\zutils.jl")
using StatsPlots

# For testing and debugging
Random.seed!(1234);

const SSD = StateSpaceDynamics

# p = "C:\\Users\\zachl\\OneDrive\\BU_YEAR1\\Research\\Tudor_Data\\Disengagement_Analysis_2025\\preprocessed_data\\TD13d_2024-11-13\\";  # Probe 2
path = "C:\\Users\\zachl\\OneDrive\\BU_YEAR1\\Research\\Tudor_Data\\Disengagement_Analysis_2025\\Processed_Encoder\\TD13d_2024-11-12\\";  # Probe 1

Neural_R4_path = path*"Probe1_R4.csv"
Neural_R1_path = path*"Probe1_R1.csv"
SVD_R4_path = path*"R4_Features.csv"
SVD_R1_path = path*"R1_Features.csv"

R4_Neural = Matrix(CSV.read(Neural_R4_path, DataFrame, header=false))
R1_Neural = Matrix(CSV.read(Neural_R1_path, DataFrame, header=false))
SVD_R4 = Matrix(CSV.read(SVD_R4_path, DataFrame, header=false))
SVD_R1 = Matrix(CSV.read(SVD_R1_path, DataFrame, header=false))

# Downsample SVD data by taking every 2nd row
SVD_R4_downsampled = SVD_R4[1:2:end, :]
SVD_R1_downsampled = SVD_R1[1:2:end, :]

# Split neural data (1200 points per trial)
R4_Neural_trials = split_into_trials(R4_Neural, 1200)
R1_Neural_trials = split_into_trials(R1_Neural, 1200)

# Split SVD features (1200 points per trial since downsampled)
SVD_R4_trials = split_into_trials(SVD_R4_downsampled, 1200)
SVD_R1_trials = split_into_trials(SVD_R1_downsampled, 1200)

# Compute the mean neural activity over timepoints for each trial
R4_Neural_avg = [mean(trial; dims=1) for trial in R4_Neural_trials]
R1_Neural_avg = [mean(trial; dims=1) for trial in R1_Neural_trials]

r4mean = [mean(r, dims=2) for r in R4_Neural_trials];
r1mean = [mean(r, dims=2) for r in R1_Neural_trials];
r1 = mean(hcat(r1mean...), dims=2)
r4 = mean(hcat(r4mean...), dims=2)

# Combined datasets for each trial
X = cat(SVD_R4_trials, SVD_R1_trials, dims=1)
Y = cat(R4_Neural_trials, R1_Neural_trials, dims=1)

X_chopped = [x[401:1000, :] for x in X];
Y_chopped = [y[401:1000, :] for y in Y];
# Assume you already have X_chopped and Y_chopped (your full data)

# Split data into train and temp
X_train, Y_train, X_temp, Y_temp = train_test_split(X_chopped, Y_chopped, 0.8)


# split temp to val and test
X_val, Y_val, X_test, Y_test = train_test_split(X_temp, Y_temp, 0.5)


# Kernelize and trim
X_train = kernelize_past_features(X_train)
Y_train = trim_Y_train_past(Y_train)

X_val = kernelize_past_features(X_val)
Y_val = trim_Y_train_past(Y_val)

X_test = kernelize_past_features(X_test)
Y_test = trim_Y_train_past(Y_test)

x_test = vcat(X_test...)
y_test = vcat(Y_test...)

# Prepare data
x_train = vcat(X_train...)
y_train = vcat(Y_train...)

x_val = vcat(X_val...)
y_val = vcat(Y_val...)

x_val_with_intercept = hcat(ones(size(x_val, 1)), x_val)

# Sweep over λ values
λ_values = [0.0001, 0.01, 0.1, 0.0, 0.1, 1, 10, 100, 1000, 10000]
results = []


encoder_model = SSD.GaussianRegressionEmission(
    input_dim = 100,
    output_dim = 46,
    include_intercept = false,
    λ = 1.0
)

SSD.fit!(encoder_model, x_train, y_train)

x_test_with_intercept = hcat(ones(size(x_test, 1)), x_test)
y_test_pred = x_test_with_intercept * encoder_model.β

r2_test = r2_score(y_test, y_test_pred)


model = SwitchingGaussianRegression(;K=2, input_dim=100, output_dim=46, include_intercept=false)
model.B[1] = encoder_model
model.A = [0.999 0.001; 0.001 0.999];
model.πₖ = [0.99; 0.01;]

XX = permutedims.(X_train);
YY = permutedims.(Y_train);

lls = fit_switching_encoder!(model, YY, XX; max_iters=100)





# for λ in λ_values
#     println("Evaluating λ: ", λ)
#     # Initialize the model
#     encoder_model = SSD.GaussianRegressionEmission(
#         input_dim = size(X_train[1])[2],
#         output_dim = size(Y_train[1])[2],
#         include_intercept = true,
#         λ = λ
#     )
    
#     # Fit the model
#     SSD.fit!(encoder_model, x_train, y_train)
    
#     # Predict on validation set
#     y_val_pred = x_val_with_intercept * encoder_model.β
    
#     # Compute R²
#     r2_val = r2_score(y_val, y_val_pred)
    
#     # Save the result: store λ, r2, and model parameters
#     push!(results, (λ=λ, r2=r2_val, β=copy(encoder_model.β)))
# end

# Results is now a vector of NamedTuples containing λ, r², and the β coefficients
r2_values = [result.r2 for result in results]
best_index = argmax(r2_values)
best_beta = results[best_index].β
best_lambda = results[best_index].λ

best_encoder_model = SSD.GaussianRegressionEmission(
        input_dim = size(X_train[1])[2],
        output_dim = size(Y_train[1])[2],
        include_intercept = true,
        λ = best_lambda
    )

best_encoder_model.β .= best_beta

x_test_with_intercept = hcat(ones(size(x_test, 1)), x_test)
y_test_pred = x_test_with_intercept * best_beta

r2_test = r2_score(y_test, y_test_pred)


function r2_score(y_true, y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - (ss_res / ss_tot)
end


