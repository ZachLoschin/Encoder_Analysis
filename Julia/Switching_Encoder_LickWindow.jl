
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
# Random.seed!(1234);

const SSD = StateSpaceDynamics

base_path = "C:\\Research\\Encoder_Modeling\\Encoder_Analysis\\Processed_Encoder\\TDsa12\\"
session_folders = filter(isdir, glob("*", base_path))

for session_path in session_folders
    # try
        session = splitpath(session_path)[end]
        session_save = replace(session, "-" => "_")
        println("session save: ", session_save)

        # Automatically detect probe number from folder name
        if endswith(session, "_P3")
            prb = 3
        elseif endswith(session, "_P1")
            prb = 1
        elseif endswith(session, "_P2")
            prb = 2
        else
            println("Skipping folder without probe suffix: $session")
            continue
        end
        println("Processing session: $session with Probe $prb")
        session_path = session_path * "\\"
        println(session_path)

        SR = 1200
        # SR=600
        PCA_P1_R1, KP_R1 = load_data_encoder_noSVD(session_path, "R1", prb, SR)  #chunk size 600 for 10ms bins, 1200 for 5ms bins
        PCA_P1_R4, KP_R4 = load_data_encoder_noSVD(session_path, "R4", prb, SR)

        PCA_P1_R1_Cut, KP_R1_Cut, FCs_R1, LRCs_R1, Tongue_mat_R1 = load_data_encoder_cut_noSVD(session_path, "R1", prb)
        PCA_P1_R4_Cut, KP_R4_Cut, FCs_R4, LRCs_R4, Tongue_mat_R4 = load_data_encoder_cut_noSVD(session_path, "R4", prb)


        println("LOADED DATA")

        # KP_R4 = [dropdims(el[:, 1:12, :]; dims=3) for el in KP_R4]
        # KP_R1 = [dropdims(el[:, 1:12, :]; dims=3) for el in KP_R1]
        # KP_R4_Cut = [dropdims(el[:, 1:12, :]; dims=3) for el in KP_R4_Cut]
        # KP_R1_Cut = [dropdims(el[:, 1:12, :]; dims=3) for el in KP_R1_Cut]

        println(size(KP_R1[1]))
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

        println("Prefitting Encoders")

        # lags=4
        # leads = 0
        # start_time = 90
        # dif = 100-lags;
        # pre=3
        # post=10
        # lags = 4
        # three_sec = 300
        # two_sec = 201


        # for 5ms bins
        lags=8
        leads = 0
        start_time = 190  # This being 190 means that we only get 10 points before the GC, or 50 ms
        dif = 200-lags;
        pre=6
        post=20
        lags=8
        three_sec = 600
        two_sec = 401


        # Get enough data to create kernel and start at GC still
        # X_R1 = [X[100-lags:end,:] for X in Jaw_R1]
        # X_R4 = [X[100-lags:end,:] for X in Jaw_R4]

        X_R1 = [X[start_time-lags+1:end,:] for X in KP_R1]
        X_R4 = [X[start_time-lags+1:end,:] for X in KP_R4]


        Y_R1 = [Y[start_time-lags+1:end, 1:10] for Y in PCA_P1_R1]
        Y_R4 = [Y[start_time-lags+1:end, 1:10] for Y in PCA_P1_R4]

        X_R1_kernel = kernelize_window_features(X_R1, lags)
        X_R4_kernel = kernelize_window_features(X_R4, lags)

        Y_R1_trimmed = kernelize_window_features(Y_R1, lags)
        Y_R4_trimmed = kernelize_window_features(Y_R4, lags)


        # Y_R1_trimmed = trim_Y_train_past(Y_R1)
        # Y_R4_trimmed = trim_Y_train_past(Y_R4)

        FCs_R4 = FCs_R4 .- start_time
        FCs_R1 = FCs_R1 .- start_time

        LRCs_R4 = LRCs_R4 .- start_time
        LRCs_R1 = LRCs_R1 .- start_time

        FCs = cat(FCs_R1, FCs_R4, dims=2)
        LRCs= cat(LRCs_R1, LRCs_R4, dims=1)

        print(FCs_R1)
        X_R1 = [X_R1_kernel[i][(FCs_R1[i]-pre):(FCs_R1[i]), :] for i in eachindex(X_R1_kernel)]
        X_R4 = [X_R4_kernel[i][(FCs_R4[i]-pre):(FCs_R4[i]+post), :] for i in eachindex(X_R4_kernel)]

        Y_R1 = [Y_R1_trimmed[i][(FCs_R1[i]-pre):(FCs_R1[i]), :] for i in eachindex(Y_R1_trimmed)]
        Y_R4 = [Y_R4_trimmed[i][(FCs_R4[i]-pre):(FCs_R4[i]+post), :] for i in eachindex(Y_R4_trimmed)]

        X_eng = cat(X_R1, X_R4, dims=1)
        Y_eng = cat(Y_R1, Y_R4, dims=1)


        # X_R1 = [X_R1_kernel[i][end-7:end, :] for i in eachindex(X_R1_kernel)]
        # X_R4 = [X_R4_kernel[i][end-7:end, :] for i in eachindex(X_R4_kernel)]

        # Y_R1 = [Y_R1_trimmed[i][end-7:end, :]  for i in eachindex(Y_R1_trimmed)]
        # Y_R4 = [Y_R4_trimmed[i][end-7:end, :]  for i in eachindex(Y_R4_trimmed)]


        # X_diseng = cat(X_R1, X_R4, dims=1)
        # Y_diseng = cat(Y_R1, Y_R4, dims=1)

        # Prefit engaged model
        X_eng = vcat(X_eng...)
        Y_eng = vcat(Y_eng...)

        β_eng, Σ_eng = weighted_ridge_regression(X_eng, Y_eng, 0.0)
        
        println("Engaged model prefitted")
        println(size(β_eng))


        # # # seems to be a problem with X
        # X_diseng = vcat(X_diseng...)
        # Y_diseng = vcat(Y_diseng...)
        
        # β_diseng, Σ_diseng = weighted_ridge_regression(X_diseng, Y_diseng, 1.0)



        """
        Set up the switching encoder model
        """
        println("Setting up switching model")

        X_R1 = [X[start_time-lags:end,:] for X in KP_R1_Cut]
        X_R4 = [X[start_time-lags:end,:] for X in KP_R4_Cut]
        X = cat(X_R1, X_R4, dims=1)

        Y = cat(PCA_P1_R1_Cut, PCA_P1_R4_Cut, dims=1)
        Y = [y[start_time-lags:end, 1:10] for y in Y]

        # X_ready = permutedims.(X_ready)
        # Y_ready = permutedims.(Y_ready)

        # X_ready = rand(size(X_ready)...)

        X_kern = kernelize_window_features(X, lags)
        Y_trim = kernelize_window_features(Y, lags)

        # Y_trim = trim_Y_train_past(Y)

        X_ready = permutedims.(X_kern)
        Y_ready = permutedims.(Y_trim)
        # Y_ready = [randn(size(y)) for y in Y_ready]


        # Initialize the Gaussian HMM-GLM
        model = SwitchingGaussianRegression(;K=2, input_dim=size(X_ready[1])[1], output_dim=size(Y_ready[1])[1], include_intercept=true)

        model.B[1].β = β_eng
        model.B[1].Σ = Σ_eng
        model.B[1].λ = 0.0
        model.B[2].λ = 0.0

        # model.B[2].β = β_diseng
        # model.B[2].Σ = Σ_diseng

        model.A = [0.99 0.01; 0.01 0.99]
        model.πₖ = [0.01; 0.99]

        lls = fit_custom!(model, Y_ready, X_ready, max_iters=50)

        plot(lls)
        title!("Training Log-Likelihood")
        xlabel!("EM Iteration")
        ylabel!("Log-Likelihood")



        """
        Plot the trial averaged inference
        """
        println("Calculating average inference")
        X_R1 = [X[start_time-lags:three_sec,:] for X in KP_R1]
        X_R4 = [X[start_time-lags:three_sec,:] for X in KP_R4]

        Y_R1 = [Y[start_time-lags:three_sec,1:10] for Y in PCA_P1_R1]
        Y_R4 = [Y[start_time-lags:three_sec,1:10] for Y in PCA_P1_R4]

        X_R1_kernel = kernelize_window_features(X_R1, lags)
        X_R4_kernel = kernelize_window_features(X_R4, lags)

        Y_R1_trimmed = kernelize_window_features(Y_R1, lags)
        Y_R4_trimmed = kernelize_window_features(Y_R4, lags)

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

        Tongue_R1 = Tongue_mat_R1[start_time:three_sec, :];
        Tongue_R4 = Tongue_mat_R4[start_time:three_sec, :];

        # Save the data to export to MATLAB figure making
        R4_Tongue = permutedims(hcat(Tongue_R4...))
        R1_Tongue = permutedims(hcat(Tongue_R1...))
        R4_States = permutedims(hcat(γ_vectors_R4...))
        R1_States = permutedims(hcat(γ_vectors_R1...))

        R4_Vit = permutedims(hcat(V4...))
        R1_Vit = permutedims(hcat(V1...))

        # Convert matrices to DataFrames, using :auto for column names (if you don't want specific column names)
        Tongue_R4 = Tongue_R4[1:two_sec, :];
        Tongue_R1 = Tongue_R1[1:two_sec, :];

        R4_Tongue_df = DataFrame(permutedims(Tongue_R4), :auto)
        R1_Tongue_df = DataFrame(permutedims(Tongue_R1), :auto)

        # Here is the data
        X_R1_kernel;
        X_R4_kernel;
        Y_R1_trimmed;
        Y_R4_trimmed;

        # Here are the states
        R1_States;
        R4_States;

        # Get predictions at each time point from the correct emission model based on the state
        trial = 1;

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

        PC = 7
        plot(Y_trial[:,PC], label="PC1")
        plot!(y_pred[:,PC], label="Pred PC1")

        r2_score(Y_trial[:,PC], y_pred[:,PC])

        """
        """

        
        # Compute R² (per trial) and accumulate NMSE data per state (per timepoint)
        num_trials_R4 = length(X_R4_kernel)
        _, O = size(Y_R4_trimmed[1])
        r2_scores_R4 = zeros(num_trials_R4, O)

        # NMSE accumulators: sum of squared errors per state, overall y stats for normalization
        ss_err_eng = zeros(O);  n_eng  = zeros(Int, O)
        ss_err_dis = zeros(O);  n_dis  = zeros(Int, O)
        sum_y = zeros(O);       sum_y2 = zeros(O);  n_tot = 0

        for trial in 1:num_trials_R4
            X_trial = X_R4_kernel[trial]
            Y_trial = Y_R4_trimmed[trial]
            T = size(X_trial, 1)
            y_pred = zeros(T, O)
            X_bias = hcat(ones(T), X_trial)
            for i in 1:T
                state = exp(R4_States[trial, i])
                if state == 1.0
                    y_pred[i, :] = reshape(X_bias[i, :], 1, :) * model.B[1].β
                    for d in 1:O
                        ss_err_eng[d] += (Y_trial[i,d] - y_pred[i,d])^2
                        n_eng[d] += 1
                    end
                else
                    y_pred[i, :] = reshape(X_bias[i, :], 1, :) * model.B[2].β
                    for d in 1:O
                        ss_err_dis[d] += (Y_trial[i,d] - y_pred[i,d])^2
                        n_dis[d] += 1
                    end
                end
                for d in 1:O
                    sum_y[d]  += Y_trial[i,d]
                    sum_y2[d] += Y_trial[i,d]^2
                end
                n_tot += 1
            end
            for pc in 1:O
                r2_scores_R4[trial, pc] = r2_score(Y_trial[:, pc], y_pred[:, pc])
            end
        end

        num_trials_R1 = length(X_R1_kernel)
        r2_scores_R1 = zeros(num_trials_R1, O)

        for trial in 1:num_trials_R1
            X_trial = X_R1_kernel[trial]
            Y_trial = Y_R1_trimmed[trial]
            T = size(X_trial, 1)
            y_pred = zeros(T, O)
            X_bias = hcat(ones(T), X_trial)
            for i in 1:T
                state = exp(R1_States[trial, i])
                if state == 1.0
                    y_pred[i, :] = reshape(X_bias[i, :], 1, :) * model.B[1].β
                    for d in 1:O
                        ss_err_eng[d] += (Y_trial[i,d] - y_pred[i,d])^2
                        n_eng[d] += 1
                    end
                else
                    y_pred[i, :] = reshape(X_bias[i, :], 1, :) * model.B[2].β
                    for d in 1:O
                        ss_err_dis[d] += (Y_trial[i,d] - y_pred[i,d])^2
                        n_dis[d] += 1
                    end
                end
                for d in 1:O
                    sum_y[d]  += Y_trial[i,d]
                    sum_y2[d] += Y_trial[i,d]^2
                end
                n_tot += 1
            end
            for pc in 1:O
                r2_scores_R1[trial, pc] = r2_score(Y_trial[:, pc], y_pred[:, pc])
            end
        end

        # Overall R² per output dim (mean/std across trials)
        r2_all = vcat(r2_scores_R1, r2_scores_R4)
        mean_r2_per_pc = vec(mean(r2_all, dims=1))
        std_r2_per_pc  = vec(std(r2_all, dims=1))

        # NMSE = MSE_state / Var(y_overall) — shared denominator makes states comparable
        var_y   = sum_y2 ./ n_tot .- (sum_y ./ n_tot).^2
        nmse_eng = (ss_err_eng ./ max.(n_eng, 1)) ./ var_y
        nmse_dis = (ss_err_dis ./ max.(n_dis, 1)) ./ var_y

        # Average over lag dims to get one value per original PC
        n_pcs = 10
        n_lags_pc = O ÷ n_pcs
        mean_r2_pc  = [mean([mean_r2_per_pc[pc + n_pcs * lag] for lag in 0:(n_lags_pc-1)]) for pc in 1:n_pcs]
        std_r2_pc   = [mean([std_r2_per_pc[pc  + n_pcs * lag] for lag in 0:(n_lags_pc-1)]) for pc in 1:n_pcs]
        nmse_eng_pc = [mean([nmse_eng[pc + n_pcs * lag] for lag in 0:(n_lags_pc-1)]) for pc in 1:n_pcs]
        nmse_dis_pc = [mean([nmse_dis[pc + n_pcs * lag] for lag in 0:(n_lags_pc-1)]) for pc in 1:n_pcs]

        println("\n--- Encoding Accuracy (R²) per PC for session: $session ---")
        println("       " * rpad("Mean R²", 12) * "Std R²")
        println("       " * repeat("-", 24))
        for pc in 1:n_pcs
            println("PC $(lpad(pc, 2)): $(rpad(round(mean_r2_pc[pc], digits=3), 12)) $(round(std_r2_pc[pc], digits=3))")
        end

        println("\n--- NMSE by State (lower = better encoding) ---")
        println("       " * rpad("Engaged", 12) * "Disengaged")
        println("       " * repeat("-", 24))
        for pc in 1:n_pcs
            println("PC $(lpad(pc, 2)): $(rpad(round(nmse_eng_pc[pc], digits=3), 12)) $(round(nmse_dis_pc[pc], digits=3))")
        end
        println("---------------------------------------------------\n")

        # --- Visualizations ---

        # Plot 1: R² per PC with ±1 std error bars
        p_r2 = bar(1:n_pcs, mean_r2_pc;
            yerr        = std_r2_pc,
            xlabel      = "PC",
            ylabel      = "R²",
            title       = "Encoding R² per PC\n$session",
            label       = "Mean ± Std",
            ylims       = (0, 1),
            xticks      = 1:n_pcs,
            color       = :steelblue,
            alpha       = 0.8)

        # Plot 2: NMSE by state with chance reference line at 1.0
        nmse_ymax = max(maximum(nmse_eng_pc), maximum(nmse_dis_pc)) * 1.15
        p_nmse = plot(1:n_pcs, nmse_eng_pc;
            marker      = :circle,
            label       = "Engaged",
            xlabel      = "PC",
            ylabel      = "NMSE  (↓ = better)",
            title       = "NMSE by State\n$session",
            ylims       = (0, nmse_ymax),
            xticks      = 1:n_pcs)
        plot!(1:n_pcs, nmse_dis_pc; marker=:square, label="Disengaged")
        hline!([1.0]; linestyle=:dash, color=:black, label="Chance (NMSE=1)")

        p_combined = plot(p_r2, p_nmse; layout=(1, 2), size=(1000, 420))
        display(p_combined)

        if !isdir(joinpath("TDsa12_ShankFiltered", session_save))
            mkpath(joinpath("TDsa12_ShankFiltered", session_save))
        end
        savefig(p_combined, joinpath("TDsa12_ShankFiltered", session_save, "encoding_accuracy.png"))

        """
        VITERBI STATES SAVED
        """

        if !isdir(joinpath("TDsa12_ShankFiltered\\" *session_save))
            mkpath(joinpath("TDsa12_ShankFiltered\\" *session_save))
        end

        println("**SAVE PATH**", (joinpath("TDsa12_ShankFiltered\\" *session_save, "R14_PC_R2_Reg.csv")))

        R4_States_Vit_df = DataFrame(R4_Vit, :auto)
        R1_States_Vit_df = DataFrame(R1_Vit, :auto)
        R4_States_df = DataFrame(R4_States, :auto)
        R1_States_df = DataFrame(R1_States, :auto)

        # Wrap vector into a DataFrame
        # Convert to DataFrame
        mean_r2_df = DataFrame(mean_r2_per_pc', :auto)  # make it a 1×12 DataFrame


        CSV.write(joinpath("TDsa12_ShankFiltered\\" *session_save, "R14_PC_R2_Reg.csv"), mean_r2_df; header=false) 

        
        CSV.write(joinpath("TDsa12_ShankFiltered\\" *session_save, "R14_States_Reg.csv"), R4_States_df; header=false)
        CSV.write(joinpath("TDsa12_ShankFiltered\\" *session_save, "R1_States_Reg.csv"), R1_States_df; header=false)
        CSV.write(joinpath("TDsa12_ShankFiltered\\" *session_save, "R14_States_Vit_Reg.csv"), R4_States_Vit_df; header=false)
        CSV.write(joinpath("TDsa12_ShankFiltered\\" *session_save, "R1_States_Vit_Reg.csv"), R1_States_Vit_df; header=false)

        CSV.write(joinpath("TDsa12_ShankFiltered\\" *session_save, "R14_Tongue_Reg.csv"), R4_Tongue_df; header=false)
        CSV.write(joinpath("TDsa12_ShankFiltered\\" *session_save, "R1_Tongue_Reg.csv"), R1_Tongue_df; header=false)

        println("SESSION DATA SAVED")


    # catch err
    #     # Prepare error message as a string
    #     error_msg = IOBuffer()
    #     println(error_msg, "Error: ", err)
    #     println(error_msg, "Stacktrace:")
    #     for frame in stacktrace(catch_backtrace())
    #         println(error_msg, frame)
    #     end

    # #     # # Save to file
    # #     # error_file = joinpath("Results_Window", session_save, "KP2PC", "error_log.txt")
    # #     # open(error_file, "w") do f
    # #     #     write(f, String(take!(error_msg)))
    # #     # end
    # end
end


