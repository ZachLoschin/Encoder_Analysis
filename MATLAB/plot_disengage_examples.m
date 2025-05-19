function plot_disengage_examples(Tongue, disengage_times, indices, save_dir, label)
    rows = 1; cols = 1;
    n_per_fig = rows * cols;

    for i = 1:n_per_fig:length(indices)
        figure;
        for j = 1:min(n_per_fig, length(indices) - i + 1)
            idx = indices(i + j - 1);
            subplot(rows, cols, j);
            plot(Tongue(:, idx), 'b'); hold on;
            xline(11, '--k'); % GC
            xline(disengage_times(idx), '--r'); % disengage
            title([label ' Trial ' num2str(idx)]);
            xlim([0 250]);
        end
        saveas(gcf, fullfile(save_dir, [label '_Tongue_Subplot_' num2str(i) '.png']));
        close;
    end
end
