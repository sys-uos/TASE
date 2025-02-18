import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns

plt.rc("font", family='serif')  # select a fitting font type
plt.rc('text', usetex=True)  # use installed latex version to render labels


def read_species_dir2(root_dir= '/path/to/root_dir', species="Common Chaffinch"):
    def list_all_files(root_dir):
        file_list = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_list.append(os.path.join(file))
        return file_list

    data = [] # contains tuples, first elem for s05, second for s075, etc.
    sensitivity_dirs = sorted(os.listdir(root_dir))

    for filename in sorted(list_all_files(root_dir)):
        confs = []
        for sens_dir in sensitivity_dirs:
            subdir_path = os.path.join(root_dir, sens_dir)
            if os.path.isdir(subdir_path):
                distance=filename.split(".")[0].split("_")[1].split("-")[0]
                file_path = os.path.join(root_dir,sens_dir, distance,filename)
                # print(file_path)
                with open(file_path, 'r') as file:
                    content = file.readlines()
                    for line in content:
                        if species in line:
                            conf = line.strip().split('\t')[-1]
                            # print(conf)
                            confs.append(float(conf))
        data.append(confs)
    return data

def create_boxplot_gain_for_sensitivies(data):
    CONF_RANGE = 0.025
    buckets = [[] for _ in range(int(1 / CONF_RANGE))]

    for line_data in data:
        if len(line_data) != 5:
            continue
        bucket_index = int(line_data[0] / CONF_RANGE)
        if 0 <= bucket_index < len(buckets):
            conf_diff = line_data[4] - line_data[0]
            buckets[bucket_index].append(conf_diff)

    yticks = [round(x, 3) for x in np.arange(-0.25, 0.3, 0.05)]

    xtick_positions = np.arange(0, len(buckets))  # Center of each bucket
    xtick_labels = [f'[{round(i * CONF_RANGE, 3)} - {round((i + 1) * CONF_RANGE, 3)})' for i in
                    range(len(buckets))]
    for i in range(1, len(xtick_labels), 2):
        xtick_labels[i] = ''

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Create the boxplot on the first subplot
    ax1.boxplot(buckets, notch=True, labels=xtick_labels, positions=xtick_positions,
                flierprops={'marker': 'x', 'markersize': 6})
    ax1.set_ylabel('BirdNet Confidence Gain')
    ax1.set_yticks(yticks)

    # Create the bar plot on the second subplot showing the number of samples in each bucket
    num_samples = [len(bucket) for bucket in buckets]
    ax2.bar(xtick_positions, num_samples, width=0.7)
    ax2.set_xlabel('BirdNet Confidence Interval')
    ax2.set_ylabel('#samples')

    # Set y-axis to logarithmic scale
    ax2.set_yscale('log')

    # Rotate x-axis labels for better readability
    ax2.set_xticks(xtick_positions)
    ax2.set_xticklabels(xtick_labels, rotation=90)

    plt.tight_layout()
    plt.show()

def create_lineplot_with_std_for_sensitivities(data, font_size=12):
    sensitivity_reference = 0  # refers to 0.5
    sensitivity_diffs = [1, 2, 3, 4]  # refers to 0.75, 1.0, 1.25, 1.5

    CONF_RANGE = 0.025
    buckets_list = []

    for sensitivity_diff in sensitivity_diffs:
        buckets = [[] for _ in range(int(1 / CONF_RANGE))]

        for line_data in data:
            if len(line_data) != 5:
                continue
            bucket_index = int(line_data[0] / CONF_RANGE)
            if 0 <= bucket_index < len(buckets):
                conf_diff = line_data[sensitivity_reference] - line_data[sensitivity_diff]
                buckets[bucket_index].append(conf_diff)

        buckets_list.append(buckets)

    yticks = [round(x, 3) for x in np.arange(-0.25, 0.3, 0.05)]

    xtick_positions = np.arange(0, len(buckets_list[0]))  # Center of each bucket
    xtick_labels = [f'[{round(i * CONF_RANGE, 3)} - {round((i + 1) * CONF_RANGE, 3)})' for i in
                    range(len(buckets_list[0]))]
    for i in range(1, len(xtick_labels), 2):
        xtick_labels[i] = ''

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Use the "Reds" colormap for the lines, focusing on the more saturated end
    cmap = cm.get_cmap('Reds')
    colors = [cmap(0.6 + i*0.1) for i in range(len(sensitivity_diffs))]  # More saturated colors
    labels = ['difference to sensitivity 0.75', 'difference to sensitivity 1.0', 'difference to sensitivity 1.25', 'difference to sensitivity 1.5']

    for i, buckets in enumerate(buckets_list):
        means = [np.mean(bucket) if bucket else 0 for bucket in buckets]
        std_devs = [np.std(bucket) if bucket else 0 for bucket in buckets]

        # Create the line plot with standard deviation shading on the first subplot
        color = colors[i]
        ax1.plot(xtick_positions, means, marker='o', linestyle='-', color=color, label=labels[i])
        ax1.fill_between(xtick_positions,
                         [m - s for m, s in zip(means, std_devs)],
                         [m + s for m, s in zip(means, std_devs)],
                         color=color, alpha=0.2)

    ax1.set_ylabel('confidence difference', fontsize=font_size)
    ax1.set_yticks(yticks)
    ax1.tick_params(axis='both', labelsize=font_size)
    ax1.legend(fontsize=font_size)

    # Create the bar plot on the second subplot showing the number of samples in each bucket
    num_samples = [len(bucket) for bucket in buckets_list[0]]
    ax2.bar(xtick_positions, num_samples, width=0.7)
    ax2.set_xlabel('confidence interval', fontsize=font_size)
    ax2.set_ylabel('#samples', fontsize=font_size)

    # Set y-axis to logarithmic scale
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', labelsize=font_size)

    # Rotate x-axis labels for better readability
    ax2.set_xticks(xtick_positions)
    ax2.set_xticklabels(xtick_labels, rotation=75, fontsize=font_size)

    # Remove margins by setting xlim
    ax1.set_xlim(xtick_positions[0] - 0.5, xtick_positions[-1] + 0.5)
    ax2.set_xlim(xtick_positions[0] - 0.5, xtick_positions[-1] + 0.5)

    plt.tight_layout()



    plt.show()


def evaluate_range_recordings():
    def list_dirs_files(path, spec="Common Chaffinch", nightingale_suffix="loud", with_filename=False):
        print(path)
        # Check if the path is a valid directory
        if not os.path.isdir(path):
            return "The specified path is not a valid directory."

        data = []
        for dirpath, dirnames, filenames in os.walk(path):
            # Print path to all subdirectories first.
            # for dirname in dirnames:
            #    print(f"Directory: {os.path.join(dirpath, dirname)}")
            # Print path to all filenames.
            for filename in filenames:
                sensitivity = float(os.path.basename(os.path.dirname(dirpath)).split('s')[1])
                distance = float(os.path.basename(dirpath).split('m')[0])
                with open(os.path.join(dirpath, filename), 'r') as f:
                    print(filename)
                    confidences = []
                    for line in f:
                        if spec in line:
                            confidences.append(float(line.strip().split('\t')[9]))
                    if spec == "Common Nightingale":
                        spec2 = spec + "_" + nightingale_suffix
                    else:
                        spec2 = spec

                    if len(confidences) != 0:
                        confidence = np.max(confidences)
                        if with_filename:
                            data.append((spec2, sensitivity, distance, confidence, filename.split('.BirdNET')[0]))
                        else:
                            data.append((spec2, sensitivity, distance, confidence))

        if with_filename:
            df = pd.DataFrame(data, columns=['species', 'sensitivity', 'distance', 'confidence', 'filename'])
        else:
            df = pd.DataFrame(data, columns=['species', 'sensitivity', 'distance', 'confidence'])
        return df

    def boxplots_per_species(font_size=12, ofile=""):
        # Replace '/path/to/directory' with the path to the directory you want to examine
        directory_paths = ['./data/20230603/processed/range_analysis/fringilla_coelebs',
                           './data/20230603/processed/range_analysis/turdus_philomelos',
                           './data/20230603/processed/range_analysis/regulus_regulus']
        specs = ["Common Chaffinch", "Song Thrush", "Goldcrest", "Common Nightingale", "Common Nightingale"]
        df = None
        df_specs = ["Common Chaffinch", "Song Thrush", "Goldcrest", "Common Nightingale_loud",
                    "Common Nightingale_silent"]
        for i, path in enumerate(directory_paths):
            df2 = list_dirs_files(path, specs[i], with_filename=False)
            df = pd.concat([df, df2], ignore_index=True)

        path = './data/20230603/processed/range_analysis/luscinia_megarhynchos_silent'
        df2 = list_dirs_files(path, specs[3], nightingale_suffix="silent", with_filename=False)
        df = pd.concat([df, df2], ignore_index=True)
        path = './data/20230603/processed/range_analysis/luscinia_megarhynchos_loud'
        df2 = list_dirs_files(path, specs[4], nightingale_suffix="loud", with_filename=False)
        df = pd.concat([df, df2], ignore_index=True)
        # print(df)

        import matplotlib.pyplot as plt
        import seaborn as sns

        for spec in df_specs:
            # Assuming df and df_specs are already defined and loaded
            filtered_df = df.loc[df['species'] == spec].copy()
            print(filtered_df)

            # Create the main figure and subplot structure with shared x-axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [3, 1]},
                                           sharex=True)

            # Main Boxplot
            palette = sns.color_palette("Reds", len(
                filtered_df['sensitivity'].unique()))  # Define palette based on number of sensitivity levels
            sns.boxplot(x='distance', y='confidence', hue='sensitivity', notch=True, data=filtered_df, palette=palette,
                        ax=ax1)
            ax1.set_ylabel('confidence', fontsize=font_size)
            ax1.legend(title='sensitivity', fontsize=font_size, title_fontsize=font_size)
            ax1.grid(axis='x', which='major')
            ax1.set_xlabel('')  # Hide x-label on the first plot
            ax1.tick_params(axis='both', labelsize=font_size)

            # Setting y-ticks at 0.1 steps
            ax1.set_yticks([y / 10.0 for y in
                            range(int(min(filtered_df['confidence']) * 10), int(max(filtered_df['confidence']) * 10) + 1,
                                  1)])
            ax1.set_ylim((-0.001, 1.001))

            # Subplot for sample counts
            counts = filtered_df.groupby(['distance', 'sensitivity']).size().reset_index(name='counts')
            print(counts)
            counts_pivot = counts.pivot(
                index="distance",
                columns="sensitivity",
                values="counts"
            )
            # Replace all values that are 1 with 0
            counts_pivot.replace(1, 0, inplace=True)
            counts_pivot.plot(kind='bar', ax=ax2, legend=False, color=palette)  # Use same color palette for consistency
            ax2.set_xlabel('distance', fontsize=font_size)
            ax2.grid(axis='x', which='major')
            ax2.set_ylabel('number of Samples',fontsize=font_size)
            ax2.tick_params(axis='both', labelsize=font_size)

            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)

            fig.tight_layout()

            fig.savefig(ofile + '.pdf')

            # Show the plot
            plt.show()

    def calc_correlation(df):
        # Calculate correlation between confidence and distance for each species and sensitivity
        correlation_results = df.groupby(['species', 'sensitivity']).apply(lambda x: x[['confidence', 'distance']].corr().iloc[0,1])

        # Create a DataFrame to display the results
        results_df = correlation_results.reset_index()
        results_df.columns = ['species', 'sensitivity', 'correlation']

        # Pivot the DataFrame to make each unique sensitivity value its own column
        pivot_df = results_df.pivot(index='species', columns='sensitivity', values='correlation')

        print(pivot_df)

        # Convert DataFrame to LaTeX table
        latex_table = pivot_df.to_latex()

        print(latex_table)

    def print_correlation():
        # Replace '/path/to/directory' with the path to the directory you want to examine
        directory_paths = ['./data/20230603/processed/range_analysis/fringilla_coelebs',
                           './data/20230603/processed/range_analysis/turdus_philomelos',
                           './data/20230603/processed/range_analysis/regulus_regulus']
        specs = ["Common Chaffinch", "Song Thrush", "Goldcrest", "Common Nightingale", "Common Nightingale"]
        df = None
        df_specs = ["Common Chaffinch", "Song Thrush", "Goldcrest", "Common Nightingale_loud", "Common Nightingale_silent"]
        for i, path in enumerate(directory_paths):
            df2 = list_dirs_files(path, specs[i])
            df = pd.concat([df, df2], ignore_index=True)

        path = './data/20230603/processed/range_analysis/luscinia_megarhynchos_silent'
        df2 = list_dirs_files(path, specs[3], nightingale_suffix="silent")
        df = pd.concat([df, df2], ignore_index=True)
        path = './data/20230603/processed/range_analysis/luscinia_megarhynchos_loud'
        df2 = list_dirs_files(path, specs[4], nightingale_suffix="loud")
        df = pd.concat([df, df2], ignore_index=True)
        print(df)
        calc_correlation(df)

    def parse_data_as_dataframe():
        # Replace '/path/to/directory' with the path to the directory you want to examine
        directory_paths = ['./data/20230603/processed/range_analysis/fringilla_coelebs',
                           './data/20230603/processed/range_analysis/turdus_philomelos',
                           './data/20230603/processed/range_analysis/regulus_regulus']
        specs = ["Common Chaffinch", "Song Thrush", "Goldcrest", "Common Nightingale", "Common Nightingale"]
        df = None
        df_specs = ["Common Chaffinch", "Song Thrush", "Goldcrest", "Common Nightingale_loud",
                    "Common Nightingale_silent"]
        for i, path in enumerate(directory_paths):
            df2 = list_dirs_files(path, specs[i], with_filename=False)
            df = pd.concat([df, df2], ignore_index=True)

        path = './data/20230603/processed/range_analysis/luscinia_megarhynchos_silent'
        df2 = list_dirs_files(path, specs[3], nightingale_suffix="silent", with_filename=False)
        df = pd.concat([df, df2], ignore_index=True)
        path = './data/20230603/processed/range_analysis/luscinia_megarhynchos_loud'
        df2 = list_dirs_files(path, specs[4], nightingale_suffix="loud", with_filename=False)
        df = pd.concat([df, df2], ignore_index=True)
        print(df)

        return df

    def plot_std_dev_boxplots(df, font_size=16, filename=""):
        # Rename the values in the species column
        df['species'] = df['species'].replace({
            'Song Thrush': 'Song Thrush (100 dB)',
            'Common Chaffinch': 'Common Chaffinch (92 dB)',
            'Common Nightingale_loud': 'Common Nightingale (86 dB)',
            'Common Nightingale_silent': 'Common Nightingale (75 dB)',
            'Goldcrest': 'Goldcrest (74 dB)'
        })

        # Calculate standard deviation of 'confidence' per 'species', 'sensitivity', and 'distance'
        std_per_group = df.groupby(['species', 'sensitivity', 'distance'])['confidence'].std().reset_index()

        # Rename the resulting column for clarity
        std_per_group.rename(columns={'confidence': 'std_dev'}, inplace=True)

        # Set up the color palette for species
        species_list = std_per_group['species'].unique()
        palette = sns.color_palette("husl", len(species_list))

        # Create a boxplot for each species and sensitivity
        plt.figure(figsize=(12, 8))

        sns.boxplot(x='sensitivity', y='std_dev', hue='species', data=std_per_group, palette=palette)

        plt.xlabel("classifier's sensitivity", fontsize=font_size)
        plt.ylabel("standard deviation of classifier's confidence", fontsize=font_size)
        # plt.title('Boxplots of Standard Deviation of Confidence by Species and Sensitivity', fontsize=font_size)

        # Get handles and labels for the legend
        handles, labels = plt.gca().get_legend_handles_labels()

        # Specify order of items in the legend
        order = [4, 0, 2, 1, 3]

        # Add the legend to the plot in the specified order
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=font_size, title='Species',
                   title_fontsize=font_size)

        plt.grid(True)
        plt.tight_layout()

        if filename:
            plt.savefig(filename)

        plt.show()

    df = parse_data_as_dataframe()

    # Remove rows where confidence is below 0.1 or above 0.9
    # df = df[(df['confidence'] >= 0.1) & (df['confidence'] <= 0.9)]

    # --- Define output directory --- #
    dirpath = "./plots/final/range_analysis/"
    os.makedirs(dirpath, exist_ok=True)


    plot_std_dev_boxplots(df, font_size=16, filename=f"{dirpath}recording_range_variation.pdf")

    boxplots_per_species(font_size=20, ofile=f"{dirpath}recording_range_variation.pdf")

    print_correlation()

def evaluate_sensitivity():
    all_data = []
    data = read_species_dir2(root_dir="./data/20230603/processed/range_analysis/fringilla_coelebs",
                     species="Common Chaffinch")
    all_data += data
    data = read_species_dir2(root_dir="./data/20230603/processed/range_analysis/regulus_regulus",
                     species="Goldcrest")
    all_data += data
    data = read_species_dir2(root_dir="./data/20230603/processed/range_analysis/turdus_philomelos",
                     species="Song Thrush")
    all_data += data
    data = read_species_dir2(root_dir="./data/20230603/processed/range_analysis/luscinia_megarhynchos_loud",
                     species="Nightingale")
    all_data += data
    data = read_species_dir2(root_dir="./data/20230603/processed/range_analysis/luscinia_megarhynchos_silent",
                     species="Nightingale")
    all_data += data

    # create_boxplot_gain_for_sensitivies(all_data)
    create_lineplot_with_std_for_sensitivities(all_data, font_size=16)

def plot_range_analysis():
    evaluate_range_recordings()
    evaluate_sensitivity()

