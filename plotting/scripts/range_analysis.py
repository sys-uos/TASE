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
    ax2.set_ylabel('number of samples')

    # Set y-axis to logarithmic scale
    ax2.set_yscale('log')

    # Rotate x-axis labels for better readability
    ax2.set_xticks(xtick_positions)
    ax2.set_xticklabels(xtick_labels, rotation=90)

    plt.tight_layout()
    plt.show()


def evaluate_range_recordings():
    def list_dirs_files(path, spec="Common Chaffinch", nightingale_suffix="", with_filename=False):
        # Check if the path is a valid directory
        if not os.path.isdir(path):
            print("The specified path is not a valid directory.")
            exit(-1)

        data = []
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                sensitivity = float(os.path.basename(os.path.dirname(dirpath)).split('sensitivity-')[1])
                distance = float(os.path.basename(dirpath).split('m')[0])
                with open(os.path.join(dirpath, filename), 'r') as f:
                    confidences = []
                    for line in f:
                        if spec in line:
                            confidences.append(float(line.strip().split('\t')[9]))
                    spec_ = spec
                    if nightingale_suffix != "":
                        spec_ += "_" + nightingale_suffix

                    if len(confidences) != 0:
                        confidence = np.max(confidences)
                        if with_filename:
                            data.append((spec_, sensitivity, distance, confidence, filename.split('.BirdNET')[0]))
                        else:
                            data.append((spec_, sensitivity, distance, confidence))

        if with_filename:
            df = pd.DataFrame(data, columns=['species', 'sensitivity', 'distance', 'confidence', 'filename'])
        else:
            df = pd.DataFrame(data, columns=['species', 'sensitivity', 'distance', 'confidence'])

        return df

    def calc_correlation(df):
        # Calculate correlation between confidence and distance for each species and sensitivity
        correlation_results = df.groupby(['species', 'sensitivity']).apply(lambda x: x[['confidence', 'distance']].corr().iloc[0,1])

        # Create a DataFrame to display the results
        results_df = correlation_results.reset_index()
        results_df.columns = ['species', 'sensitivity', 'correlation']

        # Pivot the DataFrame to make each unique sensitivity value its own column
        pivot_df = results_df.pivot(index='species', columns='sensitivity', values='correlation')

        # Convert DataFrame to LaTeX table
        latex_table = pivot_df.to_latex()

        print(latex_table)

    def print_correlation():
        # Replace '/path/to/directory' with the path to the directory you want to examine
        directory_paths = ['./TASE/data/201912/processed/classifications/92dB_Fringilla_Coelebs',
                           './TASE/data/201912/processed/classifications/100dB_Turdus_Philomelos',
                           './TASE/data/201912/processed/classifications/74dB_Regulus_Regulus']
        specs = ["Common Chaffinch", "Song Thrush", "Goldcrest", "Common Nightingale", "Common Nightingale"]
        df = None
        for i, path in enumerate(directory_paths):
            df2 = list_dirs_files(path, specs[i], with_filename=False)
            df = pd.concat([df, df2], ignore_index=True)

        directory_path = './TASE/data/201912/processed/classifications/86dB_Luscinia_Megarhynchos'
                           # './TASE/data/201912/processed/classifications/75dB_Luscinia_Megarhynchos']
        df2 = list_dirs_files(directory_path, specs[3], nightingale_suffix="86dB", with_filename=False)
        df = pd.concat([df, df2], ignore_index=True)

        directory_path = './TASE/data/201912/processed/classifications/75dB_Luscinia_Megarhynchos'
        df2 = list_dirs_files(directory_path, specs[4], nightingale_suffix="74dB", with_filename=False)
        df = pd.concat([df, df2], ignore_index=True)

        calc_correlation(df)

    def parse_data_as_dataframe():
        # Replace '/path/to/directory' with the path to the directory you want to examine
        directory_paths = ['./TASE/data/201912/processed/classifications/92dB_Fringilla_Coelebs',
                           './TASE/data/201912/processed/classifications/100dB_Turdus_Philomelos',
                           './TASE/data/201912/processed/classifications/74dB_Regulus_Regulus']
        specs = ["Common Chaffinch", "Song Thrush", "Goldcrest", "Common Nightingale", "Common Nightingale"]
        df = None
        for i, path in enumerate(directory_paths):
            df2 = list_dirs_files(path, specs[i], with_filename=False)
            df = pd.concat([df, df2], ignore_index=True)

        directory_path = './TASE/data/201912/processed/classifications/86dB_Luscinia_Megarhynchos'
                           # './TASE/data/201912/processed/classifications/75dB_Luscinia_Megarhynchos']
        df2 = list_dirs_files(directory_path, specs[3], nightingale_suffix="86dB", with_filename=False)
        df = pd.concat([df, df2], ignore_index=True)

        directory_path = './TASE/data/201912/processed/classifications/75dB_Luscinia_Megarhynchos'
        df2 = list_dirs_files(directory_path, specs[4], nightingale_suffix="74dB", with_filename=False)
        df = pd.concat([df, df2], ignore_index=True)

        return df

    def plot_std_dev_boxplots(df, font_size=16, filename=""):
        # Rename the values in the species column
        df['species'] = df['species'].replace({
            'Song Thrush': 'Song Thrush (100 dB)',
            'Common Chaffinch': 'Common Chaffinch (92 dB)',
            'Common Nightingale_86dB': 'Common Nightingale (86 dB)',
            'Common Nightingale_75dB': 'Common Nightingale (75 dB)',
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

        plt.xlabel("sensitivity", fontsize=font_size)
        plt.ylabel("standard deviation", fontsize=font_size)
        # plt.title('Boxplots of Standard Deviation of Confidence by Species and Sensitivity', fontsize=font_size)

        # Get handles and labels for the legend
        handles, labels = plt.gca().get_legend_handles_labels()

        # Specify order of items in the legend
        order = [4, 0, 2, 1, 3]

        # Add the legend to the plot in the specified order
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=font_size-10)

        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        plt.grid(True)
        plt.tight_layout()

        if filename:
            plt.savefig(filename)

        plt.show()

    # --- Parse Data into Panda Dataframe --- #
    df = parse_data_as_dataframe()

    # --- Define output directory --- #
    dirpath = "./TASE/plotting/plots/range_analysis/"
    os.makedirs(dirpath, exist_ok=True)

    # --- Make Plot and calculate correlation --- #
    plot_std_dev_boxplots(df, font_size=30, filename=f"{dirpath}recording_range_variation.pdf")
    print_correlation()


def make_plots_for_range_analysis():
    evaluate_range_recordings()

