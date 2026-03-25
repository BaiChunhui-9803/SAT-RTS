import csv
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import pyfpgrowth
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import BoundaryNorm


# 1. Read CSV file and preprocess data
def read_csv(file_path):
    sequences = []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            sequence = row[0]  # Assume each row has only one string
            elements = [
                sequence[i : i + 2] for i in range(0, len(sequence), 2)
            ]  # Each two characters as one element
            sequences.append(elements)
    return sequences


# 2. Read result file and calculate results
def process_file(file_path):
    results = []  # Used to store calculation results for each line
    with open(file_path, "r") as file:
        for line in file:
            items = line.strip().split()
            if len(items) >= 4:  # Ensure each line has at least 4 elements
                try:
                    result = int(items[2]) + int(items[3])
                    results.append(result)
                except ValueError:
                    print(
                        f"Warning: Line '{line.strip()}' contains non-integer values and will be skipped."
                    )
            else:
                print(
                    f"Warning: Line '{line.strip()}' does not have enough items and will be skipped."
                )
    return results


# 3. Extract continuous subsequence patterns
def extract_continuous_patterns(sequences, min_support=0.1):
    all_patterns = []
    for sequence in sequences:
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence) + 1):
                sub_sequence = tuple(
                    sequence[i:j]
                )  # Use tuple to represent continuous subsequence
                all_patterns.append(sub_sequence)

    pattern_support = {}
    for pattern in all_patterns:
        if pattern not in pattern_support:
            pattern_support[pattern] = 0
        pattern_support[pattern] += 1

    min_support_count = len(sequences) * min_support
    frequent_patterns = {
        pattern: support
        for pattern, support in pattern_support.items()
        if support >= min_support_count
    }

    sorted_frequent_patterns = sorted(
        frequent_patterns.items(), key=lambda x: x[1], reverse=True
    )
    return sorted_frequent_patterns


# 4. Replace sequences with sequence patterns + complete sequence list, and annotate patterns
def replace_sequences_with_patterns(sequences, patterns, results):
    new_sequences = []
    for sequence, result in zip(sequences, results):
        sequence_str = "".join(sequence)
        matched_patterns = []
        marked_sequence = sequence_str  # Initialize as original sequence
        pattern_positions = {}  # Record starting position of each pattern

        for pattern in sorted(
            patterns, key=len, reverse=True
        ):  # Process in descending order of pattern length
            pattern_str = "".join(pattern)
            if pattern_str in sequence_str:
                matched_patterns.append(pattern_str)
                marked_sequence = marked_sequence.replace(
                    pattern_str, f"[{pattern_str}]"
                )
                start_index = sequence_str.find(pattern_str)
                position = (
                    start_index / len(sequence_str) if len(sequence_str) > 0 else 0
                )
                pattern_positions[pattern_str] = position

        marked_sequence = handle_nested_patterns(marked_sequence)
        new_sequences.append(
            (matched_patterns, marked_sequence, result, pattern_positions)
        )
    return new_sequences


# 5. Function to get marked_sequences
def get_marked_sequences(log_path, result_path, min_support=0.01):
    # Read data
    sequences = read_csv(log_path)
    results = process_file(result_path)

    # Extract continuous subsequence patterns
    continuous_patterns = extract_continuous_patterns(
        sequences, min_support=min_support
    )

    # Filter patterns with length >= 2
    filtered_continuous_patterns = [
        (pattern, support)
        for pattern, support in continuous_patterns
        if len(pattern) >= 2
    ]

    # Replace sequences with sequence patterns + complete sequence list, and annotate patterns
    marked_sequences = replace_sequences_with_patterns(
        sequences, [pattern for pattern, _ in filtered_continuous_patterns], results
    )

    return marked_sequences


# 6. Handle nested patterns, keep only outermost pattern
def handle_nested_patterns(marked_sequence):
    stack = []
    new_sequence = []
    for char in marked_sequence:
        if char == "[":
            if stack:
                stack.append(char)
            else:
                new_sequence.append("【")
                stack.append(char)
        elif char == "]":
            stack.pop()
            if not stack:
                new_sequence.append("】")
        else:
            new_sequence.append(char)
    return "".join(new_sequence)


# 7. Build pattern dictionary
def build_pattern_dict(marked_sequences, top_n=None):
    pattern_dict = {}
    # Determine number of sequences to process based on top_n value
    if top_n is None:
        sequences_to_process = marked_sequences
    else:
        sequences_to_process = marked_sequences[:top_n]

    for (
        matched_patterns,
        marked_sequence,
        result,
        pattern_positions,
    ) in sequences_to_process:
        for pattern in matched_patterns:
            position = pattern_positions[pattern]
            if pattern not in pattern_dict:
                pattern_dict[pattern] = []
            found = False
            for entry in pattern_dict[pattern]:
                if np.isclose(entry[0], position) and np.isclose(entry[1], result):
                    entry[2] += 1
                    found = True
                    break
            if not found:
                pattern_dict[pattern].append([position, result, 1])
    return pattern_dict


# 8. Create DataFrame
def create_dataframe(pattern_dict):
    all_positions = []
    all_results = []
    all_patterns = []
    all_frequencies = []

    for pattern, entries in pattern_dict.items():
        for entry in entries:
            position, result, frequency = entry
            all_positions.extend([position] * frequency)
            all_results.extend([result] * frequency)
            all_patterns.extend([pattern] * frequency)
            all_frequencies.extend([frequency] * frequency)

    data = pd.DataFrame(
        {
            "Pattern": all_patterns,
            "Position": all_positions,
            "Result": all_results,
            "Frequency": all_frequencies,
        }
    )
    return data


# 9. Plotting function
def plot_pattern_analysis(data):
    num_patterns = len(data["Pattern"].unique())

    # plt.rcParams['figure.dpi'] = 1500
    fig, axes = plt.subplots(
        num_patterns, 1, figsize=(5, 0.3 * num_patterns), sharex=True
    )

    palette = sns.color_palette("viridis", n_colors=len(data["Result"].unique()))

    for i, (pattern, ax) in enumerate(zip(data["Pattern"].unique(), axes)):
        subset = data[data["Pattern"] == pattern].copy()
        subset["Result"] = -subset["Result"]

        sns.stripplot(
            x="Position",
            y="Pattern",
            hue="Result",
            data=subset,
            size=5,
            jitter=0.2,
            dodge=True,
            ax=ax,
            legend=False,
            palette="viridis_r",
        )

        sns.violinplot(
            x="Position",
            y="Pattern",
            data=subset,
            ax=ax,
            inner=None,
            color="lightgray",
            linewidth=0,
            alpha=0.75,
            density_norm="count",
            bw_method=0.2,
        )

        ax.set_ylabel("")
        ax.set_xlim(-0.05, 1.05)

        for x in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            ax.axvline(x=x, color="#d7d7d7", linestyle="dotted", linewidth=0.8)

        ax.axhline(y=0, color="#d7d7d7", linestyle="dotted", linewidth=0.8)

        for spine in ax.spines.values():
            spine.set_color("#d7d7d7")

        ax.tick_params(axis="both", colors="#d7d7d7")
        ax.tick_params(axis="both", labelcolor="black")

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.set_xticks([])
        ax.set_xticklabels([])

    last_ax = axes[-1]
    last_ax.spines["bottom"].set_visible(True)
    last_ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    last_ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    last_ax.xaxis.set_ticks_position("bottom")

    plt.subplots_adjust(hspace=0, top=0.98, bottom=0.01)
    plt.subplots_adjust(left=0.4)

    cbar_ax = fig.add_axes([0.1, 0.99, 0.8, 1 / num_patterns * 0.3])
    norm = plt.Normalize(data["Result"].min(), data["Result"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")

    result_min = data["Result"].min()
    result_max = data["Result"].max()
    tick_positions = np.linspace(0, 1, 9)
    tick_labels = np.linspace(result_min, result_max, 9)
    cbar.set_ticks(tick_labels)
    cbar.set_ticklabels(tick_labels.astype(int))

    plt.show()
    # plt.savefig('stacked_bee_swarm_plots.png', dpi=1500)


# Main program
if __name__ == "__main__":
    log_path = "action_log.csv"
    result_path = "game_result.txt"

    # Get marked_sequences
    marked_sequences = get_marked_sequences(log_path, result_path, min_support=0.01)

    # Print result for each sequence
    print("\nSequences with annotated patterns:")
    for (
        matched_patterns,
        marked_sequence,
        result,
        pattern_positions,
    ) in marked_sequences:
        if matched_patterns:
            print(
                f"Marked Sequence: {marked_sequence}, Result: {result}, Patterns: {matched_patterns}, Positions: {pattern_positions}"
            )
        else:
            print(
                f"No pattern found, Original Sequence: {marked_sequence}, Result: {result}"
            )

    # Find all sequences with highest result as global optimal solution
    if marked_sequences:
        # 1. Find global optimal solution (all sequences with highest result)
        marked_sequences.sort(key=lambda x: x[2], reverse=True)
        max_result = marked_sequences[0][2]
        best_sequences = [seq for seq in marked_sequences if seq[2] == max_result]

        print(
            f"\nGlobal optimal solution sequences (highest result value is {max_result}):"
        )
        for (
            best_sequence,
            best_marked_sequence,
            best_result,
            best_pattern_positions,
        ) in best_sequences:
            print(
                f"Marked Sequence: {best_marked_sequence}, Result: {best_result}, Pattern: {best_sequence}, Positions: {best_pattern_positions}"
            )

        # 2. Output top 0.1% results (sorted by result descending)
        marked_sequences_sorted = sorted(
            marked_sequences, key=lambda x: x[2], reverse=True
        )
        top_count = max(
            1, int(len(marked_sequences_sorted) * 0.01)
        )  # Output at least 1
        top_sequences = marked_sequences_sorted[:top_count]

        print(
            f"\nTop 0.1% sequences by result value (total {len(top_sequences)} sequences):"
        )
        for seq, marked, res, pos in top_sequences:
            print(
                f"Marked Sequence: {marked}, Result: {res}, Pattern: {seq}, Positions: {pos}"
            )
    else:
        print("\nNo sequences found matching criteria.")

    # Build pattern dictionary
    pattern_dict = build_pattern_dict(marked_sequences, top_n=None)

    # Create DataFrame
    data = create_dataframe(pattern_dict)

    # Plot
    plot_pattern_analysis(data)
