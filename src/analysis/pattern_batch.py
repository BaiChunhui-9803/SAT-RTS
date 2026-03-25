import csv
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
import pyfpgrowth
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import BoundaryNorm
import os
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_data_paths, DEFAULT_MAP_ID, DEFAULT_DATA_ID, PROJECT_ROOT

# ==================== Global Parameter Configuration ====================
# Data source path parameters - use config module defaults
_default_paths = get_data_paths(DEFAULT_MAP_ID, DEFAULT_DATA_ID)
DEFAULT_LOG_PATH = _default_paths["action_log_path"]
DEFAULT_RESULT_PATH = _default_paths["game_result_path"]
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "analysis" / "pattern")

# Pattern mining parameters
DEFAULT_MIN_SUPPORT = 0.01  # Minimum support threshold
DEFAULT_MIN_PATTERN_LENGTH = 2  # Minimum pattern length
DEFAULT_MAX_PATTERN_LENGTH = 10  # Maximum pattern length

# Analysis parameters
DEFAULT_TOP_PERCENTILE = 1.0  # Analyze top percentage of results (1.0 = 100%)
DEFAULT_ENABLE_VISUALIZATION = True  # Enable visualization
DEFAULT_SAVE_RESULTS = True  # Save results to file
DEFAULT_ANALYSIS_MODE = "all"  # 'all', 'top_percentile', 'optimal_only'

# Visualization parameters
DEFAULT_FIGURE_SIZE = (5, 0.3)  # Size of each subplot
DEFAULT_COLOR_PALETTE = "viridis"  # Color palette
DEFAULT_JITTER = 0.2  # Scatter plot jitter
DEFAULT_VIOLIN_BW = 0.2  # Violin plot bandwidth

# Output file prefix
DEFAULT_OUTPUT_PREFIX = f"pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class PatternAnalyzer:
    """
    Sequence Pattern Analyzer - Supports pattern mining across all results
    """

    def __init__(
        self,
        log_path=DEFAULT_LOG_PATH,
        result_path=DEFAULT_RESULT_PATH,
        min_support=DEFAULT_MIN_SUPPORT,
        analysis_mode=DEFAULT_ANALYSIS_MODE,
        top_percentile=DEFAULT_TOP_PERCENTILE,
        output_dir=DEFAULT_OUTPUT_DIR,
        output_prefix=DEFAULT_OUTPUT_PREFIX,
    ):
        """
        Initialize pattern analyzer

        Args:
            log_path: Log file path
            result_path: Result file path
            min_support: Minimum support threshold
            analysis_mode: Analysis mode ('all', 'top_percentile', 'optimal_only')
            top_percentile: Percentile threshold when using top_percentile mode
            output_dir: Output directory
            output_prefix: Output file prefix
        """
        self.log_path = log_path
        self.result_path = result_path
        self.min_support = min_support
        self.analysis_mode = analysis_mode
        self.top_percentile = top_percentile
        self.output_dir = output_dir
        self.output_prefix = output_prefix

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize data containers
        self.sequences = []
        self.results = []
        self.marked_sequences = []
        self.pattern_dict = {}
        self.analysis_data = None

    def load_data(self):
        """Load sequence data and analysis result data"""
        print(f"Loading data...")
        print(f"  Log file: {self.log_path}")
        print(f"  Result file: {self.result_path}")

        self.sequences = self._read_csv()
        self.results = self._process_result_file()

        print(
            f"  Loading complete: {len(self.sequences)} sequences, {len(self.results)} results"
        )

        # Validate data consistency
        if len(self.sequences) != len(self.results):
            raise ValueError(
                f"Data length mismatch: {len(self.sequences)} sequences vs {len(self.results)} results"
            )

    def _read_csv(self):
        """Read CSV format sequence file"""
        sequences = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                for row in reader:
                    if row:  # Ensure row is not empty
                        sequence = row[0]  # Assume each row has only one string
                        elements = [
                            sequence[i : i + 2] for i in range(0, len(sequence), 2)
                        ]  # Each two characters as one element
                        sequences.append(elements)
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file not found: {self.log_path}")
        except Exception as e:
            raise Exception(f"Error reading log file: {e}")
        return sequences

    def _process_result_file(self):
        """Process result file and calculate result values"""
        results = []
        try:
            with open(self.result_path, "r", encoding="utf-8") as file:
                for line_num, line in enumerate(file, 1):
                    items = line.strip().split()
                    if len(items) >= 4:  # Ensure each line has at least 4 elements
                        try:
                            result = int(items[2]) + int(items[3])
                            results.append(result)
                        except ValueError:
                            print(
                                f"Warning: Line {line_num} contains non-integer values, will be skipped: '{line.strip()}'"
                            )
                    else:
                        print(
                            f"Warning: Line {line_num} has insufficient items, will be skipped: '{line.strip()}'"
                        )
        except FileNotFoundError:
            raise FileNotFoundError(f"Result file not found: {self.result_path}")
        except Exception as e:
            raise Exception(f"Error reading result file: {e}")
        return results

    def extract_patterns(self, min_pattern_length=DEFAULT_MIN_PATTERN_LENGTH):
        """Extract sequence patterns"""
        print(
            f"Extracting sequence patterns (min support: {self.min_support}, min pattern length: {min_pattern_length})..."
        )

        # Extract all continuous subsequence patterns
        all_patterns = []
        for sequence in self.sequences:
            for i in range(len(sequence)):
                for j in range(
                    i + min_pattern_length,
                    min(
                        len(sequence) + 1,
                        min_pattern_length + DEFAULT_MAX_PATTERN_LENGTH,
                    ),
                ):
                    sub_sequence = tuple(sequence[i:j])
                    all_patterns.append(sub_sequence)

        # Calculate pattern support
        pattern_support = {}
        for pattern in all_patterns:
            pattern_support[pattern] = pattern_support.get(pattern, 0) + 1

        # Filter frequent patterns
        min_support_count = len(self.sequences) * self.min_support
        frequent_patterns = {
            pattern: support
            for pattern, support in pattern_support.items()
            if support >= min_support_count
        }

        # Sort by support
        sorted_patterns = sorted(
            frequent_patterns.items(), key=lambda x: x[1], reverse=True
        )

        # Filter by length
        filtered_patterns = [
            (pattern, support)
            for pattern, support in sorted_patterns
            if len(pattern) >= min_pattern_length
        ]

        print(f"  Found {len(filtered_patterns)} frequent patterns")
        return filtered_patterns

    def replace_sequences_with_patterns(self, patterns):
        """Replace sequences with pattern-annotated sequences"""
        print("Annotating patterns in sequences...")

        marked_sequences = []
        pattern_strings = [pattern for pattern, _ in patterns]

        for sequence, result in zip(self.sequences, self.results):
            sequence_str = "".join(sequence)
            matched_patterns = []
            marked_sequence = sequence_str
            pattern_positions = {}

            # Process in descending order of pattern length to avoid overlap issues
            for pattern in sorted(pattern_strings, key=len, reverse=True):
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

            # Handle nested patterns
            marked_sequence = self._handle_nested_patterns(marked_sequence)
            marked_sequences.append(
                (matched_patterns, marked_sequence, result, pattern_positions)
            )

        print(f"  Completed annotation for {len(marked_sequences)} sequences")
        return marked_sequences

    def _handle_nested_patterns(self, marked_sequence):
        """Handle nested patterns, keep only outermost pattern"""
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
                if stack:
                    stack.pop()
                if not stack:
                    new_sequence.append("】")
            else:
                new_sequence.append(char)
        return "".join(new_sequence)

    def filter_sequences_by_mode(self):
        """Filter sequences based on analysis mode"""
        original_count = len(self.marked_sequences)

        if self.analysis_mode == "all":
            # Keep all sequences
            filtered_sequences = self.marked_sequences
            print(f"Using 'all' mode: keeping all {len(filtered_sequences)} sequences")

        elif self.analysis_mode == "top_percentile":
            # Keep top percentage of sequences
            sorted_sequences = sorted(
                self.marked_sequences, key=lambda x: x[2], reverse=True
            )
            top_count = max(1, int(len(sorted_sequences) * self.top_percentile))
            filtered_sequences = sorted_sequences[:top_count]
            print(
                f"Using 'top_percentile' mode: keeping top {self.top_percentile * 100}% ({len(filtered_sequences)} sequences) of best sequences"
            )

        elif self.analysis_mode == "optimal_only":
            # Keep only global optimal sequences
            if self.marked_sequences:
                max_result = max(seq[2] for seq in self.marked_sequences)
                optimal_sequences = [
                    seq for seq in self.marked_sequences if seq[2] == max_result
                ]
                filtered_sequences = optimal_sequences
                print(
                    f"Using 'optimal_only' mode: keeping global optimal sequences (result value: {max_result}, total {len(filtered_sequences)} sequences)"
                )
            else:
                filtered_sequences = []
                print("Using 'optimal_only' mode: no sequences found")
        else:
            raise ValueError(f"Unknown analysis mode: {self.analysis_mode}")

        print(
            f"Filtering complete: {original_count} -> {len(filtered_sequences)} sequences"
        )
        return filtered_sequences

    def build_pattern_dict(self, sequences_to_analyze=None):
        """Build pattern dictionary"""
        if sequences_to_analyze is None:
            sequences_to_analyze = self.marked_sequences

        print("Building pattern dictionary...")

        pattern_dict = {}

        for (
            matched_patterns,
            marked_sequence,
            result,
            pattern_positions,
        ) in sequences_to_analyze:
            for pattern in matched_patterns:
                position = pattern_positions[pattern]
                if pattern not in pattern_dict:
                    pattern_dict[pattern] = []

                # Find similar (position, result) combinations
                found = False
                for entry in pattern_dict[pattern]:
                    if np.isclose(entry[0], position, atol=0.05) and np.isclose(
                        entry[1], result, atol=1
                    ):
                        entry[2] += 1
                        found = True
                        break

                if not found:
                    pattern_dict[pattern].append([position, result, 1])

        print(f"  Build complete: {len(pattern_dict)} patterns")
        return pattern_dict

    def create_dataframe(self, pattern_dict):
        """Create DataFrame for analysis"""
        print("Creating analysis DataFrame...")

        all_positions = []
        all_results = []
        all_patterns = []
        all_frequencies = []
        all_normalized_positions = []  # Normalized position

        for pattern, entries in pattern_dict.items():
            for entry in entries:
                position, result, frequency = entry

                all_positions.extend([position] * frequency)
                all_results.extend([result] * frequency)
                all_patterns.extend([pattern] * frequency)
                all_frequencies.extend([frequency] * frequency)
                all_normalized_positions.extend(
                    [position] * frequency
                )  # Already normalized

        data = pd.DataFrame(
            {
                "Pattern": all_patterns,
                "Position": all_positions,
                "Result": all_results,
                "Frequency": all_frequencies,
                "Normalized_Position": all_normalized_positions,
            }
        )

        print(f"  DataFrame created: {len(data)} rows of data")
        return data

    def visualize_patterns(self, data, save_path=None):
        """Visualize pattern analysis results"""
        if not DEFAULT_ENABLE_VISUALIZATION:
            print("Visualization disabled")
            return

        print("Generating visualization charts...")

        unique_patterns = data["Pattern"].unique()
        num_patterns = len(unique_patterns)

        if num_patterns == 0:
            print("No pattern data found, skipping visualization")
            return

        # Create subplots
        fig, axes = plt.subplots(
            num_patterns,
            1,
            figsize=(DEFAULT_FIGURE_SIZE[0], DEFAULT_FIGURE_SIZE[1] * num_patterns),
            sharex=True,
        )

        if num_patterns == 1:
            axes = [axes]  # Ensure axes is always a list

        # Set color palette
        palette = sns.color_palette(
            DEFAULT_COLOR_PALETTE, n_colors=len(data["Result"].unique())
        )

        # Create chart for each pattern
        for i, (pattern, ax) in enumerate(zip(unique_patterns, axes)):
            subset = data[data["Pattern"] == pattern].copy()
            subset["Result"] = -subset[
                "Result"
            ]  # Invert result values for color mapping

            # Create scatter plot
            sns.stripplot(
                x="Position",
                y="Pattern",
                hue="Result",
                data=subset,
                size=5,
                jitter=DEFAULT_JITTER,
                dodge=True,
                ax=ax,
                legend=False,
                palette=DEFAULT_COLOR_PALETTE + "_r",
            )

            # Create violin plot
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
                bw_method=DEFAULT_VIOLIN_BW,
            )

            # Set chart style
            ax.set_ylabel("")
            ax.set_xlim(-0.05, 1.05)
            ax.set_title(f"Pattern: {pattern}", fontsize=10, pad=10)

            # Add reference lines
            for x in [0, 0.2, 0.4, 0.6, 0.8, 1]:
                ax.axvline(x=x, color="#d7d7d7", linestyle="dotted", linewidth=0.8)
            ax.axhline(y=0, color="#d7d7d7", linestyle="dotted", linewidth=0.8)

            # Set borders and ticks
            for spine in ax.spines.values():
                spine.set_color("#d7d7d7")
            ax.tick_params(axis="both", colors="#d7d7d7")
            ax.tick_params(axis="both", labelcolor="black")
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            ax.set_xticklabels([])

        # Set x-axis for last subplot
        last_ax = axes[-1]
        last_ax.spines["bottom"].set_visible(True)
        last_ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        last_ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        last_ax.xaxis.set_ticks_position("bottom")

        # Adjust layout
        plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.1, left=0.3)

        # Add color bar
        if not data.empty:
            cbar_ax = fig.add_axes([0.15, 0.98, 0.7, 1 / num_patterns * 0.3])
            norm = plt.Normalize(data["Result"].min(), data["Result"].max())
            sm = plt.cm.ScalarMappable(cmap=DEFAULT_COLOR_PALETTE, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")

            result_min = data["Result"].min()
            result_max = data["Result"].max()
            tick_positions = np.linspace(0, 1, min(9, len(data["Result"].unique())))
            tick_labels = np.linspace(result_min, result_max, len(tick_positions))
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels([f"{int(label)}" for label in tick_labels])
            cbar.set_label("Game Result", fontsize=10)

        # Save or show chart
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Chart saved to: {save_path}")
        else:
            plt.show()

    def save_results(self, data, pattern_dict):
        """Save analysis results"""
        if not DEFAULT_SAVE_RESULTS:
            print("Save results disabled")
            return

        print("Saving analysis results...")

        # Save detailed data
        data_path = os.path.join(self.output_dir, f"{self.output_prefix}_data.csv")
        data.to_csv(data_path, index=False, encoding="utf-8")

        # Save pattern statistics
        pattern_stats = []
        for pattern, entries in pattern_dict.items():
            total_freq = sum(entry[2] for entry in entries)
            avg_result = sum(entry[1] * entry[2] for entry in entries) / total_freq
            pattern_stats.append(
                {
                    "Pattern": pattern,
                    "Total_Frequency": total_freq,
                    "Average_Result": avg_result,
                    "Unique_Positions": len(entries),
                }
            )

        stats_df = pd.DataFrame(pattern_stats)
        stats_path = os.path.join(self.output_dir, f"{self.output_prefix}_stats.csv")
        stats_df.to_csv(stats_path, index=False, encoding="utf-8")

        # Save configuration info
        config_path = os.path.join(self.output_dir, f"{self.output_prefix}_config.txt")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("Pattern Analysis Configuration\n")
            f.write("=" * 50 + "\n")
            f.write(f"Log file path: {self.log_path}\n")
            f.write(f"Result file path: {self.result_path}\n")
            f.write(f"Minimum support: {self.min_support}\n")
            f.write(f"Analysis mode: {self.analysis_mode}\n")
            f.write(f"Percentile threshold: {self.top_percentile}\n")
            f.write(f"Total sequences: {len(self.sequences)}\n")
            f.write(f"Analyzed sequences: {len(self.marked_sequences)}\n")
            f.write(f"Patterns found: {len(pattern_dict)}\n")
            f.write(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Results saved to: {self.output_dir}")
        print(f"  - Data file: {data_path}")
        print(f"  - Statistics file: {stats_path}")
        print(f"  - Configuration file: {config_path}")

    def run_analysis(self, min_pattern_length=DEFAULT_MIN_PATTERN_LENGTH):
        """Run complete pattern analysis pipeline"""
        print("=" * 80)
        print("Starting Sequence Pattern Analysis")
        print("=" * 80)

        try:
            # 1. Load data
            self.load_data()

            # 2. Extract patterns
            patterns = self.extract_patterns(min_pattern_length)

            # 3. Annotate sequences
            self.marked_sequences = self.replace_sequences_with_patterns(patterns)

            # 4. Filter sequences by mode
            sequences_to_analyze = self.filter_sequences_by_mode()

            # 5. Build pattern dictionary
            self.pattern_dict = self.build_pattern_dict(sequences_to_analyze)

            # 6. Create DataFrame
            self.analysis_data = self.create_dataframe(self.pattern_dict)

            # 7. Visualize
            if DEFAULT_ENABLE_VISUALIZATION:
                viz_path = os.path.join(
                    self.output_dir, f"{self.output_prefix}_visualization.png"
                )
                self.visualize_patterns(self.analysis_data, save_path=viz_path)

            # 8. Save results
            self.save_results(self.analysis_data, self.pattern_dict)

            print("=" * 80)
            print("Pattern Analysis Complete!")
            print("=" * 80)

            return self.analysis_data, self.pattern_dict

        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


# ==================== Convenience Functions ====================
def run_comprehensive_analysis(
    log_path=DEFAULT_LOG_PATH,
    result_path=DEFAULT_RESULT_PATH,
    min_support=DEFAULT_MIN_SUPPORT,
    analysis_modes=["all", "top_percentile", "optimal_only"],
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Run comprehensive analysis (multi-mode comparison)

    Args:
        log_path: Log file path
        result_path: Result file path
        min_support: Minimum support
        analysis_modes: List of analysis modes to run
        output_dir: Output directory

    Returns:
        dict: Analysis results for each mode
    """
    results = {}

    for mode in analysis_modes:
        print(f"\nRunning analysis mode: {mode}")
        print("-" * 50)

        analyzer = PatternAnalyzer(
            log_path=log_path,
            result_path=result_path,
            min_support=min_support,
            analysis_mode=mode,
            output_dir=output_dir,
            output_prefix=f"pattern_analysis_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        try:
            data, pattern_dict = analyzer.run_analysis()
            results[mode] = {
                "data": data,
                "pattern_dict": pattern_dict,
                "analyzer": analyzer,
            }
        except Exception as e:
            print(f"Mode {mode} analysis failed: {e}")
            results[mode] = {"error": str(e)}

    return results


# ==================== Main Program Entry ====================
if __name__ == "__main__":
    # Example usage 1: Basic analysis
    print("Example 1: Basic Pattern Analysis")
    analyzer = PatternAnalyzer(
        log_path="action_log.csv",
        result_path="game_result.txt",
        min_support=0.01,
        analysis_mode="all",  # Analyze all results, not just optimal solutions
    )

    data, pattern_dict = analyzer.run_analysis()

    # Example usage 2: Comprehensive analysis
    print("\nExample 2: Comprehensive Pattern Analysis (Multi-mode Comparison)")
    results = run_comprehensive_analysis(
        log_path="action_log.csv",
        result_path="game_result.txt",
        min_support=0.01,
        analysis_modes=["all", "top_percentile", "optimal_only"],
    )

    # Print summary
    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)

    for mode, result in results.items():
        if "error" in result:
            print(f"{mode}: Failed - {result['error']}")
        else:
            data = result["data"]
            pattern_dict = result["pattern_dict"]
            print(
                f"{mode}: Success - {len(pattern_dict)} patterns, {len(data)} rows of data"
            )
