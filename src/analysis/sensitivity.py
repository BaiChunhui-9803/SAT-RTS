import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

# Set Chinese font
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class ParameterSensitivityAnalyzer:
    """
    Parameter Sensitivity Analyzer
    Used to analyze the impact of various parameters on clustering results in BKTree clustering algorithm
    """

    def __init__(self, results_file_path):
        """
        Initialize analyzer

        Args:
            results_file_path: Results file path
        """
        self.results_file_path = results_file_path
        self.data = None
        self.quality_scores = None

    def load_results(self):
        """Load experiment results data"""
        try:
            # Read data, skip preceding description lines
            with open(self.results_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Find the line where data starts (line containing "Grid")
            start_line = 0
            for i, line in enumerate(lines):
                if "Grid" in line and "Self" in line and "Silhouette" in line:
                    start_line = i + 1
                    break

            # Read data lines
            data_lines = []
            for line in lines[start_line:]:
                line = line.strip()
                if line and not line.startswith(
                    "-"
                ):  # Skip empty lines and separator lines
                    data_lines.append(line)

            # Parse data
            data = []
            for line in data_lines:
                parts = line.split("\t")
                if len(parts) >= 9:  # Ensure sufficient columns
                    try:
                        grid = int(parts[0])
                        self_num = int(parts[1])
                        enemy_num = int(parts[2])
                        groups = int(parts[3])
                        samples = int(parts[4])
                        epi = float(parts[5])
                        clus = int(parts[6])
                        silhouette = float(parts[7])
                        accuracy = float(parts[8])

                        # Calculate Scale variable to capture problem scale
                        scale = (self_num + enemy_num) / 2

                        data.append(
                            {
                                "Grid": grid,
                                "Self": self_num,
                                "Enemy": enemy_num,
                                "Groups": groups,
                                "Samples": samples,
                                "Epi": epi,
                                "Clus": clus,
                                "Silhouette": silhouette,
                                "Accuracy": accuracy,
                                "Scale": scale,
                            }
                        )
                    except ValueError:
                        continue

            self.data = pd.DataFrame(data)
            print(f"Successfully loaded {len(self.data)} experiment records")
            print(
                f"Scale variable range: {self.data['Scale'].min():.1f} - {self.data['Scale'].max():.1f}"
            )
            return True

        except Exception as e:
            print(f"Failed to load data: {e}")
            return False

    def calculate_quality_scores(self):
        """
        Calculate comprehensive quality scores
        Combines accuracy, silhouette coefficient, and clustering reasonableness
        """
        if self.data is None:
            return False

        quality_scores = []

        for _, row in self.data.iterrows():
            accuracy = row["Accuracy"]
            silhouette = row["Silhouette"]
            groups = row["Groups"]
            clus = row["Clus"]

            # Base score: weighted average of accuracy and silhouette coefficient
            base_score = (
                0.7 * accuracy + 0.3 * (silhouette + 1) / 2
            )  # Map silhouette from [-1,1] to [0,1]

            # Anomaly detection and penalty mechanism
            penalty = 0

            # 1. Detect cases with abnormally high silhouette but low accuracy
            if silhouette >= 0.99 and accuracy < 0.8:
                penalty += 0.5  # Severe penalty, possible clustering failure
                print(
                    f"Anomaly detected: silhouette={silhouette:.3f} but accuracy={accuracy:.3f}, possible clustering issues"
                )

            # 2. Detect cases where cluster count differs significantly from true cluster count
            cluster_ratio = (
                max(clus / groups, groups / clus) if groups > 0 and clus > 0 else 1
            )
            if cluster_ratio > 2.0:
                penalty += 0.3 * (cluster_ratio - 2.0) / 2.0  # Proportional penalty
            elif cluster_ratio > 1.5:
                penalty += 0.1 * (cluster_ratio - 1.5) / 0.5

            # 3. Penalty for very low silhouette coefficient
            if silhouette < 0:
                penalty += 0.2 * abs(silhouette)

            # Calculate final score
            final_score = max(0, base_score - penalty)
            quality_scores.append(final_score)

        self.data["Quality_Score"] = quality_scores
        self.quality_scores = quality_scores
        print(
            f"Quality score calculation complete, score range: {min(quality_scores):.3f} - {max(quality_scores):.3f}"
        )
        return True

    def analyze_parameter_sensitivity(self):
        """
        Analyze parameter sensitivity
        Analyze the impact of Grid, Scale, Groups, Epi parameters on results
        """
        if self.data is None or self.quality_scores is None:
            print("Please load data and calculate quality scores first")
            return

        # Define parameters to analyze
        params_to_analyze = ["Grid", "Scale", "Groups", "Epi"]
        sensitivity_results = {}

        # Analyze impact of each parameter
        for param in params_to_analyze:
            if param not in self.data.columns:
                print(f"Warning: Parameter {param} does not exist in data")
                continue

            param_stats = (
                self.data.groupby(param)
                .agg(
                    {
                        "Silhouette": ["mean", "std", "count"],
                        "Accuracy": ["mean", "std"],
                        "Quality_Score": ["mean", "std"],
                        "Clus": ["mean", "std"],
                    }
                )
                .round(4)
            )

            print(f"\n=== {param} Parameter Sensitivity Analysis ===")
            print(param_stats)

            # Calculate sensitivity metric
            sensitivity = self._calculate_sensitivity_metric(param)
            sensitivity_results[param] = sensitivity

        # Print parameter sensitivity comparison
        print(f"\n=== Parameter Sensitivity Comparison ===")
        sorted_sensitivities = sorted(
            sensitivity_results.items(), key=lambda x: x[1], reverse=True
        )

        for param, sensitivity in sorted_sensitivities:
            print(f"{param} parameter sensitivity: {sensitivity:.4f}")

        # Determine most influential parameter
        most_sensitive_param = max(sensitivity_results, key=sensitivity_results.get)
        print(
            f"\nConclusion: {most_sensitive_param} parameter has the greatest impact on clustering results"
        )

        return sensitivity_results

    def analyze_multi_parameter_correlation(self):
        """
        Analyze correlations between multiple parameters and their combined impact on clustering results
        """
        if self.data is None or self.quality_scores is None:
            print("Please load data and calculate quality scores first")
            return

        # Calculate parameter correlation matrix
        params_to_analyze = ["Grid", "Scale", "Groups", "Epi"]
        available_params = [p for p in params_to_analyze if p in self.data.columns]

        if len(available_params) < 2:
            print("Insufficient parameters for correlation analysis")
            return

        # Calculate correlations between parameters
        param_correlations = self.data[available_params].corr()
        print("\n=== Parameter Correlation Analysis ===")
        print(param_correlations.round(4))

        # Analyze impact of parameter combinations on results
        print("\n=== Parameter Combination Impact Analysis ===")

        # Find optimal configurations for parameter combinations
        best_combinations = self.data.nlargest(5, "Quality_Score")
        print("Top 5 parameter combinations by quality score:")
        for i, (_, row) in enumerate(best_combinations.iterrows(), 1):
            print(
                f"{i}. Grid={row['Grid']}, Scale={row['Scale']:.1f}, Groups={row['Groups']}, "
                f"Epi={row['Epi']:.1f} -> Score={row['Quality_Score']:.4f}"
            )

        # Calculate ANOVA for each parameter
        print("\n=== Parameter ANOVA Analysis ===")
        for param in available_params:
            if param in self.data.columns:
                groups = self.data.groupby(param)["Quality_Score"]
                if len(groups) > 1:
                    # Simple ANOVA
                    between_group_var = groups.var().mean()
                    within_group_var = self.data["Quality_Score"].var()
                    f_statistic = between_group_var / (within_group_var + 1e-10)
                    print(f"{param}: F-statistic = {f_statistic:.4f}")

    def find_optimal_parameter_ranges(self):
        """
        Find optimal ranges for each parameter
        """
        if self.data is None or self.quality_scores is None:
            print("Please load data and calculate quality scores first")
            return

        params_to_analyze = ["Grid", "Scale", "Groups", "Epi"]
        available_params = [p for p in params_to_analyze if p in self.data.columns]

        print("\n=== Optimal Parameter Range Analysis ===")

        for param in available_params:
            # Calculate quality score distribution for each parameter value
            param_quality = self.data.groupby(param)["Quality_Score"].agg(
                ["mean", "std", "count"]
            )

            # Find parameter values with highest quality scores
            best_values = param_quality.nlargest(3, "mean")

            print(f"\nOptimal values for {param} parameter:")
            for value in best_values.index:
                mean_score = best_values.loc[value, "mean"]
                std_score = best_values.loc[value, "std"]
                count = best_values.loc[value, "count"]
                print(
                    f"  {param}={value}: Average score={mean_score:.4f} (+/-{std_score:.4f}, sample count={count})"
                )

            # Provide recommended range
            if len(best_values) >= 2:
                min_opt = best_values.index.min()
                max_opt = best_values.index.max()
                print(f"  Recommended range: {param} in [{min_opt}, {max_opt}]")

    def calculate_parameter_importance(self):
        """
        Calculate parameter importance (based on degree of impact on quality score)
        """
        if self.data is None or self.quality_scores is None:
            print("Please load data and calculate quality scores first")
            return

        params_to_analyze = ["Grid", "Scale", "Groups", "Epi"]
        available_params = [p for p in params_to_analyze if p in self.data.columns]

        importance_scores = {}

        for param in available_params:
            # Calculate correlation between parameter and quality score
            correlation = abs(self.data[param].corr(self.data["Quality_Score"]))

            # Calculate parameter sensitivity
            sensitivity = self._calculate_sensitivity_metric(param)

            # Combined importance score
            importance = correlation * 0.6 + sensitivity * 0.4
            importance_scores[param] = importance

        # Sort by importance
        sorted_importance = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )

        print("\n=== Parameter Importance Ranking ===")
        for i, (param, importance) in enumerate(sorted_importance, 1):
            print(f"{i}. {param}: {importance:.4f}")

        return importance_scores

    def _calculate_sensitivity_metric(self, param_name):
        """
        Calculate parameter sensitivity metric
        Uses coefficient of variation of results as sensitivity measure
        """
        if param_name not in self.data.columns:
            return 0

        param_groups = self.data.groupby(param_name)["Quality_Score"].agg(
            ["mean", "std"]
        )

        # Calculate coefficient of variation (std/mean)
        cv = (
            param_groups["std"].mean() / param_groups["mean"].mean()
            if param_groups["mean"].mean() > 0
            else 0
        )

        # Calculate impact of parameter range on results
        param_range = self.data[param_name].max() - self.data[param_name].min()
        if param_range == 0:
            return 0

        # Sensitivity = coefficient of variation * parameter impact weight
        sensitivity = cv * np.log1p(param_range)

        return sensitivity

    def create_visualization(self):
        """
        Create visualization charts for parameter sensitivity analysis
        Includes comprehensive analysis of Grid, Scale, Groups, Epi parameters
        """
        if self.data is None:
            print("Please load data first")
            return

        # Create larger figure to accommodate more parameters
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # First row: Single parameter impact analysis
        # 1. Impact of Grid parameter on results
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_parameter_effect("Grid", ax1, "Grid Parameter Impact")

        # 2. Impact of Scale parameter on results
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_parameter_effect("Scale", ax2, "Scale Parameter Impact")

        # 3. Impact of Groups parameter on results
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_parameter_effect("Groups", ax3, "Groups Parameter Impact")

        # 4. Impact of Epi parameter on results
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_parameter_effect("Epi", ax4, "Epi Parameter Impact")

        # Second row: Detailed relationships between parameters and different metrics
        # 5. Grid parameter distribution and quality score boxplot
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_parameter_boxplot(
            "Grid", "Quality_Score", ax5, "Grid vs Quality Score"
        )

        # 6. Scale parameter distribution and accuracy scatter plot
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_scatter_with_trend("Scale", "Accuracy", ax6, "Scale vs Accuracy")

        # 7. Groups parameter distribution and silhouette coefficient violin plot
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_parameter_violin("Groups", "Silhouette", ax7, "Groups vs Silhouette")

        # 8. Impact of Epi parameter on cluster count
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_scatter_with_trend("Epi", "Clus", ax8, "Epi vs Cluster Count")

        # Third row: Comprehensive analysis
        # 9. Four-parameter sensitivity comparison bar chart
        ax9 = fig.add_subplot(gs[2, 0])
        self._plot_sensitivity_comparison(ax9)

        # 10. Quality score distribution
        ax10 = fig.add_subplot(gs[2, 1])
        self._plot_quality_distribution(ax10)

        # 11. Parameter correlation heatmap
        ax11 = fig.add_subplot(gs[2, 2])
        self._plot_correlation_heatmap(ax11)

        # 12. Parameter importance analysis
        ax12 = fig.add_subplot(gs[2, 3])
        self._plot_parameter_importance(ax12)

        # Fourth row: Advanced analysis
        # 13. 3D scatter plot: Grid vs Scale vs Quality Score
        ax13 = fig.add_subplot(gs[3, 0:2], projection="3d")
        self._plot_3d_scatter(ax13)

        # 14. Optimal parameter combination analysis
        ax14 = fig.add_subplot(gs[3, 2])
        self._plot_optimal_combinations(ax14)

        # 15. Parameter range recommendations
        ax15 = fig.add_subplot(gs[3, 3])
        self._plot_parameter_ranges(ax15)

        plt.suptitle(
            "BKTree Clustering Algorithm Four-Parameter (Grid, Scale, Groups, Epi) Sensitivity Analysis",
            fontsize=18,
            fontweight="bold",
        )

        # Save figure
        output_path = "comprehensive_parameter_sensitivity_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Comprehensive analysis chart saved to: {output_path}")

        plt.show()

    def _plot_correlation_heatmap(self, ax):
        """Plot parameter correlation heatmap"""
        params_to_analyze = ["Grid", "Scale", "Groups", "Epi"]
        available_params = [p for p in params_to_analyze if p in self.data.columns]

        if len(available_params) < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient parameters\nCannot calculate correlation",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Calculate correlation matrix
        correlation_matrix = self.data[available_params].corr()

        # Plot heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            square=True,
            ax=ax,
        )
        ax.set_title("Parameter Correlation Matrix")

    def _plot_parameter_importance(self, ax):
        """Plot parameter importance analysis"""
        # Calculate parameter importance
        importance_scores = self.calculate_parameter_importance()

        if importance_scores:
            params = list(importance_scores.keys())
            importances = list(importance_scores.values())

            # Create gradient colors
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(params)))

            bars = ax.barh(
                params, importances, color=colors, alpha=0.8, edgecolor="black"
            )
            ax.set_xlabel("Importance Score")
            ax.set_title("Parameter Importance Ranking")

            # Add value labels
            for bar, importance in zip(bars, importances):
                width = bar.get_width()
                ax.text(
                    width + width * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{importance:.4f}",
                    ha="left",
                    va="center",
                )

    def _plot_3d_scatter(self, ax):
        """Plot 3D scatter plot analyzing relationship between three main parameters"""
        params_for_3d = ["Grid", "Scale", "Quality_Score"]

        if all(p in self.data.columns for p in params_for_3d):
            x = self.data["Grid"]
            y = self.data["Scale"]
            z = self.data["Quality_Score"]

            # Use quality score as color mapping
            scatter = ax.scatter(x, y, z, c=z, cmap="viridis", alpha=0.6, s=50)

            ax.set_xlabel("Grid")
            ax.set_ylabel("Scale")
            ax.set_zlabel("Quality Score")
            ax.set_title("Grid vs Scale vs Quality Score")

            # Add color bar
            plt.colorbar(scatter, ax=ax, shrink=0.5)
        else:
            ax.text2D(
                0.5,
                0.5,
                "Insufficient data\nCannot plot 3D scatter",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _plot_optimal_combinations(self, ax):
        """Plot optimal parameter combination analysis"""
        # Find top 10 combinations with highest quality scores
        top_combinations = self.data.nlargest(10, "Quality_Score")

        if len(top_combinations) > 0:
            # Create simplified representation of combinations
            combo_labels = [
                f"G{row['Grid']}_S{row['Scale']:.0f}_Gr{row['Groups']}_E{row['Epi']:.1f}"
                for _, row in top_combinations.iterrows()
            ]

            scores = top_combinations["Quality_Score"].values

            # Show only top 5 to avoid crowded labels
            display_limit = min(5, len(combo_labels))

            bars = ax.bar(
                range(display_limit),
                scores[:display_limit],
                color="lightgreen",
                alpha=0.7,
                edgecolor="black",
            )

            ax.set_xlabel("Parameter Combination")
            ax.set_ylabel("Quality Score")
            ax.set_title("Top 5 Optimal Parameter Combinations")
            ax.set_xticks(range(display_limit))
            ax.set_xticklabels(
                [f"Combo {i + 1}" for i in range(display_limit)], rotation=45
            )

            # Add score labels
            for bar, score in zip(bars, scores[:display_limit]):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    def _plot_parameter_ranges(self, ax):
        """Plot parameter range recommendations"""
        params_to_analyze = ["Grid", "Scale", "Groups", "Epi"]
        available_params = [p for p in params_to_analyze if p in self.data.columns]

        # Calculate recommended ranges for each parameter
        range_info = {}
        for param in available_params:
            param_quality = self.data.groupby(param)["Quality_Score"].mean()
            if len(param_quality) > 0:
                # Find parameter values in top 25% of scores
                threshold = param_quality.quantile(0.75)
                optimal_values = param_quality[param_quality >= threshold]

                if len(optimal_values) > 0:
                    range_info[param] = {
                        "min": optimal_values.index.min(),
                        "max": optimal_values.index.max(),
                        "optimal_count": len(optimal_values),
                    }

        if range_info:
            params = list(range_info.keys())
            min_vals = [range_info[p]["min"] for p in params]
            max_vals = [range_info[p]["max"] for p in params]

            # Plot range bar chart
            y_pos = range(len(params))
            ax.barh(
                y_pos,
                [max_vals[i] - min_vals[i] for i in range(len(params))],
                left=min_vals,
                alpha=0.6,
                color="lightblue",
                edgecolor="black",
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(params)
            ax.set_xlabel("Parameter Value Range")
            ax.set_title(
                "Recommended Parameter Ranges (Based on Top 25% Quality Scores)"
            )

            # Add range labels
            for i, (param, info) in enumerate(range_info.items()):
                ax.text(
                    info["min"],
                    i,
                    f"{info['min']}",
                    ha="right",
                    va="center",
                    fontsize=8,
                )
                ax.text(
                    info["max"], i, f"{info['max']}", ha="left", va="center", fontsize=8
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data\nCannot calculate parameter ranges",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _plot_parameter_boxplot(self, param_name, metric_name, ax, title):
        """Plot parameter vs metric boxplot"""
        if param_name not in self.data.columns or metric_name not in self.data.columns:
            ax.text(
                0.5,
                0.5,
                f"{title}\nInsufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Use pandas and seaborn to plot boxplot
        unique_params = sorted(self.data[param_name].unique())
        if len(unique_params) <= 10:  # Only plot when parameter values are not too many
            data_list = []
            labels = []

            for param_val in unique_params:
                metric_vals = self.data[self.data[param_name] == param_val][metric_name]
                if len(metric_vals) > 0:
                    data_list.append(metric_vals)
                    labels.append(f"{param_val}")

            if data_list:
                bp = ax.boxplot(data_list, labels=labels, patch_artist=True)

                # Set colors
                colors = plt.cm.Set3(np.linspace(0, 1, len(bp["boxes"])))
                for box, color in zip(bp["boxes"], colors):
                    box.set_facecolor(color)
                    box.set_alpha(0.7)

                ax.set_xlabel(param_name)
                ax.set_ylabel(metric_name)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{title}\nInsufficient data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            # When parameter values are too many, use scatter plot instead
            ax.scatter(self.data[param_name], self.data[metric_name], alpha=0.6, s=30)
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric_name)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    def _plot_scatter_with_trend(self, x_param, y_param, ax, title):
        """Plot scatter plot with trend line"""
        if x_param not in self.data.columns or y_param not in self.data.columns:
            ax.text(
                0.5,
                0.5,
                f"{title}\nInsufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        x = self.data[x_param]
        y = self.data[y_param]

        # Plot scatter
        ax.scatter(x, y, alpha=0.6, s=40, c="blue", edgecolors="black", linewidths=0.5)

        # Calculate trend line
        if len(x) > 1 and x.nunique() > 1:
            # Use numpy to calculate trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x.min(), x.max(), 100)
            y_trend = p(x_trend)

            ax.plot(
                x_trend,
                y_trend,
                "r--",
                alpha=0.8,
                linewidth=2,
                label=f"Trend line: y={z[0]:.3f}x+{z[1]:.3f}",
            )

            # Calculate correlation coefficient
            correlation = np.corrcoef(x, y)[0, 1]
            ax.text(
                0.05,
                0.95,
                f"Correlation: {correlation:.3f}",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if "z" in locals():
            ax.legend()

    def _plot_parameter_violin(self, param_name, metric_name, ax, title):
        """Plot parameter vs metric violin plot"""
        if param_name not in self.data.columns or metric_name not in self.data.columns:
            ax.text(
                0.5,
                0.5,
                f"{title}\nInsufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        unique_params = sorted(self.data[param_name].unique())
        if len(unique_params) <= 8:  # Only plot when parameter values are not too many
            data_list = []
            labels = []

            for param_val in unique_params:
                metric_vals = self.data[self.data[param_name] == param_val][metric_name]
                if len(metric_vals) > 0:
                    data_list.append(metric_vals.tolist())
                    labels.append(f"{param_val}")

            if data_list:
                # Use seaborn-style violin plot
                parts = ax.violinplot(
                    data_list,
                    positions=range(len(data_list)),
                    showmeans=True,
                    showmedians=True,
                )

                # Set colors
                colors = plt.cm.Set2(np.linspace(0, 1, len(parts["bodies"])))
                for pc, color in zip(parts["bodies"], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)

                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_xlabel(param_name)
                ax.set_ylabel(metric_name)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{title}\nInsufficient data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            # When parameter values are too many, use boxplot instead
            self._plot_parameter_boxplot(
                param_name, metric_name, ax, title + " (Boxplot)"
            )

    def _plot_parameter_effect(self, param_name, ax, title):
        """Plot parameter impact on results"""
        if param_name not in self.data.columns:
            return

        grouped = self.data.groupby(param_name).agg(
            {"Accuracy": "mean", "Silhouette": "mean", "Quality_Score": "mean"}
        )

        x = grouped.index
        y1 = grouped["Accuracy"]
        y2 = (grouped["Silhouette"] + 1) / 2  # Normalize to [0,1]
        y3 = grouped["Quality_Score"]

        ax.plot(x, y1, "o-", label="Accuracy", linewidth=2, markersize=6)
        ax.plot(x, y2, "s-", label="Silhouette (Normalized)", linewidth=2, markersize=6)
        ax.plot(x, y3, "^-", label="Quality Score", linewidth=2, markersize=6)

        ax.set_xlabel(param_name)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_heatmap(self, param_name, metric_name, ax, title):
        """Plot parameter vs metric relationship heatmap"""
        if param_name not in self.data.columns or metric_name not in self.data.columns:
            ax.text(
                0.5,
                0.5,
                f"{title}\nInsufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Create parameter value distribution analysis
        param_values = sorted(self.data[param_name].unique())

        if len(param_values) < 2:
            # If parameter values are few, show bar chart directly
            param_metric = self.data.groupby(param_name)[metric_name].mean()
            bars = ax.bar(
                range(len(param_metric)),
                param_metric.values,
                alpha=0.7,
                color="lightblue",
                edgecolor="black",
            )
            ax.set_xticks(range(len(param_metric)))
            ax.set_xticklabels([str(v) for v in param_metric.index])
            ax.set_ylabel(metric_name)
            ax.set_title(title)

            # Add value labels
            for bar, val in zip(bars, param_metric.values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        else:
            # If parameter values are many, create bins for analysis
            param_bins = pd.cut(self.data[param_name], bins=min(10, len(param_values)))
            binned_data = self.data.groupby(param_bins)[metric_name].agg(
                ["mean", "count"]
            )

            if len(binned_data) > 1:
                # Create heatmap data
                heatmap_data = binned_data[["mean"]].T

                # Use color mapping
                sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlBu_r", ax=ax)
                ax.set_title(title)
                ax.set_xlabel(f"{param_name} (Binned)")

                # Adjust x-axis labels
                ax.set_xticklabels(
                    [
                        f"{interval.left:.1f}-{interval.right:.1f}"
                        for interval in binned_data.index
                    ],
                    rotation=45,
                    ha="right",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{title}\nInsufficient binned data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

    def _plot_sensitivity_comparison(self, ax):
        """Plot four-parameter sensitivity comparison"""
        params_to_analyze = ["Grid", "Scale", "Groups", "Epi"]
        available_params = [p for p in params_to_analyze if p in self.data.columns]

        sensitivities = []
        colors = ["skyblue", "lightcoral", "lightgreen", "lightsalmon"]

        for param in available_params:
            sens = self._calculate_sensitivity_metric(param)
            sensitivities.append(sens)

        bars = ax.bar(
            available_params,
            sensitivities,
            color=colors[: len(available_params)],
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_ylabel("Sensitivity Metric")
        ax.set_title("Four-Parameter Sensitivity Comparison")

        # Add value labels
        for bar, sens in zip(bars, sensitivities):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{sens:.4f}",
                ha="center",
                va="bottom",
            )

    def _plot_quality_distribution(self, ax):
        """Plot quality score distribution"""
        if "Quality_Score" not in self.data.columns:
            return

        ax.hist(
            self.data["Quality_Score"],
            bins=20,
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Quality Score Distribution")
        ax.axvline(
            self.data["Quality_Score"].mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {self.data['Quality_Score'].mean():.3f}",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_cluster_analysis(self, ax):
        """Plot cluster count change analysis"""
        # Calculate difference between cluster count and true cluster count
        if "Groups" in self.data.columns and "Clus" in self.data.columns:
            cluster_diff = self.data["Clus"] - self.data["Groups"]
            ax.scatter(
                cluster_diff,
                self.data["Accuracy"],
                alpha=0.6,
                c="blue",
                label="Accuracy vs Cluster Count Diff",
            )
            ax.scatter(
                cluster_diff,
                (self.data["Silhouette"] + 1) / 2,
                alpha=0.6,
                c="red",
                label="Silhouette vs Cluster Count Diff",
            )
            ax.set_xlabel("Cluster Count Diff (Clus - Groups)")
            ax.set_ylabel("Score (Normalized)")
            ax.set_title("Impact of Cluster Count Change on Results")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axvline(
                x=0, color="green", linestyle="--", alpha=0.7, label="Perfect Match"
            )

    def generate_report(self):
        """
        Generate parameter sensitivity analysis report
        Includes comprehensive analysis of Grid, Scale, Groups, Epi four parameters
        """
        if self.data is None:
            print("Please load data first")
            return

        report = []
        report.append(
            "=== BKTree Clustering Algorithm Four-Parameter (Grid, Scale, Groups, Epi) Sensitivity Analysis Report ===\n"
        )

        # Basic statistics
        report.append("1. Data Overview:")
        report.append(f"   - Experiment records: {len(self.data)}")
        report.append(
            f"   - Grid parameter range: {self.data['Grid'].min()} - {self.data['Grid'].max()}"
        )
        if "Scale" in self.data.columns:
            report.append(
                f"   - Scale parameter range: {self.data['Scale'].min():.1f} - {self.data['Scale'].max():.1f}"
            )
        if "Groups" in self.data.columns:
            report.append(
                f"   - Groups parameter range: {self.data['Groups'].min()} - {self.data['Groups'].max()}"
            )
        report.append(
            f"   - Epi parameter range: {self.data['Epi'].min():.1f} - {self.data['Epi'].max():.1f}"
        )
        report.append(f"   - Average accuracy: {self.data['Accuracy'].mean():.4f}")
        report.append(
            f"   - Average silhouette coefficient: {self.data['Silhouette'].mean():.4f}"
        )

        # Quality assessment
        if "Quality_Score" in self.data.columns:
            high_quality = (self.data["Quality_Score"] > 0.8).sum()
            medium_quality = (
                (self.data["Quality_Score"] >= 0.5)
                & (self.data["Quality_Score"] <= 0.8)
            ).sum()
            low_quality = (self.data["Quality_Score"] < 0.5).sum()

            report.append(f"\n2. Result Quality Distribution:")
            report.append(
                f"   - High quality results (>0.8): {high_quality} ({high_quality / len(self.data) * 100:.1f}%)"
            )
            report.append(
                f"   - Medium quality results (0.5-0.8): {medium_quality} ({medium_quality / len(self.data) * 100:.1f}%)"
            )
            report.append(
                f"   - Low quality results (<0.5): {low_quality} ({low_quality / len(self.data) * 100:.1f}%)"
            )

        # Four-parameter sensitivity analysis
        params_to_analyze = ["Grid", "Scale", "Groups", "Epi"]
        available_params = [p for p in params_to_analyze if p in self.data.columns]

        report.append(f"\n3. Four-Parameter Sensitivity Analysis:")
        sensitivity_results = {}
        for param in available_params:
            sens = self._calculate_sensitivity_metric(param)
            sensitivity_results[param] = sens
            report.append(f"   - {param} parameter sensitivity: {sens:.4f}")

        if sensitivity_results:
            most_sensitive_param = max(sensitivity_results, key=sensitivity_results.get)
            report.append(
                f"   - Conclusion: {most_sensitive_param} parameter has the greatest impact on clustering results"
            )

        # Parameter importance analysis
        if len(available_params) > 1:
            importance_scores = self.calculate_parameter_importance()
            report.append(f"\n4. Parameter Importance Ranking:")
            sorted_importance = sorted(
                importance_scores.items(), key=lambda x: x[1], reverse=True
            )
            for i, (param, importance) in enumerate(sorted_importance, 1):
                report.append(f"   {i}. {param}: {importance:.4f}")

        # Optimal parameter recommendations
        best_quality_idx = self.data["Quality_Score"].idxmax()
        best_params = self.data.loc[best_quality_idx]

        report.append(f"\n5. Optimal Parameter Combination:")
        report.append(f"   - Grid: {best_params['Grid']}")
        if "Scale" in best_params.index:
            report.append(f"   - Scale: {best_params['Scale']:.1f}")
        if "Groups" in best_params.index:
            report.append(f"   - Groups: {best_params['Groups']}")
        report.append(f"   - Epi: {best_params['Epi']:.1f}")
        report.append(f"   - Accuracy: {best_params['Accuracy']:.4f}")
        report.append(f"   - Silhouette coefficient: {best_params['Silhouette']:.4f}")
        report.append(f"   - Quality score: {best_params['Quality_Score']:.4f}")

        # Parameter recommended ranges
        report.append(
            f"\n6. Parameter Recommended Ranges (Based on Top 25% Quality Scores):"
        )
        for param in available_params:
            param_quality = self.data.groupby(param)["Quality_Score"].mean()
            if len(param_quality) > 0:
                threshold = param_quality.quantile(0.75)
                optimal_values = param_quality[param_quality >= threshold]
                if len(optimal_values) > 0:
                    min_opt = optimal_values.index.min()
                    max_opt = optimal_values.index.max()
                    report.append(f"   - {param}: {min_opt} - {max_opt}")

        # Parameter correlation analysis
        if len(available_params) > 1:
            report.append(f"\n7. Parameter Correlation Analysis:")
            correlation_matrix = self.data[available_params].corr()
            for i, param1 in enumerate(available_params):
                for j, param2 in enumerate(available_params):
                    if i < j:  # Avoid duplicates
                        correlation = correlation_matrix.loc[param1, param2]
                        report.append(f"   - {param1} vs {param2}: {correlation:.4f}")

        # Top 5 optimal parameter combinations
        top_combinations = self.data.nlargest(5, "Quality_Score")
        report.append(f"\n8. Top 5 Optimal Parameter Combinations:")
        for i, (_, row) in enumerate(top_combinations.iterrows(), 1):
            scale_str = f", Scale={row['Scale']:.1f}" if "Scale" in row.index else ""
            groups_str = f", Groups={row['Groups']}" if "Groups" in row.index else ""
            report.append(
                f"   {i}. Grid={row['Grid']}{scale_str}{groups_str}, Epi={row['Epi']:.1f} "
                f"-> Score={row['Quality_Score']:.4f}"
            )

        # Anomaly detection results
        anomaly_count = 0
        for _, row in self.data.iterrows():
            if row["Silhouette"] >= 0.99 and row["Accuracy"] < 0.8:
                anomaly_count += 1

        if anomaly_count > 0:
            report.append(f"\n9. Anomaly Detection:")
            report.append(
                f"   - Found {anomaly_count} anomalous results (high silhouette but low accuracy)"
            )
            report.append(
                "   - Recommendation: Check clustering validity for these parameter combinations"
            )

        # Practical recommendations
        report.append(f"\n10. Practical Recommendations:")

        if sensitivity_results:
            # Give recommendations based on sensitivity
            most_sensitive = max(sensitivity_results, key=sensitivity_results.get)
            if most_sensitive == "Grid":
                report.append(
                    "   - Prioritize optimizing Grid parameter as it has the greatest impact on results"
                )
            elif most_sensitive == "Scale":
                report.append(
                    "   - Pay attention to problem scale (Scale) impact, recommend determining appropriate Scale range first"
                )
            elif most_sensitive == "Groups":
                report.append(
                    "   - Groups parameter has significant impact, recommend setting target cluster count based on prior knowledge"
                )
            elif most_sensitive == "Epi":
                report.append(
                    "   - Epi parameter is sensitive to results, requires careful tuning"
                )

        report.append(
            "   - Recommend conducting experiments within recommended parameter ranges"
        )
        report.append("   - Focus on quality scores rather than single metrics")
        report.append(
            "   - Periodically check parameter correlations to avoid parameter conflicts"
        )

        # Output report
        report_text = "\n".join(report)
        print(report_text)

        # Save report
        with open(
            "comprehensive_sensitivity_analysis_report.txt", "w", encoding="utf-8"
        ) as f:
            f.write(report_text)
        print(
            f"\nComprehensive analysis report saved to: comprehensive_sensitivity_analysis_report.txt"
        )


def main(results_file=None):
    """
    Main function - Perform parameter sensitivity analysis

    Args:
        results_file: Specified results file path, if None let user choose
    """
    print(
        "BKTree Clustering Algorithm Four-Parameter (Grid, Scale, Groups, Epi) Sensitivity Analysis"
    )
    print("=" * 60)

    # If no file specified, use original file selection logic
    if results_file is None:
        # Check available result files
        result_files = [
            "output/silhouette_results.txt",
            "output_difficult/silhouette_results.txt",
        ]

        available_files = []
        for file_path in result_files:
            if os.path.exists(file_path):
                available_files.append(file_path)

        if not available_files:
            print("Error: Result files not found")
            print("Please ensure at least one of the following files exists:")
            for file_path in result_files:
                print(f"  - {file_path}")
            return

        # Select file (prefer output directory file)
        selected_file = available_files[0]
        if len(available_files) > 1:
            print("Found multiple result files, selecting for analysis:")
            for i, file_path in enumerate(available_files):
                print(f"{i + 1}. {file_path}")

            try:
                choice = (
                    int(input("Please select file number (default 1): ") or "1") - 1
                )
                if 0 <= choice < len(available_files):
                    selected_file = available_files[choice]
            except (ValueError, IndexError):
                pass
    else:
        # Use specified file
        selected_file = results_file
        if not os.path.exists(selected_file):
            print(f"Error: Specified result file does not exist: {selected_file}")
            return

    print(f"\nUsing file: {selected_file}")
    print("-" * 60)

    # Create analyzer and perform analysis
    analyzer = ParameterSensitivityAnalyzer(selected_file)

    # Load data
    if not analyzer.load_results():
        print("Data loading failed, analysis terminated")
        return

    # Calculate quality scores
    if not analyzer.calculate_quality_scores():
        print("Quality score calculation failed, analysis terminated")
        return

    # Perform four-parameter sensitivity analysis
    print("\nStarting four-parameter sensitivity analysis...")
    sensitivity_results = analyzer.analyze_parameter_sensitivity()

    # Multi-parameter comprehensive analysis
    print("\nStarting multi-parameter comprehensive analysis...")
    analyzer.analyze_multi_parameter_correlation()

    # Find optimal parameter ranges
    print("\nAnalyzing optimal parameter ranges...")
    analyzer.find_optimal_parameter_ranges()

    # Calculate parameter importance
    print("\nCalculating parameter importance...")
    analyzer.calculate_parameter_importance()

    # Create comprehensive visualization charts
    print("\nGenerating comprehensive visualization charts...")
    analyzer.create_visualization()

    # Generate detailed analysis report
    print("\nGenerating detailed analysis report...")
    analyzer.generate_report()

    print("\n" + "=" * 60)
    print("Four-parameter sensitivity analysis complete!")
    print("Generated files:")
    print(
        "  - comprehensive_parameter_sensitivity_analysis.png (Comprehensive analysis charts)"
    )
    print(
        "  - comprehensive_sensitivity_analysis_report.txt (Detailed analysis report)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
