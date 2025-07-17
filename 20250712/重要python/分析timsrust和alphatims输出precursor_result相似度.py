import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class TimsTOFDataComparator:
    """
    A comprehensive tool for comparing timsTOF data processed by different software packages.
    """
    
    def __init__(self, file1_path, file2_path, file1_name="AlphaTims", file2_name="Timsrust"):
        """
        Initialize the comparator with two CSV files.
        
        Parameters:
        -----------
        file1_path : str
            Path to the first CSV file
        file2_path : str
            Path to the second CSV file
        file1_name : str
            Name/label for the first dataset
        file2_name : str
            Name/label for the second dataset
        """
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.file1_name = file1_name
        self.file2_name = file2_name
        
        # Load data
        self.df1 = pd.read_csv(file1_path)
        self.df2 = pd.read_csv(file2_path)
        
        # Standardize data types
        self._standardize_data()
        
        # Store comparison results
        self.comparison_results = {}
        
    def _standardize_data(self):
        """Ensure consistent data types between datasets."""
        # Ensure all numeric columns are float
        for col in ['rt_values_min', 'mobility_values', 'mz_values', 'intensity_values']:
            if col in self.df1.columns:
                self.df1[col] = self.df1[col].astype(float)
            if col in self.df2.columns:
                self.df2[col] = self.df2[col].astype(float)
    
    def basic_statistics(self):
        """Generate basic statistics for both datasets."""
        print(f"\n{'='*60}")
        print(f"BASIC STATISTICS COMPARISON")
        print(f"{'='*60}\n")
        
        stats_dict = {}
        
        for name, df in [(self.file1_name, self.df1), (self.file2_name, self.df2)]:
            print(f"\n{name} Dataset:")
            print(f"  Number of rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            stats_dict[name] = {}
            
            for col in df.columns:
                stats_dict[name][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'unique_count': df[col].nunique()
                }
                
                print(f"\n  {col}:")
                print(f"    Range: [{df[col].min():.6f}, {df[col].max():.6f}]")
                print(f"    Mean ± SD: {df[col].mean():.6f} ± {df[col].std():.6f}")
                print(f"    Unique values: {df[col].nunique()}")
        
        self.comparison_results['basic_stats'] = stats_dict
        return stats_dict
    
    def find_matching_features(self, rt_tolerance=0.001, mz_tolerance=1.0, im_tolerance=0.005):
        """
        Find matching features between the two datasets based on RT, m/z, and IM tolerances.
        
        Parameters:
        -----------
        rt_tolerance : float
            Tolerance for RT matching in minutes
        mz_tolerance : float
            Tolerance for m/z matching in Da (or ppm if very small)
        im_tolerance : float
            Tolerance for ion mobility matching
        
        Returns:
        --------
        pd.DataFrame
            Matched features with values from both datasets
        """
        print(f"\n{'='*60}")
        print(f"FEATURE MATCHING ANALYSIS")
        print(f"{'='*60}\n")
        
        matched_features = []
        unmatched_df1 = []
        unmatched_df2 = list(range(len(self.df2)))
        
        for idx1, row1 in self.df1.iterrows():
            match_found = False
            
            for idx2, row2 in self.df2.iterrows():
                if idx2 not in unmatched_df2:
                    continue
                
                # Check if features match within tolerances
                rt_match = abs(row1['rt_values_min'] - row2['rt_values_min']) <= rt_tolerance
                mz_match = abs(row1['mz_values'] - row2['mz_values']) <= mz_tolerance
                im_match = abs(row1['mobility_values'] - row2['mobility_values']) <= im_tolerance
                
                if rt_match and mz_match and im_match:
                    matched_features.append({
                        'rt_1': row1['rt_values_min'],
                        'rt_2': row2['rt_values_min'],
                        'mz_1': row1['mz_values'],
                        'mz_2': row2['mz_values'],
                        'im_1': row1['mobility_values'],
                        'im_2': row2['mobility_values'],
                        'intensity_1': row1['intensity_values'],
                        'intensity_2': row2['intensity_values'],
                        'rt_diff': abs(row1['rt_values_min'] - row2['rt_values_min']),
                        'mz_diff': abs(row1['mz_values'] - row2['mz_values']),
                        'im_diff': abs(row1['mobility_values'] - row2['mobility_values']),
                        'intensity_diff': abs(row1['intensity_values'] - row2['intensity_values'])
                    })
                    unmatched_df2.remove(idx2)
                    match_found = True
                    break
            
            if not match_found:
                unmatched_df1.append(idx1)
        
        matched_df = pd.DataFrame(matched_features)
        
        print(f"Matching parameters:")
        print(f"  RT tolerance: {rt_tolerance} min")
        print(f"  m/z tolerance: {mz_tolerance} Da")
        print(f"  IM tolerance: {im_tolerance}")
        print(f"\nResults:")
        print(f"  Total features in {self.file1_name}: {len(self.df1)}")
        print(f"  Total features in {self.file2_name}: {len(self.df2)}")
        print(f"  Matched features: {len(matched_df)}")
        print(f"  Unmatched in {self.file1_name}: {len(unmatched_df1)}")
        print(f"  Unmatched in {self.file2_name}: {len(unmatched_df2)}")
        print(f"  Matching rate: {len(matched_df)/max(len(self.df1), len(self.df2))*100:.1f}%")
        
        self.comparison_results['matched_features'] = matched_df
        self.comparison_results['matching_summary'] = {
            'n_matched': len(matched_df),
            'n_unmatched_1': len(unmatched_df1),
            'n_unmatched_2': len(unmatched_df2),
            'matching_rate': len(matched_df)/max(len(self.df1), len(self.df2))
        }
        
        return matched_df
    
    def calculate_correlations(self, matched_df=None):
        """Calculate correlations between matched features."""
        if matched_df is None:
            if 'matched_features' not in self.comparison_results:
                matched_df = self.find_matching_features()
            else:
                matched_df = self.comparison_results['matched_features']
        
        if len(matched_df) == 0:
            print("No matched features found for correlation analysis.")
            return None
        
        print(f"\n{'='*60}")
        print(f"CORRELATION ANALYSIS")
        print(f"{'='*60}\n")
        
        correlations = {}
        
        for param in ['rt', 'mz', 'im', 'intensity']:
            col1 = f'{param}_1'
            col2 = f'{param}_2'
            
            if col1 in matched_df.columns and col2 in matched_df.columns:
                pearson_r, pearson_p = stats.pearsonr(matched_df[col1], matched_df[col2])
                spearman_r, spearman_p = stats.spearmanr(matched_df[col1], matched_df[col2])
                
                correlations[param] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                }
                
                print(f"{param.upper()} Correlation:")
                print(f"  Pearson r: {pearson_r:.6f} (p={pearson_p:.2e})")
                print(f"  Spearman r: {spearman_r:.6f} (p={spearman_p:.2e})")
        
        self.comparison_results['correlations'] = correlations
        return correlations
    
    def calculate_errors(self, matched_df=None):
        """Calculate various error metrics between matched features."""
        if matched_df is None:
            if 'matched_features' not in self.comparison_results:
                matched_df = self.find_matching_features()
            else:
                matched_df = self.comparison_results['matched_features']
        
        if len(matched_df) == 0:
            print("No matched features found for error analysis.")
            return None
        
        print(f"\n{'='*60}")
        print(f"ERROR METRICS")
        print(f"{'='*60}\n")
        
        errors = {}
        
        for param in ['rt', 'mz', 'im', 'intensity']:
            col1 = f'{param}_1'
            col2 = f'{param}_2'
            
            if col1 in matched_df.columns and col2 in matched_df.columns:
                mae = mean_absolute_error(matched_df[col1], matched_df[col2])
                rmse = np.sqrt(mean_squared_error(matched_df[col1], matched_df[col2]))
                
                # Calculate relative errors for non-zero values
                mask = matched_df[col1] != 0
                if mask.any():
                    relative_errors = abs(matched_df.loc[mask, col1] - matched_df.loc[mask, col2]) / matched_df.loc[mask, col1]
                    mean_relative_error = relative_errors.mean() * 100
                else:
                    mean_relative_error = np.nan
                
                errors[param] = {
                    'mae': mae,
                    'rmse': rmse,
                    'mean_relative_error_percent': mean_relative_error
                }
                
                print(f"{param.upper()} Errors:")
                print(f"  Mean Absolute Error: {mae:.6f}")
                print(f"  Root Mean Square Error: {rmse:.6f}")
                if not np.isnan(mean_relative_error):
                    print(f"  Mean Relative Error: {mean_relative_error:.2f}%")
        
        self.comparison_results['errors'] = errors
        return errors
    
    def plot_comparisons(self, matched_df=None, save_path=None):
        """Generate comprehensive comparison plots."""
        if matched_df is None:
            if 'matched_features' not in self.comparison_results:
                matched_df = self.find_matching_features()
            else:
                matched_df = self.comparison_results['matched_features']
        
        if len(matched_df) == 0:
            print("No matched features found for plotting.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Comparison: {self.file1_name} vs {self.file2_name}', fontsize=16)
        
        parameters = ['rt', 'mz', 'im', 'intensity']
        labels = ['Retention Time (min)', 'm/z', 'Ion Mobility', 'Intensity']
        
        for idx, (param, label) in enumerate(zip(parameters, labels)):
            ax = axes[idx // 2, idx % 2]
            
            col1 = f'{param}_1'
            col2 = f'{param}_2'
            
            if col1 in matched_df.columns and col2 in matched_df.columns:
                # Scatter plot
                ax.scatter(matched_df[col1], matched_df[col2], alpha=0.5, s=20)
                
                # Add diagonal line
                min_val = min(matched_df[col1].min(), matched_df[col2].min())
                max_val = max(matched_df[col1].max(), matched_df[col2].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                
                # Calculate and display R²
                r2 = np.corrcoef(matched_df[col1], matched_df[col2])[0, 1]**2
                ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel(f'{label} - {self.file1_name}')
                ax.set_ylabel(f'{label} - {self.file2_name}')
                ax.set_title(f'{label} Comparison')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_distributions(self, save_path=None):
        """Plot distributions of values for both datasets."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Value Distributions Comparison', fontsize=16)
        
        parameters = ['rt_values_min', 'mz_values', 'mobility_values', 'intensity_values']
        labels = ['Retention Time (min)', 'm/z', 'Ion Mobility', 'Intensity']
        
        for idx, (param, label) in enumerate(zip(parameters, labels)):
            ax = axes[idx // 2, idx % 2]
            
            # Plot histograms
            ax.hist(self.df1[param], bins=50, alpha=0.5, label=self.file1_name, density=True)
            ax.hist(self.df2[param], bins=50, alpha=0.5, label=self.file2_name, density=True)
            
            ax.set_xlabel(label)
            ax.set_ylabel('Density')
            ax.set_title(f'{label} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Log scale for intensity
            if param == 'intensity_values':
                ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print(f"\n{'='*60}")
        print(f"SUMMARY REPORT")
        print(f"{'='*60}\n")
        
        # Overall similarity score
        if 'matching_summary' in self.comparison_results:
            matching_rate = self.comparison_results['matching_summary']['matching_rate']
            print(f"Feature Matching Rate: {matching_rate*100:.1f}%")
        
        # Average correlations
        if 'correlations' in self.comparison_results:
            avg_pearson = np.mean([v['pearson_r'] for v in self.comparison_results['correlations'].values()])
            print(f"Average Pearson Correlation: {avg_pearson:.4f}")
        
        # Average relative errors
        if 'errors' in self.comparison_results:
            rel_errors = [v['mean_relative_error_percent'] for v in self.comparison_results['errors'].values() 
                         if not np.isnan(v['mean_relative_error_percent'])]
            if rel_errors:
                avg_rel_error = np.mean(rel_errors)
                print(f"Average Relative Error: {avg_rel_error:.2f}%")
        
        # Conclusion
        print(f"\n{'='*60}")
        print("CONCLUSION:")
        print(f"{'='*60}")
        
        if 'matching_summary' in self.comparison_results and 'correlations' in self.comparison_results:
            matching_rate = self.comparison_results['matching_summary']['matching_rate']
            avg_pearson = np.mean([v['pearson_r'] for v in self.comparison_results['correlations'].values()])
            
            if matching_rate > 0.95 and avg_pearson > 0.99:
                print("✓ The datasets are HIGHLY SIMILAR and can be considered functionally identical.")
                print("  Both software packages produce essentially the same results.")
            elif matching_rate > 0.85 and avg_pearson > 0.95:
                print("⚠ The datasets are SIMILAR with minor differences.")
                print("  The results are comparable but show some systematic variations.")
            else:
                print("✗ The datasets show SIGNIFICANT DIFFERENCES.")
                print("  Further investigation is needed before using them interchangeably.")
    
    def save_results(self, output_dir='.'):
        """Save all results to files."""
        import os
        import json
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save matched features
        if 'matched_features' in self.comparison_results:
            self.comparison_results['matched_features'].to_csv(
                os.path.join(output_dir, 'matched_features.csv'), index=False
            )
        
        # Save summary statistics
        summary = {k: v for k, v in self.comparison_results.items() if k != 'matched_features'}
        with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_dir}/")

# Example usage
if __name__ == "__main__":
    # Initialize the comparator
    comparator = TimsTOFDataComparator(
        file1_path="/Users/augustsirius/Desktop/DIABERT_test_code_lib/precursor_result_after_IM_filter.csv",
        file2_path="/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250703/timstof/precursor_result_after_IM_filter.csv",
        file1_name="AlphaTims",
        file2_name="Timsrust"
    )
    
    # Run comprehensive analysis
    comparator.basic_statistics()
    matched_features = comparator.find_matching_features(
        rt_tolerance=0.001,  # 0.001 min = 0.06 seconds
        mz_tolerance=2.0,    # 2 Da tolerance (adjust based on your instrument)
        im_tolerance=0.01    # 1% of typical IM value
    )
    
    comparator.calculate_correlations()
    comparator.calculate_errors()
    
    # Generate plots
    comparator.plot_comparisons()
    comparator.plot_distributions()
    
    # Generate summary report
    comparator.generate_summary_report()
    
    # Save results
    comparator.save_results(output_dir='./timstof_comparison_results')