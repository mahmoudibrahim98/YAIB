"""Module for subgroup-based stratified sampling.

This module provides functions to create evaluation and training sets 
with stratification based on subgroups of patients.
"""

import logging
import numpy as np
import pandas as pd
import polars as pl
import random
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, List, Union, Optional, Tuple, Any


def stratified_sample(test_size, random_state, candidates, target_group, outcome):
    """Sample from a target group while stratifying by outcome variable.
    
    Args:
        test_size: Proportion or absolute number of samples to select
        random_state: Random seed for reproducibility
        candidates: DataFrame containing a 'subgroup' column
        target_group: Which subgroup to sample from
        y: Target variable for stratification
        
    Returns:
        List of indices selected for the sample
    """
    subgroup = candidates[candidates['subgroup'] == target_group]
    ids = subgroup['stay_id'].to_list()
    y_subgroup = pd.DataFrame(index=ids).join(
        outcome.set_index('stay_id')
        ).reset_index()['label'].to_list()    

    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # # Get indices of samples in the subgroup
    # if isinstance(candidates, pl.DataFrame):
    #     subgroup_indices = candidates.filter(pl.col('subgroup') == target_group).row_indices()
    #     y_subgroup = y[subgroup_indices]
    # else:  # pandas DataFrame
    #     subgroup_indices = subgroup.index.tolist()
    #     y_subgroup = y[subgroup_indices]
        
    for train_idx, test_idx in sss.split(subgroup, y_subgroup):
        
        target_ids = subgroup.iloc[test_idx]['stay_id'].tolist()
        
    return target_ids


# def create_evaluation_and_training_sets(static, subgroup_counts, y, eval_size=500, train_size=2000, random_states=None):
def create_evaluation_and_training_sets(
    static: Union[pd.DataFrame, pl.DataFrame],
    subgroup_counts: Union[pd.DataFrame, pl.DataFrame],
    outcome: Union[pd.DataFrame, pl.DataFrame],
    eval_size: int = 500,
    train_size: int = 2000,
    random_states: Optional[List[int]] = None,
    min_subgroup_size: int = 100,
    pooled_sizes: List[int] = [2000, 2500, 3000, 3500, 4000]
) -> Dict[int, Dict[str, Any]]:
    
    """Create evaluation and training sets with stratification by subgroups.
    
    Args:
        static: DataFrame containing static patient information with a 'subgroup' column
        subgroup_counts: DataFrame containing subgroup counts information with 'subgroup', 'count', and 'quantile_index' columns
        y: Target variable for stratification
        eval_size: Size of the evaluation set per subgroup
        train_size: Size of the training set per subgroup
        random_states: List of random seeds to use for different splits
        
    Returns:
        Dictionary of indices for different training and evaluation configurations
    """
    if random_states is None:
        random_states = [42]
        
    # Convert polars DataFrame to pandas if needed
    is_polars = isinstance(static, pl.DataFrame)
    if is_polars:
        static_pd = static.to_pandas()
        if isinstance(subgroup_counts, pl.DataFrame):
            subgroup_counts = subgroup_counts.to_pandas()
        outcome_pd = outcome.to_pandas()
    else:
        static_pd = static
        outcome_pd = outcome


    
    # Get high-quantile subgroups
    high_quantile_subgroups = subgroup_counts[subgroup_counts['quantile_index'] == 1]['subgroup'].tolist()
    logging.info(f"Found {len(high_quantile_subgroups)} high-quantile subgroups")
    
    
    # Step 1: Create evaluation set by sampling examples per subgroup
    indices = {}
    for eval_random_state in random_states:

        indices[eval_random_state] = {}
        indices[eval_random_state]['pooled'] = {size: [] for size in pooled_sizes}
        # Step 1: Create evaluation sets for high-quantile subgroups
        eval_indices_all = []
        
        for subgroup in high_quantile_subgroups:
            indices[eval_random_state][subgroup] = {
             'eval': [],
            '2000_subgroup_only': [],
            'base_data_only_no_subgroup': [],
            'base_data_and_500_subgroup': [],
            'base_data_and_1000_subgroup': [],
            'base_data_and_1500_subgroup': [],
            'base_data_and_2000_subgroup': []       
            }
            # Sample evaluation set
            subgroup_idx = stratified_sample(eval_size, eval_random_state, static_pd, subgroup, outcome_pd)
            indices[eval_random_state][subgroup]['eval'] = subgroup_idx
            eval_indices_all.extend(subgroup_idx)
            # Log class distribution
            log_class_distribution(outcome_pd, subgroup_idx, f"Subgroup {subgroup} evaluation set")

        # Remove evaluation examples from the static_df to get remaining candidates for training
        train_candidates = static_pd[~static_pd['stay_id'].isin(eval_indices_all)]
        logging.info(f"Evaluation set created with random state {eval_random_state}. "
                    f"Remaining candidates for training: {train_candidates.shape}")
        
        for train_random_state in random_states:
            # Step 2: For each subgroup, create a training set
            for target_group in train_candidates['subgroup'].unique():
                # For every other subgroup, select 69 examples (balanced representation)                    
                non_target_indices = sample_non_target_subgroups(
                    train_candidates, target_group, subgroup_counts,
                    train_size, train_random_state, min_subgroup_size, outcome_pd
                )
                # Remove non-target samples to avoid overlap
                remaining_candidates = train_candidates[~train_candidates['stay_id'].isin(non_target_indices)]
                # Only consider subgroups with samples in the highest quantile (quantile_index = 1)
                if target_group in high_quantile_subgroups:
                    # Select train_size examples from the target subgroup
                    target_indices = stratified_sample(
                        train_size, train_random_state, remaining_candidates, target_group, outcome_pd
                        )
                    # Check class distribution in training set
                    log_class_distribution(outcome_pd, target_indices, f"Subgroup {target_group} training set")



                    
                    # Store different mixtures of base data and target subgroup data
                    store_dataset_configurations(
                        indices[eval_random_state][target_group], 
                        target_indices, non_target_indices, train_random_state
                    )

            # Create pooled datasets with uniform representation
            for size in pooled_sizes:
                pooled_indices = create_pooled_dataset(
                    train_candidates, subgroup_counts, size, train_random_state, min_subgroup_size
                )
                indices[eval_random_state]['pooled'][size].append(pooled_indices)
                
    
    return indices

def return_valid_subgroups(df,subgroup_counts, min_size):
    valid_subgroups = []
    for subgroup in df['subgroup'].unique():
        subgroup_count = subgroup_counts[subgroup_counts['subgroup'] == subgroup]['count'].values
        if len(subgroup_count) > 0 and subgroup_count[0] >= min_size:
            valid_subgroups.append(subgroup)
    return valid_subgroups

def create_pooled_dataset(
    df: pd.DataFrame,
    subgroup_counts: pd.DataFrame,
    total_size: int,
    random_state: int,
    min_size: int
) -> List[int]:
    """Create a pooled dataset with uniform representation across subgroups.
    
    Args:
        df: DataFrame containing the data
        subgroup_counts: DataFrame with subgroup count information
        total_size: Total size of the pooled dataset
        random_state: Random seed for reproducibility
        min_size: Minimum size required for a subgroup to be included
        
    Returns:
        List of indices for the pooled dataset
    """
    pooled_indices = []

    valid_subgroups = return_valid_subgroups(df,subgroup_counts, min_size)
    # Calculate samples per subgroup
    samples_per_subgroup = int(np.round(total_size / len(valid_subgroups))) if valid_subgroups else 0
    
    # Sample from each valid subgroup
    for subgroup in valid_subgroups:
        subgroup_data = df[df['subgroup'] == subgroup]
        
        # Ensure we don't try to sample more than available
        samples = min(samples_per_subgroup, len(subgroup_data))
        
        if samples > 0:
            subgroup_indices = subgroup_data.sample(n=samples, random_state=random_state)['stay_id'].tolist()
            pooled_indices.extend(subgroup_indices)
    
    return pooled_indices
def store_dataset_configurations(
    result_dict: Dict[str, List], 
    target_indices: List[int],
    non_target_indices: List[int],
    random_state: int
) -> None:
    """Store different dataset configurations in the result dictionary.
    
    Args:
        result_dict: Dictionary to store the configurations
        target_indices: Indices from the target subgroup
        non_target_indices: Indices from non-target subgroups
        random_state: Random seed for reproducibility
    """
    # Set random seed for consistent sampling
    random.seed(random_state)
    
    # Store full subgroup-only training set
    result_dict['2000_subgroup_only'].append(target_indices)
    
    # Store base data only (no target subgroup)
    result_dict['base_data_only_no_subgroup'].append(non_target_indices)
    
    # Store different mixtures of base data and target subgroup
    result_dict['base_data_and_500_subgroup'].append(
        random.sample(target_indices, min(500, len(target_indices))) + non_target_indices
    )
    
    result_dict['base_data_and_1000_subgroup'].append(
        random.sample(target_indices, min(1000, len(target_indices))) + non_target_indices
    )
    
    result_dict['base_data_and_1500_subgroup'].append(
        random.sample(target_indices, min(1500, len(target_indices))) + non_target_indices
    )
    
    result_dict['base_data_and_2000_subgroup'].append(
        target_indices + non_target_indices
    )

def log_class_distribution(outcome_pd: pd.DataFrame, indices: List[int], prefix: str) -> None:
    """Log the class distribution of a dataset.
    
    Args:
        y: Target variable
        indices: Indices of the dataset
        prefix: Prefix for the log message
    """
    if len(indices) == 0:
        logging.debug(f"{prefix}: Empty dataset")
        return
        
    subgroup_y = outcome_pd[outcome_pd['stay_id'].isin(indices)]
    value_counts = subgroup_y['label'].value_counts(normalize=True).round(3)
    
    
        
    # unique_values, counts = np.unique(y_subset, return_counts=True)
    # normalized_counts = counts / counts.sum()
    # value_counts = dict(zip(unique_values, normalized_counts))
    logging.debug(f"{prefix} class distribution: {value_counts}")
    # print(f"{prefix}: class distribution: {dict(value_counts)}")

def sample_non_target_subgroups(
    df: pd.DataFrame, 
    target_subgroup: str,
    subgroup_counts: pd.DataFrame,
    total_size: int,
    random_state: int,
    min_size: int,
    y: np.ndarray
) -> List[int]:
    """Sample from non-target subgroups.
    
    Args:
        df: DataFrame containing the data
        target_subgroup: The target subgroup to exclude
        subgroup_counts: DataFrame with subgroup count information
        samples_per_subgroup: Number of samples to take from each subgroup
        random_state: Random seed for reproducibility
        min_size: Minimum size required for a subgroup to be included
        y: Target variable for stratification
        
    Returns:
        List of indices from non-target subgroups
    """
    non_target_indices = []
    valid_subgroups = return_valid_subgroups(df,subgroup_counts, min_size)
    samples_per_subgroup = int(np.round(total_size / (len(valid_subgroups)-1))) if valid_subgroups else 0
    for subgroup in valid_subgroups:
        # Skip target subgroup and small subgroups
        if subgroup == target_subgroup:
            continue
    
    
        # Sample from this subgroup
        subgroup_indices = stratified_sample(samples_per_subgroup, random_state, df, subgroup, y)
        non_target_indices.extend(subgroup_indices)
    
    return non_target_indices

def add_subgroup_column(input_data, static_features=None, polars=True):
    """Add a subgroup column to the static dataframe based on combinations of specified features.
    
    Args:
        data: Dictionary containing data segments including static data
        static_features: List of static features to use for subgroup creation
        polars: Whether data is in Polars (True) or Pandas (False) format
        
    Returns:
        Updated data dictionary with subgroup column added to static data
    """
    data = input_data.copy()
    if 'STATIC' not in data:
        logging.warning("No static segment found in data. Cannot create subgroups.")
        return data
    
    if static_features is None:
        # Default to using gender, age_group, and ethnicity_cat if available
        static_features = []
        potential_features = ['gender_cat', 'age_group', 'ethnicity_cat', 'bmi_group']
        
        if polars:
            available_features = data['STATIC'].columns
        else:
            available_features = data['STATIC'].columns.tolist()
            
        for feature in potential_features:
            if feature in available_features:
                static_features.append(feature)
                
        if not static_features:
            logging.warning("No suitable categorical features found for subgroup creation.")
            return data
    
    logging.info(f"Creating subgroups based on features: {static_features}")
    
    if polars:
        # Polars implementation
        # Convert all features to strings and concatenate them
        static_df = data['STATIC']
        for feature in static_features:
            # Ensure feature exists
            if feature not in static_df.columns:
                logging.warning(f"Feature {feature} not found in static data. Skipping.")
                continue
                
            # Convert to string
            static_df = static_df.with_columns(pl.col(feature).cast(pl.Utf8))
            
        # Combine features to create subgroup identifier
        if len(static_features) > 0:
            static_df = static_df.with_columns(
                pl.concat_str([pl.col(feature) for feature in static_features], separator="_").alias("subgroup")
            )
            
            # Count subgroups and create quartiles
            subgroup_counts = static_df.group_by('subgroup').count().sort('count', descending=True)
            count_quartiles = np.percentile(subgroup_counts['count'].to_numpy(), [75])
            
            # Add quartile information to subgroup_counts
            subgroup_counts = subgroup_counts.with_columns(
                pl.lit(0).alias('quantile_index')
            )
            
            # Mark top quartile with quantile_index = 1
            subgroup_counts = subgroup_counts.with_columns(
                pl.when(pl.col('count') >= count_quartiles[0])
                .then(pl.lit(1))
                .otherwise(pl.col('quantile_index'))
                .alias('quantile_index')
            )
            
            # Update data dictionary
            data['STATIC'] = static_df
            data['SUBGROUP_COUNTS'] = subgroup_counts
            
    else:
        # Pandas implementation
        static_df = data['STATIC'].copy()
        
        # Convert all features to strings and concatenate them
        for feature in static_features:
            # Ensure feature exists
            if feature not in static_df.columns:
                logging.warning(f"Feature {feature} not found in static data. Skipping.")
                continue
                
            # Convert to string
            static_df[feature] = static_df[feature].astype(str)
            
        # Combine features to create subgroup identifier
        if len(static_features) > 0:
            static_df['subgroup'] = static_df[static_features].agg('_'.join, axis=1)
            
            # Count subgroups and create quartiles
            subgroup_counts = static_df['subgroup'].value_counts().reset_index()
            subgroup_counts.columns = ['subgroup', 'count']
            subgroup_counts = subgroup_counts.sort_values('count', ascending=False)
            
            count_quartiles = np.percentile(subgroup_counts['count'], [75])
            subgroup_counts['quantile_index'] = 0
            subgroup_counts.loc[subgroup_counts['count'] >= count_quartiles[0], 'quantile_index'] = 1
            
            # Update data dictionary
            data['STATIC'] = static_df
            data['SUBGROUP_COUNTS'] = subgroup_counts
            
    return data


def extract_label_vector(data, label_column, polars=True):
    """Extract the label vector from the outcome data.
    
    Args:
        data: Dictionary containing data segments including outcome data
        label_column: Name of the label column
        polars: Whether data is in Polars (True) or Pandas (False) format
        
    Returns:
        Numpy array of labels
    """
    if 'OUTCOME' not in data:
        logging.error("No outcome segment found in data. Cannot extract labels.")
        return None
    
    if polars:
        if label_column not in data['OUTCOME'].columns:
            logging.error(f"Label column '{label_column}' not found in outcome data.")
            return None
        
        # Extract unique labels per stay_id (take maximum value to handle sequence data)
        outcome_df = data['OUTCOME']
        grouped = outcome_df.group_by('stay_id').agg(pl.col(label_column).max().alias(label_column))
        return grouped[label_column].to_numpy()
    else:
        if label_column not in data['OUTCOME'].columns:
            logging.error(f"Label column '{label_column}' not found in outcome data.")
            return None
        
        # Extract unique labels per stay_id (take maximum value to handle sequence data)
        outcome_df = data['OUTCOME']
        grouped = outcome_df.groupby('stay_id')[label_column].max()
        return grouped.values


def apply_subgroup_stratification(preprocessed_data, subgroup_features=None, random_states=None, 
                                 eval_size=500, train_size=2000, categorize_static=True):
    """Apply subgroup stratification to preprocessed data from preprocess_data.
    
    This function takes the output from preprocess_data and applies subgroup stratification
    on top of it, without modifying the original data splits.
    
    Args:
        preprocessed_data: Dictionary containing data from preprocess_data
        label_column: Name of the label column in the outcome data
        subgroup_features: List of static features to use for subgroup creation
        random_states: List of random seeds to use for different splits
        eval_size: Size of the evaluation set per subgroup
        train_size: Size of the training set per subgroup
        categorize_static: Whether to apply categorization to static features
        
    Returns:
        Dictionary containing preprocessed_data with added subgroup stratification
    """
    result = preprocessed_data.copy()
    
    # Check if static data is available in all splits
    if 'train' not in result or 'STATIC' not in result['train']:
        logging.error("Static data not found in preprocessed data. Cannot apply subgroup stratification.")
        return result
    
    # Determine if data is in Polars format
    is_polars = isinstance(result['train']['STATIC'], pl.DataFrame)
    
    # Combine static data from all splits for subgroup creation
    # We combine them to ensure consistent subgroup definitions
    if is_polars:
        combined_static = pl.concat([
            result['train']['STATIC'], 
            result['val']['STATIC'] if 'val' in result else pl.DataFrame(),
            result['test']['STATIC'] if 'test' in result else pl.DataFrame()
        ])
        
        combined_outcome = pl.concat([
            result['train']['OUTCOME'], 
            result['val']['OUTCOME'] if 'val' in result else pl.DataFrame(),
            result['test']['OUTCOME'] if 'test' in result else pl.DataFrame()
        ])
    else:
        combined_static = pd.concat([
            result['train']['STATIC'], 
            result['val']['STATIC'] if 'val' in result else pd.DataFrame(),
            result['test']['STATIC'] if 'test' in result else pd.DataFrame()
        ])
        
        combined_outcome = pd.concat([
            result['train']['OUTCOME'], 
            result['val']['OUTCOME'] if 'val' in result else pd.DataFrame(),
            result['test']['OUTCOME'] if 'test' in result else pd.DataFrame()
        ])
    

    
    # Create a temporary data dict for subgroup creation
    temp_data = {
        'STATIC': combined_static,
        'OUTCOME': combined_outcome
    }
    
    # Add subgroup column
    temp_data = add_subgroup_column(temp_data, static_features=subgroup_features, polars=is_polars)
    
    
    if 'SUBGROUP_COUNTS' not in temp_data or y is None:
        logging.error("Failed to create subgroups or extract labels.")
        return result
    
    # Create evaluation and training sets based on subgroups
    if random_states is None:
        random_states = [42]
        
    indices = create_evaluation_and_training_sets(
        temp_data['STATIC'], 
        temp_data['SUBGROUP_COUNTS'], 
        temp_data['OUTCOME'], 
        eval_size=eval_size, 
        train_size=train_size, 
        random_states=random_states
    )
    
    # Add subgroup information to each split in the result
    for split in ['train', 'val', 'test']:
        if split in result:
            # Add subgroup column to static data
            if is_polars:
                result[split]['STATIC'] = result[split]['STATIC'].join(
                    temp_data['STATIC'].select(['stay_id', 'subgroup']),
                    on='stay_id',
                    how='left'
                )
            else:
                result[split]['STATIC'] = result[split]['STATIC'].merge(
                    temp_data['STATIC'][['stay_id', 'subgroup']],
                    on='stay_id',
                    how='left'
                )
    
    # Add subgroup information to result
    result['SUBGROUP_COUNTS'] = temp_data['SUBGROUP_COUNTS']
    result['SUBGROUP_INDICES'] = indices
    
    # Log information about created sets
    eval_random_state = random_states[0]  # Use first random state for logging
    
    # Count number of subgroups in top quartile
    if is_polars:
        top_quartile_count = result['SUBGROUP_COUNTS'].filter(pl.col('quantile_index') == 1).height
    else:
        top_quartile_count = result['SUBGROUP_COUNTS'][result['SUBGROUP_COUNTS']['quantile_index'] == 1].shape[0]
        
    logging.info(f"Number of subgroups in top quartile (quantile_index=1): {top_quartile_count}")
    
    # Count total samples in evaluation set
    eval_count = 0
    for subgroup in indices[eval_random_state]:
        if subgroup != 'pooled' and 'eval' in indices[eval_random_state][subgroup]:
            eval_count += len(indices[eval_random_state][subgroup]['eval'])
    
    logging.info(f"Total samples in subgroup evaluation set: {eval_count}")
    
    return result 


import sys
import numpy as np
from IPython.display import display, HTML
import pandas as pd

def analyze_results_dictionary(results):
    """
    Analyze and display the content and size of the results dictionary.
    
    Args:
        results: The nested dictionary returned by create_evaluation_and_training_sets
    """
    # Get total size in memory
    size_bytes = sys.getsizeof(results)
    
    # Initialize counters
    total_indices = 0
    total_sets = 0
    eval_sets = 0
    train_sets = 0
    
    # Create summary dataframes
    summary_rows = []
    detailed_rows = []
    
    # Analyze structure
    print(f"Results Dictionary Structure:")
    print(f"{'='*50}")
    
    for eval_seed, seed_data in results.items():
        print(f"\nEvaluation Seed: {eval_seed}")
        print(f"{'-'*30}")
        
        # Process pooled datasets
        if 'pooled' in seed_data:
            pooled_data = seed_data['pooled']
            for size, size_data in pooled_data.items():
                for i, indices in enumerate(size_data):
                    set_size = len(indices)
                    total_indices += set_size
                    total_sets += 1
                    train_sets += 1
                    
                    # Add to detailed summary
                    detailed_rows.append({
                        'eval_seed': eval_seed,
                        'dataset_type': 'pooled',
                        'subgroup': 'all',
                        'configuration': f'size_{size}',
                        'train_seed_index': i,
                        'num_samples': set_size
                    })
                    
                    print(f"  Pooled (size {size}), Train Set {i}: {set_size} samples")
        
        # Process subgroup-specific datasets
        for key, value in seed_data.items():
            if key != 'pooled':
                subgroup = key
                print(f"  Subgroup: {subgroup}")
                
                for config, indices_list in value.items():
                    if config == 'eval':
                        # Evaluation set
                        set_size = len(indices_list)
                        total_indices += set_size
                        total_sets += 1
                        eval_sets += 1
                        
                        # Add to detailed summary
                        detailed_rows.append({
                            'eval_seed': eval_seed,
                            'dataset_type': 'evaluation',
                            'subgroup': subgroup,
                            'configuration': config,
                            'train_seed_index': None,
                            'num_samples': set_size
                        })
                        
                        # Add to summary
                        summary_rows.append({
                            'eval_seed': eval_seed,
                            'subgroup': subgroup,
                            'eval_samples': set_size,
                            'train_configs': len(value) - 1  # Subtract 1 for 'eval'
                        })
                        
                        print(f"    Evaluation Set: {set_size} samples")
                    else:
                        # Training sets
                        for i, indices in enumerate(indices_list):
                            set_size = len(indices)
                            total_indices += set_size
                            total_sets += 1
                            train_sets += 1
                            
                            # Add to detailed summary
                            detailed_rows.append({
                                'eval_seed': eval_seed,
                                'dataset_type': 'training',
                                'subgroup': subgroup,
                                'configuration': config,
                                'train_seed_index': i,
                                'num_samples': set_size
                            })
                            
                            print(f"    {config}, Train Set {i}: {set_size} samples")
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_rows)
    detailed_df = pd.DataFrame(detailed_rows)
    
    # Calculate size in MB
    size_mb = size_bytes / (1024 * 1024)
    
    # Display overall statistics
    print("\nOverall Statistics:")
    print(f"{'='*50}")
    print(f"Total dictionary size: {size_mb:.2f} MB")
    print(f"Total number of datasets: {total_sets}")
    print(f"  - Evaluation sets: {eval_sets}")
    print(f"  - Training sets: {train_sets}")
    print(f"Total indices stored: {total_indices}")
    print(f"Average indices per dataset: {total_indices/total_sets:.1f}")
    
    # Display summary dataframe
    print("\nSummary by Subgroup:")
    print(f"{'='*50}")
    display(summary_df)
    
    # Display detailed statistics
    print("\nDetailed Dataset Statistics:")
    print(f"{'='*50}")
    
    # Group by configuration and calculate statistics
    config_stats = detailed_df.groupby(['dataset_type', 'configuration']).agg(
        count=('num_samples', 'count'),
        min_samples=('num_samples', 'min'),
        max_samples=('num_samples', 'max'),
        avg_samples=('num_samples', 'mean')
    ).reset_index()
    
    display(config_stats)
    
    # Return the detailed dataframe for further analysis
    return detailed_df

# Example usage:
# detailed_stats = analyze_results_dictionary(results)