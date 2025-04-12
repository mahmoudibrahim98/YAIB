import copy
import logging
import os

import gin
import json
import hashlib
import pandas as pd
import polars as pl
from pathlib import Path
import pickle
from timeit import default_timer as timer
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, ShuffleSplit
from icu_benchmarks.data.preprocessor import Preprocessor, PandasClassificationPreprocessor, PolarsClassificationPreprocessor
from icu_benchmarks.constants import RunMode
from icu_benchmarks.run_utils import check_required_keys
import numpy as np
from .constants import DataSplit as Split, DataSegment as Segment, VarType as Var


import pickle

import torch

from recipys.recipe import Recipe
from recipys.selector import all_numeric_predictors, all_outcomes, has_type, all_of
from recipys.step import (
    Step,
    StepScale,
    StepImputeFastForwardFill,
    StepImputeFastZeroFill,
    StepImputeFill,
    StepSklearn,
    StepHistorical,
    Accumulator,
    StepImputeModel,
)

from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, MinMaxScaler

from icu_benchmarks.wandb_utils import update_wandb_config
from icu_benchmarks.data.loader import ImputationPredictionDataset
import abc
from .constants import DataSplit as Split, DataSegment as Segment

# @gin.configurable("base_classification_preprocessor")
class CustomPreprocessor(Preprocessor):
    def __init__(
        self,
        generate_features: bool = False,
        scaling: bool = True,
        use_static_features: bool = True,
        save_cache=None,
        load_cache=None,
        vars_to_exclude=None,
        keep_static_features: bool = True,
    ):
        """
        Args:
            generate_features: Generate features for dynamic data.
            scaling: Scaling of dynamic and static data.
            use_static_features: Use static features.
            save_cache: Save recipe cache from this path.
            load_cache: Load recipe cache from this path.
            vars_to_exclude: Variables to exclude from missing indicator/ feature generation.
        Returns:
            Preprocessed data.
        """
        self.generate_features = generate_features
        self.scaling = scaling
        self.use_static_features = use_static_features
        self.keep_static_features = keep_static_features
        self.imputation_model = None
        self.save_cache = save_cache
        self.load_cache = load_cache
        self.vars_to_exclude = vars_to_exclude

    def apply(self, data, vars) -> dict[dict[pl.DataFrame]]:
        """
        Args:
            data: Train, validation and test data dictionary. Further divided in static, dynamic, and outcome.
            vars: Variables for static, dynamic, outcome.
        Returns:
            Preprocessed data.
        """
        # Check if dynamic features are present
        if (
            Segment.static in  data.keys()
            and len(vars[Segment.static]) > 0
        ):
            logging.info("Preprocessing static features.")
            
            data = self._process_static(data, vars)

            data = categorize_static_features(data, polars=True)
        else:
            self.use_static_features = False

        if Segment.dynamic  in data.keys():
            logging.info("Preprocessing dynamic features.")
            logging.info(data.keys())

            data = self._fix_time_unit(data)
            data = self._process_dynamic(data, vars)
            if self.use_static_features:
                # Join static and dynamic data.
                data[Segment.dynamic] = data[Segment.dynamic].join(
                    data[Segment.static], on=vars["GROUP"]
                )


            if not self.keep_static_features:
                # Remove static features from splits
                data[Segment.features] = data.pop(Segment.static)


            # Create feature splits
            data[Segment.features] = data.pop(Segment.dynamic)

        elif not self.keep_static_features:
            data[Segment.features] = data.pop(Segment.static)

        else:
            raise Exception(f"No recognized data segments data to preprocess. Available: {data.keys()}")
        logging.debug("Data head")
        logging.debug(data[Segment.features].head())
        logging.debug(data[Segment.outcome])
        
        if vars["SEQUENCE"] in data[Segment.outcome] and len(data[Segment.features]) != len(
            data[Segment.outcome]
        ):
            raise Exception(
                f"Data and outcome length mismatch: "
                f"features: {len(data[Segment.features])}, outcome: {len(data[Segment.outcome])}"
            )
        data[Segment.features] = data[Segment.features].unique()


        logging.info(f"Generate features: {self.generate_features}")
        return data

    def _process_static(self, data, vars):
        sta_rec = Recipe(data[Segment.static], [], vars[Segment.static])
        sta_rec.add_step(StepSklearn(MissingIndicator(features="all"), sel=all_of(vars[Segment.static]), in_place=False))
        # if self.scaling:
        #     sta_rec.add_step(StepScale())
        sta_rec.add_step(StepImputeFill(sel=all_numeric_predictors(), strategy="zero"))
        # sta_rec.add_step(StepImputeFastZeroFill(sel=all_numeric_predictors()))
        # if len(data[Segment.static].select_dtypes(include=["object"]).columns) > 0:
        types = ["String", "Object", "Categorical"]
        sel = has_type(types)
        if len(sel(sta_rec.data)) > 0:
            # if len(data[Segment.static].select(cs.by_dtype(types)).columns) > 0:
            sta_rec.add_step(StepSklearn(SimpleImputer(missing_values=None, strategy="most_frequent"), sel=has_type(types)))
            # sta_rec.add_step(StepSklearn(LabelEncoder(), sel=has_type(types), columnwise=True))
        data = apply_recipe_to_splits(sta_rec, data, Segment.static, self.save_cache, self.load_cache)

        return data
    def _fix_time_unit(self, data):
        
        data[Segment.dynamic] = data[Segment.dynamic].with_columns(
        (pl.col("time") * 3600_000).cast(pl.Duration(time_unit="ms")).alias("time")
        )

        return data
    
    
    def _model_impute(self, data, group=None):
        dataset = ImputationPredictionDataset(data, group, self.imputation_model.trained_columns)
        input_data = torch.cat([data_point.unsqueeze(0) for data_point in dataset], dim=0)
        self.imputation_model.eval()
        with torch.no_grad():
            logging.info(f"Imputing with {self.imputation_model.__class__.__name__}.")
            imputation = self.imputation_model.predict(input_data)
            logging.info("Imputation done.")
        assert imputation.isnan().sum() == 0
        data = data.copy()
        data.loc[:, self.imputation_model.trained_columns] = imputation.flatten(end_dim=1).to("cpu")
        if group is not None:
            data.drop(columns=group, inplace=True)
        return data

    def _process_dynamic(self, data, vars):
        dyn_rec = Recipe(data[Segment.dynamic], [], vars[Segment.dynamic], vars["GROUP"], vars["SEQUENCE"])
        if self.scaling:
            dyn_rec.add_step(StepScale())
        if self.imputation_model is not None:
            dyn_rec.add_step(StepImputeModel(model=self.model_impute, sel=all_of(vars[Segment.dynamic])))
        if self.vars_to_exclude is not None:
            # Exclude vars_to_exclude from missing indicator/ feature generation
            vars_to_apply = list(set(vars[Segment.dynamic]) - set(self.vars_to_exclude))
        else:
            vars_to_apply = vars[Segment.dynamic]
        dyn_rec.add_step(StepSklearn(MissingIndicator(features="all"), sel=all_of(vars_to_apply), in_place=False))
        # dyn_rec.add_step(StepImputeFastForwardFill())
        dyn_rec.add_step(StepImputeFill(strategy="forward"))
        # dyn_rec.add_step(StepImputeFastZeroFill())
        dyn_rec.add_step(StepImputeFill(strategy="zero"))
        if self.generate_features:
            dyn_rec = self._dynamic_feature_generation(dyn_rec, all_of(vars_to_apply))
        data = apply_recipe_to_splits(dyn_rec, data, Segment.dynamic, self.save_cache, self.load_cache)
        
        return data

    def _dynamic_feature_generation(self, data, dynamic_vars):
        logging.debug("Adding dynamic feature generation.")
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MIN, suffix="min_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MAX, suffix="max_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.COUNT, suffix="count_hist"))
        data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.MEAN, suffix="mean_hist"))
        return data

    def to_cache_string(self):
        return (
            super().to_cache_string()
            + f"_classification_{self.generate_features}_{self.scaling}_{self.imputation_model.__class__.__name__}"
        )
@staticmethod
def apply_recipe_to_splits(
    recipe: Recipe, data, type: str, save_cache=None, load_cache=None
) :
    """Fits and transforms the training features, then transforms the validation and test features with the recipe.
     Works with both Polars and Pandas versions of recipys.

    Args:
        load_cache: Load recipe from cache, for e.g. transfer learning.
        save_cache: Save recipe to cache, for e.g. transfer learning.
        recipe: Object containing info about the features and steps.
        data: Dict containing 'train', 'val', and 'test' and types of features per split.
        type: Whether to apply recipe to dynamic features, static features or outcomes.

    Returns:
        Transformed features divided into 'train', 'val', and 'test'.
    """

    if isinstance(load_cache, str):
        # Load existing recipe
        recipe = restore_recipe(load_cache)
        data[type] = recipe.bake(data[type])
    elif isinstance(save_cache, str):
        # Save prepped recipe
        data[type] = recipe.prep()
        cache_recipe(recipe, save_cache)
    else:
        # No saving or loading of existing cache
        data[type] = recipe.prep()

    return data


def cache_recipe(recipe: Recipe, cache_file: str) -> None:
    """Cache recipe to make it available for e.g. transfer learning."""
    recipe_cache = copy.deepcopy(recipe)
    recipe_cache.cache()
    if not (cache_file / "..").exists():
        (cache_file / "..").mkdir()
    cache_file.touch()
    with open(cache_file, "wb") as f:
        pickle.dump(recipe_cache, f, pickle.HIGHEST_PROTOCOL)
    logging.info(f"Cached recipe in {cache_file}.")


def restore_recipe(cache_file: str) -> Recipe:
    """Restore recipe from cache to use for e.g. transfer learning."""
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            logging.info(f"Loading cached recipe from {cache_file}.")
            recipe = pickle.load(f)
            return recipe
    else:
        raise FileNotFoundError(f"Cache file {cache_file} not found.")

"""Module for static feature categorization.

This module provides functions to categorize and bin static features
for use in stratification during dataset splitting.
"""



def ethnicity_map(item):
    """Map ethnicity strings to integer categories.
    
    Args:
        item: Ethnicity string
        
    Returns:
        Integer category (0: white, 1: black, 2: asian, 3: other)
    """
    if isinstance(item, str):
        if 'white' in item.lower():
            return 0
        elif 'black' in item.lower():
            return 1
        elif 'asian' in item.lower():
            return 2
        else:
            return 3
    else:
        return 3  # Default for None or NaN


def categorize_static_features(data, polars=True):
    """Apply categorization to static features.
    
    Args:
        data: Dictionary containing data segments including static data
        polars: Whether data is in Polars (True) or Pandas (False) format
        
    Returns:
        Dictionary with the same data structure but with additional categorized static features
    """

    
    # logging.info("Categorizing static features for stratification...")
    if polars:
        return _categorize_static_features_polars(data)
    else:
        return _categorize_static_features_pandas(data)


def _categorize_static_features_polars(data):

    static_df = data

    static_df = _categorize_static_features_split_polars(static_df)
    data[Segment.static] = static_df

    
    return data

def _categorize_static_features_split_polars(data):
    """Apply categorization to static features in Polars DataFrame.
    
    Args:
        data: Dictionary containing data segments including static data in Polars format
        
    Returns:
        Dictionary with the same data structure but with additional categorized static features
    """
    if 'STATIC' not in data:
        logging.warning("No static segment found in data. Skipping categorization.")
        return data
    static_df = data[Segment.static]
    

    # Process gender if available
    if 'sex' in static_df.columns:
        static_df = static_df.with_columns(
            pl.col("sex").map_elements(lambda x: 0 if x == 'Female' else 1, return_dtype=pl.Int64).alias("sex_cat")
        )
    
    # Process ethnicity if available
    if 'ethnic' in static_df.columns:
        static_df = static_df.with_columns(
            pl.col("ethnic").fill_null("unknown").map_elements(ethnicity_map, return_dtype=pl.Int64).alias("ethnicity_cat")
        )
    
    # Process age if available
    if 'age' in static_df.columns:
        # Cap age at 90
        static_df = static_df.with_columns(
            pl.col("age").map_elements(lambda x: min(x, 90) if x is not None else None, return_dtype=pl.Float64).alias("age_capped")
        )
        
        # Create age groups
        age_bins = [0, 30, 50, 70, 100]
        
        # Define a function to bin ages
        def bin_age(age):
            if age is None:
                return None
            for i, upper in enumerate(age_bins[1:], 0):
                if age <= upper:
                    return i
            return len(age_bins) - 2
        
        static_df = static_df.with_columns(
            pl.col("age_capped").map_elements(bin_age, return_dtype=pl.Int64).alias("age_group")
        )
        
        logging.info(f"Age groups: 0: (0, 30], 1: (30, 50], 2: (50, 70], 3: (70, 100]")
    
    # Process BMI if weight and height are available
    if all(col in static_df.columns for col in ['weight', 'height']):
        # Calculate BMI
        # static_df = static_df.with_columns(
        #     (10000 * pl.col("weight") / (pl.col("height") * pl.col("height"))).alias("bmi")
        # )
        
        # Define BMI bins
        bmi_bins = [0, 18.5, 24.9, 29.9, 100]
        
        # Define a function to bin BMI
        def bin_bmi(bmi):
            if bmi is None:
                return None
            for i, upper in enumerate(bmi_bins[1:], 0):
                if bmi <= upper:
                    return i
            return len(bmi_bins) - 2
        
        static_df = static_df.with_columns(
            pl.col("bmi").map_elements(bin_bmi, return_dtype=pl.Int64).alias("bmi_group")
        )
        
        logging.info(f"BMI groups: 0: (0, 18.5] (Underweight), 1: (18.5, 24.9] (Normal), "
                     f"2: (24.9, 29.9] (Overweight), 3: (29.9, 100] (Obese)")
    
    
    return static_df


def _categorize_static_features_pandas(data):
    
    static_df = data[Segment.static].copy()

    static_df = _categorize_static_features_split_pandas(static_df)

    data[Segment.static] = static_df_train

    return data

def _categorize_static_features_split_pandas(static_df):
    """Apply categorization to static features in Pandas DataFrame.
    
    Args:
        data: Dictionary containing data segments including static data in Pandas format
        
    Returns:
        Dictionary with the same data structure but with additional categorized static features
    """
    

    
    # Process gender if available
    if 'sex' in static_df.columns:
        static_df['sex_cat'] = [0 if i == 'Female' else 1 for i in static_df['sex']]
    
    # Process ethnicity if available
    if 'ethnic' in static_df.columns:
        static_df['ethnicity'] = static_df['ethnicity'].fillna('unknown')
        static_df['ethnicity_cat'] = list(map(ethnicity_map, static_df['ethnicity']))
    
    # Process age if available
    if 'age' in static_df.columns:
        # Cap age at 90
        static_df['age_capped'] = [min(i, 90) if pd.notna(i) else np.nan for i in static_df['age']]
        
        # Create age groups
        age_bins = [0, 30, 50, 70, 100]
        static_df['age_group'] = pd.cut(static_df['age_capped'], bins=age_bins, labels=False)
        
        age_intervals = pd.cut(pd.Series([1, 31, 51, 71]), bins=age_bins).cat.categories
        age_interval_strings = [str(interval) for interval in age_intervals]
        age_interval_mapping = {interval: idx for idx, interval in enumerate(age_interval_strings)}
        logging.info(f"Age groups: {age_interval_mapping}")
    
    # Process BMI if weight and height are available
    if all(col in static_df.columns for col in ['weight', 'height']):
        # Calculate BMI
        static_df['bmi'] = 10000 * static_df['weight'] / (static_df['height'] * static_df['height'])
        
        # Create BMI groups
        bmi_bins = [0, 18.5, 24.9, 29.9, 100]
        static_df['bmi_group'] = pd.cut(static_df['bmi'], bins=bmi_bins, labels=False)
        
        bmi_intervals = pd.cut(pd.Series([1, 19, 25, 30]), bins=bmi_bins).cat.categories
        bmi_interval_strings = [str(interval) for interval in bmi_intervals]
        bmi_interval_mapping = {interval: idx for idx, interval in enumerate(bmi_interval_strings)}
        logging.info(f"BMI groups: {bmi_interval_mapping}")
    
    # Update data dictionary with modified static dataframe
    return static_df 

def filter_data_by_stay_ids(data_dict, stay_ids):
    """
    Filter a dictionary of Polars DataFrames to include only the specified stay IDs.
    
    Args:
        data_dict (dict): Dictionary with keys 'OUTCOME', 'STATIC', 'FEATURES' containing Polars DataFrames
        stay_ids (list): List of stay_ids to keep
    
    Returns:
        dict: Dictionary with the same structure but filtered DataFrames
    """
    if not stay_ids:
        raise ValueError("stay_ids list cannot be empty")
    
    filtered_data = {}
    
    # Create a filter expression for the stay_ids
    stay_id_filter = pl.col("stay_id").is_in(stay_ids)
    
    # Filter each DataFrame in the dictionary
    for key, df in data_dict.items():
        if key in ['OUTCOME', 'STATIC']:
            # These DataFrames typically have one row per stay_id
            filtered_data[key] = df.filter(stay_id_filter)
        elif key == 'FEATURES':
            # FEATURES DataFrame might have multiple rows per stay_id
            filtered_data[key] = df.filter(stay_id_filter)
        else:
            # Copy any other keys as-is
            filtered_data[key] = df
    
    # Verify that we found data for all requested stay_ids
    for key in ['OUTCOME', 'STATIC']:
        if key in filtered_data:
            found_ids = filtered_data[key].select('stay_id').unique().to_series().to_list()
            missing_ids = set(stay_ids) - set(found_ids)
            if missing_ids:
                print(f"Warning: {len(missing_ids)} stay_ids not found in {key} DataFrame")
    
    return filtered_data

def make_train_val_test(
    data: dict[pd.DataFrame],
    unfiltered_data: dict[pd.DataFrame],
    test_stay_ids : list[int],
    vars: dict[str],
    train_size=0.8,
    seed: int = 42,
    debug: bool = False,
    runmode: RunMode = RunMode.classification,
    polars: bool = True,
) -> dict[dict[pl.DataFrame]]:
    """Randomly split the data into training and validation sets for fitting a full model.

    Args:
        data: dictionary containing data divided int OUTCOME, STATIC, and DYNAMIC.
        vars: Contains the names of columns in the data.
        train_size: Fixed size of train split (including validation data).
        seed: Random seed.
        debug: Load less data if true.
    Returns:
        Input data divided into 'train', 'val', and 'test'.
    """
    # ID variable
    id = vars[Var.group]

    if debug:
        # Only use 1% of the data
        logging.info("Using only 1% of the data for debugging. Note that this might lead to errors for small datasets.")
        if polars:
            data[Segment.outcome] = data[Segment.outcome].sample(fraction=0.01, seed=seed)
        else:
            data[Segment.outcome] = data[Segment.outcome].sample(frac=0.01, random_state=seed)

    # Get stay IDs from outcome segment
    stays = _get_stays(data, id, polars)

    # If there are labels, and the task is classification, use stratified k-fold
    if Var.label in vars and runmode is RunMode.classification:
        # Get labels from outcome data (takes the highest value (or True) in case seq2seq classification)
        labels = _get_labels(data, id, vars, polars)
        train_val = StratifiedShuffleSplit(train_size=train_size, random_state=seed, n_splits=1)
        train, val = list(train_val.split(stays, labels))[0]

    else:
        # If there are no labels, use random split
        train_val = ShuffleSplit(train_size=train_size, random_state=seed)
        train, val = list(train_val.split(stays))[0]

    if polars:
        split = {
            Split.train: stays[train].cast(pl.datatypes.Int64).to_frame(),
            Split.val: stays[val].cast(pl.datatypes.Int64).to_frame(),
        }
    else:
        split = {Split.train: stays.iloc[train], Split.val: stays.iloc[val]}

    data_split = {}

    for fold in split.keys():  # Loop through splits (train / val / test)
        # Loop through segments (DYNAMIC / STATIC / OUTCOME)
        # set sort to true to make sure that IDs are reordered after scrambling earlier
        if polars:
            data_split[fold] = {
                data_type: split[fold]
                .join(data[data_type].with_columns(pl.col(id).cast(pl.datatypes.Int64)), on=id, how="left")
                .sort(by=id)
                for data_type in data.keys()
            }
        else:
            data_split[fold] = {
                data_type: data[data_type].merge(split[fold], on=id, how="right", sort=True) for data_type in data.keys()
            }
            
            

    # Maintain compatibility with test split
    if test_stay_ids:
        test_data = filter_data_by_stay_ids(unfiltered_data, test_stay_ids)
        data_split[Split.test] = test_data
    return data_split

def add_test_split(data,
                      unfiltered_data,
                      test_stay_ids):
    """
    Change the test split to a new test split.
    """
    data[Split.test] = filter_data_by_stay_ids(unfiltered_data, test_stay_ids)
    return data


def _get_stays(data, id, polars):
    return (
        pl.Series(name=id, values=data[Segment.outcome][id].unique())
        if polars
        else pd.Series(data[Segment.outcome][id].unique(), name=id)
    )


def _get_labels(data, id, vars, polars):
    # Get labels from outcome data (takes the highest value (or True) in case seq2seq classification)
    if polars:
        return data[Segment.outcome].group_by(id).max()[vars[Var.label]]
    else:
        return data[Segment.outcome].groupby(id).max()[vars[Var.label]].reset_index(drop=True)

