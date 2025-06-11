# %%
import sys, pysqlite3
sys.modules['sqlite3'] = pysqlite3

import pandas as pd
import numpy as np

import geopandas as gpd
from shapely.ops import unary_union

from pqdm.processes import pqdm
from multiprocessing import Pool, cpu_count
from mpi4py import MPI
import swifter
import dask.dataframe as dd
import dask.dataframe as dd
from tqdm import tqdm

import os
import re
import pickle
import sys
import time
import csv
import json
from math import e, sin, pi, ceil, log

from word2number import w2n

from textblob import TextBlob
from dateutil.parser import parse
from spellchecker import SpellChecker
from collections import defaultdict
from itertools import combinations
from datetime import datetime

import pickle

import statsmodels.api as sm
from sklearn.utils import resample
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from linearmodels.iv.absorbing import AbsorbingLS
from statsmodels.tsa.api import VAR

from scipy.optimize import least_squares
import scipy.stats as stats

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from stargazer.stargazer import Stargazer

# from memory_profiler import profile

import warnings

from pandas.errors import PerformanceWarning

#### Silence irrelevant warnings
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="kurtosistest only valid for n>=20")
warnings.filterwarnings(
    "ignore", message="omni_normtest is not valid with less than 8 observations"
)
warnings.filterwarnings("ignore", message=".*kurtosistest.*")
warnings.filterwarnings(
    "ignore",
    message="DataFrame is highly fragmented. This is usually the result of calling `frame.insert` many times",
)

#### Set pandas settings
tqdm.pandas()
pd.set_option("future.no_silent_downcasting", True)

#### Set matplotlib settings
mpl.rc("font", family="Times New Roman", size=30)
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = "Times New Roman"
mpl.rcParams["mathtext.it"] = "Times New Roman:italic"
mpl.rcParams["mathtext.bf"] = "Times New Roman:bold"

# Make color map
viridis = mpl.colormaps["viridis"]
newcolors = viridis(np.linspace(0, 1, 100))[0:80]
cmap = ListedColormap(newcolors)

#### Set recursion limit
sys.setrecursionlimit(5000)

#### Define parameters
n_jobs = int(os.cpu_count()) - 2

output_folder = "/Users/vbp/Dropbox (Personal)/Apps/Overleaf/UK Duration"
figures_folder = os.path.join(output_folder, "Figures")
tables_folder = os.path.join(output_folder, "Tables")


hedonics_rm_full = [
    "bedrooms",
    "bathrooms",
    "floorarea",
    "yearbuilt",
    "livingrooms",
    "parking",
    "heatingtype",
    "condition",
]
hedonics_rm = ["bedrooms", "bathrooms", "floorarea", "yearbuilt", "livingrooms"]
hedonics_zoop = [
    "bedrooms_zoop",
    "bathrooms_zoop",
    "floors",
    "receptions",
]

# Connect to Dask scheduler on the main node (change to scheduler address if needed)
# client = Client("tcp://$(hostname -s):8786")  # This uses the SLURM main node


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

    def are_connected(self, x, y):
        return self.find(x) == self.find(y)

    def build_groups(self):
        groups = {}
        for element in self.parent:
            root = self.find(element)
            if root not in groups:
                groups[root] = set()
            groups[root].add(element)
        return groups

    def get_group(self, x):
        root = self.find(x)
        group = set()
        for element in self.parent:
            if self.find(element) == root:
                group.add(element)
        return group

    def find_smallest_connector(self, group1, group2, min_diff=1):
        smallest_diff = float("inf")
        smallest_pair = None

        for elem1 in group1:
            for elem2 in group2:
                diff = abs(elem1 - elem2)
                if diff < smallest_diff:
                    smallest_diff = diff
                    smallest_pair = (elem1, elem2)

                    if smallest_diff == min_diff:
                        break

        return smallest_pair


def remove_punt(s):
    for punct in [".", ",", "'", "(", ")", "FLAT ", "APARTMENT "]:
        s = s.replace(punct, "")
    s = s.replace(" - ", "-")
    return s


def create_string_id(
    df, key_name="property_id", columns=["flat_number", "street_number", "postcode"]
):
    df[key_name] = df[columns].fillna("").agg(" ".join, axis=1)
    df[key_name] = df[key_name].str.upper()
    df[key_name] = df[key_name].apply(remove_punt)
    df[key_name] = df[key_name].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def load_residuals(data_folder):
    dfs = []
    for file in os.listdir(f"{data_folder}/working/residuals"):
        # print(file)
        if not file.startswith("residual"):
            continue
        df = pd.read_pickle(os.path.join(data_folder, "working", "residuals", file))
        dfs.append(df)
    return pd.concat(dfs)


def years_between_dates(s):
    return s.apply(lambda x: x.n / 365 if pd.notna(x) else np.nan)


def month_str(n):
    # Ensure n is within the valid range of 1-12
    if 1 <= n <= 12:
        # Create a date object for the first day of the nth month in any year (e.g., 2020)
        date = datetime(2020, n, 1)
        # Return the month name
        return date.strftime("%B")  # %B for full month name, %b for abbreviated name
    else:
        return str(n)


def get_union(df):
    uf = UnionFind()
    for i, row in df.iterrows():
        uf.union(row["date"], row["L_date"])
    return uf


def baseline_ystar_function(ystar, T, k):
    # Compute the exponents
    exponent_A = (ystar / 100) * (T + k)
    exponent_B = (ystar / 100) * T

    # Calculate the expressions, ensuring numerical stability
    expr1 = np.log(np.clip(1 - np.exp(-exponent_A), 1e-15, None))
    expr2 = np.log(np.clip(1 - np.exp(-exponent_B), 1e-15, None))

    return expr1 - expr2


def estimate_ystar(
    df,
    lhs_var="did_rsi",
    model_function=baseline_ystar_function,
    get_se=True,
    n_boot=50,
):

    def residuals(params, T, k, lhs_var):
        # Calculate residuals
        return lhs_var - model_function(params, T, k)

    # Drop missing observations
    df.drop(
        df[(df["T"].isna()) | (df["k"].isna()) | (df[lhs_var].isna())].index,
        inplace=True,
    )

    # Extract key columns
    did = df[lhs_var]
    T = df["T"]
    k = df["k"]

    # Initial guess for ystar
    initial_guess = [3]

    # Perform the least squares optimization
    try:
        result = least_squares(
            residuals,
            initial_guess,
            args=(T, k, did),
            bounds=([0], [np.inf]),  # Ensure ystar stays positive
            method="trf",  # Trust Region Reflective algorithm
            jac="2-point",  # Numerical estimation of the Jacobian
        )

        # Extract the estimated parameter
        ystar_estimate = result.x[0]

        # Option 1: Robust SEs
        if get_se is True or get_se == "robust":
            # Calculate residuals at the solution
            resid = result.fun  # This is lhs_var - model_function(params, T, k)

            # Obtain the Jacobian matrix at the solution
            J = result.jac  # Shape: (number of observations, number of parameters)

            # Number of observations and parameters
            n_obs = len(did)
            n_params = len(result.x)

            # Compute the robust covariance matrix using the sandwich estimator
            # Inverse of (J^T J)
            JTJ_inv = np.linalg.inv(J.T @ J)

            # Middle matrix: J^T * diag(residuals^2) * J
            middle_matrix = J.T @ np.diag(resid**2) @ J

            # Robust covariance matrix
            robust_cov = JTJ_inv @ middle_matrix @ JTJ_inv

            # Extract the robust standard error for ystar
            ystar_std_error = np.sqrt(np.diag(robust_cov))[0]

        # Option 2: Boostrap SEs
        elif get_se == "boot":
            boot_estimates = []
            for _ in range(n_boot):
                # Sample with replacement
                df_sample = df.sample(frac=1, replace=True)

                # Extract columns from bootstrap sample
                did_b = df_sample[lhs_var].values
                T_b = df_sample["T"].values
                k_b = df_sample["k"].values

                # Try to run the same estimation on the bootstrap sample
                try:
                    res_b = least_squares(
                        residuals,
                        initial_guess,
                        args=(T_b, k_b, did_b),
                        bounds=([0], [np.inf]),
                        method="trf",
                        jac="2-point",
                    )
                    boot_estimates.append(res_b.x[0])
                except:
                    pass

            if len(boot_estimates) > 1:
                ystar_std_error = np.std(boot_estimates, ddof=1)
            else:
                ystar_std_error = np.nan

        else:
            ystar_std_error = np.nan

        return ystar_estimate, ystar_std_error

    except:
        print("XXX NLLS estimation failed.")
        return np.nan, np.nan


def residualize(df_full, dependent_var, indep_vars, absorb_vars, residual_name):
    """
    Runs an OLS regression with fixed effects (absorbed variables), computes residuals,
    and adjusts them by adding back the mean of the dependent variable.

    Parameters:
    - df: pandas DataFrame containing the data.
    - dependent_var: string, name of the dependent variable.
    - absorb_vars: list of strings, names of variables to absorb (fixed effects).
    - residual_name: string, name of the residuals column to be created in df.
    """

    df = df_full.dropna(subset=[dependent_var] + indep_vars + absorb_vars)

    y = df[dependent_var]
    X = sm.add_constant(df[indep_vars])
    absorb_df = df[absorb_vars].copy()

    for var in absorb_vars:
        if absorb_df[var].dtype.name != "category":
            absorb_df[var] = absorb_df[var].astype("category")

    if len(absorb_vars) > 0:
        model = AbsorbingLS(y, X, absorb=absorb_df)
        res = model.fit()
        df_full[residual_name] = res.resids + y.mean()
    else:
        model = sm.OLS(y, X)
        res = model.fit()
        df_full[residual_name] = res.resid + y.mean()

    return df_full


def log_time_elapsed(time_elapsed, part, data_folder):
    """
    Logs the time elapsed for a specific part of the data construction process.
    The log is saved in a file named 'time_elapsed.txt' in the specified data folder.
    """

    os.makedirs(os.path.join(data_folder, "log"), exist_ok=True)

    t = np.round(time_elapsed / 60)
    log_file = os.path.join(
        data_folder, "log", f"log_DC{part}_{time.strftime('%Y_%m_%d')}.txt"
    )
    with open(log_file, "w") as f:
        f.write(f"Time elapsed in data construction (part {part}): {t} minutes\n")


# %%
