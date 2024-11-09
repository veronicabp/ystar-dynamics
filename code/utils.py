# %%

import pandas as pd
import numpy as np

import dask.dataframe as dd
from dask.distributed import Client, progress
from dask.diagnostics import ProgressBar
from dask import delayed, compute
from pqdm.processes import pqdm
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

import os
import re
import sys
import time
import csv
from math import ceil

from word2number import w2n
from textblob import TextBlob
from dateutil.parser import parse
from spellchecker import SpellChecker

import pickle

import statsmodels.api as sm
from sklearn.utils import resample
from scipy.optimize import curve_fit
from scipy.optimize import least_squares

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="kurtosistest only valid for n>=20")
warnings.filterwarnings(
    "ignore", message="omni_normtest is not valid with less than 8 observations"
)
warnings.filterwarnings("ignore", message=".*kurtosistest.*")

tqdm.pandas()
pd.set_option("future.no_silent_downcasting", True)

n_jobs = int(os.cpu_count()) - 2
hedonics_rm = ["bedrooms", "floorarea", "bathrooms", "livingrooms", "yearbuilt"]

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


def estimate_ystar(df, lhs_var="did_rsi"):
    def model_function(ystar, T, k):
        # Compute the exponents
        exponent_A = (ystar / 100) * (T + k)
        exponent_B = (ystar / 100) * T

        # Calculate the expressions, ensuring numerical stability
        expr1 = np.log(np.clip(1 - np.exp(-exponent_A), 1e-15, None))
        expr2 = np.log(np.clip(1 - np.exp(-exponent_B), 1e-15, None))

        return expr1 - expr2

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

    return ystar_estimate, ystar_std_error


# %%
