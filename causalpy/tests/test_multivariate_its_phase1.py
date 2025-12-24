#   Copyright 2022 - 2025 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Tests for Phase 1.1: Multivariate ITS Formula Handling

Tests the extension of InterruptedTimeSeries to accept and parse multiple formulas.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import causalpy as cp
from causalpy.custom_exceptions import FormulaException

# Fast sampling for PyMC tests
sample_kwargs = {
    "chains": 2,
    "draws": 100,
    "tune": 50,
    "progressbar": False,
    "random_seed": 42,
}


@pytest.fixture
def simple_data(rng):
    """Create simple time series data."""
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="M")
    n_months = len(dates)

    # Create two outcome variables
    y1 = 100 + np.linspace(0, 20, n_months) + rng.normal(0, 5, n_months)
    y2 = 50 + np.linspace(0, 10, n_months) + rng.normal(0, 3, n_months)

    # Create time variable
    t = np.arange(n_months)

    df = pd.DataFrame(
        {
            "y1": y1,
            "y2": y2,
            "t": t,
        },
        index=dates,
    )
    return df


def test_single_string_formula_backward_compatibility(simple_data):
    """Test that single string formula still works (backward compatibility)."""
    treatment_time = pd.Timestamp("2022-01-01")

    result = cp.InterruptedTimeSeries(
        simple_data,
        treatment_time,
        formula="y1 ~ 1 + t",
        model=LinearRegression(),
    )

    # Should have list attributes
    assert hasattr(result, "outcome_variable_names")
    assert hasattr(result, "_y_design_info_list")
    assert hasattr(result, "_x_design_info_list")

    # Lists should contain one element
    assert len(result.outcome_variable_names) == 1
    assert len(result._y_design_info_list) == 1
    assert len(result._x_design_info_list) == 1
    assert result.outcome_variable_names[0] == "y1"


def test_list_of_formulas(simple_data):
    """Test that list of formulas is accepted and parsed correctly."""
    treatment_time = pd.Timestamp("2022-01-01")

    formulas = [
        "y1 ~ 1 + t",
        "y2 ~ 1 + t",
    ]

    result = cp.InterruptedTimeSeries(
        simple_data,
        treatment_time,
        formula=formulas,
        model=LinearRegression(),
    )

    # Should have list attributes
    assert hasattr(result, "outcome_variable_names")
    assert hasattr(result, "_y_design_info_list")
    assert hasattr(result, "_x_design_info_list")

    # Lists should contain correct number of elements
    assert len(result.outcome_variable_names) == 2
    assert len(result._y_design_info_list) == 2
    assert len(result._x_design_info_list) == 2

    # Outcome variable names should be correct
    assert result.outcome_variable_names[0] == "y1"
    assert result.outcome_variable_names[1] == "y2"


def test_list_of_formulas_different_predictors(simple_data):
    """Test that different formulas with different predictors work."""
    treatment_time = pd.Timestamp("2022-01-01")

    # Add a month variable
    simple_data["month"] = simple_data.index.month

    formulas = [
        "y1 ~ 1 + t",
        "y2 ~ 1 + t + C(month)",
    ]

    result = cp.InterruptedTimeSeries(
        simple_data,
        treatment_time,
        formula=formulas,
        model=LinearRegression(),
    )

    # Should parse both formulas successfully
    assert len(result.outcome_variable_names) == 2
    assert len(result._y_design_info_list) == 2
    assert len(result._x_design_info_list) == 2

    # X design info should have different number of columns
    assert len(result._x_design_info_list[0].column_names) < len(
        result._x_design_info_list[1].column_names
    )


def test_empty_formula_list_raises_error(simple_data):
    """Test that empty formula list raises FormulaException."""
    treatment_time = pd.Timestamp("2022-01-01")

    with pytest.raises(FormulaException, match="cannot be empty"):
        cp.InterruptedTimeSeries(
            simple_data,
            treatment_time,
            formula=[],
            model=LinearRegression(),
        )


def test_non_string_in_formula_list_raises_error(simple_data):
    """Test that non-string elements in formula list raise FormulaException."""
    treatment_time = pd.Timestamp("2022-01-01")

    with pytest.raises(FormulaException, match="must be strings"):
        cp.InterruptedTimeSeries(
            simple_data,
            treatment_time,
            formula=["y1 ~ 1 + t", 123],  # Second element is not a string
            model=LinearRegression(),
        )


def test_invalid_formula_type_raises_error(simple_data):
    """Test that invalid formula type raises FormulaException."""
    treatment_time = pd.Timestamp("2022-01-01")

    with pytest.raises(FormulaException, match="must be a string or list of strings"):
        cp.InterruptedTimeSeries(
            simple_data,
            treatment_time,
            formula=123,  # Invalid type
            model=LinearRegression(),
        )


def test_invalid_formula_syntax_raises_error(simple_data):
    """Test that invalid formula syntax raises FormulaException."""
    treatment_time = pd.Timestamp("2022-01-01")

    with pytest.raises(FormulaException, match="Error parsing formula"):
        cp.InterruptedTimeSeries(
            simple_data,
            treatment_time,
            formula="y1 ~ invalid_syntax!!!",
            model=LinearRegression(),
        )


def test_multiple_formulas_with_invalid_one_raises_error(simple_data):
    """Test that if one formula in list is invalid, error is raised."""
    treatment_time = pd.Timestamp("2022-01-01")

    with pytest.raises(FormulaException, match="Error parsing formula"):
        cp.InterruptedTimeSeries(
            simple_data,
            treatment_time,
            formula=[
                "y1 ~ 1 + t",  # Valid
                "y2 ~ invalid_syntax!!!",  # Invalid
            ],
            model=LinearRegression(),
        )


def test_formula_stored_correctly(simple_data):
    """Test that the formula is stored correctly."""
    treatment_time = pd.Timestamp("2022-01-01")

    formula_str = "y1 ~ 1 + t"
    result = cp.InterruptedTimeSeries(
        simple_data,
        treatment_time,
        formula=formula_str,
        model=LinearRegression(),
    )
    assert result.formula == [formula_str]

    formula_list = ["y1 ~ 1 + t", "y2 ~ 1 + t"]
    result = cp.InterruptedTimeSeries(
        simple_data,
        treatment_time,
        formula=formula_list,
        model=LinearRegression(),
    )
    assert result.formula == formula_list
