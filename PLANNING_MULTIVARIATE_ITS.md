# Planning Document: Multivariate Interrupted Time Series (ITS)

## Key Decisions

- **Model Approach**: Single multivariate model with full covariance matrix (not independent models)
- **Model Instance**: Always use the same model instance for all outcomes
- **Supported Models**: LinearRegression from PyMCModel and LinearRegression from sklearn (must support multivariate outcomes)
- **Covariance**: Full covariance matrix (not optional, not diagonal)
- **Dimension Name**: Use `outcomes` dimension in xarray DataArrays
- **Formula Flexibility**: Allow different predictors (RHS) across formulas - each outcome can have different predictor variables

## API Design

Since we use patsy for formula parsing, and patsy does not support the formulae syntax (e.g., `c(y1, y2, y3) ~ x + z`), we will use a list of formulas as the API.

```python
result = cp.InterruptedTimeSeries(
    df,
    treatment_time,
    formula=[
        "prod1_sales ~ 1 + t + C(month)",
        "prod2_sales ~ 1 + t + C(month)",
        "prod3_sales ~ 1 + t + C(month)",
    ],
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"random_seed": seed}
    ),
)
```

**Requirements:**
- Works with existing patsy infrastructure (no extensions needed)
- Flexible: Each outcome can have different RHS formulas
- Backward compatible (single string formula still works for univariate case)

## Implementation Plan

### Phase 1: Core Data Structure Changes

#### 1.1 Formula Handling
- [ ] Modify `__init__` to accept `formula: str | list[str]`
- [ ] Validate formula input (single string or list of strings)
- [ ] Parse each formula using `dmatrices()` from patsy
- [ ] Store multiple outcome variable names
- [ ] Store multiple design info objects (`_y_design_info_list`, `_x_design_info_list`)

#### 1.2 Data Structure
- [ ] Create `pre_y` and `post_y` as xarray DataArrays with new dimension `outcomes`
  - Shape: `(n_obs, n_outcomes)` instead of `(n_obs, 1)`
  - Coords: `{"obs_ind": ..., "outcomes": ["outcome_0", "outcome_1", ...]}`
- [ ] Create `pre_X` and `post_X` as list of DataArrays (one per outcome)
  - Each can have different number of coefficients (since different predictors are allowed)
  - Each outcome can have different predictor variables
- [ ] Store outcome variable names: `self.outcome_variable_names: list[str]`

#### 1.3 Model Fitting Strategy

**Selected Approach: Single Multivariate Model**
- Use single model instance for all outcomes
- Extend LinearRegression (PyMC and sklearn) to handle multivariate outcomes
- Use full covariance matrix for residuals (captures correlations between outcomes)
- Model must support multivariate y: shape `(n_obs, n_outcomes)`
- Store single model: `self.model: PyMCModel | RegressorMixin`

**Implementation Notes:**
- Same model instance used for all outcomes (not a list of models)
- Models must be extended to support multivariate outcomes
- Full covariance matrix is used (not optional for now)
- Supported model types: LinearRegression from PyMCModel and LinearRegression from sklearn

#### 1.4 Testing
- [ ] Test formula parsing (single string vs list)
- [ ] Test formula validation (invalid inputs)
- [ ] Test data structure creation (pre_y, post_y with outcomes dimension)
- [ ] Test backward compatibility (single string formula works)
- [ ] Test single outcome case (behaves identically to current implementation)
- [ ] Test that existing univariate tests still pass

### Phase 2: Model Extensions

#### 2.1 Multivariate Model Support
- [ ] Extend LinearRegression (PyMC) to handle multivariate outcomes
  - Support full covariance matrix for residuals:
    ```python
    # Full covariance matrix
    L = pm.LKJCholeskyCov("L", n=n_outcomes, eta=2, sd_dist=pm.HalfNormal.dist(sigma=1))
    cov = pm.Deterministic("cov", L @ L.T)
    ```
  - Update `build_model()` to handle multivariate y (shape: `(n_obs, n_outcomes)`)
  - Update `fit()`, `predict()`, `score()` methods to work with multivariate outcomes
- [ ] Extend LinearRegression (sklearn) to handle multivariate outcomes
  - Update `fit()`, `predict()`, `score()` methods to work with multivariate outcomes
  - Handle multivariate y (shape: `(n_obs, n_outcomes)`)
- [ ] Ensure same model instance is used for all outcomes

#### 2.2 Model Selection
- [ ] Models must support multivariate outcomes (check model capabilities)
- [ ] Supported models: LinearRegression from PyMCModel and LinearRegression from sklearn
- [ ] Full covariance matrix is used by default for PyMC models (not optional for now)

#### 2.3 Testing
- [ ] Test model fitting with multivariate outcomes (PyMC LinearRegression)
- [ ] Test model fitting with multivariate outcomes (sklearn LinearRegression)
- [ ] Test that same model instance is used for all outcomes
- [ ] Test full covariance matrix structure in PyMC model
- [ ] Test model fit with different formulas per outcome
- [ ] Test model fit with same formula for all outcomes

### Phase 3: Prediction and Impact Calculation

#### 3.1 Counterfactual Predictions
- [ ] For each outcome, generate counterfactual predictions
- [ ] Store as xarray with `outcomes` dimension
- [ ] Shape: `(n_post_obs, n_outcomes)` or `(n_chains, n_draws, n_post_obs, n_outcomes)`

#### 3.2 Impact Calculation
- [ ] Calculate impact for each outcome separately
- [ ] Store as xarray with `outcomes` dimension
- [ ] Shape: `(n_post_obs, n_outcomes)` or `(n_chains, n_draws, n_post_obs, n_outcomes)`
- [ ] Cumulative impact per outcome

#### 3.3 Three-Period Design Support
- [ ] Extend `_split_post_period()` to handle multivariate outcomes
- [ ] Split intervention and post-intervention impacts per outcome

#### 3.4 Testing
- [ ] Test counterfactual predictions for all outcomes
- [ ] Test prediction shapes (correct outcomes dimension)
- [ ] Test impact calculation for each outcome
- [ ] Test cumulative impact per outcome
- [ ] Test three-period design with multivariate outcomes
- [ ] Test impact calculation with different formulas per outcome

### Phase 4: Visualization

#### 4.1 Plotting Strategy
- [ ] Separate subplot per outcome (recommended)
  - `fig, ax = plt.subplots(n_outcomes, 3, ...)` (3 plots per outcome)
  - Each outcome gets its own row of 3 plots (fit, impact, cumulative)

#### 4.2 Plot Methods
- [ ] Update `_bayesian_plot()` to handle multivariate outcomes
- [ ] Update `_ols_plot()` to handle multivariate outcomes
- [ ] Add parameter `outcome: str | int | None = None` to plot specific outcome
- [ ] Add parameter `plot_all: bool = True` to plot all outcomes

#### 4.3 Plot Data Export
- [ ] Update `get_plot_data_bayesian()` and `get_plot_data_ols()`
- [ ] Return DataFrame with outcome identifier column
- [ ] Or: Return dict of DataFrames keyed by outcome name

#### 4.4 Testing
- [ ] Test plotting with multivariate outcomes (PyMC model)
- [ ] Test plotting with multivariate outcomes (sklearn model)
- [ ] Test separate subplots per outcome
- [ ] Test `outcome` parameter (plot specific outcome)
- [ ] Test `plot_all` parameter
- [ ] Test plot data export (DataFrame structure)

### Phase 5: Reporting and Summaries

#### 5.1 Effect Summary
- [ ] Update `effect_summary()` to handle multivariate outcomes
- [ ] Add parameter `outcome: str | int | None = None`
  - `None`: Summarize all outcomes (return dict or multi-index DataFrame)
  - `str/int`: Summarize specific outcome
- [ ] Return structure:
  - Single outcome: `EffectSummary` (current behavior)
  - Multiple outcomes: `dict[str, EffectSummary]` or `MultiOutcomeEffectSummary`

#### 5.2 Summary Tables
- [ ] Create summary table comparing effects across outcomes
- [ ] Include columns: outcome name, mean effect, HDI/CI, cumulative effect
- [ ] Option: Include correlation matrix if using multivariate model

#### 5.3 Persistence Analysis
- [ ] Extend `analyze_persistence()` for multivariate outcomes
- [ ] Return dict keyed by outcome name
- [ ] Or: Return DataFrame with outcome as index

#### 5.4 Testing
- [ ] Test `effect_summary()` with multivariate outcomes
- [ ] Test `effect_summary(outcome=None)` returns all outcomes
- [ ] Test `effect_summary(outcome="outcome_name")` returns specific outcome
- [ ] Test summary tables structure
- [ ] Test persistence analysis with multivariate outcomes
- [ ] Test return structures (dict vs DataFrame)

### Phase 6: Integration and Edge Case Testing

#### 6.1 Integration Tests
- [ ] Test full workflow with 2-3 outcomes (end-to-end)
- [ ] Test complete workflow with PyMC LinearRegression
- [ ] Test complete workflow with sklearn LinearRegression
- [ ] Test workflow with different formulas per outcome
- [ ] Test workflow with same formula for all outcomes

#### 6.2 Edge Cases
- [ ] Test with many outcomes (10+)
- [ ] Test with missing data in some outcomes
- [ ] Test with different time ranges per outcome (if applicable)
- [ ] Test error handling for invalid inputs

### Phase 7: Documentation

#### 7.1 API Documentation
- [ ] Update docstring for `InterruptedTimeSeries`
- [ ] Document new `formula` parameter (str | list[str])
- [ ] Document multivariate model options
- [ ] Add examples for multivariate use cases

#### 7.2 Example Notebooks
- [ ] Create `its_multivariate_pymc.ipynb`
  - Marketing campaign example (multiple products)
  - Education policy example (multiple test scores)
- [ ] Create `its_multivariate_skl.ipynb`
  - Same examples using sklearn LinearRegression
- [ ] Update existing ITS notebooks to mention multivariate option

## Technical Considerations

### Data Structure Design

**Current (Univariate):**
```python
pre_y: xr.DataArray  # Shape: (n_obs, 1), dims: ["obs_ind", "treated_units"]
```

**Proposed (Multivariate):**
```python
pre_y: xr.DataArray  # Shape: (n_obs, n_outcomes), dims: ["obs_ind", "outcomes"]
```

**Decision**: Use new `outcomes` dimension for clarity and to avoid confusion with synthetic control's `treated_units`

### Model Fitting

**Multivariate Model Approach:**
```python
# Single model fit with multivariate outcomes
self.model.fit(X=pre_X_combined, y=pre_y, coords=coords)
# X_combined: stacked or list of X arrays (one per outcome, or combined structure)
# y: (n_obs, n_outcomes) - xarray DataArray with "outcomes" dimension
# Model uses full covariance matrix for residuals
```

**Note**: Same model instance is used for all outcomes. Models must support multivariate outcomes (LinearRegression from PyMCModel and LinearRegression from sklearn).

### Backward Compatibility

- [ ] Single string formula should still work
- [ ] Convert to list internally: `formula = [formula] if isinstance(formula, str) else formula`
- [ ] Single outcome case should behave identically to current implementation
- [ ] All existing tests should pass without modification

## Implementation Order

1. **Phase 1**: Core data structure changes + Phase 1.4 Testing
2. **Phase 2**: Model extensions + Phase 2.3 Testing
3. **Phase 3**: Predictions and impacts + Phase 3.4 Testing
4. **Phase 4**: Visualization + Phase 4.4 Testing
5. **Phase 5**: Reporting and summaries + Phase 5.4 Testing
6. **Phase 6**: Integration and edge case testing
7. **Phase 7**: Documentation (user enablement)

## Success Criteria

- [ ] Can fit multivariate ITS with list of formulas
- [ ] Backward compatible with single formula
- [ ] Generates counterfactuals for all outcomes
- [ ] Calculates impacts for all outcomes
- [ ] Plots all outcomes (separate subplots)
- [ ] Effect summaries work for multivariate case
- [ ] Three-period design works with multivariate outcomes
- [ ] Comprehensive test coverage
- [ ] Documentation and examples complete
