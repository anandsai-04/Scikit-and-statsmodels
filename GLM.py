import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fit a GLM for claim amount
model = smf.glm(formula='claim_amount ~ age + gender + policy_type + ...', data=data, family=sm.families.Gaussian()).fit()
print(model.summary())
Model Diagnostics and Hypothesis Testing:

Check the model's assumptions, conduct statistical tests, and interpret the results.
