"""
Statistical Measures for Model Evaluation

This module contains the statisticalMeasures class with methods for:
- Prediction accuracy metrics (RMSE, MAE)
- Stability/Robustness metrics (Standard Deviation of Errors)
- Statistical significance tests (Paired t-test, Wilcoxon Signed-Rank test)
"""

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


class statisticalMeasures:
    """
    A class containing statistical measures for evaluating and comparing
    prediction models, particularly for linear regression models and transformers.
    """
    
    @staticmethod
    def calculateRMSE(actualVals: list[float], predictedVals: list[float]) -> float:
        """
        Calculate Root Mean Squared Error (RMSE)
        
        Inputs:
            actualVals: list[float] - The actual/true values from the dataset
            predictedVals: list[float] - The predicted values from linear regression 
                                          models and transformers
        
        Output:
            float - The RMSE value (lower is better, 0 means perfect prediction)
        
        When to use:
            Use RMSE to measure prediction accuracy when you want to:
            - Penalize larger errors more heavily (due to squaring)
            - Evaluate model performance in the same units as the target variable
            - Compare different models where outliers and large errors are critical
            
        Note:
            Ensure actualVals and predictedVals are aligned and have the same length.
        """
        actualVals = np.array(actualVals)
        predictedVals = np.array(predictedVals)
        
        if len(actualVals) != len(predictedVals):
            raise ValueError("actualVals and predictedVals must have the same length")
        
        squared_errors = (actualVals - predictedVals) ** 2
        mean_squared_error = np.mean(squared_errors)
        rmse = np.sqrt(mean_squared_error)
        
        return float(rmse)
    
    @staticmethod
    def calculateMAE(actualVals: list[float], predictedVals: list[float]) -> float:
        """
        Calculate Mean Absolute Error (MAE)
        
        Inputs:
            actualVals: list[float] - The actual/true values from the dataset
            predictedVals: list[float] - The predicted values from linear regression 
                                          models and transformers
        
        Output:
            float - The MAE value (lower is better, 0 means perfect prediction)
        
        When to use:
            Use MAE to measure prediction accuracy when you want to:
            - Treat all errors equally (linear penalty)
            - Get a more robust metric that's less sensitive to outliers than RMSE
            - Understand the average magnitude of errors in the same units as target
            - Compare models where outliers should not dominate the evaluation
            
        Note:
            Ensure actualVals and predictedVals are aligned and have the same length.
        """
        actualVals = np.array(actualVals)
        predictedVals = np.array(predictedVals)
        
        if len(actualVals) != len(predictedVals):
            raise ValueError("actualVals and predictedVals must have the same length")
        
        absolute_errors = np.abs(actualVals - predictedVals)
        mae = np.mean(absolute_errors)
        
        return float(mae)
    
    @staticmethod
    def calculateSDE(actualVals: list[float], predictedVals: list[float]) -> float:
        """
        Calculate Standard Deviation of Errors (SDE)
        
        Inputs:
            actualVals: list[float] - The actual/true values from the dataset
            predictedVals: list[float] - The predicted values from linear regression 
                                          models and transformers
        
        Output:
            float - The SDE value (lower is better, indicates more consistent predictions)
        
        When to use:
            Use SDE to measure model stability/robustness when you want to:
            - Assess the consistency and reliability of predictions
            - Understand the variability in prediction errors
            - Determine if a model makes consistently accurate predictions or has
              erratic behavior with high variance
            - Compare the stability of different models (lower SDE = more stable)
            
        Note:
            Ensure actualVals and predictedVals are aligned and have the same length.
        """
        actualVals = np.array(actualVals)
        predictedVals = np.array(predictedVals)
        
        if len(actualVals) != len(predictedVals):
            raise ValueError("actualVals and predictedVals must have the same length")
        
        errors = actualVals - predictedVals
        sde = np.std(errors, ddof=1)  # Using ddof=1 for sample standard deviation
        
        return float(sde)
    
    @staticmethod
    def calculatePairedTTest(model1Errors: list[float], model2Errors: list[float]) -> tuple[float, float]:
        """
        Calculate Paired t-test for comparing two models
        
        Inputs:
            model1Errors: list[float] - Prediction errors from the first model
                                         (calculated as actualVals - predictedVals)
            model2Errors: list[float] - Prediction errors from the second model
                                         (calculated as actualVals - predictedVals)
        
        Output:
            tuple[float, float] - (t_statistic, p_value)
                t_statistic: The calculated t-statistic
                p_value: The two-tailed p-value (typically use α=0.05 for significance)
        
        When to use:
            Use paired t-test for statistical significance testing when:
            - Data is normally distributed (check with Shapiro-Wilk or Q-Q plots)
            - You want to compare two models on the same dataset
            - Sample size is reasonably large (n > 30 typically safe)
            - You need to determine if one model performs significantly better than another
            
        Interpretation:
            - If p_value < 0.05: The difference between models is statistically significant
            - If p_value >= 0.05: No significant difference between models
            
        Note:
            Ensure model1Errors and model2Errors are aligned (errors from same samples).
            Use Wilcoxon test instead if data is not normally distributed.
        """
        model1Errors = np.array(model1Errors)
        model2Errors = np.array(model2Errors)
        
        if len(model1Errors) != len(model2Errors):
            raise ValueError("model1Errors and model2Errors must have the same length")
        
        t_statistic, p_value = ttest_rel(model1Errors, model2Errors)
        
        return (float(t_statistic), float(p_value))
    
    @staticmethod
    def wilcoxon_test(model1Errors: list[float], model2Errors: list[float]) -> tuple[float, float]:
        """
        Calculate Wilcoxon Signed-Rank test for comparing two models
        
        Inputs:
            model1Errors: list[float] - Prediction errors from the first model
                                         (calculated as actualVals - predictedVals)
            model2Errors: list[float] - Prediction errors from the second model
                                         (calculated as actualVals - predictedVals)
        
        Output:
            tuple[float, float] - (statistic, p_value)
                statistic: The Wilcoxon test statistic
                p_value: The two-tailed p-value (typically use α=0.05 for significance)
        
        When to use:
            Use Wilcoxon Signed-Rank test for statistical significance testing when:
            - Data is NOT normally distributed (non-parametric alternative to t-test)
            - You have paired samples from two models on the same dataset
            - Sample size is small or distribution is skewed
            - You need a robust comparison that doesn't assume normality
            
        Interpretation:
            - If p_value < 0.05: The difference between models is statistically significant
            - If p_value >= 0.05: No significant difference between models
            
        Note:
            Ensure model1Errors and model2Errors are aligned (errors from same samples).
            This is the non-parametric alternative to the paired t-test.
            Use when normality assumption of t-test is violated.
        """
        model1Errors = np.array(model1Errors)
        model2Errors = np.array(model2Errors)
        
        if len(model1Errors) != len(model2Errors):
            raise ValueError("model1Errors and model2Errors must have the same length")
        
        statistic, p_value = wilcoxon(model1Errors, model2Errors)
        
        return (float(statistic), float(p_value))
