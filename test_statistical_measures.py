"""
Unit tests for statisticalMeasures class.

This test suite demonstrates the usage of statistical measures for comparing
prediction models in machine learning contexts.
"""

import unittest
from statistical_measures import statisticalMeasures


class TestStatisticalMeasures(unittest.TestCase):
    
    # ========================================================================
    # RMSE Tests
    # ========================================================================
    
    def test_RMSE_perfect_prediction(self):
        actual = [10.0, 20.0, 30.0, 40.0]
        perfect_pred = [10.0, 20.0, 30.0, 40.0]
        rmse = statisticalMeasures.calculateRMSE(actual, perfect_pred)
        
        self.assertEqual(rmse, 0.0, "Perfect predictions should have RMSE = 0")
    
    
    def test_RMSE_small_errors(self):
        actual = [10.0, 20.0, 30.0, 40.0]
        good_pred = [10.5, 19.8, 30.2, 39.9]
        rmse = statisticalMeasures.calculateRMSE(actual, good_pred)
        
        self.assertGreater(rmse, 0, "Small errors should result in RMSE > 0")
        self.assertLess(rmse, 1.0, "Small errors should have relatively small RMSE")
    
    
    def test_RMSE_large_errors(self):
        actual = [10.0, 20.0, 30.0, 40.0]
        good_pred = [10.5, 19.8, 30.2, 39.9]
        poor_pred = [12.0, 18.0, 33.0, 37.0]
        
        rmse_good = statisticalMeasures.calculateRMSE(actual, good_pred)
        rmse_poor = statisticalMeasures.calculateRMSE(actual, poor_pred)
        
        self.assertGreater(rmse_poor, rmse_good, "Poor model should have higher RMSE")
    
    
    # ========================================================================
    # MAE Tests
    # ========================================================================
    
    def test_MAE_perfect_prediction(self):
        actual = [100, 200, 300, 400]
        perfect_pred = [100, 200, 300, 400]
        mae = statisticalMeasures.calculateMAE(actual, perfect_pred)
        
        self.assertEqual(mae, 0.0, "Perfect predictions should have MAE = 0")
    
    
    def test_MAE_consistent_small_errors(self):
        actual = [100, 200, 300, 400]
        consistent_pred = [102, 198, 303, 397]
        mae = statisticalMeasures.calculateMAE(actual, consistent_pred)
        
        self.assertGreater(mae, 0, "Small errors should result in MAE > 0")
        self.assertLess(mae, 5.0, "Consistent small errors should have low MAE")
    
    
    def test_MAE_with_outlier(self):
        actual = [100, 200, 300, 400]
        consistent_pred = [102, 198, 303, 397]
        outlier_pred = [101, 199, 301, 450]  # Last prediction way off
        
        mae_consistent = statisticalMeasures.calculateMAE(actual, consistent_pred)
        mae_outlier = statisticalMeasures.calculateMAE(actual, outlier_pred)
        
        self.assertGreater(mae_outlier, mae_consistent, "MAE should increase with outlier")
    
    
    # ========================================================================
    # SDE Tests
    # ========================================================================
    
    def test_SDE_no_variation(self):
        actual = [10, 20, 30, 40, 50]
        consistent_pred = [12, 22, 32, 42, 52]  # All off by +2
        sde = statisticalMeasures.calculateSDE(actual, consistent_pred)
        
        self.assertEqual(sde, 0.0, "Consistent errors should have SDE = 0")
    
    
    def test_SDE_low_variation(self):
        actual = [10, 20, 30, 40, 50]
        stable_pred = [11, 21, 29, 41, 49]
        sde = statisticalMeasures.calculateSDE(actual, stable_pred)
        
        self.assertGreater(sde, 0, "Varying errors should have SDE > 0")
        self.assertLess(sde, 2.0, "Low variation should have relatively small SDE")
    
    
    def test_SDE_high_variation(self):
        actual = [10, 20, 30, 40, 50]
        stable_pred = [11, 21, 29, 41, 49]
        unstable_pred = [8, 25, 28, 45, 47]
        
        sde_stable = statisticalMeasures.calculateSDE(actual, stable_pred)
        sde_unstable = statisticalMeasures.calculateSDE(actual, unstable_pred)
        
        self.assertGreater(sde_unstable, sde_stable, "Unstable model should have higher SDE")
    
    
    # ========================================================================
    # Paired t-test Tests
    # ========================================================================
    
    def test_paired_t_test_no_difference(self):
        model1_errors = [0.5, 0.6, 0.4, 0.7, 0.5]
        model2_errors = [0.5, 0.6, 0.4, 0.7, 0.5]
        t_stat, p_val = statisticalMeasures.calculatePairedTTest(model1_errors, model2_errors)
        
        # When errors are identical, p_val should be 1.0 or nan (both mean no difference)
        # scipy's ttest_rel returns nan when std=0
        self.assertTrue(p_val == 1.0 or (p_val != p_val), "Identical errors should show no significance (p=1.0 or nan)")    
    
    def test_paired_t_test_significant_difference(self):
        model1_errors = [0.5, 0.6, 0.4, 0.7, 0.5, 0.6]  # Small consistent overestimation
        model2_errors = [2.5, 2.8, 2.4, 2.6, 2.7, 2.5]  # Large consistent overestimation
        t_stat, p_val = statisticalMeasures.calculatePairedTTest(model1_errors, model2_errors)
        
        self.assertLess(p_val, 0.05, "Consistent bias difference should be significant")

    
    
    # ========================================================================
    # Wilcoxon Test Tests
    # ========================================================================
    
    def test_wilcoxon_no_difference(self):
        model1_errors = [0.5, -0.4, 0.6, -0.5, 0.4]
        model2_errors = [0.5, -0.4, 0.6, -0.5, 0.4]
        w_stat, p_val = statisticalMeasures.wilcoxon_test(model1_errors, model2_errors)
        
        self.assertGreater(p_val, 0.05, "Identical errors should NOT be significant")
    
   
    # ========================================================================
    # Error Handling Tests
    # ========================================================================
    
    def test_error_handling(self):
        actual = [1.0, 2.0, 3.0]
        predicted = [1.1, 2.2]  # Different length
        
        with self.assertRaises(ValueError):
            statisticalMeasures.calculateRMSE(actual, predicted)
        
        with self.assertRaises(ValueError):
            statisticalMeasures.calculateMAE(actual, predicted)
        
        with self.assertRaises(ValueError):
            statisticalMeasures.calculateSDE(actual, predicted)
        
        errors1 = [0.1, 0.2]
        errors2 = [0.1, 0.2, 0.3]  # Different length
        
        with self.assertRaises(ValueError):
            statisticalMeasures.calculatePairedTTest(errors1, errors2)
        
        with self.assertRaises(ValueError):
            statisticalMeasures.wilcoxon_test(errors1, errors2)


def run_example_showcase():
    """
    Practical example: Comparing Linear Regression vs Transformer Model
    
    This demonstrates how to use all statistical measures in a real workflow.
    """        
    actual_prices = [145.2, 167.8, 189.3, 202.5, 234.7, 256.2, 278.9, 301.3]
    linear_reg_pred = [146.0, 168.5, 188.0, 203.0, 235.0, 255.0, 280.0, 300.0]
    transformer_pred = [145.5, 167.5, 189.5, 202.0, 234.5, 256.5, 279.0, 301.5]
    
    # Accuracy
    print("\nAccuracy Comparison")
    print("-" * 70)
    lr_rmse = statisticalMeasures.calculateRMSE(actual_prices, linear_reg_pred)
    tf_rmse = statisticalMeasures.calculateRMSE(actual_prices, transformer_pred)
    lr_mae = statisticalMeasures.calculateMAE(actual_prices, linear_reg_pred)
    tf_mae = statisticalMeasures.calculateMAE(actual_prices, transformer_pred)
    
    print(f"Linear Regression  → RMSE: {lr_rmse:.3f}, MAE: {lr_mae:.3f}")
    print(f"Transformer        → RMSE: {tf_rmse:.3f}, MAE: {tf_mae:.3f}")
    winner_acc = 'Transformer' if tf_rmse < lr_rmse else 'Linear Regression'
    print(f"Winner (Accuracy): {winner_acc} ✓\n")
    
    # Stability
    print("\nStability Analysis")
    print("-" * 70)
    lr_sde = statisticalMeasures.calculateSDE(actual_prices, linear_reg_pred)
    tf_sde = statisticalMeasures.calculateSDE(actual_prices, transformer_pred)
    
    print(f"Linear Regression SDE: {lr_sde:.3f}")
    print(f"Transformer SDE:       {tf_sde:.3f}")
    winner_stable = 'Transformer' if tf_sde < lr_sde else 'Linear Regression'
    print(f"Winner (Stability): {winner_stable} ✓\n")
    
    # Statistical significance
    print("\nStatistical Significance Test")
    print("-" * 70)
    lr_errors = [a - p for a, p in zip(actual_prices, linear_reg_pred)]
    tf_errors = [a - p for a, p in zip(actual_prices, transformer_pred)]
    
    w_stat, p_val = statisticalMeasures.wilcoxon_test(lr_errors, tf_errors)
    print(f"Wilcoxon test p-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print(f"Result: Models are SIGNIFICANTLY different ✓")
    else:
        print(f"Result: No significant difference")
    
    print("\nFINAL CONCLUSION")
    print(f"Best Accuracy:  {winner_acc}")
    print(f"Most Stable:    {winner_stable}")
    print(f"Significant:    {'Yes' if p_val < 0.05 else 'No'}")
    print("\n")


if __name__ == "__main__":
    # Run the practical example first
    run_example_showcase()
    
    print("\n" + "="*70)
    print("RUNNING UNIT TESTS")
    print("="*70 + "\n")
    
    # Custom test runner for clean output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestStatisticalMeasures)
    
    # Run tests with custom output
    for test in suite:
        result = unittest.TestResult()
        test.run(result)
        test_name = test._testMethodName
        
        if result.wasSuccessful():
            print(f"{test_name} ... PASS ✓")
        else:
            print(f"{test_name} ... FAIL ✗")
            for failure in result.failures + result.errors:
                print(f"  {failure[1]}")