# tests/test_preprocessing.py

import pandas as pd
import pytest

# Example function to test. Let's imagine we put this in a separate utils.py file later.
def convert_labels_to_zero_indexed(series: pd.Series) -> pd.Series:
    """Converts a pandas Series of labels (e.g., [1, 2, 3]) to be zero-indexed ([0, 1, 2])."""
    if series.min() == 1:
        return series - 1
    return series

# Unit test for the function above
def test_label_conversion():
    """
    Tests that the label conversion function correctly subtracts 1
    from a series that starts at 1.
    """
    # Arrange: Create sample data
    input_data = pd.Series([1, 2, 3, 4, 5, 1])
    expected_output = pd.Series([0, 1, 2, 3, 4, 0])
    
    # Act: Run the function
    actual_output = convert_labels_to_zero_indexed(input_data)
    
    # Assert: Check if the result is as expected
    pd.testing.assert_series_equal(actual_output, expected_output)

def test_label_conversion_already_zero_indexed():
    """
    Tests that the label conversion function does nothing if the
    series already contains a 0.
    """
    # Arrange
    input_data = pd.Series([0, 1, 2, 3])
    expected_output = pd.Series([0, 1, 2, 3])
    
    # Act
    actual_output = convert_labels_to_zero_indexed(input_data)
    
    # Assert
    pd.testing.assert_series_equal(actual_output, expected_output)