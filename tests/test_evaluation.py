import pytest
import numpy as np
from batchdetect.evaluation import accuracy_and_kappa_with_flip

def test_accuracy_kappa_valid_inputs():
    """Test with standard binary inputs."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0] # 75% accuracy
    
    acc, kappa, used_flip = accuracy_and_kappa_with_flip(y_true, y_pred)
    
    assert acc == 0.75
    assert not used_flip
    assert isinstance(kappa, float)

def test_accuracy_kappa_perfect_match():
    """Test where y_true == y_pred."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    
    acc, kappa, used_flip = accuracy_and_kappa_with_flip(y_true, y_pred)
    
    assert acc == 1.0
    assert kappa == 1.0
    assert not used_flip

def test_accuracy_kappa_perfect_flip():
    """Test where y_true == 1 - y_pred (should detect flip)."""
    y_true = [0, 1, 0, 1]
    y_pred = [1, 0, 1, 0] # Perfectly flipped
    
    acc, kappa, used_flip = accuracy_and_kappa_with_flip(y_true, y_pred)
    
    assert acc == 1.0
    assert kappa == 1.0
    assert used_flip

def test_accuracy_kappa_mixed():
    """Test a mixed case where flip might be better."""
    y_true = [0, 0, 1, 1]
    # y_pred = [1, 1, 0, 1] -> acc=0.25
    # flipped= [0, 0, 1, 0] -> acc=0.75
    y_pred = [1, 1, 0, 1]
    
    acc, kappa, used_flip = accuracy_and_kappa_with_flip(y_true, y_pred)
    
    assert acc == 0.75
    assert used_flip

def test_accuracy_kappa_shape_mismatch():
    """Test error on different shapes."""
    with pytest.raises(ValueError, match="y_true and y_pred must have the same shape"):
        accuracy_and_kappa_with_flip([0, 1], [0, 1, 0])

def test_accuracy_kappa_empty():
    """Test error on empty inputs."""
    with pytest.raises(ValueError, match="Empty input arrays"):
        accuracy_and_kappa_with_flip([], [])

def test_accuracy_kappa_degenerate():
    """Test case where kappa is undefined (e.g., all same class)."""
    # If all predictions are 0 (after flip or not) and all true are 0.
    # pe = 1.0. kappa = (1-1)/(1-1) = nan
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    
    acc, kappa, used_flip = accuracy_and_kappa_with_flip(y_true, y_pred)
    
    assert acc == 1.0
    assert np.isnan(kappa)
