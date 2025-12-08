import numpy as np

def accuracy_and_kappa_with_flip(y_true, y_pred):
    """
    Given binary labels y_true and y_pred (0/1),
    choose between y_pred and 1 - y_pred based on which
    gives higher accuracy versus y_true. Then compute
    accuracy and Cohen's kappa using the better orientation.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels (0 or 1).
    
    Returns
    -------
    accuracy : float
        Accuracy of the best-oriented predictions.
    kappa : float
        Cohen's kappa for the best-oriented predictions.
    used_flip : bool
        True if 1 - y_pred was used, False if original y_pred was used.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    
    # Ensure binary {0,1}
    if not np.array_equal(np.unique(y_true), np.intersect1d(np.unique(y_true), [0, 1])):
        raise ValueError("y_true must contain only 0 and 1.")
    if not np.array_equal(np.unique(y_pred), np.intersect1d(np.unique(y_pred), [0, 1])):
        raise ValueError("y_pred must contain only 0 and 1.")
    
    n = y_true.size
    if n == 0:
        raise ValueError("Empty input arrays.")

    # Accuracy with original predictions
    acc_orig = np.mean(y_true == y_pred)
    
    # Accuracy with flipped predictions
    y_pred_flip = 1 - y_pred
    acc_flip = np.mean(y_true == y_pred_flip)
    
    # Choose best orientation (tie -> original)
    if acc_flip > acc_orig:
        used_flip = True
        y_pred_best = y_pred_flip
        acc_best = acc_flip
    else:
        used_flip = False
        y_pred_best = y_pred
        acc_best = acc_orig

    # Confusion matrix components for best orientation
    tp = np.sum((y_true == 1) & (y_pred_best == 1))
    tn = np.sum((y_true == 0) & (y_pred_best == 0))
    fp = np.sum((y_true == 0) & (y_pred_best == 1))
    fn = np.sum((y_true == 1) & (y_pred_best == 0))
    
    # Observed agreement
    po = (tp + tn) / float(n)
    
    # Expected agreement under independence
    p_true_pos = (tp + fn) / float(n)
    p_true_neg = (tn + fp) / float(n)
    p_pred_pos = (tp + fp) / float(n)
    p_pred_neg = (tn + fn) / float(n)
    
    pe = p_true_pos * p_pred_pos + p_true_neg * p_pred_neg
    
    if np.isclose(1.0 - pe, 0.0):
        kappa = np.nan  # Degenerate case
    else:
        kappa = (po - pe) / (1.0 - pe)
    
    return acc_best, kappa, used_flip

