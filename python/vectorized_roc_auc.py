import numpy as np
import pytest


def vectorized_roc_auc(y_trues: np.ndarray, y_preds: np.ndarray) -> np.ndarray:
    """Calculate ROC AUC scores for multiple sets of predictions using a vectorized approach.
    This will avoid looping over different sets of predictions and be faster than explicitly
    parallelizing.

    Args:
        y_trues (np.ndarray): Ground truth binary labels (1D or 2D array).
        y_preds (np.ndarray): Prediction scores (2D array, each row is a set of predictions).

    Returns:
        np.ndarray: ROC AUC scores for each set of predictions.

    Examples:
        >>> y_trues = np.array([0, 1, 1, 0, 1])
        >>> y_preds = np.array([[0.1, 0.23, 0.39, 0.86, 0.66]])
        >>> vectorized_roc_auc(y_trues, y_preds)
    """

    if y_trues.ndim == 1:
        y_trues = np.tile(y_trues, (y_preds.shape[0], 1))
    elif y_trues.shape[0] != y_preds.shape[0]:
        raise ValueError(
            "y_true and y_preds must have the same number of rows when y_trues is 2D"
        )

    # Sort the predictions and the corresponding y_true values
    sorted_indices = np.argsort(y_preds, axis=1)[:, ::-1]
    y_trues_sorted = np.take_along_axis(y_trues, sorted_indices, axis=1)

    # Calculate TPR and FPR
    tps = np.cumsum(y_trues_sorted, axis=1)
    fps = np.cumsum(1 - y_trues_sorted, axis=1)

    tpr = tps / tps[:, -1][:, np.newaxis]  # True positive rate
    fpr = fps / fps[:, -1][:, np.newaxis]  # False positive rate

    # Add (0,0) point to TPR and FPR
    tpr = np.hstack([np.zeros((tpr.shape[0], 1)), tpr])
    fpr = np.hstack([np.zeros((fpr.shape[0], 1)), fpr])

    # Compute AUC using the trapezoidal rule
    auc = np.trapz(tpr, fpr, axis=1)
    return auc


@pytest.fixture(params=[1, 2])
def sample_data(request):
    np.random.seed(42)
    n_samples = 1000
    n_pred_sets = 5

    if request.param == 1:
        # 1D y_true
        y_true = np.random.randint(0, 2, n_samples)
        y_preds = np.random.rand(n_pred_sets, n_samples)
    else:
        # 2D y_true
        y_true = np.random.randint(0, 2, (n_pred_sets, n_samples))
        y_preds = np.random.rand(n_pred_sets, n_samples)

    return y_true, y_preds


def test_vectorized_roc_auc(sample_data):
    y_true, y_preds = sample_data
    custom_scores = vectorized_roc_auc(y_true, y_preds)

    from sklearn.metrics import roc_auc_score

    if y_true.ndim == 1:
        sklearn_scores = np.array(
            [roc_auc_score(y_true, y_preds[i]) for i in range(y_preds.shape[0])]
        )
    else:
        sklearn_scores = np.array(
            [roc_auc_score(y_true[i], y_preds[i]) for i in range(y_preds.shape[0])]
        )

    np.testing.assert_allclose(custom_scores, sklearn_scores, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__])
