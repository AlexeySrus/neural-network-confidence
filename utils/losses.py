def l2(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()