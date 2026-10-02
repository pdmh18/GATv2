import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_roc(curves: dict, path="reports/roc.png"):
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=name)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.savefig(path); plt.close()
