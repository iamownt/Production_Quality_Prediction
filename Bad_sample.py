# from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression
#
#
# # 实际上LR不能只接受一个类别。
# kf = KFold(n_splits=5)
#
# bad_ind = []
# threshold = 0.4
# for tr_ind, va_ind in kf.split(tr):
#     train_x, train_y = tr[tr_ind], tr[va_ind]
#     train_y = np.ones(len(train_y))
#     val_x, val_y = tr[va_ind], tr[val_ind]
#     log_reg = LogisticRegression(random_state=42)
#     log_reg.fit(train_x, train_y)
#     y_proba = log_reg.predict_proba(val_x)
#     bad_ind.append(va_ind[y_proba <= threshold])
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dropout", type=float, default=0, help="dropout_rate")
args = parser.parse_args()
print(args.dropout)
