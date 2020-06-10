#!/home/juangonzalezvelez/.linuxbrew/bin/python3
# PBS -N CV_Tesis_JCGV
# PBS -o CV_out.txt
# PBS -e CV_err.txt

import numpy as np
from numpy.core._multiarray_umath import ndarray
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
from sklearn.metrics import accuracy_score as acc
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE
from imblearn.metrics import classification_report_imbalanced, sensitivity_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime as dt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve, roc_auc_score
from numpy import interp
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold, GroupShuffleSplit, GroupKFold, train_test_split

sns.set(style="ticks", color_codes=True)


def svc_param_selection(X, y, nfolds):
    Cs = 2. ** np.arange(-4, 4)  # Array of C's for cross-validation
    Gm = 10. ** np.arange(-5, 3)  # Array of gammas for cross-validation
    param_grid = {'C': Cs, 'gamma': Gm}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=nfolds, scoring='roc_auc')
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


parameters = {'criterion': ['entropy', 'gini'],
              'max_depth': list(np.linspace(10, 1200, 10, dtype=int)) + [None],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'min_samples_leaf': [1, 4, 12],
              'min_samples_split': [2, 5, 10],
              'n_estimators': list(np.linspace(151, 1200, 10, dtype=int))}

start = dt.now()

print("Started at: ", str(start))

num_features = 17

'''
Carga de Datos
'''

df_train = pd.read_csv('Databases\HS_Entrenamiento_LaUnion.csv')
# Select columns with characteristics
feature_name_train = df_train.columns.tolist()
feature_name_train = feature_name_train[:-1]
X_frame_train = df_train.iloc[:, :-1]
X = df_train.values[:, :-1]  # returns a numpy array
y = df_train.values[:, -1]

df_train2 = pd.read_csv('Databases\HS_Validacion_LaUnion.csv')
# Select columns with characteristics
feature_name_train2 = df_train2.columns.tolist()
feature_name_train2 = feature_name_train2[:-1]
X_frame_train2 = df_train2.iloc[:, :-1]
X2 = df_train2.values[:, :-1]  # returns a numpy array
y2 = df_train2.values[:, -1]

X = np.append(X, X2, axis=0)
y = np.append(y, y2, axis=0)

groups = np.zeros(4000)

groups[:1000] = 1
groups[1000:2000] = 2
groups[2000:3000] = 3
groups[3000:4000] = 4

# sns.pairplot(df_train.iloc[0:1000,5:],hue="Label")
# plt.savefig('C:/Users/jkgv1/OneDrive/Escritorio/Pairplot_Train_tramo1.png', dpi=300)
#
# sns.pairplot(df_train.iloc[1000:2000,5:],hue="Label")
# plt.savefig('C:/Users/jkgv1/OneDrive/Escritorio/Pairplot_Train_tramo2.png', dpi=300)
#
# sns.pairplot(df_train.iloc[2000:,5:],hue="Label")
# plt.savefig('C:/Users/jkgv1/OneDrive/Escritorio/Pairplot_Train_tramo3.png', dpi=300)
#
# sns.pairplot(df_test.iloc[:,5:],hue="Label")
# plt.savefig('C:/Users/jkgv1/OneDrive/Escritorio/Pairplot_Test.png', dpi=300)


'''
Reserva de memoria para metricas
'''

f1_test_RF = np.zeros(4)
acc_test_RF = np.zeros(4)
auc_test_RF = np.zeros(4)
kappa_test_RF = np.zeros(4)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

fold = 0  # inicio contador fold

gss = GroupKFold(n_splits=4)

for train_idx, test_idx in gss.split(X, y, groups):

    Xtrain = X[train_idx, :]
    ytrain = y[train_idx]
    Xtest = X[test_idx, :]
    ytest = y[test_idx]

    X_tr, Xtest, y_tr, ytest = train_test_split(Xtest, ytest, test_size=0.80, random_state=42)
    Xtrain = np.append(Xtrain, X_tr, axis=0)
    ytrain = np.append(ytrain, y_tr, axis=0)

    Xtest_df = pd.DataFrame(Xtest, columns=feature_name_train)
    ytest_df = pd.DataFrame(ytest, columns=['Label test'])

    pathX = 'C:/Users/jkgv1/OneDrive/Escritorio/' + 'Xtest' + 'fold' + str(fold) + '.xlsx'
    pathy = 'C:/Users/jkgv1/OneDrive/Escritorio/' + 'ytest' + 'fold' + str(fold) + '.xlsx'

    Xtest_df.to_excel(pathX)
    ytest_df.to_excel(pathy)

    Xtrain = Xtrain[:, 5:]
    Xtest = Xtest[:, 5:]

    scaler = MinMaxScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    n_neigh = 27
    print('n adasyn', n_neigh)
    ada = ADASYN(random_state=91, n_neighbors=n_neigh, sampling_strategy=1, n_jobs=6)
    Xtrain, ytrain = ada.fit_resample(Xtrain, ytrain)

    '''Optimización RF'''

    tpot_classifier = TPOTClassifier(generations=5, population_size=10, offspring_size=5,
                                     verbosity=2, early_stop=3,
                                     config_dict={'sklearn.ensemble.RandomForestClassifier': parameters},
                                     cv=4, scoring='roc_auc', n_jobs=12)

    '''Ajuste del modelo'''
    tpot_classifier.fit(Xtrain, ytrain)

    '''Predicción'''
    rf_y_pred = tpot_classifier.predict(Xtest)
    rf_y_prob = [probs[1] for probs in tpot_classifier.predict_proba(Xtest)]

    ypred_df = pd.DataFrame(rf_y_pred, columns=['Label pred'])

    pathy_pred = 'C:/Users/jkgv1/OneDrive/Escritorio/' + 'ypred' + 'fold' + str(fold) + '.xlsx'

    ypred_df.to_excel(pathy_pred)

    '''Validación y metricas de desempeño'''
    print('RF')
    print(confusion_matrix(ytest, rf_y_pred))
    print('kappa', cohen_kappa_score(ytest, rf_y_pred))
    report = precision_recall_fscore_support(ytest, rf_y_pred, average='weighted')
    auc_test_RF[fold] = roc_auc_score(ytest, rf_y_pred, average='weighted')
    kappa_test_RF[fold] = cohen_kappa_score(ytest, rf_y_pred)
    f1_test_RF[fold] = report[2]
    acc_test_RF[fold] = report[0]

    '''Cálculo del AUC'''
    fpr, tpr, _ = roc_curve(ytest, rf_y_pred)
    roc_auc = auc(fpr, tpr)

    interp_tpr = interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)
    fold = fold + 1

'''Calculo promedio de AUC'''
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

'''Grafica AUC'''
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="AUC-ROC - RF - TRANSFER")
ax.legend(loc="lower right")
plt.savefig('Graphs/ROC_AUC_TRANSFER_99_porc.png', dpi=300)

'''Print de metricas'''
print('------------------------')
print('F1 RF', np.mean(f1_test_RF))
print('Accuracy RF', np.mean(acc_test_RF))
print('Kappa RF', np.mean(kappa_test_RF))
print('AUC-ROC', np.mean(auc_test_RF))
print('------------------------')
print('                        ')

end = dt.now()
print("Finished at: ", str(end))
total = end - start
print("Total time spent: ", total)
