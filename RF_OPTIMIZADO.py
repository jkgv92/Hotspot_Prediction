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

'''Definición de parametros para optimización'''


def svc_param_selection(X, y, nfolds):
    Cs = 2. ** np.arange(-4, 4)  # Array of C's for cross-validation
    Gm = 10. ** np.arange(-5, 3)  # Array of gammas for cross-validation
    param_grid = {'C': Cs, 'gamma': Gm}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=nfolds, scoring='roc_auc')
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


def DL_Model(activation1='linear', neurons1=5, neurons2=5, neurons3=5, optimizer='Adam', activation2='linear',
             activation3='linear'):
    model = Sequential()
    model.add(Dense(neurons1, input_dim=17, activation=activation1))
    model.add(Dense(neurons2, activation=activation2))
    model.add(Dense(neurons3, activation=activation3))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


parameters = {'criterion': ['entropy', 'gini'],
              'max_depth': list(np.linspace(10, 1200, 10, dtype=int)) + [None],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'min_samples_leaf': [4, 12],
              'min_samples_split': [5, 10],
              'n_estimators': list(np.linspace(151, 1200, 10, dtype=int))}

activation1 = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
activation2 = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
activation3 = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
neurons1 = [5, 10, 20, 50, 60, 80]
neurons2 = [5, 10, 20, 50, 60, 80]
neurons3 = [5, 10, 20, 50, 60, 80]
optimizer = ['SGD', 'Adam', 'Adamax']
param_grid = dict(activation1=activation1, activation2=activation2, activation3=activation3, neurons1=neurons1,
                  neurons2=neurons2, neurons3=neurons3, optimizer=optimizer)

start = dt.now()

print("Started at: ", str(start))

num_features = 17

'''
Carga de Datos
'''

df = pd.read_csv('Databases\HS_Aprendizaje.csv')
# Select columns with characteristics
feature_name = df.columns.tolist()
feature_name = feature_name[5:-1]
X_frame = df.iloc[:, 5:-1]
X = df.values[:, 5:]  # returns a numpy array
Xn = np.zeros((12, 250, 91))
l = 250

for i in range(12):
    Xn[i, :, :] = X[l * i:l * (i + 1), :]

y = np.zeros(12)
y[4:8] = 1
y[8:12] = 2

'''
Reserva de memoria para metricas
'''

f1_test_svm = np.zeros(4)
acc_test_svm = np.zeros(4)
auc_test_svm = np.zeros(4)
kappa_test_svm = np.zeros(4)

f1_test_knn = np.zeros(4)
acc_test_knn = np.zeros(4)
auc_test_knn = np.zeros(4)
kappa_test_knn = np.zeros(4)

f1_test_RF = np.zeros(4)
acc_test_RF = np.zeros(4)
auc_test_RF = np.zeros(4)
kappa_test_RF = np.zeros(4)

f1_test_rnn = np.zeros(4)
acc_test_rnn = np.zeros(4)
auc_test_rnn = np.zeros(4)
kappa_test_rnn = np.zeros(4)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()

fold = 0  # inicio contador fold

'''
Selección de función de clasificación para SelectKBest
'''
score_func = mutual_info_classif
# score_func = chi2
# score_func = f_classif
print('la funcion para seleccion de caracteristicas es: ', score_func)

loo_test = StratifiedKFold(n_splits=4, random_state=None, shuffle=False)  # Selección del tipo de validación cruzada

for train_index, test_index in loo_test.split(np.squeeze(Xn[:, 0, :]), y):

    '''Carga de datos por segmentos'''

    print('Train index', train_index)
    print('Test index', test_index)

    Xtrain = Xn[train_index, :, :]
    Xtest = Xn[test_index, :, :]
    Xtrain2 = np.squeeze(Xtrain[0, :, :])

    for iii in range(1, Xtrain.shape[0]):
        Xtrain2 = np.append(Xtrain2, np.squeeze(Xtrain[iii, :, :]), axis=0)
    Xtrain = np.copy(Xtrain2)
    ytrain = Xtrain[:, -1]
    Xtrain = Xtrain[:, 0:-1]
    Xtest2: ndarray = np.squeeze(Xtest[0, :, :])

    for iii in range(1, Xtest.shape[0]):
        Xtest2 = np.append(Xtest2, np.squeeze(Xtest[iii, :, :]), axis=0)
    Xtest = np.copy(Xtest2)
    ytest = Xtest[:, -1]
    Xtest = Xtest[:, 0:-1]

    '''Normalización de datos'''

    scaler = MinMaxScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    '''Optimización SMOTE'''

    best_params_smote = svc_param_selection(Xtrain, ytrain, 5)
    SVM_smote = svm.SVC(kernel='rbf', C=best_params_smote['C'], gamma=best_params_smote['gamma'],
                        class_weight='balanced')

    print('valor c ideal SVM SMOTE', best_params_smote['C'], 'valor gamma ideal SVM SMOTE', best_params_smote['gamma'])

    border_sm = BorderlineSMOTE(k_neighbors=27, random_state=91, sampling_strategy=1)

    sm = SVMSMOTE(random_state=91, k_neighbors=2, sampling_strategy=1, svm_estimator=SVM_smote)

    ada = ADASYN(random_state=91, n_neighbors=27, sampling_strategy=1, n_jobs=6)

    Kmeans = KMeansSMOTE(random_state=91, k_neighbors=2, sampling_strategy=1, n_jobs=6,
                         kmeans_estimator=MiniBatchKMeans(n_clusters=20))

    '''Muestreo Sintetico'''

    #Xtrain, ytrain = SMOTE().fit_resample(Xtrain, ytrain)
    Xtrain, ytrain = border_sm.fit_resample(Xtrain, ytrain)

    '''Selección de caracteristicas'''

    # rel_MI = SelectKBest(score_func=score_func, k=num_features)
    # Xtrain = rel_MI.fit_transform(Xtrain, ytrain)
    # Xtest = rel_MI.transform(Xtest)
    # rel_MI_support = rel_MI.get_support()
    # rel_MI_feature = X_frame.loc[:, rel_MI_support].columns.tolist()
    # rel_MI_scores = rel_MI.scores_[rel_MI_support].tolist()
    # feature_selection_df = pd.DataFrame({'Feature': rel_MI_feature, 'Score':rel_MI_scores})

    Xtrain = Xtrain[:, [71, 83, 88, 70, 89, 56, 86, 53, 58, 59, 29, 28, 69, 41, 74, 23, 87]]
    Xtest = Xtest[:, [71, 83, 88, 70, 89, 56, 86, 53, 58, 59, 29, 28, 69, 41, 74, 23, 87]]

    '''  
    RF 
    '''

    # RF = RandomForestClassifier(n_estimators=1000, bootstrap=True, class_weight='balanced')
    # RF.fit(Xtrain, ytrain)
    # rf_y_pred = RF.predict(Xtest)



    '''Optimización RF'''
    tpot_classifier = TPOTClassifier(generations=5, population_size=24, offspring_size=12,
                                     verbosity=2, early_stop=12,
                                     config_dict={'sklearn.ensemble.RandomForestClassifier': parameters},
                                     cv=4, scoring='balanced_accuracy')

    '''Ajuste del modelo'''
    tpot_classifier.fit(Xtrain, ytrain)

    '''Predicción'''
    rf_y_pred = tpot_classifier.predict(Xtest)
    rf_y_prob = [probs[1] for probs in tpot_classifier.predict_proba(Xtest)]

    '''Validación y metricas de desempeño'''
    print('RF')
    print(confusion_matrix(ytest, rf_y_pred))
    print('kappa', cohen_kappa_score(ytest, rf_y_pred))
    report = precision_recall_fscore_support(ytest, rf_y_pred, average='weighted')
    auc_test_RF[fold] = roc_auc_score(ytest, rf_y_pred, average='weighted')
    kappa_test_RF[fold] = cohen_kappa_score(ytest, rf_y_pred)
    f1_test_RF[fold] = report[2]
    acc_test_RF[fold] = report[0]

    # Compute area under the curve
    fpr, tpr, _ = roc_curve(ytest, rf_y_prob)
    roc_auc = auc(fpr, tpr)

    interp_tpr = interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)


    # # only use if you can visualise
    #
    # # Set default figure size
    # plt.rcParams['figure.figsize'] = (8, 8)
    # # Plot ROC curve
    # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title("Title")
    # plt.legend(loc="lower right")
    # plt.show()

    '''Contador de iteracciones'''
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
       title="AUC-ROC - RF - BORDERLINE")
ax.legend(loc="lower right")
plt.savefig('Graphs/ROC_AUC_RF_OPTIMIZADA_BORDERLINE.png', dpi=300)


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
