import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier

def get_data_sets(x, y):
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size = 0.15, random_state = 0)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size = 0.15, random_state = 0)
    # Data de entrenamiento
    print(f'Forma de x_train : {x_train.shape}')
    print(f'Forma de y_train : {y_train.shape}')
    #Data de validación
    print(f'Forma de x_test : {x_val.shape}')
    print(f'Forma de y_test : {y_val.shape}')
    #Data de prueba
    print(f'Forma de x_test : {x_test.shape}')
    print(f'Forma de y_test : {y_test.shape}')
    return x_train_val, y_train_val, x_test, y_test, x_train, y_train, x_val, y_val

def prepare_data(dataset):
    # Tranformar boolean a Integer (Falso = 0, Verdadero = 1)
    le = LabelEncoder()
    revenue = le.fit_transform(dataset['Revenue'])

    # Separar en datos y clases
    x = pd.get_dummies(dataset)
    x = x.drop(['Revenue'], axis = 1)
    y = revenue

    print(f'Forma de x: {x.shape}')
    print(f'Forma de y: {y.shape}')
    return x, y

def get_report(y_test, y_pred):
    confu_matrix = confusion_matrix(y_test, y_pred)
    TN = confu_matrix[0,0]
    FN = confu_matrix[1,0]
    FP = confu_matrix[0,1]
    TP = confu_matrix[1,1]

    print ('              +-----------------+')
    print ('              |   Predicción    |')
    print ('              +-----------------+')
    print ('              |    +   |    -   |')
    print ('+-------+-----+--------+--------+')
    print ('| Valor |  +  |  {:5d} |  {:5d} |'.format(TP, FN) )
    print ('| real  +-----+--------+--------+')
    print ('|       |  -  |  {:5d} |  {:5d} |'.format(FP, TN) )
    print ('+-------+-----+--------+--------+')
    print()

    print(classification_report(y_test, y_pred))

def get_optimal_values_KNN(x_train, y_train, x_val, y_val):
    neighbors = [x for x in range(1,50) if x % 2 != 0]
    p_values = list(range(1,4))
    weights = ['uniform', 'distance']
    max_score = 0
    optimal_k = 1
    optimal_p = 1

    # Evaluamos para escoger el mejor parámetro
    for k in neighbors:
        for p in p_values:
            for weight in weights:
                knn = KNeighborsClassifier(n_neighbors=k, p=p, weights=weight, n_jobs=-1)
                knn.fit(x_train, y_train)
                y_pred = knn.predict(x_val)
                if max_score < accuracy_score(y_val,y_pred)*100:
                    optimal_k = k
                    optimal_p = p
                    optimal_weight = weight
                    max_score = accuracy_score(y_val,y_pred)*100
    print(max_score, optimal_k, optimal_p, optimal_weight)
    return max_score, optimal_k, optimal_p, optimal_weight

def get_optimal_values_Random_Forest(x_train, y_train, x_val, y_val):
    n_estimators = [x for x in range(0,160,5) if x]
    criterions = ['gini', 'entropy']
    max_features = [5, 'sqrt', 'log2', None]
    max_score = 0
    warm_starts = [True, False]
    optimal_n_estimator = 1
    optiomal_criteria = 'gini'
    optiomal_m_feature = 1
    optimal_warm_start = False

    # Evaluamos para escoger el mejor parámetro
    for k in n_estimators:
        for c in criterions:
            for m_feature in max_features:
                for warm_start in warm_starts:
                    random_forest = RandomForestClassifier(n_estimators=k, criterion=c, max_features=m_feature, warm_start=warm_start, random_state=0, n_jobs=-1)
                    random_forest.fit(x_train, y_train)
                    y_pred = random_forest.predict(x_val)
                    if max_score < accuracy_score(y_val, y_pred)*100:
                        optimal_n_estimator = k
                        optiomal_criteria = c
                        optiomal_m_feature = m_feature
                        optimal_warm_start = warm_start
                        max_score = accuracy_score(y_val,y_pred)*100
    print(max_score, optimal_n_estimator, optiomal_criteria, optiomal_m_feature, optimal_warm_start)
    return max_score, optimal_n_estimator, optiomal_criteria, optiomal_m_feature, optimal_warm_start

def get_optimal_values_decision_tree(x_train, y_train, x_val, y_val):
    criterions = ['gini', 'entropy']
    splitters = ['best', 'random']
    max_features = [5, 'sqrt', 'log2', None]
    class_weights = ['balanced' , None]
    max_score = 0
    optimal_splitter = 'best'
    optiomal_criteria = 'gini'
    optiomal_m_feature = 1
    optimal_class_weight = None

    # Evaluamos para escoger el mejor parámetro
    for k in splitters:
        for c in criterions:
            for m_feature in max_features:
                for class_w in class_weights:
                    decesion_tree = DecisionTreeClassifier(splitter=k, criterion=c, max_features=m_feature, class_weight=class_w, random_state=0)
                    decesion_tree.fit(x_train, y_train)
                    y_pred = decesion_tree.predict(x_val)
                    if max_score < accuracy_score(y_val, y_pred)*100:
                        optimal_splitter = k
                        optiomal_criteria = c
                        optiomal_m_feature = m_feature
                        optimal_class_weight = class_w
                        max_score = accuracy_score(y_val,y_pred)*100
    print(max_score, optimal_splitter, optiomal_criteria, optiomal_m_feature, optimal_class_weight)
    return max_score, optimal_splitter, optiomal_criteria, optiomal_m_feature, optimal_class_weight

def get_optimal_values_ComplementNB(x_train, y_train, x_val, y_val):
    alphas = [x/10 for x in range(0,11)]
    fit_priors = [True, False]
    norms = [True, False]
    max_score = 0
    optimal_fit_prior = True
    optimal_alpha = 1.0
    optiomal_norm = False
    
    # Evaluamos para escoger el mejor parámetro
    for alpha in alphas:
        for fit_prior in fit_priors:
            for norm in norms:
                naive = ComplementNB(alpha=alpha, fit_prior=fit_prior, norm=norm)
                naive.fit(x_train, y_train)
                y_pred = naive.predict(x_val)
                if max_score < accuracy_score(y_val, y_pred)*100:
                    optimal_alpha = alpha
                    optimal_fit_prior = fit_prior
                    optiomal_norm = norm
                    max_score = accuracy_score(y_val, y_pred)*100
    print(max_score, optimal_alpha, optimal_fit_prior, optiomal_norm)
    return max_score, optimal_alpha, optimal_fit_prior, optiomal_norm

def get_optimal_values_SVC(x_train, y_train, x_val, y_val):
    Cs = [x for x in range(0,101,10) if x ]
    Cs.append(1)
    gammas = ['auto', 'scale']
    probabilities = [False, True]
    shrinkings = [False, True]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    max_score = 0
    optimal_C = 1
    optimal_gamma = 'auto'
    optiomal_prob = False
    optimal_shrinking = True
    optimal_kernel = 'rbf'
    
    # Evaluamos para escoger el mejor parámetro
    for c in Cs:
        for gamma in gammas:
            for prob in probabilities:
                for shrk in shrinkings:
                    for kernel in kernels:
                        svc = SVC(kernel=kernel, C=c, gamma=gamma, random_state=0, probability=prob, shrinking=shrk)
                        svc.fit(x_train, y_train)
                        y_pred = svc.predict(x_val)
                        if max_score < accuracy_score(y_val, y_pred)*100:
                            optimal_C = c
                            optimal_gamma = gamma
                            optiomal_prob = prob
                            optimal_shrinking = shrk
                            optimal_kernel = kernel
                    
                    max_score = accuracy_score(y_val, y_pred)*100
    print(max_score, optimal_C, optimal_gamma, optiomal_prob, optimal_shrinking, optimal_kernel)
    return max_score, optimal_C, optimal_gamma, optiomal_prob, optimal_shrinking, optimal_kernel

