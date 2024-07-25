from sklearn.pipeline import Pipeline
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def one_hot_encoder1(dataframe, categorical_cols):  # bu yeni datayla farklılık olmasın diye sonra da işimize yarayacak.
    #https://drlee.io/surviving-the-one-hot-encoding-pitfall-in-data-science-62d8254cf3f6

    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded_cols = ohe.fit_transform(dataframe[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, index=dataframe.index)

    if hasattr(ohe, 'get_feature_names_out'):
        encoded_df.columns = ohe.get_feature_names_out(categorical_cols)
    else:
        encoded_df.columns = ohe.get_feature_names(categorical_cols)
    encoded_columns = encoded_df.columns
    dataframe = pd.concat([dataframe.drop(columns=categorical_cols), encoded_df], axis=1)
    return dataframe, encoded_columns

def carettas_data_prep(df):
    df = df.drop(columns=["Comments", "Notes", "Unnamed: 42", "Species", "Key", "ExDate"])
    df = df[0:93]
    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)
    cat_cols.append("VegType")
    # missing value handling:
    for col in num_cols:
        if df[col].isnull().any():
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
    df["Divisions"].fillna(df["Divisions"].mode()[0], inplace=True)
    df["VegType"].fillna("no", inplace=True)
    df["PlantRoot"].fillna("no", inplace=True)

    # outlier handling
    for col in num_cols:
        #print(col, check_outlier(df, col))
        if check_outlier(df, col):  # eğer true ise
            replace_with_thresholds(df, col)
    # VegType'taki çoklu bitki çeşitlerini ayrı samplelar haline getirme:
    df_aa = df.copy()

    def split_vegtype(df_aa):
        rows = []
        for _, row in df.iterrows():
            veg_types = str(row['VegType']).split('\n') if pd.notnull(row['VegType']) else []
            for veg in veg_types:
                new_row = row.copy()
                new_row['VegType'] = veg.strip()
                rows.append(new_row)
        return pd.DataFrame(rows)

    new_df = split_vegtype(df_aa)
    multi_entries_mask = df['VegType'].str.contains('\n', na=False)
    cleaned_df = df[~multi_entries_mask]
    df = pd.concat([cleaned_df, new_df], ignore_index=True)  # merged_df --> df
    # VegType yazım hatalarını düzeltme:

    corrections = {
        " -railorad vine": "-railroad vine",
        "-railorad vine": "-railroad vine",
        " -railroad vine": "-railroad vine",
        " -sea oats on dune": "-sea oats",
        "-sea grape": "-sea grapes",
        "-beach purslane": "-sea purslane",
        "-salt grass or torpedo grass (seashore dropseed)": "-salt grass",
        "-salt grass/torpedo grass/seashore dropseed": "-salt grass",
        "-sea oats on dune": "-sea oats",
        "-crested saltbush": "-crested saltbrush",
        " -sea oats": "-sea oats"
    }
    df["VegType"] = df["VegType"].replace(corrections)
    # duplicate data handling:
    df = df.drop_duplicates(keep='last')  # biri hariç diğer tekrarlayanları sildim.
    df.reset_index(drop=True, inplace=True)
    # MultiVegNum feature oluşturma:
    group_sizes = df.groupby('NestID').size()
    df.loc[:, 'MultiVegNum'] = df['NestID'].map(group_sizes - 1)
    # vegpresence'ı 0 olup vegtype girili olanlar varsa droplamak için:
    faulty_veg = df[(df['VegPresence'] == 0) & (df['VegType'] != 'no')]
    df = df.drop(faulty_veg.index)
    df.reset_index(drop=True, inplace=True)
    # encoding
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    df = df.drop(columns=["NestID"])  # kardinal ama önceden işimize yaradığından şimdi dropladım.
    df, colnames_1h = one_hot_encoder1(df, cat_cols)
    # normalization of numeric cols
    # X_scaled = RobustScaler().fit_transform(df[num_cols])
    # df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)
    y = df["HS"]
    X = df.drop(["HS"], axis=1)
    num_cols = [col for col in num_cols if col != 'HS']
    X_scaled = RobustScaler().fit_transform(X[num_cols])
    X[num_cols] = pd.DataFrame(X_scaled, columns=X[num_cols].columns)
    return X, y
#df = pd.read_csv(r"C:\Users\kilic\PycharmProjects\cerenPyProje\Impact-of-Dune-Plants-on-Sea-Turtles-Machine-Learning-App-main\ReddingRootsCaseStudy22_csv.csv")
#X, y, cn = carettas_data_prep(df)

scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']


def base_models(X, y, scoring):
    print("Base Models....")
    classifiers = [('LR', LinearRegression()),
                   ("SVR", SVR()),
                   ("CART", DecisionTreeRegressor()),
                   ("RF", RandomForestRegressor()),
                   ("ENet", ElasticNet()),
                   ("XGBRegressor", XGBRegressor())
                   ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    for name, classifier in classifiers:
        classifier.fit(X_train, y_train)
        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)

        print(f"showing TRAIN SCORES for {name}")
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred_train))
        print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred_train))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))
        print('r2_score:', metrics.r2_score(y_train, y_pred_train))

        print(f"showing TEST SCORES for {name}")
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_test))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_test))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))
        print('r2_score:', metrics.r2_score(y_test, y_pred_test))


############# defining hyperparameters ###############
lr_params = {}
svr_params = {'kernel': ["rbf", "linear", "poly"],
              'epsilon': range(0, 1),
              'C': [10, 30, 40, 50, 60, 70, 80, 90, 100]}
rf_params = {'n_estimators': [50, 100, 200],
             'max_depth': [None, 10, 20],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4]}
dtree_params = {"max_depth": range(1, 11),
                "min_samples_split": range(2, 20)}
en_params = {'alpha': [0.1, 1, 10, 0.01],
             'l1_ratio': np.arange(0.30, 1.00, 0.15),
             'tol': [0.0001, 0.001]}
xgb_params = {'n_estimators': [100, 200, 300],
              'learning_rate': [0.01, 0.05, 0.1],
              'max_depth': [3, 5, 7],
              'min_child_weight': [1, 3, 5],
              'subsample': [0.8, 0.9, 1.0],
              'colsample_bytree': [0.8, 0.9, 1.0]}

classifiers = [('LR', LinearRegression(), lr_params),
               ("SVR", SVR(), svr_params),
               ("CART", DecisionTreeRegressor(), dtree_params),
               ("RF", RandomForestRegressor(), rf_params),
               ("ENet", ElasticNet(), en_params),
               ("XGBRegressor", XGBRegressor(), xgb_params)
               ]

import pickle
def hyperparameter_optimization(X, y, cv=5, scoring=scoring, rfe_num=20):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print("before hp tuning:")
        print("Negative Mean Squared Error (MSE):", np.mean(cv_results['test_neg_mean_squared_error']))
        print("Negative Mean Absolute Error (MAE):", np.mean(cv_results['test_neg_mean_absolute_error']))
        print("R^2 Score:", np.mean(cv_results['test_r2']))

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        final_model.fit(X, y)
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print("after hp tuning:")
        print("Negative Mean Squared Error (MSE):", np.mean(cv_results['test_neg_mean_squared_error']))
        print("Negative Mean Absolute Error (MAE):", np.mean(cv_results['test_neg_mean_absolute_error']))
        print("R^2 Score:", np.mean(cv_results['test_r2']))
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
        # joblib.dump(final_model, f"{name}_model_080724.pkl")
        with open(f"nn{name}_model_080724.pkl","wb") as file:
            pickle.dump(final_model, file)

        print(f"results using RFE ({rfe_num} features):")
        rfe = RFE(estimator=final_model, n_features_to_select=rfe_num)
        pipeline = Pipeline(steps=[('s', rfe), ('m', final_model)])
        cv_rfe_res = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
        print("Negative Mean Squared Error (MSE):", np.mean(cv_rfe_res['test_neg_mean_squared_error']))
        print("Negative Mean Absolute Error (MAE):", np.mean(cv_rfe_res['test_neg_mean_absolute_error']))
        print("R^2 Score:", np.mean(cv_rfe_res['test_r2']))
        rfecv = RFECV(final_model, min_features_to_select=1, cv=cv)
    return best_models

##############################################################
def rfecv_plots(df, model):
    # https://medium.com/@hsu.lihsiang.esth/feature-selection-with-recursive-feature-elimination-rfe-for-parisian-bike-count-data-23f0ce9db691
    classifiers = [('LR', LinearRegression(), lr_params),
                   ("CART", DecisionTreeRegressor(), dtree_params),
                   ("RF", RandomForestRegressor(), rf_params),
                   ("ENet", ElasticNet(), en_params),
                   ("XGBRegressor", XGBRegressor(), xgb_params)
                   ]
    for name, classifier, params in classifiers:
        X, y = carettas_data_prep(df)
        estimator = classifier
        min_features_to_select = 1
        rfecv = RFECV(estimator, min_features_to_select=min_features_to_select, cv=5)
        rfecv.fit(X, y)
        feature_index = rfecv.get_support(indices=True)
        feature_mask = rfecv.support_
        feature_names = rfecv.get_feature_names_out()
        feature_number = rfecv.n_features_
        results = pd.DataFrame(rfecv.cv_results_)
        rfecv_score = rfecv.score(X, y)
        print(f"rfecv results for {name} model:")
        print('Original feature number:', len(X.columns))
        print('Optimal feature number:', feature_number)
        print('Selected features:', feature_names)
        print('Score:', rfecv_score)

        print(f"saving plot figure to directory for {name} model:")
        plt.figure(figsize=(10, 6))
        plt.plot(range(min_features_to_select, len(results['mean_test_score']) + min_features_to_select),
                 results['mean_test_score'])
        plt.title(f'{name} Model RFECV Scores')
        plt.xlabel('Number of features')
        plt.ylabel('Mean Test Score')
        plt.grid(True)
        #plt.show(block=True)
        plt.savefig(f'{name}_rfecv_visualization.png')
        print(f"plot of {name} model is saved:")

#rfecv_plots(df, classifiers)
#############################################################

def main():
    df = pd.read_csv(
        r"C:\Users\kilic\PycharmProjects\cerenPyProje\Impact-of-Dune-Plants-on-Sea-Turtles-Machine-Learning-App-main\ReddingRootsCaseStudy22_csv.csv")
    X, y = carettas_data_prep(df)
    print(df.columns)
    print(X.columns)
    print(X.shape)
    print(X.columns)
    print(y.shape)
    print(y.head())
    base_models(X, y, scoring=scoring)
    best_models = hyperparameter_optimization(X, y)

if __name__ == "__main__":
    print("İşlem başladı")
    main()

