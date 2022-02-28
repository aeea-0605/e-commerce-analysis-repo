import numpy as np
import pandas as pd


def filtering_df(df, movie_cnt, user_cnt):
    filter_movies = df.movieId.value_counts() > movie_cnt
    filter_movies = filter_movies[filter_movies].index.tolist()
    
    filter_users = df.userId.value_counts() > user_cnt
    filter_users = filter_users[filter_users].index.tolist()
    
    filtered_df = df[(df['userId'].isin(filter_users)) & (df['movieId'].isin(filter_movies))]
    
    print(f"필터링 전 데이터 수 :{len(df)}")
    print(f"필터링 후 데이터 수 :{len(filtered_df)}")
    
    return filtered_df


def eval_model(result_df, target, return_score=False):
    from sklearn.metrics import mean_squared_error
    
    y_true = result_df.rating.values
    y_pred = result_df.prediction.values
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"RMSE of recommendation by {target}-{target} collaborative filtering:", rmse)
    if return_score:
        return rmse


def trans_interaction_matrix(df, impute=None):
    pivot_df = pd.pivot_table(df, index='userId', columns='movieId', values='rating')

    print(f"Return {pivot_df.index.name}-{pivot_df.columns.name} Matrix")

    if impute == "zero":
        return pivot_df.fillna(0)
    elif impute == "mean":
        return pivot_df.T.fillna(pivot_df.mean(axis=1)).T
    else:
        return pivot_df