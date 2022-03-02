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


def search_n_similarities(cosine_matrix, target_id, df_index, n):
    target_index = np.where(df_index == target_id)[0][0]

    similarities = (
        pd.Series(cosine_matrix[target_index], index=df_index)
        .drop(target_id, axis=0)
    ).sort_values(ascending=False)

    n_similarities = similarities[similarities.values == 1.0]
    if len(n_similarities) < n:
        n_similarities = similarities[:n]

    return n_similarities


def predict_target_scores(zero_ratings, mean_scores, n_similarities, target):
    scores_dict = {target: [], 'prediction': []}
    for target_id in zero_ratings.columns:
        target_idx = np.where(zero_ratings.loc[:, target_id].values != 0.0)[0]

        scores_dict[target].append(target_id)
        if len(target_idx) == 0:
            scores_dict['prediction'].append(mean_scores.loc[target_id])
        else:
            zero_ratings_val = [zero_ratings.loc[:, target_id].values[idx] for idx in target_idx]
            n_similarities_val = [n_similarities.values[idx] for idx in target_idx]

            score = np.sum(zero_ratings_val) / np.sum(n_similarities_val)
            scores_dict['prediction'].append(score)

    return scores_dict