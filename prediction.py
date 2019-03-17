from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from team_maker import prep_dataframe

def predict_batter_scores():
    """
    Takes in current information and predicts batter scores
    """

    model = joblib.load('rf_batter_model.pkl')

def get_batter_prediction_data(df):
    """
    Scrapes web for relevant data to feed into model.
    """
    df = prep_dataframe()
    pstats = pd.read_csv('./pitcher_season_stats.csv')
