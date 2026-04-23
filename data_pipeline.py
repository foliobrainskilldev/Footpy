import pandas as pd
import numpy as np

class FootballDataPipeline:
    def __init__(self, span=5):
        self.span = span

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['home_team', 'away_team', 'home_goals', 'away_goals', 'date'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    def calculate_temporal_form(self, df: pd.DataFrame) -> pd.DataFrame:
        teams = pd.concat([df['home_team'], df['away_team']]).unique()
        form_data = {team: {'goals_scored': [], 'goals_conceded': [], 'points': []} for team in teams}
        
        home_form = []
        away_form = []
        
        for idx, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            if len(form_data[home]['points']) == 0:
                home_form.append(1.0)
            else:
                ewma_home = pd.Series(form_data[home]['points']).ewm(span=self.span).mean().iloc[-1]
                home_form.append(ewma_home)
                
            if len(form_data[away]['points']) == 0:
                away_form.append(1.0)
            else:
                ewma_away = pd.Series(form_data[away]['points']).ewm(span=self.span).mean().iloc[-1]
                away_form.append(ewma_away)
            
            home_goals = row['home_goals']
            away_goals = row['away_goals']
            
            if home_goals > away_goals:
                home_pts, away_pts = 3, 0
            elif home_goals == away_goals:
                home_pts, away_pts = 1, 1
            else:
                home_pts, away_pts = 0, 3
                
            form_data[home]['points'].append(home_pts)
            form_data[home]['goals_scored'].append(home_goals)
            form_data[home]['goals_conceded'].append(away_goals)
            
            form_data[away]['points'].append(away_pts)
            form_data[away]['goals_scored'].append(away_goals)
            form_data[away]['goals_conceded'].append(home_goals)

        df['home_form'] = home_form
        df['away_form'] = away_form
        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.clean_data(df)
        df = self.calculate_temporal_form(df)
        
        df['goal_diff'] = df['home_goals'] - df['away_goals']
        df['result'] = np.where(df['goal_diff'] > 0, 0, np.where(df['goal_diff'] == 0, 1, 2))
        
        df['total_goals'] = df['home_goals'] + df['away_goals']
        df['over_25'] = np.where(df['total_goals'] > 2.5, 1, 0)
        
        return df