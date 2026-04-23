import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson
import joblib

class FootballMLPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model_1x2 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
        self.home_attack = {}
        self.away_attack = {}
        self.home_defense = {}
        self.away_defense = {}
        self.global_home_avg = 0
        self.global_away_avg = 0
        self.is_trained = False

    def _feature_engineering(self, df):
        df['goal_diff'] = df['home_goals'] - df['away_goals']
        df['result'] = np.where(df['goal_diff'] > 0, 0, np.where(df['goal_diff'] == 0, 1, 2))
        return df

    def train(self, df):
        df = self._feature_engineering(df)
        features = ['home_xg', 'away_xg', 'home_form', 'away_form', 'h2h_home_wins', 'h2h_away_wins']
        X = df[features]
        y = df['result']

        X_scaled = self.scaler.fit_transform(X)
        self.model_1x2.fit(X_scaled, y)

        self.global_home_avg = df['home_goals'].mean()
        self.global_away_avg = df['away_goals'].mean()

        for team in df['home_team'].unique():
            team_home_matches = df[df['home_team'] == team]
            self.home_attack[team] = team_home_matches['home_goals'].mean() / self.global_home_avg if self.global_home_avg > 0 else 1
            self.home_defense[team] = team_home_matches['away_goals'].mean() / self.global_away_avg if self.global_away_avg > 0 else 1

        for team in df['away_team'].unique():
            team_away_matches = df[df['away_team'] == team]
            self.away_attack[team] = team_away_matches['away_goals'].mean() / self.global_away_avg if self.global_away_avg > 0 else 1
            self.away_defense[team] = team_away_matches['home_goals'].mean() / self.global_home_avg if self.global_home_avg > 0 else 1

        self.is_trained = True
        joblib.dump(self.model_1x2, 'model_1x2.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')

        return {"accuracy": float(self.model_1x2.score(X_scaled, y))}

    def predict(self, match_data):
        if not self.is_trained:
            try:
                self.model_1x2 = joblib.load('model_1x2.joblib')
                self.scaler = joblib.load('scaler.joblib')
                self.is_trained = True
            except:
                raise Exception("MODEL_NOT_TRAINED")

        features = np.array([[
            match_data['home_xg'],
            match_data['away_xg'],
            match_data['home_form'],
            match_data['away_form'],
            match_data['h2h_home_wins'],
            match_data['h2h_away_wins']
        ]])

        X_scaled = self.scaler.transform(features)
        probs_1x2 = self.model_1x2.predict_proba(X_scaled)[0]

        home_team = match_data['home_team']
        away_team = match_data['away_team']

        home_exp_goals = self.home_attack.get(home_team, 1) * self.away_defense.get(away_team, 1) * self.global_home_avg
        away_exp_goals = self.away_attack.get(away_team, 1) * self.home_defense.get(home_team, 1) * self.global_away_avg

        prob_under_25 = 0
        prob_over_25 = 0

        for i in range(6):
            for j in range(6):
                prob = poisson.pmf(i, home_exp_goals) * poisson.pmf(j, away_exp_goals)
                if i + j < 2.5:
                    prob_under_25 += prob
                else:
                    prob_over_25 += prob

        return {
            "prob_home": float(probs_1x2[0]),
            "prob_draw": float(probs_1x2[1]),
            "prob_away": float(probs_1x2[2]),
            "prob_over_25": float(prob_over_25),
            "prob_under_25": float(prob_under_25)
        }

    def backtest(self, df):
        df = self._feature_engineering(df)
        initial_bankroll = 1000.0
        current_bankroll = initial_bankroll
        stake = 10.0
        wins = 0
        total_bets = 0
        max_bankroll = initial_bankroll
        max_drawdown = 0.0

        for _, row in df.iterrows():
            match_data = row.to_dict()
            try:
                preds = self.predict(match_data)
                implied_home = 1 / row['odd_home']
                if preds['prob_home'] - implied_home > 0.05:
                    total_bets += 1
                    if row['result'] == 0:
                        current_bankroll += stake * (row['odd_home'] - 1)
                        wins += 1
                    else:
                        current_bankroll -= stake
                    
                    if current_bankroll > max_bankroll:
                        max_bankroll = current_bankroll
                    drawdown = (max_bankroll - current_bankroll) / max_bankroll
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
            except:
                continue

        roi = ((current_bankroll - initial_bankroll) / initial_bankroll) * 100
        hit_rate = (wins / total_bets) * 100 if total_bets > 0 else 0

        return {
            "roi_percentage": float(roi),
            "max_drawdown_percentage": float(max_drawdown * 100),
            "hit_rate_percentage": float(hit_rate),
            "total_bets": int(total_bets),
            "final_bankroll": float(current_bankroll)
        }