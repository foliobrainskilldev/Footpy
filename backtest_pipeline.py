import pandas as pd

class BacktestEngine:
    def __init__(self, initial_bankroll=1000.0, stake_percent=0.02, edge_threshold=0.05):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.stake_percent = stake_percent
        self.edge_threshold = edge_threshold
        self.peak_bankroll = initial_bankroll
        self.max_drawdown = 0.0
        self.wins = 0
        self.total_bets = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0

    def calculate_stake(self) -> float:
        return self.bankroll * self.stake_percent

    def register_bet(self, is_won: bool, odds: float, stake: float):
        self.total_bets += 1
        if is_won:
            profit = stake * (odds - 1)
            self.bankroll += profit
            self.gross_profit += profit
            self.wins += 1
        else:
            self.bankroll -= stake
            self.gross_loss += stake

        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll

        current_drawdown = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

    def run(self, predictions_df: pd.DataFrame) -> dict:
        for idx, row in predictions_df.iterrows():
            if self.bankroll <= 0:
                break

            stake = self.calculate_stake()
            
            implied_home = 1 / row['odd_home']
            if row['prob_home'] - implied_home > self.edge_threshold:
                is_won = (row['actual_result'] == 0)
                self.register_bet(is_won, row['odd_home'], stake)
                continue

            implied_away = 1 / row['odd_away']
            if row['prob_away'] - implied_away > self.edge_threshold:
                is_won = (row['actual_result'] == 2)
                self.register_bet(is_won, row['odd_away'], stake)
                continue

            implied_draw = 1 / row['odd_draw']
            if row['prob_draw'] - implied_draw > self.edge_threshold:
                is_won = (row['actual_result'] == 1)
                self.register_bet(is_won, row['odd_draw'], stake)

        roi = ((self.bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        hit_rate = (self.wins / self.total_bets) * 100 if self.total_bets > 0 else 0
        profit_factor = (self.gross_profit / self.gross_loss) if self.gross_loss > 0 else float('inf')

        return {
            "initial_bankroll": float(self.initial_bankroll),
            "final_bankroll": float(self.bankroll),
            "total_bets": int(self.total_bets),
            "wins": int(self.wins),
            "hit_rate_pct": float(hit_rate),
            "roi_pct": float(roi),
            "max_drawdown_pct": float(self.max_drawdown * 100),
            "profit_factor": float(profit_factor)
        }