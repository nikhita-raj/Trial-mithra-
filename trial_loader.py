import pandas as pd
import os

DATA_PATH = "data/ctg-studies.csv"


class TrialRepository:

    def __init__(self):

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Dataset not found at {DATA_PATH}"
            )

        self.df = pd.read_csv(DATA_PATH)
        self.df.columns = [c.strip() for c in self.df.columns]

    def get_all_trials(self):
        return self.df.copy()

    def get_recruiting_trials(self):

        if "Study Status" in self.df.columns:
            return self.df[
                self.df["Study Status"].str.contains(
                    "RECRUITING", case=False, na=False
                )
            ]

        return self.df.copy()