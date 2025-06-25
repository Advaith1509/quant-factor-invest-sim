# factor_model_trainer.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

class FactorModelTrainer:

    def __init__(self, data_dir=".", model_dir="models", factor_files=None, composite_file="/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/composite_scores.csv"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.factor_files = factor_files if factor_files is not None else {
            "value": "/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/value_scores.csv",
            "momentum": "/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/momentum_scores.csv",
            "quality": "/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/quality_scores.csv",
            "volatility": "/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/volatility_scores.csv",
            "volume": "/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/volume_scores.csv"
        }
        self.composite_file = composite_file
        self.df = None
        self.model = None
        self.best_model = None

    def _load_factors(self):
        dfs = []
        for factor, fname in self.factor_files.items():
            fpath = os.path.join(self.data_dir, fname)
            try:
                df = pd.read_csv(fpath)
                if df.shape[1] > 2:
                    df = df.rename(columns={df.columns[-1]: factor})
                else:
                    print(f"Warning: {fname} does not have expected score column structure.")
                    continue
                dfs.append(df)
            except FileNotFoundError:
                print(f"Error: Factor file not found at {fpath}")
                return None
            except Exception as e:
                print(f"An error occurred while reading {fpath}: {e}")
                return None
        return dfs

    def _merge_factors(self, factor_dfs):
        if not factor_dfs:
            print("Warning: No factor dataframes to merge.")
            return pd.DataFrame()
        merged = factor_dfs[0]
        for df in factor_dfs[1:]:
            merged = pd.merge(merged, df, on=["symbol", "date"], how="inner")
        return merged

    def _load_labels(self):
        fpath = os.path.join(self.data_dir, self.composite_file)
        try:
            return pd.read_csv(fpath)
        except FileNotFoundError:
            print(f"Error: Labels file not found at {fpath}")
            return None
        except Exception as e:
            print(f"An error occurred while reading {fpath}: {e}")
            return None

    def prepare_dataset(self):
        print("Preparing dataset...")
        factor_dfs = self._load_factors()
        if factor_dfs is None:
            self.df = pd.DataFrame()
            return self.df
        features = self._merge_factors(factor_dfs)

        labels = self._load_labels()
        if labels is None:
            self.df = pd.DataFrame()
            return self.df

        if features.empty or labels.empty:
            print("Warning: Features or labels DataFrame is empty after loading/merging.")
            self.df = pd.DataFrame()
            return self.df

        df = pd.merge(features, labels, on=["symbol", "date"], how="inner")
        df = df.dropna()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'composite_score' in numeric_cols:
            numeric_cols.remove('composite_score')

        large_value_threshold = 1e6
        decimal_places_to_keep = 2
        for col in numeric_cols:
            large_mask = df[col].abs() > large_value_threshold
            if large_mask.any():
                df.loc[large_mask, col] = df.loc[large_mask, col].round(decimal_places_to_keep)
                print(f"Reduced decimal places for large values in '{col}'.")

        self.df = df.copy()
        print("Dataset prepared.")
        return self.df

    def train_model(self):
        if self.df is None or self.df.empty:
            print("Error: No data available for training. Run prepare_dataset first.")
            return

        print("Training model...")
        X = self.df[["value", "momentum", "quality", "volatility", "volume"]]
        y = self.df["composite_score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        self.model = model

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test R^2 Score: {r2:.4f}")

        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "rf_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Store for later use
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred

    def train_model_cv(self, cv=5):
        if self.df is None or self.df.empty:
            print("Error: No data available for cross-validation. Run prepare_dataset first.")
            return
        if self.model is None:
            print("Warning: Model not trained yet. Training a default model for CV.")
            model_for_cv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model_for_cv = self.model

        print(f"Performing {cv}-fold cross-validation...")
        X = self.df[["value", "momentum", "quality", "volatility", "volume"]]
        y = self.df["composite_score"]

        cv_scores = cross_val_score(model_for_cv, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse_scores = np.sqrt(-cv_scores)

        mean_rmse = np.mean(cv_rmse_scores)
        std_rmse = np.std(cv_rmse_scores)

        print(f"Cross-Validation RMSE (Mean): {mean_rmse:.4f}")
        print(f"Cross-Validation RMSE (Std Dev): {std_rmse:.4f}")

    def tune_and_train_model(self, param_grid, cv=5):
        if self.df is None or self.df.empty:
            print("Error: No data available for tuning. Run prepare_dataset first.")
            return

        print("Performing hyperparameter tuning with GridSearchCV...")
        X = self.df[["value", "momentum", "quality", "volatility", "volume"]]
        y = self.df["composite_score"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_for_tuning = RandomForestRegressor(random_state=42)

        grid_search = GridSearchCV(estimator=model_for_tuning, param_grid=param_grid,
                                   cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)

        import time
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        tuning_time = end_time - start_time
        print(f"GridSearchCV completed in {tuning_time:.4f} seconds")

        print("\nBest hyperparameters found:")
        print(grid_search.best_params_)

        self.best_model = grid_search.best_estimator_
        print("\nTraining final model with best hyperparameters...")
        start_time = time.time()
        self.best_model.fit(X_train, y_train)
        end_time = time.time()
        final_training_time = end_time - start_time
        print(f"Final model training completed in {final_training_time:.4f} seconds")

        y_pred = self.best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("\nEvaluation on Test Set (Best Model):")
        print(f"RMSE: {rmse:.4f}")
        print(f"R^2 Score: {r2:.4f}")

        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "rf_model_tuned.pkl")
        joblib.dump(self.best_model, model_path)
        print(f"Tuned Model saved to {model_path}")

        # Store for later use
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred

    def plot_actual_vs_predicted(self):
        if not hasattr(self, 'y_test') or not hasattr(self, 'y_pred'):
            print("Error: No predictions available. Run train_model or tune_and_train_model first.")
            return
        plt.figure(figsize=(7, 7))
        plt.scatter(self.y_test, self.y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual Composite Score")
        plt.ylabel("Predicted Composite Score")
        plt.title("Actual vs. Predicted Composite Scores")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        model = self.best_model if self.best_model is not None else self.model
        if model is None:
            print("Error: No trained model available.")
            return
        features = ["value", "momentum", "quality", "volatility", "volume"]
        importances = model.feature_importances_
        plt.figure(figsize=(8, 5))
        plt.bar(features, importances)
        plt.title("Feature Importances")
        plt.ylabel("Importance")
        plt.xlabel("Factor")
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize with your data directory and files if needed
    trainer = FactorModelTrainer(
        data_dir=".",  # Change to your data directory
        model_dir="models",
        factor_files={
            "value": "/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/value_scores.csv",
            "momentum": "/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/momentum_scores.csv",
            "quality": "/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/quality_scores.csv",
            "volatility": "/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/volatility_scores.csv",
            "volume": "/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/volume_scores.csv"
        },
        composite_file="/Users/advaith/Desktop/Projects/quant_factor_simulator/data/processed_not_normalized/factor_scores/composite_scores.csv"
    )

    trainer.prepare_dataset()
    trainer.train_model()
    trainer.train_model_cv(cv=5)
    trainer.plot_actual_vs_predicted()
    trainer.plot_feature_importance()
