# examination_timetabling.py

# Import necessary libraries
import pandas as pd
import catboost as cb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import PartialDependenceDisplay
import shap

class ExaminationTimetabling:
    """Class for predicting student enrollment for an exam within a given examination term."""
    
    def __init__(self, file_name):
        self.dataset = self.load_data(file_name)
        self.remove_feature('LendaKodi')
        self.convert_columns_to_numeric()
        self.train_data, self.validation_data = self.split_data()
        self.model = self.train_model()
        self.evaluate_model()
        self.visualize_feature_correlations(self.dataset)
        self.show_permutation_importance(self.model, self.validation_data)
        self.show_partial_dependence(self.model, self.validation_data, 'Programi')
        self.show_2d_partial_dependence(self.model, self.validation_data, ['Niveli', 'Afati'])
        self.explain_with_shap(self.model, self.validation_data)
        # self.print_main_features()

    def load_data(self, file_name):
        """Loads dataset from the specified CSV file."""
        return pd.read_csv(f"{file_name}.csv", encoding="ISO-8859-1")

    def remove_feature(self, feature_name):
        """Removes the specified feature from the dataset."""
        if feature_name in self.dataset.columns:
            self.dataset.drop(columns=[feature_name], inplace=True)

    def convert_columns_to_numeric(self):
        """Converts categorical columns to numeric using Label Encoding."""
        columns_to_convert = ['Afati', 'VitiAkademik', 'Niveli', 'Semestri', 'Programi', 'Lenda', 'Profesori']
        for column in columns_to_convert:
            encoder = LabelEncoder()
            self.dataset[column] = encoder.fit_transform(self.dataset[column])

    def split_data(self):
        """Splits the dataset into training and validation sets (80/20 ratio)."""
        return train_test_split(self.dataset, test_size=0.2, random_state=42)

    def train_model(self):
        """Trains a CatBoostRegressor model to predict 'NrFleteparaqitjeve'."""
        X_train = self.train_data.drop(columns=['NrFleteparaqitjeve'])
        y_train = self.train_data['NrFleteparaqitjeve']
        model = cb.CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=8, verbose=0)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self):
        """Evaluates the trained model on the validation dataset."""
        X_val = self.validation_data.drop(columns=['NrFleteparaqitjeve'])
        y_val = self.validation_data['NrFleteparaqitjeve']
        predictions = self.model.predict(X_val)
        print(f"MSE: {mean_squared_error(y_val, predictions)}")
        print(f"R^2 Score: {r2_score(y_val, predictions)}")

    def visualize_feature_correlations(self, dataset):
        """Displays a heatmap of feature correlations."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.show()

    def show_permutation_importance(self, model, validation_set):
        """Computes and displays permutation feature importance."""
        X_val = validation_set.drop(columns=['NrFleteparaqitjeve'])
        y_val = validation_set['NrFleteparaqitjeve']
        perm = PermutationImportance(model, random_state=42).fit(X_val, y_val)
        print(eli5.format_as_text(eli5.explain_weights(perm, feature_names=X_val.columns.tolist())))

    def show_partial_dependence(self, model, validation_set, feature):
        """Plots a partial dependence plot for a given feature."""
        X_val = validation_set.drop(columns=['NrFleteparaqitjeve'])
        PartialDependenceDisplay.from_estimator(model, X_val, [feature])
        plt.title(f'Partial Dependence of {feature}')
        plt.show()

    def show_2d_partial_dependence(self, model, validation_set, features):
        """Plots a 2D partial dependence plot for two features."""
        X_val = validation_set.drop(columns=['NrFleteparaqitjeve'])
        fig, ax = plt.subplots(figsize=(8, 6))
        PartialDependenceDisplay.from_estimator(model, X_val, [tuple(features)], ax=ax)
        plt.show()

    def explain_with_shap(self, model, validation_set):
        """Computes and visualizes SHAP values for model interpretation."""
        X_val = validation_set.drop(columns=['NrFleteparaqitjeve'])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        shap.initjs()
        shap.summary_plot(shap_values, X_val)

    def print_main_features(self):
        """Displays dataset information and statistics."""
        print("Dataset Head:")
        print(self.dataset.head())
        print("\nDataset Info:")
        print(self.dataset.info())
        print("\nDataset Description:")
        print(self.dataset.describe())

if __name__ == "__main__":
    file_name = "dataset_examination_timetabling"
    ExaminationTimetabling(file_name)
