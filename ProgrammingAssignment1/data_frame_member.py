import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import custom_linear_regression_model
import numpy as np
class Data_Frame_Controller:
    def __init__(self, file_to_read, name):
        self.name = name
        self.dataFrame = pd.read_csv(file_to_read)
        # Select only columns we need
        self.dataFrame = self.dataFrame[["ZIP OR POSTAL CODE", "PRICE", "SQUARE FEET", "LOT SIZE", "YEAR BUILT"]]
        # Drop rows with missing data
        self.dataFrame = self.dataFrame.dropna()

    def print_first_few_rows(self):
        print(self.dataFrame.head())

    def show_home_price_histogram(self):
        plt.hist(self.dataFrame["PRICE"], bins=100)
        plt.title("Distribution of Home Prices")
        plt.xlabel("Home Price (Per 1,000,000 USD)")
        plt.ylabel("Number of Homes")
        plt.xlim(0, 7000000)
        plt.ylim(0, 20)
        plt.show()

    def show_square_footage_histogram(self):
        plt.hist(self.dataFrame["SQUARE FEET"], bins=100)
        plt.title("Distribution of Square Footage")
        plt.xlabel("Square Footage")
        plt.ylabel("Number of Homes")
        plt.xlim(0, 5000)
        plt.ylim(0, 30)
        plt.show()

    def calculate_mean_median_std_deviation(self):
        numeric_cols = ["PRICE", "SQUARE FEET"]

        stats = {}
        for col in numeric_cols:
            stats[col] = {
                "mean": self.dataFrame[col].mean(),
                "median": self.dataFrame[col].median(),
                "std_dev": self.dataFrame[col].std()
            }

        for col, values in stats.items():
            print(f"\n--- {col} ---")
            print(f"Mean   : {values['mean']:.2f}")
            print(f"Median : {values['median']:.2f}")
            print(f"Std Dev: {values['std_dev']:.2f}")

        return stats

    def visualize_scatter_plot(self):
        plt.scatter(self.dataFrame["PRICE"], self.dataFrame["SQUARE FEET"])
        plt.title("Price vs Square Footage")
        plt.xlabel("PRICE")
        plt.ylabel("SQUARE FEET")
        plt.xlim(0, 7000000)
        plt.ylim(0, 7000)
        plt.show()

    def plot_regression_line(self, beta):
        # 1️⃣ Extract variables
        X = self.dataFrame[["SQUARE FEET", "LOT SIZE", "YEAR BUILT"]].values
        y = self.dataFrame["PRICE"].values

        # 2️⃣ Fix other variables at their mean
        avg_lot_size = np.mean(self.dataFrame["LOT SIZE"])
        avg_year_built = np.mean(self.dataFrame["YEAR BUILT"])

        # 3️⃣ Create a range of square footage values
        sqft_range = np.linspace(0, 7000, 100)

        # 4️⃣ Build input matrix with average values
        X_plot = np.c_[np.ones((100, 1)),  # intercept
        sqft_range,  # variable on x-axis
        np.full(100, avg_lot_size),
        np.full(100, avg_year_built)]

        # 5️⃣ Compute predicted prices
        y_pred = X_plot @ beta

        # 6️⃣ Plot actual scatter + regression line
        plt.scatter(self.dataFrame["SQUARE FEET"], y, alpha=0.5, label="Actual Data")
        plt.plot(sqft_range, y_pred, color="red", linewidth=2, label="Regression Line")
        plt.title("Price vs Square Footage (with Regression Line)")
        plt.xlabel("Square Footage")
        plt.ylabel("Home Price (USD)")
        plt.legend()
        plt.show()

    def perform_linear_regression(self):
        x = self.dataFrame[["SQUARE FEET", "LOT SIZE", "YEAR BUILT"]]
        y = self.dataFrame["PRICE"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        linear_regression = custom_linear_regression_model.CustomLinearRegression( x_train, x_test, y_train, y_test);

        linear_regression.train_linear_regression()

        y_pred =  linear_regression.predict()

        print("Beta coefficients:", linear_regression.beta)
        mean_price = self.dataFrame["PRICE"].mean()
        print("Mean Home Price:", mean_price)
        print("RMSE:", linear_regression.getRMSE(y_pred))
        print("RMSE as % of Mean Price:", (linear_regression.getRMSE(y_pred) / mean_price) * 100, "%")

        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Predicted vs Actual Prices")
        plt.plot([0, max(y_test)], [0, max(y_test)], color='red')  # y=x line
        plt.show()

        self.plot_regression_line(linear_regression.beta)