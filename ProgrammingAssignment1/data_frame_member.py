import pandas as pd
import matplotlib.pyplot as plt

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