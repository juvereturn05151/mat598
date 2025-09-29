import data_frame_member
from scipy import stats
from statsmodels.stats.weightstats import ztest

class DataScienceRunner:
    def __init__(self):
        self.dataFrame_redmond = (data_frame_member.DataFrameController
                                  ("redfin_Redmond_98052.csv", "Redmond"))
        self.dataFrame_bothell = (data_frame_member.DataFrameController
                                  ("redfin_Bothell_98012.csv", "Bothell"))
        self.dataFrame_woodinville = (data_frame_member.DataFrameController
                                  ("redfin_Woodinville_98072.csv", "Woodinville"))

    def print_first_few_rows(self):
        self.dataFrame_redmond.print_first_few_rows()
        self.dataFrame_bothell.print_first_few_rows()
        self.dataFrame_woodinville.print_first_few_rows()

    def show_home_price_histogram(self):
        self.dataFrame_redmond.show_home_price_histogram()
        self.dataFrame_bothell.show_home_price_histogram()
        self.dataFrame_woodinville.show_home_price_histogram()

    def show_square_footage_histogram(self):
        self.dataFrame_redmond.show_square_footage_histogram()
        self.dataFrame_bothell.show_square_footage_histogram()
        self.dataFrame_woodinville.show_square_footage_histogram()

    def print_calculate_mean_median_std_deviation(self):
        self.dataFrame_redmond.calculate_mean_median_std_deviation()
        self.dataFrame_bothell.calculate_mean_median_std_deviation()
        self.dataFrame_woodinville.calculate_mean_median_std_deviation()

    def perform_hypothesis_test(self):
        print("Let's perform a hypothesis test")
        print("H0: prices are the same")
        print("HA: prices are significantly different")

        z_stat, p_value = ztest(self.dataFrame_redmond.dataFrame["PRICE"], self.dataFrame_bothell.dataFrame["PRICE"], alternative='two-sided')
        alpha = 0.05
        z_critical = stats.norm.ppf(1 - alpha / 2)

        print(f"Z-statistic: ", z_stat)
        print(f"Z-critical: ", z_critical)
        print(f"P-value: ", p_value)

        print(f"\nSince |{z_stat}| > {z_critical}, we reject the null hypothesis.")
        print(f"\nSince {p_value} is low, we reject the null hypothesis.")

    def visualize_scatter_plot(self):
        self.dataFrame_redmond.visualize_scatter_plot()
        self.dataFrame_bothell.visualize_scatter_plot()
        self.dataFrame_woodinville.visualize_scatter_plot()

    def perform_regression(self):
        self.dataFrame_redmond.perform_linear_regression()
        self.dataFrame_bothell.perform_linear_regression()
        self.dataFrame_woodinville.perform_linear_regression()

    def run(self):
        self.print_first_few_rows()
        self.show_home_price_histogram()
        self.show_square_footage_histogram()
        self.print_calculate_mean_median_std_deviation()
        self.perform_hypothesis_test()
        self.visualize_scatter_plot()
        self.perform_regression()