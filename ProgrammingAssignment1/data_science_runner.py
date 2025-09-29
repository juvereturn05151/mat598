import data_frame_member
from scipy import stats
from statsmodels.stats.weightstats import ztest
class Data_Science_Runner:
    def __init__(self):
        self.dataFrame_redmond = (data_frame_member.Data_Frame_Controller
                                  ("redfin_Redmond_98052.csv", "Redmond"))
        self.dataFrame_bothell = (data_frame_member.Data_Frame_Controller
                                  ("redfin_Bothell_98012.csv", "Bothell"))
        self.dataFrame_woodinville = (data_frame_member.Data_Frame_Controller
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

    def run(self):
        self.print_first_few_rows()
        self.show_home_price_histogram()
        self.show_square_footage_histogram()
        self.print_calculate_mean_median_std_deviation()
        self.perform_hypothesis_test()
