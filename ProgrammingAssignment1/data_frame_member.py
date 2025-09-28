import pandas as pd

class Data_Frame_Controller:
    def __init__(self, file_to_read):
        self.dataFrame = pd.read_csv(file_to_read)
        # Select only columns we need
        self.dataFrame = self.dataFrame[["ZIP OR POSTAL CODE", "PRICE", "SQUARE FEET", "LOT SIZE", "YEAR BUILT"]]
        # Drop rows with missing data
        self.dataFrame = self.dataFrame.dropna()

    def print_first_few_rows(self):
        print(self.dataFrame.head())