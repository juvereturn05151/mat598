import data_frame_member

class Data_Science_Runner:
    def __init__(self):
        self.dataFrame_redmond = (data_frame_member.Data_Frame_Controller
                                  ("redfin_Redmond_98052.csv"))
        

    def run(self):
        self.dataFrame_redmond .print_first_few_rows()

