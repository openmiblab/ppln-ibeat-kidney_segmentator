import ibeat_kidney_segmentator as ppln
from ibeat_kidney_segmentator.utils import pipe
import os

PIPELINE = 'kidney_segmentator'

def run(build, pp_input_dir, pp_output_dir, test_i, test_o, fold, model_for_inf=None):
    
    ppln.stage_1_build_canvas.run(build)
    ppln.stage_5_display.run(build)
    ppln.stage_7_data_split.run()
    ppln.stage_6_data_prep.run()
    ppln.stage_8_preprocessing.run()
    ppln.stage_9_train.run()
    ppln.stage_10_postprocessing.run(VALIDATION, pp_input_dir, pp_output_dir)
    ppln.stage_11_test_model.run(test_i, test_o, fold, model_for_inf)

    #optional 
    #ppln.stage_2_view_and_edit.run()

if __name__=='__main__':

    BUILD = os.path.join(os.getcwd(), 'build')
    VALIDATION = []
    pp_input_dir = []
    pp_output_dir = []
    test_i = []
    test_o = [] 
    fold = 0
    model_for_inf='checkpoint_best.pth'

    pipe.run_script(run, 
                    BUILD, 
                    PIPELINE, 
                    pp_input_dir,
                    pp_output_dir,
                    test_i,
                    test_o,
                    fold,
                    model_for_inf)