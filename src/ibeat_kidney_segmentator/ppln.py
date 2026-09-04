import ibeat_kidney_segmentator as ppln
from ibeat_kidney_segmentator.utils import pipe
import os

PIPELINE = 'kidney_segmentator'

def run(build, pp_input_dir, pp_output_dir, test_i, test_o, fold, model_for_inf=None):
    
    ppln.stage_1_build_canvas.run(build)
    ppln.stage_3_display.run(build)
    ppln.stage_4_data_split.run()
    ppln.stage_5_data_prep.run()
    ppln.stage_6_preprocessing.run()
    ppln.stage_7_train.run()
    ppln.stage_8_postprocessing.run(VALIDATION, pp_input_dir, pp_output_dir)
    ppln.stage_9_test_model.run(test_i, test_o, fold, model_for_inf)

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