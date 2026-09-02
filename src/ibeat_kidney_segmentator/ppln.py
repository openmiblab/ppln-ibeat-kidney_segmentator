import ibeat_kidney_segmentator as ppln
from ibeat_kidney_segmentator.utils import pipe

PIPELINE = 'kidney_segmentator'

def run(build):
    
    ppln.stage_1_hello_world.run(build)
    ppln.stage_2_hello_world_back.run(build)


if __name__=='__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_script(run, BUILD, PIPELINE)