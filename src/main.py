
import os
import pipelines


DATAPATH = "C:\\Users\\md1spsx\\Documents\\Data\\TRISTAN_TWO_COMPOUND"
RESULTSPATH = "C:\\Users\\md1spsx\\Documents\\Results\\TRISTAN_2_compounds_image_analysis"


if __name__ == '__main__':

    pipelines.onescandev(os.path.join(DATAPATH, 'MEDCIC_02_Visit1Scan1'), RESULTSPATH)
    #pipelines.onescan(DATAPATH, RESULTSPATH, 2,1,1)