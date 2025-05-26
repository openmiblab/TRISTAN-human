import time
import traceback
from compute import (
    io,
    log, 
    segment, 
    csv, 
    motioncorrect, 
    plotting, 
    mapping, 
    analysis,
    helper,
)

## SETUP LOGGER

def setup_logger(subject, visit, scan):
    try:
        logger = log.create_logger(subject, visit, scan)
        logger.info('Logger setup successfully')
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        error_message = traceback.format_exc()
        print(error_message)
        logger = None

    return logger

## DETAILS SETUP

def setup_detail_dict(datapath, resultspath, subject, visit, scan, logger):
    try:
        info = helper.details_dict(datapath, resultspath, subject, visit, scan, logger)
        if logger:
            logger.info('Details dictionary setup successfully')
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        error_message = traceback.format_exc()
        if logger:
            logger.error('Error setting up details dictionary')
            logger.error(error_message)
        info = None

    return info


## DATA EXTRACTION

def extract_arrays(info):
    start_time = time.time()
    try:
        io.arrays_from_dicom(info)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Data extraction completed in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        error_message = traceback.format_exc()
        if info['logger']:
            info['logger'].error('Error extracting arrays from dicom')
            info['logger'].error(error_message)
        pass

    return

## SEGMENTATION

def segment_ROI(info, seg_type):
    start_time = time.time()
    try:
        segment.setup_segmentation(info, seg_type)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('ROI segmentation completed in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        error_message = traceback.format_exc()
        if info['logger']:
            info['logger'].error('Error segmenting ROI')
            info['logger'].error(error_message)
        pass

    return

## MOTION CORRECTION

def correct_motion(info, fit_interval=None):
    start_time = time.time()
    try:
        motioncorrect.setup_mdreg(info, fit_interval)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Completed Motion Correction via mdreg in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occured:")
        traceback.print_exc()
        error_message = traceback.format_exc()
        if info['logger']:
            info['logger'].error('Error completing motion correction')
            info['logger'].error(error_message)
        pass
    return
    
def t1mapping_molli(info, scan_type):
    start_time = time.time()
    try:
        mapping.map_molli(info, scan_type)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Completed Motion Correction via mdreg in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occured:")
        traceback.print_exc()
        error_message = traceback.format_exc()
        if info['logger']:
            info['logger'].error('Error completing motion correction')
            info['logger'].error(error_message)
    
## REFORMAT TO CSV

def format_to_csv(info):
    start_time = time.time()
    try:
        csv.create_csv(info)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Completed csv file creation via pds in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occured:")
        traceback.print_exc()
        error_message = traceback.format_exc()
        if info['logger']:
            info['logger'].error('Error completing data reformatting to CSV')
            info['logger'].error(error_message)
        pass
    return

## SHOW ROIS

def show_ROIS(info, checkpoint):
    start_time = time.time()
    try:
        plotting.overlay_masks(info, checkpoint)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Completed showing ROIS in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occured:")
        traceback.print_exc()
        error_message = traceback.format_exc()
        if info['logger']:
            info['logger'].error('Error showing ROIS')
            info['logger'].error(error_message)
        pass
    return

## CONVERT XLSX TO DCMRI FORMAT

def run_dcmri(info):
    start_time = time.time()
    try:
        analysis.dcmri_analysis(info)
        total_time = time.time() - start_time
        if info['logger']:
            info['logger'].info('Completed dcmri analysis in {} seconds'.format(total_time))
    except Exception as e:
        print("An error occured:")
        traceback.print_exc()
        error_message = traceback.format_exc()
        if info['logger']:
            info['logger'].error('Error in dcmri')
            info['logger'].error(error_message)
        pass

    if info['logger']:
        info['logger'].info('IMPLEMENT DCMRI ANALYSIS FUNCTIONALITY HERE')

    return













