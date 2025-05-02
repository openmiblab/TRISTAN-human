import time
import os
from tqdm import tqdm
import zarr
import numpy as np
import napari
import dask.array as da
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
import pandas as pd
import zipfile

import vreg
import dbdicom as db
import mdreg
import wezel
from wezel.plugins import pyvista, transform, segment

import stages
from compute import io


ROI_SUBSET = ['liver', 'aorta', 'spleen', 'kidney_right', 'kidney_left', 
             'inferior_vena_cava', 'portal_vein_and_splenic_vein']

ROI_MOLLI = ['liver', 'aorta', 'spleen']

# from wezel import app

def all(datapath, resultspath):
    subjects = [5]
    visits = [1]
    scans = [1,2]
    for subject in subjects:
        for visit in visits:
            for scan in scans:
                onescan(datapath, resultspath, subject, visit, scan)


def gui():
    proj = 'TRISTAN'
    wzl = wezel.app(project=proj)
    wzl.add_menu(proj)
    wzl.add_action(pyvista.action_show_mask_surface, proj)
    wzl.add_action(pyvista.action_show_mask_surfaces, proj)
    wzl.add_action(pyvista.action_show_mask_surfaces_with_reference, proj)
    wzl.add_menu(segment.menu)
    wzl.add_menu(transform.menu)
    wzl.add_menu(wezel.menubar.about.menu) 
    return wzl


def merge_series(datapath):
    db.merge_series(
        datapath,
        '3D_DISCO_Dyn_cor_fb', 
        merged='3D_DISCO_Dyn_cor_fb_all',
    )
    # db.merge_series(
    #     datapath,
    #     ['MOLLI_S0_1', 'MOLLI_S0_2'], 
    #     merged='MOLLI_S0', 
    #     study='Tmp',
    # )


def merge_zarr(datapath, resultspath):
    # # Create merged z-array on disk
    dcm = db.database(datapath)
    series = dcm.series(SeriesDescription='3D_DISCO_Dyn_cor_fb')
    # TODO: store affine in header (saves having to read series again)
    store = os.path.join(resultspath, 'disco.zarr')
    for i, s in tqdm(enumerate(series), desc='Merging series..'):
        array = s.pixel_values(['SliceLocation', 'InstanceNumber'])
        if i==0:
            zarray = zarr.create(
                shape=array.shape, dtype='float32', 
                chunks=array.shape[:2] + (1,1),
                store=store)
            zarray[:] = array
        else:
            zarray.append(array, axis=3)

    # # Compute coronal images
    # dyn = da.from_zarr(os.path.join(resultspath, 'disco.zarr'))
    # if image=='mean':
    #     img = da.mean(dyn, axis=-1).compute()


def unzip_data(datapath, resultspath):
    """
    Unzips all ZIP files from 'datapath' into 'resultspath'.
    Each ZIP file will be extracted into a folder with the same name as the ZIP file.
    
    Parameters:
    - datapath: str -> Path where the ZIP files are located.
    - resultspath: str -> Path where the extracted files should be stored.
    """
    # Ensure the result path exists
    if not os.path.exists(resultspath):
        os.makedirs(resultspath)
    
    # Iterate through all files in the datapath
    file = datapath + '.zip'
    # Check if the file is a ZIP file
    #zip_file_path = os.path.join(datapath, file)
    
    # Create a folder with the same name as the ZIP file (without .zip extension)
    extract_folder = os.path.join(resultspath, os.path.basename(datapath))
    
    # Ensure the folder does not already exist
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    
    # Extract the ZIP file
    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        print(f"✅ Extracted: {file} → {extract_folder}")
    except zipfile.BadZipFile:
        print(f"❌ Error: {file} is not a valid ZIP file")
    except Exception as e:
        print(f"❌ Error extracting {file}: {e}")



def compute_mean(datapath):

    vol = db.volume(
        datapath, 
        '3D_DISCO_Dyn_cor_fb', 
        index=0, 
        dims='TriggerTime',
    )
    db.write_volume(
        datapath, 
        vreg.mean(vol, axis=3), 
        series='DISCO_mean', 
        study='Maps',
        ref='3D_DISCO_Dyn_cor_fb', 
        ref_index=0,
    )


def segment_mean(datapath, resultspath):

    dcm = db.database(datapath)

    print('Saving axial image as nifti..')
    coronal = dcm.volume('DISCO_mean')
    axial = coronal.reslice(orient='axial')
    nifti = os.path.join(resultspath, 'DISCO_mean'+'_axial.nii.gz')
    vreg.write_nifti(axial, nifti)

    # Perform segmentation on axial average
    print('Segmenting organs..')
    totalsegmentator(
        nifti, 
        resultspath, 
        task="total_mr", 
        roi_subset=ROI_SUBSET, 
        quiet=True, 
        device='cpu',
    )
    # Save coronal results to database
    for roi in tqdm(ROI_SUBSET, desc='Writing to DICOM..'):
        filepath = os.path.join(resultspath, roi + '.nii.gz')
        dcm.write_volume(
            vreg.read_nifti(filepath).slice_like(coronal), 
            series=roi+'_DISCO', 
            study='Segment DISCO',
            ref='DISCO_mean', 
        )
        
    dcm.save().close()


def export_time_curves(datapath, resultspath):

    dims = ('SliceLocation', 'InstanceNumber')
    times = ((0x0009,0x10E9), 'TriggerTime')
    dcm = db.database(datapath)
    
    # Read masks
    mask = {}
    sig = {'time': []}
    for roi in tqdm(ROI_SUBSET, desc='Reading masks..'):
        img = dcm.pixel_values(roi+'_DISCO', dims='SliceLocation')
        mask[roi] = img > 0.5
        sig[roi] = []
    
    # Read time curves
    for series in dcm.series("3D_DISCO_Dyn_cor_fb"):
        s, t = series.pixel_values(dims, return_vals=times) 
        sig['time'] += list(t[times[0]][0,0] + t[times[1]][0,:] / 1000)
        for roi in tqdm(ROI_SUBSET, desc='Extracting signal averages..'):
            sig[roi] += [
                np.mean(s[...,k][mask[roi]]) 
                for k in range(t[times[0]].shape[1])
            ]

    # Save to excel
    pd.DataFrame(data=sig).to_excel(
        os.path.join(resultspath, dcm.PatientID + '.xlsx'), 
        sheet_name='dyn1', 
        index=False,
    )
    

def map_molli(datapath):

    # Get MOLLI series
    dcm = db.database(datapath)
    mollis = dcm.series('[Loc:110.51] T1Map_LL_tra_mbh')

    # Loop over the mollis
    for i, molli in tqdm(enumerate(mollis), desc=f'Fitting MOLLI'):

        # Get data
        vol, crds = molli.volume('InversionTime', return_coords=True)

        # Compute
        reg, _, fit, pars = mdreg.fit(
            vol.values[:,:,0,:], 
            fit_image = {
                'func': mdreg.abs_exp_recovery_2p,
                'TI': crds['InversionTime']/1000,
                'bounds':([0,0], [np.inf,3]),
            }, 
            fit_coreg = {
                'package': 'elastix',
                'FinalGridSpacingInPhysicalUnits': 50.0,
                'spacing': vol.spacing[:2],
            },
            maxit=3, 
            verbose=2,
        )

        # Save results
        dcm.write_volume(
            vreg.volume(np.expand_dims(pars[:,:,0], 2), vol.affine), 
            series=f'MOLLI_S0_{i+1}', 
            study='MOLLI maps',
            ref=molli, 
        )
        dcm.write_volume(
            vreg.volume(np.expand_dims(pars[:,:,1], 2), vol.affine), 
            series=f'MOLLI_T1_{i+1}',
            study='MOLLI maps',
            ref=molli, 
        )
        dcm.write_volume(
            vreg.volume(np.expand_dims(reg, 2), vol.affine), 
            series=f'MOLLI_reg_{i+1}', 
            study='MOLLI maps', 
            ref=molli, 
            coords=crds,
        )
        dcm.write_volume(
            vreg.volume(np.expand_dims(fit, 2), vol.affine), 
            series=f'MOLLI_fit_{i+1}', 
            study='MOLLI maps', 
            ref=molli, 
            coords=crds,
        )

    dcm.save().close()
    

def segment_molli(datapath, resultspath):

    # Get MOLLI series
    dcm = db.database(datapath)
    mollis = dcm.series('[Loc:110.51] T1Map_LL_tra_mbh')

    for i, molli in enumerate(mollis):

        # Use the image with longest TI for segmentation
        nifti = os.path.join(resultspath, 'molli.nii.gz')
        vol = molli.volume('InversionTime').separate(axis=3)
        vreg.write_nifti(vol[0], nifti)

        # Perform segmentation
        print('Segmenting MOLLI' + str(i+1))
        totalsegmentator(
            nifti, resultspath, task="total_mr", roi_subset=ROI_MOLLI, 
            quiet=True, device='cpu')
        
        # Save segmentation data
        dcm.write_volume(
            vol[-1], 
            series='MOLLI' + str(i+1),
            study='Segment MOLLI', 
            ref=molli,
        )
        # Save segmentation results
        for roi in tqdm(ROI_MOLLI, desc=f'Saving MOLLI{i+1}'):
            filepath = os.path.join(resultspath, roi + '.nii.gz')
            dcm.write_volume(
                vreg.read_nifti(filepath), 
                series=roi+f'_MOLLI{i+1}',
                study='Segment MOLLI',
                ref=molli,
            )     

    dcm.save().close()



def onescandev(datapath, resultspath):
    start_time = time.time()

    # unzip_data(datapath, resultspath)
    # compute_mean(resultspath)
    # segment_mean(datapath, resultspath)
    # export_time_curves(datapath, resultspath)
    # map_molli(datapath)
    # segment_molli(datapath, resultspath)
    # TODO coreg MOLLI with DISCO?

    # merge_series(datapath)
    # merge_zarr(datapath, resultspath)
    
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    gui().open(resultspath).use()
    #gui().open(datapath).display('DISCO_mean').use()





def onescan(datapath, resultspath, subject, visit, scan):
    pass

    # # setup logger
    # logger = stages.setup_logger(subject, visit, scan)

    # # setup details dictionary
    # info = stages.setup_detail_dict(datapath, resultspath, subject, visit, scan, logger)

    # # extract arrays from dicom
    # stages.extract_arrays(info)

    # # segment ROI pre coregistration
    # stages.segment_ROI(info, 'pre_coreg')

    # # show ROIS
    # stages.show_ROIS(info, 'pre_coreg')

    # # setup mdreg
    # stages.correct_motion(info, [20,90])

    # # segment ROI post coregistration
    # stages.segment_ROI(info, 'post_coreg')

    # # show ROIS
    # stages.show_ROIS(info, 'post_coreg')

    # # Map T1
    # stages.t1mapping_molli(info, 'pre_coreg')

    # if scan == 1:
    #     stages.t1mapping_molli(info, 'post_coreg')

    # if scan == 2:
    #     # export CSV
    #     # involves both scans 1 and 2
    #     stages.format_to_csv(info)
    #     stages.run_dcmri(info)



