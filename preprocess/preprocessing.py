from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import bids_utils

# Predefined variables
datapath ="Neuroinformatics/pre-dataset/"
output_path = "Neuroinformatics/dataset/" 

def preprocess_axon(axon_mask, px_size):
    '''
    Preprocesses axon mask to extract relevant features.
    Returns morphometrics, index image, and instance segmentation.
    '''
    morph_output = get_axon_morphometrics(
        axon_mask, 
        pixel_size=px_size,
        return_index_image=True,
        return_border_info=True,
        return_instance_seg=True
    )
    return morph_output

def preprocess_myelin(axon_mask, myelin_mask, px_size):
    '''
    Preprocesses myelin mask to extract relevant features.
    Returns myelin map, myelin image, and prompts dataframe.
    '''
    morph_output = get_axon_morphometrics(
        axon_mask, 
        im_myelin=myelin_mask, 
        pixel_size=px_size,
        return_index_image=True,
        return_border_info=True,
        return_instance_seg=True
    )
    morphometrics, index_im, instance_seg_im, instance_map = morph_output

    mask = myelin_mask == 255
    myelin_map = instance_map.astype(np.uint16) * mask
    myelin_im = instance_seg_im * np.repeat(mask[:,:,np.newaxis], 3, axis=2)

    bboxes = morphometrics.iloc[:, -4:]
    centroids = morphometrics.iloc[:, :2]

    prompts_df = pd.concat([centroids.astype('int64'), bboxes], axis=1)
    prompts_df.rename(columns={'x0 (px)': 'x0', 'y0 (px)': 'y0'}, inplace=True)
    col_order = ['x0', 'y0', 'bbox_min_x', 'bbox_min_y', 'bbox_max_x', 'bbox_max_y']
    prompts_df = prompts_df.reindex(columns=col_order)

    return myelin_map, myelin_im, prompts_df

def save_bbox_img(myelin_img, prompts_df, index_im, fname):
    '''
    Saves the image with bounding boxes overlaid for quality control.
    '''
    mask = Image.fromarray(myelin_img)
    rgbimg = Image.new("RGBA", mask.size)
    rgbimg.paste(mask)

    draw = ImageDraw.Draw(rgbimg)
    for i in range(prompts_df.shape[0]):
        draw.rectangle(
            [
                prompts_df.iloc[i, 2], 
                prompts_df.iloc[i, 3],
                prompts_df.iloc[i, 4],
                prompts_df.iloc[i, 5],
            ],
            outline='red',
            width=2
        )
    index_im = Image.fromarray(index_im)
    rgbimg.paste(index_im, mask=index_im)
    rgbimg.save(fname)

def main(datapath, output_path=None):
    data_dict = bids_utils.index_bids_dataset(datapath)

    # Set the output path
    if output_path is None:
        output_path = Path().cwd() / 'derivatives' / 'maps'
    else:
        output_path = Path(output_path) / 'derivatives' / 'maps'
    output_path.mkdir(parents=True, exist_ok=True)
    
    for sub in tqdm(data_dict.keys()):
        subject_path = output_path / sub
        subject_path.mkdir(exist_ok=True)

        px_size = data_dict[sub]['px_size']
        samples = (s for s in data_dict[sub].keys() if 'sample' in s)
        for sample in samples:
            samples_path = subject_path / 'micr'
            samples_path.mkdir(exist_ok=True)

            axon_mask = data_dict[sub][sample]['axon']
            myelin_mask = data_dict[sub][sample]['myelin']

            # Preprocess axon mask
            axon_output = preprocess_axon(axon_mask, px_size)
            idx_im_axon, _, _, _ = axon_output

            # Preprocess myelin mask
            myelin_output = preprocess_myelin(axon_mask, myelin_mask, px_size)
            myelin_map, myelin_im, prompts = myelin_output

            # Define filenames for output files
            qc_fname = samples_path / f'{sub}_{sample}_qc.png'
            map_fname = samples_path / f'{sub}_{sample}_myelinmap.png'
            prompts_fname = samples_path / f'{sub}_{sample}_prompts.csv'

            # Save all derivatives
            save_bbox_img(myelin_im, prompts, idx_im_axon, qc_fname)
            ads_utils.imwrite(map_fname, myelin_map.astype(np.uint16))
            prompts.to_csv(prompts_fname)
            nb_axons = prompts.shape[0]
            if nb_axons >= 256:
                print(f'WARNING: {sub}_{sample} has {nb_axons} axons. It will be saved in 16bit format.')

if __name__ == '__main__':
    main(datapath, output_path)
