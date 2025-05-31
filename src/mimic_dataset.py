import os
import torch
import pandas as pd
import numpy as np
import logging

from torch.utils.data import Dataset
from PIL import Image, ImageFile  
ImageFile.LOAD_TRUNCATED_IMAGES = True # Enable truncated loads

from data_utils import MIMIC_CXR_Processor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MIMICDatasetCreator:
    def __init__(self, image_dir, report_dir, struct_path, metadata_path):
        """
        Args:
            image_dir (str): Directory with all the images.
            report_dir (str): Directory with all the reports.
            struct_path (str): Path to the structured data CSV file.
            metadata_path (str): Path to the metadata CSV file.
        """
        self.image_dir = image_dir
        self.report_dir = report_dir
        self.struct_data = pd.read_csv(struct_path)
        self.metadata = pd.read_csv(metadata_path)
        # Pre-filter metadata for faster lookups if needed, or handle potential type issues
        self.metadata['subject_id'] = self.metadata['subject_id'].astype(int)
        self.metadata['study_id'] = self.metadata['study_id'].astype(int)

    def create_dataset_table(self, output_path, n_samples=None):
        """
        Creates and saves a dataset table with all paths
        to images and reports, along with structured data.
        Filters for PA view images.
        """
        dataset_table = self.struct_data.copy()

        # Create image paths using subject_id and study_id, filtering for PA view
        def get_image_path(row):
            subject_id = int(row['subject_id'])
            study_id = int(row['study_id'])
            # Extract first digits for folder structure (p10, p11, etc.)
            p_folder = f"p{str(subject_id)[:2]}"
            p_id_folder = f"p{subject_id}"
            img_dir_path = os.path.join(self.image_dir, p_folder, p_id_folder, f"s{study_id}")

            if not os.path.exists(img_dir_path):
                return None

            # Filter metadata for the current study
            study_metadata = self.metadata[
                (self.metadata['subject_id'] == subject_id) &
                (self.metadata['study_id'] == study_id)
            ]

            for file in os.listdir(img_dir_path):
                if file.endswith('.jpg'):
                    dicom_id = file.split('.')[0]
                    # Find the image's metadata entry
                    image_metadata = study_metadata[study_metadata['dicom_id'] == dicom_id]
                    if not image_metadata.empty:
                        # Check if the ViewPosition is 'PA'
                        if image_metadata.iloc[0]['ViewPosition'] in ['PA']:
                            return os.path.join(img_dir_path, file)
            # Return None if no PA image is found for this study
            return None

        # Create report paths using subject_id and study_id
        def get_report_path(row):
            subject_id = str(int(row['subject_id']))
            study_id = str(int(row['study_id']))
            # Extract first digits for folder structure
            p_folder = f"p{subject_id[:2]}"
            p_id_folder = f"p{subject_id}"
            report_path = os.path.join(self.report_dir, p_folder, p_id_folder, f"s{study_id}.txt")
            # Check if report file exists
            if os.path.exists(report_path):
                return report_path
            return None

        dataset_table['image_path'] = dataset_table.apply(get_image_path, axis=1)
        dataset_table['report_path'] = dataset_table.apply(get_report_path, axis=1)

        # Filter out rows with missing image or report files
        initial_count = len(dataset_table)
        dataset_table = dataset_table[dataset_table['image_path'].notnull()]
        logging.info(f"Filtered out {initial_count - len(dataset_table)} rows due to missing image paths.")
        initial_count = len(dataset_table)
        dataset_table = dataset_table[dataset_table['report_path'].notnull()]
        logging.info(f"Filtered out {initial_count - len(dataset_table)} rows due to missing report paths.")

        # Limit to the first n_samples if specified
        if n_samples:
            dataset_table = dataset_table.head(n_samples)
        dataset_table.to_csv(output_path, index=False)

        return dataset_table


class MIMICDataset(Dataset):
    """
    Dataset for MIMIC data with image paths, report text, and structured data
    """
    def __init__(self, dataset_csv, transform=None, preload_reports=False):
        """
        Args:
            dataset_csv (str): Path to the CSV file with image_path, report_path, and structured data
            transform (callable, optional): Optional transform to be applied on an image
            processor (MIMIC_CXR_Processor): Processor for parsing reports
        """
        logging.info(f"Loading dataset table from {dataset_csv}")
        self.dataset_table = pd.read_csv(dataset_csv)
        self.transform = transform
        self.processor = MIMIC_CXR_Processor()

        # Filter out entries with empty reports after cleaning
        initial_count = len(self.dataset_table)
        logging.info(f"Initial dataset size: {initial_count}")
        indices_to_keep = []
        empty_report_count = 0
        for index, row in self.dataset_table.iterrows():
            try:
                with open(row['report_path'], 'r') as file:
                    report_text = file.read()
                cleaned_report = self.processor.get_cleaned_report(report_text)
                # Check if cleaned report is not empty (strip whitespace)
                if self.processor.is_simple_report(cleaned_report):
                    indices_to_keep.append(index)
                else:
                    empty_report_count += 1
            except FileNotFoundError:
                 logging.warning(f"Report file not found, skipping: {row['report_path']}")
                 empty_report_count += 1 # Count as empty/invalid
            except Exception as e:
                 logging.error(f"Error processing report {row['report_path']}: {e}")
                 empty_report_count += 1 # Count as empty/invalid

        self.dataset_table = self.dataset_table.loc[indices_to_keep].reset_index(drop=True)
        final_count = len(self.dataset_table)
        logging.info(f"Filtered out {initial_count - final_count} entries with empty or problematic reports.")
        logging.info(f"Final dataset size: {final_count}")


    def __len__(self):
        return len(self.dataset_table)

    def __getitem__(self, idx):
        # Get the paths and structured data
        row = self.dataset_table.iloc[idx]
        img_path = row['image_path']
        resize_factor = 4
        try:
            image = Image.open(img_path).convert('RGB')
            # Resize the image to half its original size to save memory
            image = image.resize((image.width // resize_factor, image.height // resize_factor))
        except Exception as e:
            logging.error(f"Error loading or processing image {img_path}: {e}")
            # Handle error: return None or raise exception, or return a placeholder
            # For simplicity, let's raise it for now, or you could skip this item
            raise RuntimeError(f"Failed to load image at index {idx}: {img_path}") from e


        try:
            with open(row['report_path'], 'r') as file:
                report_text = file.read()
                report_text = self.processor.get_cleaned_report(report_text)
        except Exception as e:
            raise RuntimeError(f"Failed to load report at index {idx}: {row['report_path']}") from e

        # Extract structured data
        struct_data = row.drop(['image_path', 'report_path', 'subject_id', 'study_id']).to_dict()
        struct_text = self.processor.get_structured_report_text(struct_data)

        # Create structured data tensor
        raw_vals = [row.get(c, np.nan) for c in  self.processor.chexpert_columns]
        chex = []
        for v in raw_vals:
            if np.isnan(v):
                chex.append(-1.0)
            else:
                chex.append(float(v))
        chexpert_labels = torch.tensor(chex, dtype=torch.float)

        if self.transform:
            image = self.transform(image)
            
        return image, report_text, struct_text, chexpert_labels

if __name__ == "__main__":
    image_dir = '../data/mimic-cxr/mimic-cxr-jpg/2.1.0/files'
    report_dir = '../data/mimic-cxr/mimic-cxr-reports/files'
    struct_path = '../data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv'
    metadata_path = '../data/mimic-cxr/mimic-cxr-2.0.0-metadata.csv'
    output_csv = '../data/data_8k.csv'

    # First create the dataset table
    create_dataset = True
    if create_dataset:
        logging.info("Creating dataset table...")
        creator = MIMICDatasetCreator(image_dir, report_dir, struct_path, metadata_path)
        dataset_table = creator.create_dataset_table(output_path=output_csv, n_samples=8000)
        logging.info(f"Created dataset table with {len(dataset_table)} entries at {output_csv}")

    ds = MIMICDataset(dataset_csv=output_csv)
    for i in range(3):
        try:
            img, rpt, struct_txt, chex = ds[i]
            print("Image shape:", img)
            print("Report text:", rpt)
            print("CheXpert labels:", chex.tolist())
        except Exception as e:
            logging.error(f"Error processing item {i}: {e}")
