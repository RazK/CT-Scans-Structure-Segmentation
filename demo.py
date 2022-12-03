import nibabel as nib

from threshold_segmentation import DEMO_INPUT_FILE_1, DEMO_OUTPUT_FILE


def demo():
    # loads the nifti files
    nifti_file = nib.load(f'data/{DEMO_INPUT_FILE_1}')
    # getting a pointer to the data
    img_data = nifti_file.get_fdata()
    # returning a boolean matrix with True in place where the intensity level were below 500
    low_values_flags = img_data < 500
    # returning a boolean matrix with True in place where the intensity level were above or equal to 500
    high_values_flags = img_data >= 500

    # Turning boolean to {0,1}
    img_data[low_values_flags] = 0
    img_data[high_values_flags] = 1

    # creating new NiftiImage
    new_nifti = nib.Nifti1Image(img_data, nifti_file.affine)
    # saving the nifti file.
    nib.save(new_nifti, f'out/{DEMO_OUTPUT_FILE}')

if __name__ == "__main__":
    demo()