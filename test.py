import img_preproc

VALIDATE_FILE = '/root/dataset/cornell_grasping_dataset/validation-cgd'

def test():
    data_files_ = VALIDATE_FILE
    images, bboxes = img_preproc.inputs([data_files_])