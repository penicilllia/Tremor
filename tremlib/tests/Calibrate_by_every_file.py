from trem_researcher import TremResearcher, sep
import time

from multiprocessing import Pool, cpu_count
from functools import partial

calib_images_path = fr'..{sep}Calibration{sep}Calibration photo{sep}Accurate attempt 5'
researcher = TremResearcher(marks_num=5, marks_size_mm=10)
calib_images = researcher.photo_files(calib_images_path)
files_to_check = len(calib_images)

def f(calib_images_path, i):
    from trem_researcher import TremResearcher
    import numpy as np
    import cv2
    _researcher = TremResearcher(marks_num=5, marks_size_mm=10)
    calib_images = _researcher.photo_files(calib_images_path)

    FRAMES_PER_STEP = 5
    STEP_MM = 5

    # folders_to_check = [r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 2\X',
    #                     r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 2\Y']
    folders_to_check = [fr'..{sep}Calibration{sep}Check calibration photo{sep}Focus distance 50 mm 3{sep}X 0',
                        fr'..{sep}Calibration{sep}Check calibration photo{sep}Focus distance 50 mm 3{sep}X 15',
                        fr'..{sep}Calibration{sep}Check calibration photo{sep}Focus distance 50 mm 3{sep}X 30',
                        fr'..{sep}Calibration{sep}Check calibration photo{sep}Focus distance 50 mm 3{sep}X 45',
                        fr'..{sep}Calibration{sep}Check calibration photo{sep}Focus distance 50 mm 3{sep}X 60',
                        fr'..{sep}Calibration{sep}Check calibration photo{sep}Focus distance 50 mm 3{sep}Y 0',
                        fr'..{sep}Calibration{sep}Check calibration photo{sep}Focus distance 50 mm 3{sep}Y 15',
                        fr'..{sep}Calibration{sep}Check calibration photo{sep}Focus distance 50 mm 3{sep}Y 30',
                        fr'..{sep}Calibration{sep}Check calibration photo{sep}Focus distance 50 mm 3{sep}Y 45',
                        fr'..{sep}Calibration{sep}Check calibration photo{sep}Focus distance 50 mm 3{sep}Y 60']

    print(f"Calibrate with {i} file. Started.")
    _researcher.calibrate(images_paths=[calib_images[i]])
    print(f"Calibrate with {i} file. Finished.")

    get_errors = _researcher.get_errors_finder(
        frames_per_step=FRAMES_PER_STEP,
        step_mm=STEP_MM,
        text_for_log=f'Find errors for {calib_images[i]}.')

    return {calib_images[i]: {folder : get_errors(_researcher.photo_files(folder)) for folder in folders_to_check}}


if __name__ == '__main__':
    t1 = time.time()
    with Pool(64) as pool: #min(files_to_check, 20)
        new_f = partial(f, calib_images_path)
        res_f = pool.map(new_f, range(files_to_check))

    t2 = time.time()
    print(f'Time elapsed: {t2-t1} seconds')

    import pickle

    res_dict = {}
    for x in res_f:
        res_dict.update(x)

    calib_folder = calib_images_path.rsplit(sep, 1)[-1]
    with open(f'..{sep}BIN FILES{sep}Errors for {calib_folder} by every file (corrected).pkl', 'wb') as f:
        pickle.dump(res_dict, f)
