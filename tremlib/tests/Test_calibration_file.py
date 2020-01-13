from trem_researcher import TremResearcher, sep
import time

from multiprocessing import Pool, cpu_count
from functools import partial

calib_settings_path = fr'D:\Projects\Tremor\BIN FILES\Calibration for Accurate attempt 5 (all) (corrected).pkl'
researcher = TremResearcher(marks_num=5, marks_size_mm=10)

test_folders = [r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\X 0',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\X 15',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\X 30',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\X 45',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\X 60',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\X 75',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\Y 0',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\Y 15',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\Y 30',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\Y 45',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\Y 60',
                r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\Y 75']

def f(calib_settings_path, test_folder):
    from trem_researcher import TremResearcher
    import numpy as np
    import cv2
    _researcher = TremResearcher(marks_num=5, marks_size_mm=10)
    _researcher.load_calib_settings(calib_settings_path)

    FRAMES_PER_STEP = 5
    STEP_MM = 5

    get_errors = _researcher.get_errors_finder(
        frames_per_step=FRAMES_PER_STEP,
        step_mm=STEP_MM,
        text_for_log=f'Find errors for {test_folder.split(sep)[-1]}.')

    return {test_folder : get_errors(_researcher.photo_files(test_folder))}


if __name__ == '__main__':
    t1 = time.time()
    with Pool(min(len(test_folders), 20)) as pool:
        new_f = partial(f, calib_settings_path)
        res_f = pool.map(new_f, test_folders)

    t2 = time.time()
    print(f'Time elapsed: {t2-t1} seconds')

    import pickle

    res_dict = {}

    for x in res_f:
        res_dict.update(x)

    fname = calib_settings_path.rsplit(sep)[-1].rsplit('.')[0]
    with open(f'..{sep}BIN FILES{sep}Errors for {fname} (z).pkl', 'wb') as f:
        pickle.dump(res_dict, f)
