from trem_researcher import TremResearcher, sep, mk_dir_if_not_exists
import numpy as np
import time
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial

errors_by_every_file_path = fr'D:\Projects\Tremor\BIN FILES\Errors for Accurate attempt 5 by every file (corrected).pkl'
researcher = TremResearcher(marks_num=5, marks_size_mm=10)


def f(errors_by_every_file_path, allowable_error):
    from trem_researcher import TremResearcher
    import numpy as np

    _researcher = TremResearcher(marks_num=5, marks_size_mm=10)

    folders_to_check = [r'D:\Projects\Tremor\Calibration\Check calibration photo\Focus distance 50 mm 3\X 0',
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

    with open((errors_by_every_file_path), 'rb') as file:
        errors = pickle.load(file)

    files_to_calib = []
    for file in errors:
        errors_for_file = np.asarray([np.mean(errors[file][folder]) for folder in errors[file]])
        is_allowable_error = np.abs(errors_for_file) < allowable_error
        if list(is_allowable_error).count(True) == len(is_allowable_error):
            files_to_calib += [file]

    if len(files_to_calib) == 0:
        return {allowable_error: {folder: np.nan for folder in folders_to_check}}

    print(f"Calibrate sample with allowable_error={allowable_error}. Started.\n")
    _researcher.calibrate(images_paths=files_to_calib)
    print(f"Calibrate sample with allowable_error={allowable_error}. Finished.\n")

    FRAMES_PER_STEP = 5
    STEP_MM = 5
    get_errors = _researcher.get_errors_finder(
        frames_per_step=FRAMES_PER_STEP,
        step_mm=STEP_MM,
        text_for_log=f'Find errors for subsample with allowable error:{allowable_error}.')

    _res = {allowable_error: {folder: get_errors(_researcher.photo_files(folder)) for folder in folders_to_check}}
    return _res


if __name__ == '__main__':
    t1 = time.time()
    with Pool(20) as pool:
        errors_range = np.arange(7, 0, -7 / 64)
        new_f = partial(f, errors_by_every_file_path)
        res_f = pool.map(new_f, errors_range)

    t2 = time.time()
    print(f'Time elapsed: {t2 - t1} seconds')

    import pickle

    res_dict = {}
    for x in res_f:
        res_dict.update(x)

    calib_folder = errors_by_every_file_path.rsplit(sep, 1)[-1]
    with open(f'..{sep}BIN FILES{sep}Errors for {calib_folder} (hard test).pkl', 'wb') as f:
        pickle.dump(res_dict, f)
