import os, glob
import platform
import numpy as np

# if platform.system() == "Windows":
#     OS_SLASH = '\\'
# else:
#     OS_SLASH = '/'
OS_SLASH = '/'

class FilesHelper:
    @staticmethod
    def mk_dir_if_not_exists(dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    @staticmethod
    def photo_files(path, sort=True) -> np.ndarray:
        extensions = ['jpg', 'JPG']
        return FilesHelper.find_with_extensions(path, extensions, sort)

    @staticmethod
    def video_files(path, sort=True) -> np.ndarray:
        extensions = ['mp4', 'MP4']
        return FilesHelper.find_with_extensions(path, extensions, sort)

    @staticmethod
    def find_with_extensions(path, extensions, sort=True) -> np.ndarray:
        before_dir = os.path.abspath(os.curdir)
        os.chdir(path)
        glob_res = []
        for ext in extensions:
            for file in glob.glob(f"*.{ext}"):
                if file.rsplit('.', maxsplit=1)[-1] == ext:
                    glob_res.append(f"%s{OS_SLASH}%s" % (path, file))
        os.chdir(before_dir)
        glob_res = np.asarray(glob_res)
        return np.sort(glob_res) if sort else glob_res