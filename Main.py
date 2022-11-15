import re
from os import path, listdir, walk, get_terminal_size
from time import localtime
from typing import List

import cv2 as cv2
import numpy as np
from joblib import Parallel, delayed

from Decorators import st_time  # Used to time main run function, for testing
from FileHandler import add_to_json
from Private import DATA_DIR
from Util import replace_goofy_slashes, save_date_str, transpose_list_of_lists, SUPPORTED_IMGS


class ImgData:
    """ A Class used for all pictures, target image and search image both. """

    def __init__(self, img_path: str = None, error_dir: str = None, is_target: bool = False):
        self.path = img_path
        self.file = path.basename(self.path)
        self.error_dir = error_dir or path.join(DATA_DIR, "errors/IMAGE_ERRORS.json")
        self.is_target = is_target
        self.width = None
        self.height = None
        self.scaled_width = None
        self.scaled_height = None
        self.gray = None  # --------------------- np array
        self.color = self.create_img_from_path()  # np array

    def create_img_from_path(self):
        """ Checks the file set as `self.path` for being an image and is supported per OpenCV docs.

        If so, converts the file to a cv2 image, set's the `self.width` and `self.height` parameters, scales down a
        copy of the image to 500px on the longer edge, converts it to gray and set's it as the `self.gray` parameter.
        """

        def scale_to_px(img, px_amt=300):
            """ Scales shortest side of image to 300px if bigger currently, longer side to same ratio.

            'target image' is `scale * 1.25` or `scale * 1.75`
            Used for the gray image that will be used to get key points.
            """
            if self.width > px_amt or img.shape[1] > px_amt:
                if self.width > self.height:
                    scale = int((px_amt / self.height) * 100)
                else:
                    scale = int((px_amt / self.width) * 100)

                if self.is_target:
                    # CHANGE THIS BACK TO 1.25 FOR NON-PORTRAIT IMAGES! - 1.75 for portraits
                    img = self.scale_img_pct(img, (scale * 1.25))
                else:
                    img = self.scale_img_pct(img, scale)
            return img

        # File type of the image based on it's extension.
        img_file_type = re.search(re.compile('\.\w{3,4}$'), self.path.lower()).group()

        if path.isfile(self.path) and img_file_type in SUPPORTED_IMGS:
            image = cv2.imread(self.path, cv2.IMREAD_COLOR)  # 3 channel BGR

            img_shape = image.shape
            self.width = img_shape[1]
            self.height = img_shape[0]

            scaled_img = scale_to_px(image, 500)

            self.gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)  # 1 channel GRAY

            return scaled_img
        else:
            return self._save_error()

    def scale_img_pct(self, image, scale=20.):
        """ Scales an image by a percentage.

        `scale` is the desired, new percent from it's previous size (ie: 20% of original size)
        """
        width = round(self.width * (scale / 100))
        height = round(self.height * (scale / 100))

        dim = (width, height)
        resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        new_dim_confirm = resized_img.shape
        self.scaled_width = new_dim_confirm[1]
        self.scaled_height = new_dim_confirm[0]

        return resized_img

    def show_img(self):
        """ Shows the image on the screen.

        Used for testing the functions in the class while building.
        """
        # Resize window if it's too big to fit on screen
        w_dim = self._calc_new_win_dims((self.width, self.height))

        cv2.namedWindow(self.file, cv2.WINDOW_NORMAL)
        cv2.imshow(self.file, self.color)

        cv2.resizeWindow(self.file, w_dim)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __short_repr__(self):
        """ Removes the np arrays set as the `self.color` and `self.gray` parameters.
        """
        return dict((k, v) for k, v in self.__dict__.items() if k not in ['color', 'gray'])

    @staticmethod
    def _calc_new_win_dims(dim: tuple):
        """ Calculates window dimensions before showing the image. Intended to only be used with the `self.show_img`
        function.

        Assumes dim[0] is width -&- dim[1] is height.
        """
        try:
            # Get the (w, h) for the monitors
            import ctypes
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            mon_w, mon_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except:
            mon_w, mon_h = (1920, 1080)

        w, h = dim
        it_will_fit = (lambda: bool(h <= mon_h and w <= mon_w))

        if w > mon_w:
            new_w = mon_w - 500
            h = (new_w / w) * h
            w = new_w

        if h > mon_h:
            new_h = mon_h - 300
            w = (new_h / h) * w
            h = new_h

        if it_will_fit:
            return int(w), int(h)
        else:
            print(f"\nIMAGE WILL NOT FIT!\nDIM:\t({w, h})")
            return 200, 200

    def _save_error(self):
        """ Saves data when an error is presented without stopping the process. """
        if self.is_target:
            error_obj = {
                "error": f"ERROR READING FILE!",
                "image": self.__dict__,
                "stats": {"saved_at": save_date_str()}
                }
            return add_to_json(error_obj, self.error_dir, is_error=True)
        else:
            print(f'Could not create ImgData obj from:\t{self.path}')
            return self.path

    def __getstate__(self):
        state = self.__dict__.copy()
        return state


class FlannMatch:
    """ Expected to have `input of img1 = TARGET_IMG, img2 = SEARCH_IMG`

    SIFT index_params: dict(algorithm=1, trees=5)

    ORB index_params: dict(
        algorithm=6,
        table_number=6,
        key_size=12,
        multi_probe_level=1)

    """

    def __init__(self,
                 img1: ImgData,
                 img2: ImgData,
                 threshold: float = 0.15,
                 k: int = 2,
                 error_dir: str = path.join(DATA_DIR, "errors/IMAGE_ERRORS.json"),
                 index_params=dict(
                     algorithm=6,
                     table_number=6,
                     key_size=12,
                     multi_probe_level=1),
                 search_params=dict(checks=50)
                 ):
        self.img1 = img1  # target img
        self.img2 = img2  # search img
        self.all_matches = None
        self.good_matches = []
        self.threshold = threshold or 0.15
        self.k = k
        self.held_time = None
        self.display_time = self._get_display_time()
        self.error_dir = error_dir
        self.index_params = index_params
        self.search_params = search_params
        # self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

    def match_key_points(self):
        """ Expected to have input of `img1 = TARGET_IMG, img2 = SEARCH_IMG`
            return format: `(SEARCH_IMG, TARGET_IMG, pct_matched)` OR `(0, 0, 0)`
        """
        # When using ORB:
        kp1, desc1 = self.orb.detectAndCompute(self.img1.gray, None)
        kp2, desc2 = self.orb.detectAndCompute(self.img2.gray, None)

        # When using SIFT:
        # kp1, desc1 = self.sift.detectAndCompute(self.img1.gray, None)
        # kp2, desc2 = self.sift.detectAndCompute(self.img2.gray, None)

        if len(kp1) <= 1 and len(kp2) <= 1:
            print('\n\n', "=" * 90, sep='\n'), print("ERROR!"), print("=" * 90, '\n', sep='\n')
            return self._save_error()

        self.all_matches = self.flann.knnMatch(desc2, desc1, self.k)

        [self.good_matches.append(m[0]) for m in self.all_matches if len(
            m) == 2 and m[0].distance <= (self.threshold * m[1].distance)]

        if len(self.good_matches) > 0:
            print('\nMATCHES FOUND!\nSee the `good_matches` or `all_matches` attributes on this object to access!')
            pct_matched = str(abs((len(self.good_matches) / len(self.all_matches)) * 100))

            # (SEARCH_IMG, TARGET_IMG, pct_matched)
            return self.img2, self.img1, pct_matched
        else:
            return 0, 0, 0

    def _get_display_time(self):
        tmp_time = localtime()
        self.held_time = tmp_time
        return f'{tmp_time.tm_hour}-{tmp_time.tm_min}-{tmp_time.tm_sec}'

    def _save_error(self):
        """ Saves data when an error is presented without stopping the process. """
        # error_file_name = f"ERROR_{self.display_time}.json"
        # error_full_pth = path.join(replace_goofy_slashes(self.error_dir), error_file_name)
        error_obj = {
            "error": "ERROR attempting to get key-points in one or more image!".upper(),
            "img1": self.img1.path,
            "img2": self.img2.path
            }

        add_to_json(error_obj)
        self.good_matches = None
        self.all_matches = None
        return error_obj


class FlannRunner:
    def __init__(self,
                 target_path: str or list,
                 search_path: str or list,
                 target_limiter: tuple = None,
                 search_limiter: tuple = None,
                 save_path: str = None,
                 error_dir: str = None,
                 threshold: float = 0.15,
                 run_parallel: bool = False,
                 parallel_cores: int = 4):

        self.targetPath = target_path
        self.searchPath = search_path

        self.fileInc = len(listdir("../data/matches/"))
        self.savePath = save_path or path.join(DATA_DIR, f"matches/{self.fileInc}.json")
        self.error_dir = error_dir or path.join(DATA_DIR, "errors/RUN_ERRORS.json")

        self.threshold = threshold
        self.runParallel = run_parallel
        self.parallelCores = parallel_cores

        self.targetImgPaths = self.get_files_from_dir(target_path, target_limiter)
        self.targetImgDataObjs = [ImgData(tPath, is_target=True) for tPath in self.targetImgPaths]

        self.searchImgPaths = self.get_files_from_dir(search_path, search_limiter)
        self.check_and_print_ready()

    def get_files_from_dir(self, pth, limiter: tuple = None):
        if type(pth) is list:
            # Multiple paths
            all_paths_lists = [self.get_files_from_dir(p, limiter) for p in pth]

            # Transpose the list to mix up multiple directories, especially important for many drives.
            transposed_path_list = transpose_list_of_lists(all_paths_lists)
            return transposed_path_list

        if path.isfile(pth):
            return [replace_goofy_slashes(pth)]

        if path.isdir(pth):
            paths = [replace_goofy_slashes(path.join(pth, file))
                     for file in list(next(walk(pth)))[2]]

            if limiter:
                paths = paths[limiter[0]: limiter[1]]
            return paths

        # throws error if `pth` can not be found!
        return self._save_error(pth)

    @st_time  # Times the function, used to test the parallel setup
    def get_matches(self) -> None:
        for targetImg in self.targetImgDataObjs:
            print('\n\n'), print('*' * get_terminal_size()[0])
            print(f"Looking for {path.basename(targetImg.path)}", '\n')

            if self.runParallel:
                flann_res = self.run_parallel_flann(targetImg, self.searchImgPaths)
            else:
                flann_res = self.run_std_flann(targetImg, self.searchImgPaths)

            if len(flann_res) > 0:
                # FILTER out non-matched objs
                matches = list(filter(lambda r: r[0] != 0, flann_res))

                # FILTER for filetypes not found -> Remove from self.searchImgPaths
                bad_files = list(filter(lambda bf: bf[0] == 0 and type(bf[1]) is str, flann_res))
                bad_files = [bf[1] for bf in bad_files]
                self.searchImgPaths = list(
                    filter(lambda pth: pth not in bad_files, self.searchImgPaths))

                if self.savePath:
                    # `matches` format expected to be `(SEARCH_IMG, TARGET_IMG, pct_matched)`
                    matched_save_fmt = [(match[0], match[2]) for match in matches]
                    saved = self.save_flann_matches(matched_save_fmt, targetImg)

                    print('\n\n'), print('=' * get_terminal_size()[0])
                    print("SAVED!"), print(saved)
                    print('=' * get_terminal_size()[0], '\n\n')

    def run_parallel_flann(self, target_img: ImgData, search_paths: List[str]):
        """ Multi core application of creating FlannMatch objs and executing it's getMatches function. """
        matches = Parallel(n_jobs=self.parallelCores, verbose=2)(
            delayed(self.many_search_paths_wrapper)(target_img, sImg) for sImg in search_paths)

        return matches

    def run_std_flann(self, target_img: ImgData, search_paths: [str]):
        """ Standard for loop for each path in `searchPaths` executing getMatches function on each """
        matches = [self.many_search_paths_wrapper(target_img, sPth)
                   for sPth in search_paths]
        return matches

    def many_search_paths_wrapper(self, target_img: ImgData, search_img_path: str):
        """ Wrapper to use in either prior `run...` function.
            Requires `tImg` to be of type `ImgData`.
            Converts a `sImgPth` single path-like string to an `ImgData` obj.
            Creates a `FlannMatch` object to compare the `ImgData` objs and
                runs it's `getMatches` function.
        """
        s_img = ImgData(search_img_path)  # ------- Convert path str to `ImgData` Obj

        if type(s_img.color) is not np.ndarray:
            # File is not supported -> Filter for and remove outside of loop.
            return 0, s_img.path

        if s_img.gray is not None:
            f_match_obj = FlannMatch(target_img, s_img, threshold=self.threshold)  # Load Flann Obj
            # Run matcher between two Images - CAN BE (0, 0, 0) ie: Err
            ret = f_match_obj.match_key_points()
            return ret
        else:
            # error occurred & should have been logged to error directory passed to ImgData constructor.
            # same as NO MATCH returned in `FlannMatch.matchKeyPoints()` - filtered in next step
            return 0, 0, 0

    def save_flann_matches(self, matched: [(ImgData, float or int or None)], target_img: ImgData):
        """ Saves match object data in json format.

        Expecting input as:
            `matched = [(ImgData, %match), ...], targetImg = ImgData`  -OR-
            `matched = [], targetImg = ImgData`  -(if no matches found)-
        """
        save_path = path.abspath(self.savePath)
        save_obj = {'target': target_img.__short_repr__(), 'matches': [], 'paths_checked': []}

        # Skips this and use empty list from above if no matches!
        for s_img, pct in matched:
            m_obj = s_img.__short_repr__()
            m_obj['pct_matched'] = pct
            save_obj['matches'].append(m_obj)

        if type(self.searchPath) is list:
            [save_obj['paths_checked'].append(sp) for sp in self.searchPath]
        elif type(self.searchPath) is str:
            save_obj['paths_checked'].append(self.searchPath)

        if type(self.searchPath) is not str:
            tmp_search_img_paths = [list(next(walk(p)))[2] for p in self.searchPath]
            tmp_lengths = [len(lst) for lst in tmp_search_img_paths]
            search_img_paths_possible_path_count = sum(tmp_lengths)
        else:
            tmp_search_img_paths = list(next(walk(self.searchPath)))[2]
            search_img_paths_possible_path_count = len(tmp_search_img_paths)

        save_obj['stats'] = {
            "amt_files_checked": len(self.searchImgPaths),
            "possible_files": search_img_paths_possible_path_count,
            "amt_matches": len(matched),
            "saved_at": save_date_str()
            }
        return add_to_json(save_obj, save_path)

    def check_and_print_ready(self):
        chk_1 = bool(type(self.targetPath) in [list, str])  # - targetPath is list or str
        chk_2 = bool(type(self.searchPath) in [list, str])  # - searchPath is list or str

        # targetImgPaths OK
        chk_3 = bool(type(self.targetImgPaths) == list and len(self.targetImgPaths) > 0)

        # searchImgPaths OK
        chk_4 = bool(type(self.targetImgPaths) in [
            list, np.ndarray] and len(self.searchImgPaths) > 0)

        # targetImgPaths converted into ImgData objs, some may have not due to wrong type
        chk_5 = bool(len(self.targetImgDataObjs) > 0)

        chk_6 = bool(path.isdir(path.dirname(self.savePath))  # ---- savePath is OK
                     and path.basename(self.savePath).endswith('.json'))

        chk_7 = bool(path.isdir(path.dirname(self.error_dir))  # -- error_dir is OK
                     and path.basename(self.error_dir).endswith('.json'))

        ready_lst = [chk_1, chk_2, chk_3, chk_4, chk_5, chk_6, chk_7]
        ready = sum(ready_lst) == len(ready_lst)

        def print_ready():
            term_width = get_terminal_size()[0]
            print('\n\n'), print('=' * term_width * 2)
            print('FlannRunner INITIALIZED!'.center(term_width)), print('_' * term_width)
            print(f'No. Target Img Paths:\t\t{len(self.targetImgPaths)}')
            print(f'No. Search Img Paths:\t\t{len(self.searchImgPaths)}')
            print(f'No. Target Img Objs:\t\t{len(self.targetImgDataObjs)}')
            print(f'Matches Save Path:\t\t{self.savePath}')
            print(f'Matches Error Path:\t\t{self.error_dir}')
            print(f'Image Error Path:\t\t{self.targetImgDataObjs[0].error_dir}')
            print('\n'), print("Ready to get_matches!".center(term_width))

            print('=' * term_width), print('\n\n')

        def print_abort():
            term_width = get_terminal_size()[0]
            print('\n\n'), print('=' * term_width * 2), print(f'ERROR!'.center(term_width))

            print(f"chk_1:\t{chk_1}\tTarget Path\t\t{type(self.targetPath)}")
            print(f"chk_2:\t{chk_2}\tSearch Path\t\t{type(self.targetPath)}")
            print(
                f"chk_3:\t{chk_3}\tTarget Img Paths\t{type(self.targetImgPaths)}\t{len(self.targetImgPaths)}")
            print(
                f"chk_4:\t{chk_4}\tSearch Img Paths\t{type(self.targetImgPaths)}\t{len(self.searchImgPaths)}")
            print(f"chk_5:\t{chk_5}\tTarget Img Convert\t{len(self.targetImgDataObjs)}")
            print(
                f"chk_6:\t{chk_6}\tSave Path\t{path.dirname(self.savePath)}\t\t{path.basename(self.savePath)}")
            print(
                f"chk_7:\t{chk_7}\tError Path\t{path.dirname(self.error_dir)}\t{path.basename(self.error_dir)}")
            print(f"ready_lst:\t{ready_lst}")

            print('=' * term_width), print('\n\n')

            raise SystemError("Please check the invalid path before continuing!\n")

        if ready:
            return print_ready()
        else:
            return print_abort()

    def _save_error(self, reading_path=None):
        if reading_path:
            error_obj = {
                "error": f"FILE COULD NOT BE FOUND!",
                "path": reading_path
                }

            add_to_json(error_obj, self.error_dir, is_error=True)
            raise Exception(error_obj)


if __name__ == "__main__":
    from Private import checking_dirs, targets, save_path

    # CHANGE LINE 47 TO ALTER target IMG SIZE!!!
    f = FlannRunner(target_path=targets,
                    search_path=checking_dirs,
                    target_limiter=None,  # (420, 425),
                    search_limiter=(0, 10),
                    save_path=save_path,
                    error_dir=None,
                    threshold=.16,
                    run_parallel=True,
                    parallel_cores=4)

    f.get_matches()
