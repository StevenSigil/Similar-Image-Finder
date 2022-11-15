# Same Image Finder

A program that compares one target image to a/many directory(s) to search for closely-matching images using the [OpenCV](https://opencv.org/) packages, and sped up by being able to run on multiple cores thanks to [joblib](https://joblib.readthedocs.io/).

Currently, this program is setup to use the ORB key-point detection, see below the _"Setup"_ for details.

Results are not guaranteed, many hours were spent finding optimal parameters for key-point detection, FLANN recognition, and score comparisons. This works fine for the project it was built for, but lossless experimentation is always a good idea for your projects.

<br />

---

---

<br />

## Notes

- This program assumes you have a data directory for saving purposes above the directory containing _`Main.py`_. This will be where errors and matches will be saved in json format.

- SIFT/ORB key-point detection was decided to be used over other methods of similarity detection so it wouldn't affect the results if an image is rotated, warped, etc...

- Some of the script is currently not at peak optimization. For instance, when two identical images are to be matched, the best percentage of a match given thus far is around 40% using SIFT; ORB yields close to a 15% match.

- Not all functions in each class are used all the time. Some are helper functions when testing, such as the `show_img` function in the `ImgData` class. It can be used to show the picture of either the `target` or `search_image`. Due to needing the program to search entire directories quickly, and the potential for many matches, I opted to have all `FlannMatch` objects written to a json file for later, manual comparison.

- While valuable in the early stages of construction, the `save_error` functions are no longer very valuable and need to be redone.

---

<br />

## Setup

Below is an explanation of how I currently set input variables for the main running function. This was stored in `Private.py`, not included in git.

The first part of _Private.py_ are the directories that I intend to check against the target image. A single directory can be used as well, however multiple are accepted in list format.

#### _Private.py_

```python
# `searchPath` param setup - Add all directories you want to search for a match with target image(s) (ie: `targetPath`)

checkingDir1 = "F:/Path/To/Look/For/Images/01"
checkingDir2 = "F:/Path/To/Look/For/Images/02"
checkingDir3 = "F:/Path/To/Look/For/Images/03"

checking_dirs = [checkingDir1, checkingDir2, checkingDir3]
```

<br />

Secondly is the setup for the target image(s).

Note: if you only intend to check a single image against the directory(s), just enter the full path as the `target` image as a string or in list format. Then you can import that to the main script.

```python
target_img = ["C:/Path/To/Retrieve/An/Image.jpg" ]
```

<br />

My use case needed to match many potential targets with the same directories that were setup previously.

As noted in the code, I setup a specific directory holding all target images; entered in the file name of the first image to check; then wrote a small script to get all images in the directory and filter out images not between the index of the `start_img` up to the `amt_to_check`.

My use case also needed to unsure there was an extension for each file name, the extension was supported by OpenCV, and because some copies were made ending in `_00`, those were to be filtered out as well. Finally, my list of `valid_targets` were set as the variable `targets` to be imported in the main script.

#### _Private.py_

```python
# `targetPath` param setup

from os import listdir, path
import re
from Util import SUPPORTED_IMGS

target_dir = "C:/Path/To/Retrieve/Many/Target/Images"

# First image to check against search images in `target_dir`
start_img = "SCN_0001.jpg"

# Get all images in `target_dir` and find the index of `start_img`
target_dir_cont = [i for i in listdir(target_dir) if '.' in i]
start_img_idx = target_dir_cont.index(start_img)

# 5 targets will be checked against images in `checking_dirs`
amt_to_check = 5

# Adjust this lambda function to fit your needs. My `target_dir` has some unsupported images in the directory and I also don't want any images with `_00` on the end of the filename.
valid_targets = list(
    filter(
        lambda f_name: bool(
            re.search(re.compile('\.\w{3,4}$'), f_name).group()
            in SUPPORTED_IMGS and '_00' not in f_name), target_dir_cont))

# List of full paths pointing to `valid_target` images, restricted to the `amt_to_check` amount of images.
targets = [path.join(target_dir, f_name) for f_name
           in valid_targets[start_img_idx: start_img_idx + amt_to_check]]
```

<br />

Finally for `Private.py`, I declare my `DATA_DIR` and `SAVE_PATH` for saving the matching json data and errors.

#### _Private.py_

```python
# `savePath` param setup
DATA_DIR = "C:/Path/To/Data/Storing"
SAVE_PATH = path.join(DATA_DIR, "matches_01", "TEST_01.json")
```

<br />

Once these variables are setup, you can import them into the main script, and await results.

#### _Main.py_

```python
if __name__ == "__main__":
    from Private import checking_dirs, targets, save_path

    # Build the Runner object with input variables
    f = FlannRunner(target_path=targets,
                    search_path=checking_dirs,
                    target_limiter=None,  # Limits amt. of targets, of what is imported
                    search_limiter=None,  # Limits amt. of search images, per directory imported as `checking_dirs`
                    save_path=save_path,  # Path to save results to
                    error_dir=None,       # Defaults to `DATA_DIR`/errors/IMAGE_ERRORS.json or `DATA_DIR`/errors/RUN_ERRORS.json
                    threshold=.16,        #       Change for optimizations
                    run_parallel=True,    #   Parallel processes on CPU
                    parallel_cores=4)

    # Run matching program
    f.get_matches()
```

<br />

---

<br />

## ORB vs. SIFT

Currently the application is set to use ORB as the key-point detector as the licensing is easier to distribute and is much quicker. However, the accuracy when testing shows a significant decrease of true positives and of the positives matched, a lower score of matching key-points. If able to per the license on the SIFT package, I would recommend using that over ORB.

As noted in the script, additional parameters are needed when initializing the `FlannBasedMatcher` when using ORB:

#### _Main.py_

```python
class FlannMatch:
    """ ...
        ...

    SIFT index_params: dict(algorithm=1, trees=5)

    ORB index_params: dict(
        algorithm=6,
        table_number=6,
        key_size=12,
        multi_probe_level=1)
    """

    def __init__(self,
                ...
                ...

                # If using ORB:
                index_params= dict(
                    algorithm=6,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1),

                 # Can be changed for ORB or SIFT
                search_params= dict(checks=50)
    )
    ...
    ...

    # change to `self.sift = cv2.SIFT_create()`
    self.orb = cv2.ORB_create()

    # keep same regardless of ORB/SIFT
    self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
```

<br />

After setting up either ORB or SIFT, you will want to make sure the proper `detectAndCompute` function is used based on whichever you called the `create` function on.

### _Main.py_

```python
class FlannMatch:
    def __init__:
        ...

    def match_key_points(self):
        ...

        # When using ORB:
        kp1, desc1 = self.orb.detectAndCompute(self.img1.gray, None)
        kp2, desc2 = self.orb.detectAndCompute(self.img2.gray, None)

        # When using SIFT:
        # kp1, desc1 = self.sift.detectAndCompute(self.img1.gray, None)
        # kp2, desc2 = self.sift.detectAndCompute(self.img2.gray, None)
```
