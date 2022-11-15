import json
import os

from Private import DATA_DIR

MASTER_DATA_FILE = "MAIN.json"


def read_json_data(f_name=MASTER_DATA_FILE):
    """
    Used to read json data files. If the file name is
    not found, an empty list will be returned instead.
    """

    def read_file(f):
        if os.path.isfile(f):
            try:
                with open(f, 'r') as rFile:
                    obj = json.load(rFile)
                    rFile.close()
                    return list(obj)
            except:
                raise Exception(f"ERROR\tUNKNOWN ERROR READING!\nCould not find file:\t{f}")
        else:
            print('\n', f'Could not read/find file:\t{f}\nReturning empty list...\n')
            return []

    def get_file_ext(f):
        """ `f` can be a file or path-like ending in a file with an extension. """
        return os.path.splitext(f)[1]

    print('\n', f'Attempting to find file:\t\t{f_name}')

    # fName is file: read file -> return contents
    if os.path.isfile(f_name):
        contents = read_file(f_name)
        return contents

    # fName is string and not directory (ie: 'test.json' or 'test')
    # (assumes intent was to use DATA_DIR + fName as file) ->
    #   Check if ends in .json:
    #       True: Should be file -> Return file content or empty list
    #       False: Check if fName has an extension:
    #           True: Can't be use, throw
    #           False: Add '.json' to DATA_DIR + fName ->
    #               Should be file -> Return file content or empty list
    if type(f_name) is str and not os.path.isdir(f_name):
        # Join fName with Data_Dir
        data_dir_f_name = os.path.abspath(os.path.join(DATA_DIR, f_name))

        # Ends in .json?
        if f_name[-5:] == '.json':
            contents = read_file(data_dir_f_name)
            return contents
        else:
            # Check for any filetype extension in fName - wont be '.json'
            if len(get_file_ext(data_dir_f_name)) > 0:
                print(f"ERROR:\tCANNOT READ FILE:\n\t{data_dir_f_name}\n\tAS JSON FILE!")
                return None
            else:
                new_f_name = f_name + '.json'
                print(f'No extension detected on input:\t{f_name}\nChanging to:\t\t\t{new_f_name}')
                print('Trying again...'.center(os.get_terminal_size()[0]))

                contents = read_json_data(new_f_name)
                return contents


def write_json_data(data=None, f_name=MASTER_DATA_FILE, indent=4):
    if data is None:
        data = []

    os.path.normpath(f_name)
    json_obj = json.dumps(data, indent=indent)

    if os.path.isfile(f_name):
        with open(f_name, 'w') as wFile:
            wFile.write(json_obj)
            print(f'\n\nSaved to {f_name}')
    else:
        try:
            with open(os.path.join(DATA_DIR, f_name), 'w') as wFile:
                wFile.write(json_obj)
        except:
            raise FileNotFoundError(f"ERROR SAVING!\nCould not find file '{f_name}'")

    return json_obj


def add_to_json(data, f_name=MASTER_DATA_FILE, indent=4, is_error=False):
    """Appends matching data objects/dicts to an existing data-file list-object and saves it."""
    existing_data = read_json_data(f_name)

    if is_error:
        existing_data.append(data)
        return write_json_data(existing_data, f_name, indent)

    # search `existing_data` for same `target`, append if found
    if len(existing_data) > 0:  # would be empty list if file not found.
        found_same_target = list(filter(lambda x: 'target' in x.keys()
                                                  and x['target'] == data['target'], existing_data))

        if len(found_same_target) > 0:
            for tar in found_same_target:
                ex_idx = existing_data.index(tar)

                # Add newest matches to existing matches & sort
                [tar['matches'].append(m) for m in data['matches'] if m not in tar['matches']]
                tar['matches'] = sorted(tar['matches'], key=lambda x: x['path'])

                # Add additional paths_checked & sort
                tmp_pth_chk = set(tar['paths_checked'])
                [tmp_pth_chk.add(chkPth) for chkPth in data['paths_checked']]
                tar['paths_checked'] = sorted(list(tmp_pth_chk))

                # Update 'stats'
                original_stats = tar['stats']
                new_stats_updated = data['stats']['saved_at']

                found_updated = list(filter(lambda k: k[:7] == 'updated',
                                            list(original_stats.keys())))
                amt_updated = len(found_updated)

                tmp_update_obj = {f'updated_{amt_updated + 1}': new_stats_updated}
                tar['stats'].update(tmp_update_obj)

                # Check for new stat keys in `data['stats']`
                [tar['stats'].update({k: v})
                 for k, v in data['stats'].items()
                 if k not in original_stats.keys()]

                # Replace `target` obj inside of `existing_data`
                existing_data[ex_idx] = tar
        else:
            existing_data.append(data)
    else:
        existing_data.append(data)

    # Sort existing_data MAIN list by `target`,`path` in dict
    sorted_data = sorted(existing_data, key=lambda d: os.path.basename(d['target']['path']))

    ret = write_json_data(sorted_data, f_name, indent)
    return ret
