from pathlib import Path

def file_exist(file_path):
    return Path(file_path).is_file()

def check_file_exist(file_path):
    assert file_exist(file_path)

def dir_exist(dir_path):
    return Path(dir_path).is_dir()

def check_dir_exist(dir_path):
    assert dir_exist(dir_path)

def remove_file(file_path):
    assert file_exist(file_path)
    file_path_obj = Path(file_path)
    file_path_obj.unlink() # missing_ok=False for Python>=3.4

def get_file_name_and_suffix(file_name):
    import re
    if file_name.endswith("/") or file_name.endswith("\\"):
        raise Exception()
    match_result = re.match(r"(.*)\.(.*)", file_name)
    if match_result is None:
        return file_name, ""
    else:
        return match_result.group(1), match_result.group(2)

def get_file_path_without_suffix(file_path, EndWithSlash=True):
    file_path_no_suffix, suffix = get_file_name_and_suffix(file_path)
    return file_path_no_suffix

def current_script_path_without_suffix(script_file_path):
    return get_file_path_without_suffix(script_file_path)