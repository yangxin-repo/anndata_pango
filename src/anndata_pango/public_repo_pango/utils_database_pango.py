
import json
import subprocess
import shlex
from importlib.resources import files
from platformdirs import user_data_dir
from pathlib import Path

def load_data_sources():

    # find the data_sources file
    config_file = files("anndata_pango.public_repo_pango").joinpath("data_sources.json")
    
    # load the data_sources file
    with config_file.open("r", encoding="utf-8") as f:
        return json.load(f)

def open_chrome(chrome_path:str=None,
                usr_dir:str=None,
                headless:bool=False,
                remote_port:int=9527):

    # verify chrome_path,usr_dir and download_path
    if chrome_path is None:
        chrome_path = "google-chrome-stable"

    if usr_dir is None:
        usr_dir = user_data_dir("public_repo_pango")+"/chrome_usr_dir"
    Path(usr_dir).mkdir(parents=True, exist_ok=True)

    # the bash script
    if headless:
        script_bash = [chrome_path,
                       f"--user-data-dir={usr_dir}",
                       f"--remote-debugging-port={remote_port}",
                       f"--headless=new"]
    else:
        script_bash = [chrome_path,
                       f"--user-data-dir={usr_dir}",
                       f"--remote-debugging-port={remote_port}"]

    # establish the chorme process
    proc = subprocess.Popen(["bash","-c",shlex.join(script_bash)],
                            start_new_session=True, 
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


    return proc


