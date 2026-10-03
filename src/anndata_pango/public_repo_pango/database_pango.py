
import requests
import asyncio
from tqdm import tqdm
from platformdirs import user_data_dir
from pathlib import Path
from urllib.parse import urlparse, unquote
from playwright.async_api import (async_playwright,
                                  Error as PlaywrightError)

from .utils_database_pango import (load_data_sources,
                                   open_chrome)

class DataBase_Pango:

    def __init__(self,
                 save_path:str = None):

        # load data_sources
        self.data_sources = load_data_sources()

        # add save path
        if save_path is None:
            save_path = user_data_dir("public_repo_pango")+"/database"
        self.save_path = save_path

    def download(self,
                 dataset:str,
                 download_pipe:str="chrome",
                 asynchronous:bool=True):

        # download through chrome
        if download_pipe == "chrome":
            if asynchronous:
                proc = download_from_chrome_pango(url=self.data_sources[dataset],
                                                  download_path=self.save_path)

                return proc
            else:
                asyncio.run(download_from_chrome_pango(url=self.data_sources[dataset],
                                                       download_path=self.save_path))
        # download through requests
        elif download_pipe == "requests":
            download_from_requests(url=self.data_sources[dataset],
                                   download_path=self.save_path)            

        return None

class Chrome_Session_Pango:

    def __init__(self,
                 chrome_path:str=None,
                 usr_dir:str=None,
                 download_path:str=None,
                 headless:bool=False,
                 remote_port:int=9527):

        # build slot to store params
        self.chrome_path = chrome_path
        self.usr_dir = usr_dir
        self.download_path=download_path
        self.headless=headless
        self.remote_port=remote_port
        self._chrome_proc=None

        # verify download_path
        if self.download_path is None:
            self.download_path=user_data_dir("public_repo_pango")+"/database"
        Path(self.download_path).mkdir(parents=True, exist_ok=True)

    def __enter__(self):

        self._chrome_proc = open_chrome(chrome_path=self.chrome_path,
                                        usr_dir=self.usr_dir,
                                        headless=self.headless,
                                        remote_port=self.remote_port)

        return self

    def __exit__(self, exc_type, exc, tb):

        # kill the chorme process
        self._chrome_proc.terminate()
        self._chrome_proc.wait()

        return False

async def download_from_chrome_pango(url:str,
                                     download_path:str=None,
                                     headless:bool=False):

    # extract the file name
    path = urlparse(url).path
    filename = unquote(path.split("/")[-1])

    # set the remote port
    remote_port=9527
    # register the chrome process
    with Chrome_Session_Pango(remote_port=remote_port,
                              download_path=download_path,
                              headless=headless) as csp:

        # register the playwright process
        async with async_playwright() as p:

            # connect to chrome
            browser = await p.chromium.connect_over_cdp(f"http://localhost:{remote_port}")

            # get the browser context
            context = browser.contexts[-1] if browser.contexts else await browser.new_context()

            # open a new page
            page = await context.new_page()

            # triger download
            async with page.expect_download() as dl:
                try:
                    await page.goto(url)
                except PlaywrightError:
                    pass

            # save the download information
            download = await dl.value
            await download.save_as((Path(csp.download_path)/filename))
            print(csp.download_path,filename)
            print(await download.path())

            # kill the chorme process
            await browser.close()

    return None

def download_from_requests(url:str,
                           download_path:str=None):

        # verify download_path
        if download_path is None:
            download_path=user_data_dir("public_repo_pango")+"/database"
        Path(download_path).mkdir(parents=True, exist_ok=True)

        # extract the file name
        path = urlparse(url).path
        filename = unquote(path.split("/")[-1])

        # submit the requests
        with requests.get(url=url, 
                          stream=True, 
                          timeout=30) as r:
            # raise error when encounter the http error
            r.raise_for_status() 

            # get the total size
            total = int(r.headers.get("Content-Length", 0))

            # write to local chunk by chunk
            with open((Path(download_path)/filename), "wb") as f,tqdm(
                total=total,
                unit="B",
                unit_scale=True,     
                unit_divisor=1024,
                desc=(Path(download_path)/filename)) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return None