#!/usr/bin/env python3

try:
    import os, pathlib, shutil
    import tempfile
    import re
    import queue
    import pandas as pd
    from requests_html import HTMLSession
    from bs4 import BeautifulSoup
    from selenium import webdriver
    from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.wait import WebDriverWait
except (ImportError, AttributeError, NameError, ModuleNotFoundError): 
    raise Exception('Unknown module')   


''' GENERATE A MOCK DATASET TO BE USED WITH MarTo.py (i.e. having the required schema)
    
    ---
    1. Collect 'Back to the Future III' movie data from web using Selenium and BeautifulSoup
        - subtitles (SRC_SRT) contain timestamps and dialogues
        - script (SRC_SCRIPT) contains parts and dialogues
    2. Process raw data and concatenate on key 'dialogue'
    3. Store in a table using Pandas and output in xls format
    
    ---
    NB. <class CollectDataFromWeb> implements selenium.webdriver() with Firefox. 
        To use another web browser, refer to the Selenium 4.0 documentation.
'''


# URLs web src 
SRC_SRT = "https://yts-subs.com/subtitles/back-to-the-future-part-iii-1990-english-yify-129087"
SRC_SCRIPT = "https://movies.fandom.com/wiki/Back_to_the_Future_Part_III/Transcript"

# Path to xls storage
DIR_XLS_DATA = pathlib.Path(__file__).parent.joinpath('data/bttf.xlsx').resolve()



class Toolbox(object):
    @staticmethod
    def staging_tmpDir_create() -> pathlib.Path:
        staging_dir = tempfile.mkdtemp(dir=pathlib.Path(__file__).parent.resolve())
        return pathlib.Path(staging_dir)

    @staticmethod
    def staging_tmpDir_clear(staging_dir) -> None:
        shutil.rmtree(staging_dir)

    @staticmethod
    def unzip_archive(staging_dir) -> None:
        for archive in staging_dir.glob('*.zip'):
            shutil.unpack_archive(archive, staging_dir)
            os.remove(archive)
        for srt in staging_dir.glob('*srt'):
            os.rename(srt, staging_dir.joinpath('bttf_srt.txt'))
    
    @staticmethod
    def get_txt_files(staging_dir) -> list[pathlib.Path]:
        files = []
        for txt in staging_dir.glob('*.txt'):
            files.append(txt)
        return files
    
    @staticmethod
    def numbered_file(format) -> pathlib.Path:
        counter = 0
        while True:
            counter += 1
            path = DIR_XLS_DATA.parent.joinpath(f'bttf_({counter}).{format}')
            if not path.exists():
                return path


class CollectDataFromWeb(object):
    def __init__(self, staging_dir:pathlib.Path):
        self.staging_dir = staging_dir
    
    # Execute a JS function using Firefox engine 
    # to generate a get request with proper headers 
    def download_srt_archive(self, url:str=SRC_SRT) -> None:
        options = Options()
        # 0 -> desktop; 1 -> default directory; 2 -> user defined directory
        options.set_preference("browser.download.folderList", 2)
        options.set_preference('browser.download.dir', str(self.staging_dir))
        driver = webdriver.Firefox(options=options) 
        driver.get(url)
        e = driver.find_element(by=By.ID, value='btn-download-subtitle')
        driver.execute_script('arguments[0].click();', e)
        WebDriverWait(driver, timeout=3)
        driver.quit()
        Toolbox.unzip_archive(self.staging_dir)

    # Extract hard-coded script from a webpage
    def download_script_txt(self, url:str=SRC_SCRIPT) -> None:
        session = HTMLSession()
        raw_html = session.get(url)
        if raw_html.status_code == 200:
            parsed_html = BeautifulSoup(raw_html.content, 'html5lib')\
                            .find('div', class_="mw-parser-output")\
                            .text
        with open(self.staging_dir.joinpath('bttf_script.txt'), 'w+', encoding='utf-8') as f:
            f.write(parsed_html)
    
    # Runner function
    def collect(self) -> None:
        self.download_srt_archive()
        self.download_script_txt()


class ProcessData(object):
    def __init__(self, staging_dir:pathlib.Path):
        self.srt = staging_dir.joinpath('bttf_srt.txt')
        self.script = staging_dir.joinpath('bttf_script.txt')

    # Extract timecodes and dialogues
    def extract_srt(self) -> pd.DataFrame:
        q = queue.Queue()
        with open(self.srt, 'r+', encoding='utf-8') as f:
            for line in f.readlines():
                t = re.search(r'^([0-9]+:[0-9]+:[0-9]+)', line)
                if t != None: q.put(t.group())
                d = re.findall(r'^(?:[\D]+).', line)
                if len(d) > 0 : q.put(*d)
        table = []
        z, l = [], []
        c = 1
        while not q.empty(): 
            item = q.get()
            if not re.search(r'^[0-9]', item):
                item = re.sub(r"\'", ' ', item)
                l.append(item)
                c+=1 
            else: z.append(item)
            if len(z) > 1:
                table.append({z[0] : ' '.join(l[-c:])})
                z.pop(0)
                l.clear()
        table.append({z[0] : ' '.join(l[-c:])})
        df = pd.concat([pd.DataFrame.from_dict(_, columns=['dialogue'], orient='index') for _ in table])   
        df = df.reset_index().rename(columns={'index':'timecode'})

        # Targeted elements to remove
        if '@'or'encod' in df.iat[0, 0]: df = df[1:]
        
        return df[2:].reset_index(drop=True)
    
    # Extract parts and dialogues
    def extract_script(self) -> pd.DataFrame:
        q = queue.Queue() 
        with open(self.script, 'r+', encoding='utf-8') as f:
            for line in f.readlines():
                t = re.search(r'^[\w]+\:', line)
                if t != None: q.put(t.group())
                e = re.findall(r'\w[A-Za-z\s,\.\!\?\']{10,}(?!\:)\w.', line)
                if len(e) > 0 : q.put(*e)
   
        table = []
        z, l = [], []
        c = 1
        while not q.empty(): 
            item = q.get()
            if not re.search(r'^[\w]+(?=\:)', item):
                l.append(item)
                c+=1 
            else: 
                z.append(item.strip(':'))
            if len(z) > 1:
                for _ in l[-c:]:
                    table.append({z[0] : _})
                z.pop(0)
                l.clear()
        table.append({z[0] : ' '.join(l[-c:])})
        df = pd.concat([pd.DataFrame.from_dict(_, columns=['dial'], orient='index') for _ in table])   
        df = df.reset_index().rename(columns={'index':'part'})

        # Targeted elements to remove 
        df = df[df['part'] != '10']
        df = df[df['part'] != 'Scene']
        df = df.drop(index=[19, 20, 21, 23, 24, 25])
        #print(df)
        return df[3:].reset_index(drop=True)

    # Join timecodes and parts on dialogues
    def concat(self) -> pd.DataFrame:
        pd.set_option("display.min_rows", 100)
        df_srt = self.extract_srt()
        df_script = self.extract_script()
        df_concat = pd.concat([df_srt, df_script], axis=1)\
                      .drop(columns=['dial'], axis=1)

        return df_concat[['timecode', 'part', 'dialogue']]

    # Store dataframe in xls format
    def to_excel(self, path_out:pathlib.Path) -> None:
        df = self.concat()
        df = df.dropna().reset_index()
        
        if not path_out.exists(): path = path_out
        else: path = Toolbox.numbered_file('xlsx')
        
        with pd.ExcelWriter(path, engine='openpyxl') as w:
            df.to_excel(w, sheet_name='bttf')



def main():
    STAGING_DIRECTORY = Toolbox.staging_tmpDir_create()
    CollectDataFromWeb(staging_dir=STAGING_DIRECTORY).collect()
    ProcessData(staging_dir=STAGING_DIRECTORY).to_excel(path_out=DIR_XLS_DATA)
    Toolbox.staging_tmpDir_clear(staging_dir=STAGING_DIRECTORY)



if __name__ == "__main__":
    main()
