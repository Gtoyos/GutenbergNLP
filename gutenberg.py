#This module is used to get the books from Gutenberg's server.
#It downloads the selected books in case they not exist in the local directory.

import pandas as pd
from tqdm.autonotebook import tqdm
import os.path
import requests
import re
import nltk
nltk.download('punkt')
import string
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
import numpy as np

class Gutenberg(Dataset):
    def __init__(self,catalog_file="catalog.csv",check_integrity=True,books_folder="books",
        book_transform=None,subject_transform=None):
        self.book_transform = book_transform
        self.subject_transform = subject_transform
        self.books_folder = books_folder

        if not type(catalog_file) is str:
            self.catalog = catalog_file
            return
        if(not os.path.isfile(".processed_"+catalog_file)):
            if(os.path.isfile(catalog_file)):
                print("Using existing",catalog_file)
            else:
                print("Downloading catalog...")
                open(catalog_file,"wb").write(requests.get("https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv").content)
            self.catalog = pd.read_csv(catalog_file)
            self.catalog = self._extractsample()
            self.catalog.to_csv(".processed_"+catalog_file)
        else:
            print("Using cached catalog...")
            self.catalog = pd.read_csv(".processed_"+catalog_file)
            self.catalog = self.catalog.set_index('Text#')
        if(not os.path.isdir(books_folder)):
            os.mkdir(books_folder)
        print("Downloading & loading books... This may take a while")
        failed = []
        for book_id in tqdm(self.catalog.index):
            if(not os.path.isfile(books_folder+"/"+str(book_id)+".txt")):
                if(self._downloadbook(book_id)>0):
                    print("Warning: Couldn't download book",book_id)
                    failed.append(book_id)
            elif(check_integrity):
                if(self._checkbookintegrity(book_id)>0):
                    print("Warning: Can't read book",book_id)
                    failed.append(book_id)
        self.catalog = self.catalog.drop(index=failed)
        print("Dataset & catalog are ready!")
        if(len(failed)>0):
            print("Warning:",len(failed),"books weren't downladed/loaded successfully and are not included in the catalog.")

    def _extractsample(self):
        #Get only english books.
        catalog = self.catalog[self.catalog.Type == "Text"]
        catalog = catalog[catalog.Language == "en"]
        catalog = catalog[["Text#","Title","Subjects"]]
        catalog = catalog.dropna()
        catalog = catalog.set_index('Text#')
        #Extract subjects from multivalued column
        def exists_cat(v,row):
            l = re.split('; |-- |,',str(row["Subjects"]))
            l = [x.strip(" ')(") for x in l]
            l = [x for x in l if not any(c.isdigit() for c in x)]
            l = [x.split("(")[0].strip() for x in l]
            for y in v:
                if y in l:
                    return 1
            return 0
        for k,v in subject_map.items():
            catalog[k] = catalog.apply(lambda row: exists_cat(v,row), axis=1)
        catalog = catalog.drop("Subjects",axis=1)

        #Check if there are books that have no category, create a special label for them
        def istherecat(keys,row):
            for k in keys:
                if row[k]==1:
                    return 0
            return 1
        catalog["Others"] = catalog.apply(lambda row: istherecat(subject_map.keys(),row),axis=1)

        #We observed that categories are imbalanced. We take a proportional sample for each category
        sps = []
        for k in subject_map.keys():
            sp = catalog[catalog[k]==1]
            sp = sp.sample(min(1000,len(sp)),random_state=42)
            sps.append(sp.index)
        sp = catalog[catalog["Others"]==1]
        sp = sp.sample(min(1000,len(sp)),random_state=42)
        sps.append(sp.index)
        sampled = [item for sublist in sps for item in sublist]
        catalogsample = catalog[catalog.index.isin(sampled)]

        #We now have a catalog with better balanced categories for each selected subject.
        return catalogsample

    def _downloadbook(self,book_id,tries=3):
        if(book_id in BANNEDBOOKS):
            return 2
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        urls = [
            "https://www.gutenberg.org/cache/epub/"+str(book_id)+"/pg"+str(book_id)+".txt",
            "https://www.gutenberg.org/files/"+str(book_id)+"/"+str(book_id)+".txt",
            "https://www.gutenberg.org/files/"+str(book_id)+"/"+str(book_id)+"-0.txt",
            "https://www.gutenberg.org/cache/epub/"+str(book_id)+"/pg"+str(book_id)+"-0.txt",
        ]
        while(tries>0):
            for u in urls:
                response = requests.get(u,headers=headers)
                if(response.status_code == 200):
                    open(self.books_folder+"/"+str(book_id)+".txt", "wb").write(response.content)
                    return 0
            tries-=1
        return 1

    def _checkbookintegrity(self,book_id):
        try:
            self.getBook(book_id)
            return 0
        except Exception as e:
            return 1

    def _cleanbook(self,txt):
        while(True):
            start = max([txt.find(COPYRIGHT0),txt.find("*** START OF "),txt.find("***START OF")])
            if start==-1:
                break
            txt = txt[start:]
            txt = txt[txt.find("\n"):]
        while(True):
            end = min([txt.find("*** END OF"),txt.find("***END OF")])
            if end==-1:
                break
            txt = txt[:end]
        return txt
    
    def getBook(self,book_id):
        with open(self.books_folder+"/"+str(book_id)+".txt") as f:
            txt = f.read()
        return self._cleanbook(txt)

    def getBookExtract(self,book_id,max_words=-1):
        with open("books/"+str(book_id)+".txt") as f:
            txt = f.read()
        txt = self._cleanbook(txt)
        tokens = word_tokenize(txt, language='english')
        tokens = list(filter(lambda token: token not in string.punctuation, tokens))
        gap = int(len(tokens)*0.1)
        tokens = tokens[gap:]
        if max_words>0:
            return tokens[:max_words]
        return tokens

    def getCatalog(self,with_embedding=False):
        if with_embedding:
            failed = []
            for book_id in self.catalog.index:
                if(self.getBookEmbedding(book_id) is None):
                    failed.append(book_id)
            return self.catalog.drop(index=failed)
        else:
            return self.catalog

    #Get book embedding, if it exsits.
    def getBookEmbedding(self,book_id):
        try:
            return np.load(self.books_folder+"/emb-"+str(book_id)+".npy")
        except Exception as e:
            return None

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self,idx):
        book = self.getBook(self.catalog.index[idx])
        subjects = self.catalog.iloc[idx,1:]
        if self.book_transform:
            book = self.book_transform(self.catalog.index[idx],book)
        if self.subject_transform:
            subjects = self.subject_transform(subjects)
        return book, subjects


#This is the selected categories that we want to calssify. 
subject_map = {
    "Juvenile fiction": ["Juvenile fiction"],
    "History": ["History","Natural history","Civil war","World War","History and criticism"],
    "Poetry": ["Poetry","English poetry"],
    "Politics and gouvernment": ["Politics and government"],
    "Cooking": ["Cooking"],
    "Mistery": ["Mistery","Mystery fiction","Detective and mystery stories"],
    "Philosophy": ["Philosophy"],
    "Christian": ["Christian life","Christian"],
    "Love stories": ['Love stories'],
    "Periodicals": ['Periodicals'],
    "Humor": ['English wit and humor','Humorous stories'],
    "Travelling": ['Description and travel','Voyages and travels',"Travel"],
    "Correspondence": ["Correspondence"],
    "Adventure": ['Adventure stories','Adventure and adventurers'],
    "Drama": ["Drama"],
    "Biography": ['Biography'],
    "Historical fiction": ['Historical fiction'],
    "Science fiction": ['Science fiction'],
    "Fantasy fiction": ['Fairy tales','Fantasy fiction'],
    'Science': ['Science']
}

#Strings used for cleaning books
COPYRIGHT0 = '''Project Gutenberg Etexts or other materials be they
hardware or software or any other related product without express
permission.'''

# This books can't be downloaded properly. Ignore them.
BANNEDBOOKS = [928,9942,10547,20073,25222,25387,26147,29785,30174,37157,37354,41654]

#TODO:
#-improve book cleaning?