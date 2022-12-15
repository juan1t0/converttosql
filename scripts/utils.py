import json
import pandas as pd
import time
import re

import unicodedata
import math

from deep_translator import GoogleTranslator
from cdifflib import CSequenceMatcher

SOS_token = 0
EOS_token = 1

symbols = {'<': 'lessthan', '>':'greaterthan',
           '<=':'greaterthanorequalto', '>=':'greaterthanorequalto',
           '!=':'notequalto', '=':'equalto', '<>':'notequalto',
           '&':'and','|':'or','^':'xor',
           '*':'all','.':' in '}

class Dataset:
    '''
    class that encapsule the processing of the dataset and maintain the database structure

    '''
    def __init__(self, data_path, structure_path):
        self.DataPath = data_path
        self.StructurePath = structure_path
        self.load_data()
    
    def load_data(self):
        datafile = open(self.DataPath, 'r')
        strcfile = open(self.StructurePath, 'r')
        self.Data = json.load(datafile)
        self.Structure = self.del_tables_redundance(json.load(strcfile))

    def del_tables_redundance(self, structure_dict):
        tables = {}
        for table in structure_dict:
            columns = set()
            for col in table['column_names']:
                columns.add(col[1])
            tables[table['db_id']] = list(columns)
        return tables
    
    def gen_dataframe(self, transform=False, translate=None, onlysql=False):
        nls = []
        sql = []
        tbs = []
        for sample in self.Data:
            nls.append(sample['question'])
            sql.append(sample['query'])
            tbs.append(sample['db_id'])
        
        if transform:
            nls, sql = self.transform_data(nls, sql)
        if translate:
            nls = self.translate_data(nls, translate)
        if onlysql:
            idx = self.evaluate_sql(sql)
            for i in reversed(idx):
                del nls[i]
                del sql[i]
                del tbs[i]

        self.DataFrame = pd.DataFrame({'NL_query': nls, 'SQL_query':sql, 'SQLtable':tbs})
        self.InputHeader = 'NL_query'
        self.TargetHeader = 'SQL_query'
        self.TableIdHeader = 'SQLtable'
        return self.DataFrame

    def transform_data(self, data_list, sql_list):
        self.Transformed = True
        new_data_list = []
        new_sql_list = []
        for dline in data_list:
            nline = ''
            for w in dline.split():
                if bool(re.search(r'\d', w)):#w.isnumeric():
                    w = ' '.join([*w])
                nline += w + ' '
            new_data_list.append(nline)
        
        for dline in sql_list:
            nline = ''
            for w in dline.split():
                if w.isnumeric():
                    w = ' '.join([*w])
                nline += w + ' '
            new_sql_list.append(nline)

        return new_data_list, new_sql_list
        
    def translate_data(self, data_list, language):
        self.Translated = True
        self.Translator = GoogleTranslator(source='auto', target=language)
        new_data_list = []
        for i, dline in enumerate(data_list):
            new_data_list.append(self.Translator.translate(dline))
            if (i+1)%500 == 0:
                print(f'translated {(i+1)*100/len(data_list):.2f}%')
        return new_data_list
        
    def evaluate_sql(self, sql_list):
        pass

    def get_columns(self, table_id):
        return self.Structure[table_id]

#
def as_minutes(s):
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds'%(m,s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s/(percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

class Corpus:
    def __init__(self, tables=None):
        self.Word2Index = {}
        self.WordsCount = {}
        self.Index2Word = {0: 'SOS', 1: 'EOS'}
        self.NWords = 2

        if tables:
            self.Tables = tables
            for tab in self.Tables:
                for col in self.Tables[tab]:
                    self.add_word(col.replace(' ',''))
    
    def add_word(self, word):
        if word not in self.Word2Index:
            self.Word2Index[word] = self.NWords
            self.WordsCount[word] = 1
            self.Index2Word[self.NWords] = word
            self.NWords += 1
        else:
            self.WordsCount[word] += 1
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

def unicode_2_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s):
    s = unicode_2_ascii(s.lower().strip())
    s = re.sub(r'([.!?|])', r' \1', s)
    s = re.sub(r'([_])', r'', s)
    s = re.sub(r'[^a-zA-Z0-9.?|]+', r' ', s)
    return s

def replace_symbols_sql(sql_query):
    for s in symbols:
        if s in sql_query:
            sql_query = sql_query.replace(s, symbols[s])
    return sql_query

def similars(sentence, lis, topk=5):
    ratios = {}
    for el in lis:
        for s in sentence.split():
            if el not in ratios:
                ratios[el] = 0.0    
            ratios[el] += CSequenceMatcher(None, s, el).ratio()
    ratios = {k: v for k, v in sorted(ratios.items(), key=lambda item: item[1])}
    return list(ratios.keys())[-topk:]

def read_row(row, dataset,reverse=False):
    in_query = row[dataset.InputHeader]
    output = row[dataset.TargetHeader]
    tableid = row[dataset.TableIdHeader]
    columns = [c.replace(' ','') for c in dataset.get_columns(tableid)]
    columns = ' '.join(similars(in_query, columns))
    sql_query = replace_symbols_sql(output)
    return [in_query +' | '+ tableid +' '+ columns, sql_query]

def create_corpuses_from_dataframe(dataset, reverse=False):
    df = dataset.DataFrame
    pairs = [[normalize_string(s) for s in read_row(
        r,dataset,reverse)] for _,r in df.iterrows()]
    
    return Corpus(dataset.Structure), Corpus(dataset.Structure), pairs

def filter_pair(pair, max_len):
    return len(pair[0].split()) < max_len and len(pair[1].split()) < max_len

def filter_pairs(pairs, max_len):
    return [pair for pair in pairs if filter_pair(pair, max_len)]

def prepare_data_from_dataframe(dataset, max_len):
    input_corpus, output_corpus, pairs = create_corpuses_from_dataframe(dataset)
    pairs = filter_pairs(pairs, max_len)
    for pair in pairs:
        input_corpus.add_sentence(pair[0])
        output_corpus.add_sentence(pair[1])
    
    print('Input corpus with %d words'%(input_corpus.NWords))
    print('Output corpus with %d words'%(output_corpus.NWords))

    return input_corpus, output_corpus, pairs