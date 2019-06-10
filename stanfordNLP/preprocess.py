#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from pyltp import SentenceSplitter

__doc__ = 'description'
__author__ = '13314409603@163.com'


import sys
import os
from stanfordcorenlp import StanfordCoreNLP
import stanfordnlp
from stanfordnlp.utils.resources import DEFAULT_MODEL_DIR

model_dir = r'E:\pyWorkspace\EMNLP2018-JMEE\stanford-corenlp-full-2018-10-05'
# nlp = StanfordCoreNLP(model_dir,lang='zh',quiet=False, logging_level=logging.DEBUG)
nlp = StanfordCoreNLP(model_dir,quiet=False, logging_level=logging.DEBUG)

def process():
    sentence = "The skeleton of a second baby has been found at a rural Wisconsin home where a 22-year-old woman 's dead infant was discovered in a blue plastic container June 8 officials said Monday."
    # sentence = "原、被告于2004年4月份相识并确立恋爱关系,2005年6月22日登记结婚。"
    words = nlp.word_tokenize(sentence)
    print(words)
    # poses = nlp.pos_tag(sentence)
    # print(poses)
    # print(nlp.ner(sentence))
    # pipe = stanfordnlp.Pipeline(models_dir=DEFAULT_MODEL_DIR,lang='zh')
    # stanfordnlp.download('zh',resource_dir=DEFAULT_MODEL_DIR,confirm_if_exists=True)
    # pipeline = stanfordnlp.Pipeline(lang='zh')
    # doc = pipeline(sentence)
    # print(doc.sentences[0])
    # print(nlp.parse(sentence))
    print(nlp.dependency_parse(sentence))

#使用stanford分词,分词结果全部存在一个文件中
def segment_words(source,savePath,stop_words_path):

    savePath = os.path.join(savePath,'result.txt')
    fw = open(savePath,'w', encoding='utf8')
    #停用词
    with open(stop_words_path, 'r', encoding='utf8') as f:
        stopWords = set(f.read().split())

    def handFile(path):
        if(os.path.isdir(path)):
            for fileNmae in os.listdir(path):
                handFile(os.path.join(path,fileNmae))
        else:
            handleSingleFile(path)

    def handleSingleFile(path):
        with open(path,'r',encoding='utf8') as f:
            content = f.read()
            sentences = SentenceSplitter.split(content)
            for str in sentences:
                str = str.strip()
                if(str==None or len(str)==0):
                    continue
                # 分词
                # print(str)
                words = nlp.word_tokenize(str)

                # 去停用词
                words = list(filter(lambda x:False if(x in stopWords) else True,words))
                if (len(words) == 0):
                    continue

                fw.write(' '.join(list(words)))
                fw.write('\n')
            fw.flush()

    #分别处理三类文件夹
    qstsbl = os.path.join(source, 'qstsbl')
    handFile(qstsbl)
    qsz = os.path.join(source)
    handFile(qsz)
    cpws = os.path.join(source)
    handFile(cpws)


if __name__ == '__main__':

    rootdir = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF\\'
    # segment_words(source=os.path.join(rootdir, '原始_待分句_样例'),
    #               savePath=os.path.join(os.path.join(rootdir, 'glove'), 'train'),
    #               stop_words_path=os.path.join(rootdir, 'litterStopWords.txt'))
    process()
    sys.exit()
