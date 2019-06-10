#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__doc__ = 'description将brat生成的ann文件和源文件.txt结合，生成人工标注的样子的文件'
__author__ = '13314409603@163.com'

from stanfordnlp.server import CoreNLPClient
from stanfordcorenlp import StanfordCoreNLP
import copy
import os
import sys
import json



#将源文件和标注文件合一
def formLabelData(labelFilePath,savePath):

    if(not os.path.exists(savePath)):
        os.mkdir(savePath)
    def handlderDir(nlp,dirPath):
        for fileName in os.listdir(dirPath):
            newPath = os.path.join(dirPath, fileName)
            if (os.path.isdir(newPath)):
                handlderDir(nlp,newPath)
            else:
                handlerSingleFile(nlp,newPath)

    #获取event列表，其中包括每个事件含有的原文句子
    def getEvents(filePath,originFile):
        # 读取ann文件，获取标注记录
        relations = []  # 存储事件和参数关系Relation
        entitiesDict = {}  # 存储参数实体Entity
        with open(filePath, 'r', encoding='utf8') as fLabel:
            for line in fLabel.readlines():
                if (line.startswith('T')):
                    entity = Entity(line)
                    entitiesDict[entity.getId()] = entity
                if (line.startswith('E')):
                    relations.append(Relation(line))

        events = []  # 存储事件

        # 根据初始化的relations和entitiesDict完善entites的name，构造event
        for relation in relations:
            event = None
            for index, paramter in enumerate(relation.getParameters()):  # 形如[['Marry','T3'],['Time','T1']]
                if (index == 0):  # 第一个是描述事件触发词的entity
                    # 构造事件object
                    event = Event(relation.id, paramter[0])
                    # 获得触发词对应的entity
                    entity = entitiesDict.get(paramter[1])
                    entity = copy.copy(entity)

                    # 设置触发词的名称：事件类型_Trigger
                    entity.setName(paramter[0] + '_Trigger')
                    # 填入触发词
                    event.setTrigger(entity)
                else:
                    # 事件参数处理
                    entity = entitiesDict.get(paramter[1])
                    entity = copy.copy(entity)
                    entity.setName(event.getType() + '_' + paramter[0])
                    event.addArgument(entity)
            events.append(event)

        # 将事件按标注索引最小位排序
        events.sort(key=lambda x: x.getBegin())

        # 把每个事件涉及的原文语句填入
        for event in events:
            # eventBeginLineIndex = 0
            with open(originFile, 'r', encoding='utf8') as fData:
                cursor = 0
                for line in fData.readlines():
                    line = line.replace('\n', '\r\n')
                    beginIndexOfTheLine = cursor
                    endIndexOfTheLine = cursor + len(line)

                    # 标注起止范围都在在当前句子内
                    if (endIndexOfTheLine <= event.beginIndex):
                        cursor = endIndexOfTheLine
                        continue
                    if (beginIndexOfTheLine <= event.beginIndex and event.beginIndex <= endIndexOfTheLine
                            and beginIndexOfTheLine <= event.endIndex and event.endIndex <= endIndexOfTheLine):
                        event.addSentence(line)
                        event.setBeginLineIndex(beginIndexOfTheLine)
                        break
                    # 只有起始范围在当前句子
                    elif (beginIndexOfTheLine <= event.beginIndex and event.beginIndex <= endIndexOfTheLine and
                          endIndexOfTheLine < event.endIndex):
                        event.addSentence(line)
                        event.setBeginLineIndex(beginIndexOfTheLine)
                        # 只有截止范围在当前句子
                    elif (event.beginIndex < beginIndexOfTheLine and beginIndexOfTheLine <= event.endIndex and
                          event.endIndex <= endIndexOfTheLine):
                        event.addSentence(line)
                        break
                    cursor = endIndexOfTheLine
        return events
    # 分割成一个句子只标注一个事件
    def handlerSingleFile(nlp, filePath):
        if (filePath.find('.ann') == -1):
            return
        # 查看源文件是否存在，如果不存在直接跳过
        originFile = os.path.join(filePath, filePath.replace('.ann', '.txt'))
        if (not os.path.exists(originFile)):
            return
        print(filePath)
        # 获取事件list
        events = getEvents(filePath, originFile)
        if (events == None or len(events) == 0):
            return

        strs = conver_event_to_sample2(nlp, events)

        # 存储
        theSavePath = ''
        if (filePath.find('qsz') != -1):
            theSavePath = os.path.join(savePath, 'qsz_' + os.path.basename(filePath).replace('.ann', '.txt'))
        if (filePath.find('cpws') != -1):
            theSavePath = os.path.join(savePath, 'cpws' + os.path.basename(filePath).replace('.ann', '.txt'))
        if (filePath.find('qstsbl') != -1):
            theSavePath = os.path.join(savePath, 'qstsbl' + os.path.basename(filePath).replace('.ann', '.txt'))
        with open(theSavePath, 'w', encoding='utf8') as fw:
            fw.write('\n'.join(strs))
    # model_dir = r'C:\Users\13314\Desktop\Bi-LSTM+CRF\stanford-corenlp-full-2018-10-05'
    # nlp = StanfordCoreNLP(model_dir, lang='zh')
    annotators = ['tokenize', 'pos', 'lemma', 'ner', 'depparse']
    with CoreNLPClient(annotators=annotators, properties='zh', timeout=150000, memory='6G') as client:
        if (os.path.isdir(labelFilePath)):
            handlderDir(client,labelFilePath)
        else:
            handlerSingleFile(client,labelFilePath)
def del_br(event):
    sentence = ''.join(event.getSentences())
    # 去换行符
    br_index = sentence.find('\r\n')

    def update_index(entity, field_index, value):
        if (entity.getBegin() > field_index):
            entity.beginIndex = entity.getBegin() + value
        if (entity.getEnd() > field_index):
            entity.endIndex = entity.getEnd() + value

    while (br_index != -1):
        sentence = sentence[0:br_index] + sentence[br_index + 2:]
        update_index(event.getTrigger(), br_index, -2)
        if (event.getArguments() != None):
            for argu_entity in event.getArguments():
                update_index(argu_entity, br_index, -2)
        br_index = sentence.find('\r\n')
    event.sentence = sentence
def conver_event_to_sample2(client,events):
    last_sentence = ''
    last_words = None
    json_strs = []
    annotators = ['tokenize', 'pos', 'lemma', 'ner', 'depparse']
    result_dict = None
    for event in events:
        del_br(event)
        sentence = event.sentence
        if(sentence==last_sentence):
            '''同一个句子,多个事件'''
            event_mention = simple_event(event, last_words)
            result_dict['golden-event-mentions'].append(event_mention.__dict__)
        else:
            '''新句子，现把上一个句子的结果存入'''
            if(result_dict!=None):
                json_strs.append(json.dumps(result_dict, ensure_ascii=False))

            '''开始新的句子处理'''
            result_dict = {}
            ann = client.annotate(sentence, output_format='json', annotators=annotators)
            ann_sentence = ann['sentences'][0]

            '''依赖树'''
            stanford_colcc = []
            for item in ann_sentence['enhancedPlusPlusDependencies']:
                #root设为-1，全部减一
                stanford_colcc.append(item['dep']+'/dep='+str(item['governor']-1)+'/gov='+str(item['dependent']-1))
            result_dict['stanford-colcc'] = stanford_colcc

            '''ner'''
            golden_entity_mentions = []
            for entity in ann_sentence['entitymentions']:
                golden_entity_mentions.append({'start':entity['tokenBegin'],'end':entity['tokenEnd'],'entity-type':entity['ner'],'text':entity['text']})
            result_dict['golden-entity-mentions'] = golden_entity_mentions

            '''lemma,words,pos_tags'''
            lemma = []
            words = []
            pos_tags = []
            for item in ann_sentence['tokens']:
                lemma.append(item['lemma'])
                words.append(item['word'])
                pos_tags.append(item['pos'])
            result_dict['lemma'] = lemma
            result_dict['words'] = words
            result_dict['pos-tags'] = pos_tags

            event_mention = simple_event(event,words)
            result_dict['golden-event-mentions'] = [event_mention.__dict__]

            last_sentence = sentence
            last_words = words

    json_strs.append(json.dumps(result_dict, ensure_ascii=False))
    return json_strs

def convert_event_to_sample(nlp,event):
    del_br(event)
    sentence = event.sentence
    annotators = ['tokenize', 'pos', 'lemma', 'ner', 'depparse']
    properties = {'annotators': annotators, 'outputFormat': 'json'}
    words = list(nlp.word_tokenize(sentence))

    pos_tags = list(nlp.pos_tag(sentence))
    pos_tags = [x[1] for x in pos_tags]
    stanford_colcc = list(nlp.dependency_parse(sentence))
    stanford_colcc = list(map(lambda x: x[0] + '/dev=' + str(x[1]) + '/gov=' + str(x[2]), stanford_colcc))
    lemma = list(nlp.word_tokenize(sentence))
    event_mention = simple_event(event,words)
    result_dict = {}
    result_dict['stanford-colcc'] = stanford_colcc
    result_dict['lemma'] = lemma
    result_dict['words'] = words
    result_dict['pos-tags'] = pos_tags
    result_dict['golden-event-mentions']=[event_mention.__dict__]

    string = json.dumps(result_dict,ensure_ascii=False)
    return string

#构造停用词表，否定词不能作为停用词去掉
def stopWords(base_path):
    stopWords = set()
    stopPath = os.path.join(base_path,'stopWords')
    for file in os.listdir(stopPath):
        with open(os.path.join(stopPath,file),'r',encoding='utf8') as f:
            content = f.read()
            stopWords = stopWords.union(set(content.split('\n')))
    negativePath = os.path.join(base_path,'negativeWords')
    with open(os.path.join(negativePath,'dict_negative.txt'),'r',encoding='utf8') as f:
        negativeWords = set(map(lambda line:line.split('\t')[0],f.readlines()))
    stopWords = stopWords.difference(negativeWords)
    with open(os.path.join(base_path,'newStopWords.txt'),'w',encoding='utf8') as fw:
        fw.write('\n'.join(stopWords))

#记录一个标注体，形如T1   Person 17 19    双方
#表示标注体ID为T1，标注体类型为Person，标注范围为[17,19)，标注的值为“双方”
class Entity(object):
    def __init__(self,str):
        splits = str.strip().split('\t')
        self.id = splits[0]
        self.type = splits[1].split()[0] #参数类型，比如Person
        self.beginIndex = int(splits[1].split()[1])
        self.endIndex = int(splits[1].split()[2])
        self.value = splits[2]
        self.name = None #参数在具体事件中的名称，比如Participant_Person
    def getId(self):
        return self.id
    def getBegin(self):
        return self.beginIndex
    def getEnd(self):
        return self.endIndex
    def setName(self,str):
        self.name = str
    def getValue(self):
        return self.value
    def getName(self):
        return self.name
    def getType(self):
        return self.type

class Event(object):
    def __init__(self,id,type):
        self.id = id
        self.type = type
        self.arguments = []
        self.trigger = None

        #该事件标注索引最小最大位
        self.beginIndex = sys.maxsize
        self.endIndex = -1
        self.sentence = []
        self.beginLineIndex = 0 #该事件在原文本中涉及范围第一行的起始索引,因为将每个事件涉及的句子单独提出来之后，去标标签时需要指定这个句子在原文中的起始索引

        self.words = []
        self.tags = []
        self.posTags = []
    def addSentence(self,sentence):
        self.sentence.append(sentence)
    def setType(self,type):
        self.type = type
    def getType(self):
        return self.type
    def setTrigger(self,entity):
        self.trigger = entity
        self.beginIndex = min(self.beginIndex,entity.getBegin())
        self.endIndex = max(self.endIndex,entity.getEnd())
    def getTrigger(self):
        return self.trigger
    def addArgument(self,entity):
        self.arguments.append(entity)
        self.beginIndex = min(self.beginIndex,entity.getBegin())
        self.endIndex = max(self.endIndex,entity.getEnd())
    def getArguments(self):
        return self.arguments
    def getBegin(self):
        return self.beginIndex
    def getEnd(self):
        return self.endIndex
    def setBeginLineIndex(self,index):
        self.beginLineIndex = index
    def getBeginLineIndex(self):
        return self.beginLineIndex
    def getSentences(self):
        return self.sentence
    def setTags(self,tags):
        self.tags = tags
    def getTags(self):
        return self.tags
    def setWords(self,words):
        self.words = words
    def getWords(self):
        return self.words
    def setPosTags(self,posTags):
        self.posTags = posTags
    def getPosTags(self):
        return self.posTags

class simple_event(object):
    def __init__(self,event,words):
        self.trigger = {}
        trigger_entity = event.getTrigger()
        self.trigger['start'],self.trigger['end'] = sentence_index_to_word_index(words,trigger_entity.getBegin()-event.getBeginLineIndex(),trigger_entity.getEnd()-event.getBeginLineIndex())
        self.trigger['text'] = trigger_entity.getValue()

        self.arguments = []
        for argu_entity in event.getArguments():
            argu = {}
            argu['start'],argu['end'] = sentence_index_to_word_index(words,argu_entity.getBegin()-event.getBeginLineIndex(),argu_entity.getEnd()-event.getBeginLineIndex())
            argu['role'] = argu_entity.getName()
            argu['text'] = argu_entity.getValue()
            self.arguments.append(argu)

        self.event_type = event.getType()

#把在原句中的index转为分词后词语的index
def sentence_index_to_word_index(words,start,end):

    word_start = 0
    word_end = 0

    #游标
    begin_index = 0

    for index,word in enumerate(words):
        end_index = begin_index+len(word)

        #查看start是否在当前word范围内
        if(start>=begin_index and start<end_index):
            word_start = index
        if(end>=begin_index and end<end_index):
            word_end = index
            break
        begin_index = end_index
    if(word_end==word_start):
        word_end+=1
    return word_start,word_end

# 记录一个事件的关系，源数据形如：E1	Marry:T2 Time:T3 Participant:T1
# 表示事件Marry:T2,有参数Time:T3和Participant:T1
class Relation(object):
    def __init__(self,str):
        splits = str.split('\t')
        self.id = splits[0]
        self.parameters = list(map(lambda str:str.split(':'),splits[1].split())) #[[Marray,T2].[Time,T3}...]
    def getParameters(self):
        return self.parameters


def main():
    base_path = 'C:\\Users\\13314\\Desktop\\Bi-LSTM+CRF'
    brat_base_path = os.path.join(base_path, 'brat')
    save_path = r'E:\pyWorkspace\EMNLP2018-JMEE\samples_label'
    formLabelData(
        labelFilePath=os.path.join(brat_base_path, 'labeled'),
        savePath=save_path)

if __name__ == '__main__':
    main()
    print ('end')
    sys.exit(0)
