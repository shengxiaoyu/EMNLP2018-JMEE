#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import sys

__doc__ = 'description'
__author__ = '13314409603@163.com'

def main(dir,save_file):
    fw = open(save_file,'w',encoding='utf8')
    for file_name in os.listdir(dir):
        with open(os.path.join(dir, file_name), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                jl = json.loads(line, encoding="utf-8")
                words = jl["words"]
                tags = ['O' for _ in words]
                events = jl['golden-event-mentions']
                for event in events:
                    trigger = event['trigger']
                    for index in range(trigger['start'],trigger['end']):
                        tags[index] = event['event_type']
                for word,tag in zip(words,tags):
                    fw.write('('+word+','+tag+') ')
                fw.write('\n')

if __name__ == '__main__':
    main(r'E:\pyWorkspace\EMNLP2018-JMEE\samples_label\test',r'C:\Users\13314\Desktop\Bi-LSTM+CRF\labeled\test.txt')
    sys.exit()
