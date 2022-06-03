from cProfile import label
from distutils.command.config import config

from langcodes import Language
from factory.database import Database
import spacy
from spacy import displacy
from spacy.tokens import DocBin
import pandas as pd
from spacy.training.example import Example
from spacy.scorer import Scorer
from spacy.pipeline import EntityRuler

class SpacyTrain(object):
    def __init__(self):
        self.db = Database()
        self.collection_name = 'nlpdb'

    def ner2(self, first):
            if first == True:
                nlp = spacy.load('en_core_web_lg')
                TRAIN_DATA = pd.read_csv("data/trainData.csv",encoding='unicode_escape',  error_bad_lines=False, sep=';').values
            else:
                nlp = spacy.load(r".\output/model-best")
                TRAIN_DATA = self.db.get()
            db= DocBin()
            scorer = Scorer()
            examples = []
            losses = {}
            for id, text, category, start_position_1,start_position_2,start_position_3,stop_position_1,stop_position_2,stop_position_3 in TRAIN_DATA:
                doc = nlp.make_doc(text)
                ents= []
                split = category.split(',')
                example=[]
                if(len(split) == 1):
                    sentence = str(text)
                    word = sentence[int(start_position_1):].split(' ')[0]
                    example = Example.from_dict(doc, {'entities': [(int(start_position_1), int(start_position_1)+len(word),category)]})
                    print(doc, {'entities': [(int(start_position_1), int(start_position_1)+len(word),category)]})
                    span = doc.char_span(int(start_position_1),int(start_position_1)+len(word), label=category, alignment_mode="contract")
                    if span is None:
                        print('skipping')
                    else:
                        ents.append(span)
                    
                elif(len(split) == 2 and start_position_2 >=0 and stop_position_2 >=0):
                    sentence = str(text)
                    word1 = sentence[int(start_position_1):].split(' ')[0]
                    span = doc.char_span(int(start_position_1),int(start_position_1)+len(word1), label=split[0], alignment_mode="contract")
                    if span is None:
                        print('skipping')
                    else:
                        ents.append(span)
                    sentence = str(text)
                    word2 = sentence[int(start_position_2):].split(' ')[0]
                    span = doc.char_span(int(start_position_2),int(start_position_2)+len(word2), label=split[1], alignment_mode="contract")
                    if span is None:
                        print('skipping')
                    else:
                        ents.append(span)
                elif(len(split) == 3 and start_position_2 >=0 and stop_position_2 >=0 and start_position_3 >=0 and stop_position_3 >=0):
                    sentence = str(text)
                    word1 = sentence[int(start_position_1):].split(' ')[0]
                    span = doc.char_span(int(start_position_1),int(start_position_1)+len(word1), label=split[0], alignment_mode="contract")
                    if span is None:
                        print('skipping')
                    else:
                        ents.append(span)
                    sentence = str(text)
                    word2 = sentence[int(start_position_2):].split(' ')[0]
                    span = doc.char_span(int(start_position_2),int(start_position_2)+len(word1), label=split[1], alignment_mode="contract")
                    if span is None:
                        print('skipping')
                    else:
                        ents.append(span)
                    sentence = str(text)
                    word3 = sentence[int(start_position_3):].split(' ')[0]
                    span = doc.char_span(int(start_position_3),int(start_position_3)+len(word3), label=split[2], alignment_mode="contract")
                    if span is None:
                        print('skipping')
                    else:
                        ents.append(span)
                else:
                    continue
                doc.ents = ents
                db.add(doc)
            db.to_disk("./train.spacy")
            print(losses)
            print(scorer.score(examples))
            return scorer.score(examples)

        
    def findData(self, request):
        nlp = spacy.load(r".\output/model-best")
        test_text = str(request["text"])
        doc = nlp(test_text)
        scorer = Scorer()
        examples = []
        ners =['SECRET','SEXUAL_LIFE', 'MAIL', 'POLITICS', 'WEALTH']
        disp = displacy.render(doc, style="ent")
        responses=[]
        for ent in doc.ents:
            responses.append({"label":ent.label_, "text":ent.text})
            if ent.label_ in ners:
                examples.append(Example.from_dict(doc, {'entities': [(int(test_text.index(ent.text)), int(test_text.index(ent.text))+len(ent.text),ent.label_ )]}))
                self.db.insert({"text": test_text, "start_position_1": test_text.index(ent.text), "stop_position_1":test_text.index(ent.text) + len(ent.text), "category": ent.label_ , "score": scorer.score(examples), "start_position_2": 0,"start_position_3": 0,"stop_position_2": 0,"stop_position_3":0   })
                self.controlData()
        score = scorer.score(examples)

        return {"data": responses, "displacy": disp, "score": score}

    def controlData(self):
        data = self.db.get()
        print(data)
        if(len(data)%200 == 0):
            print("controldata")
            self.ner2(False)

    def tweetData(self):
        nlp = spacy.load(r".\output/model-best")
        tweetsData = pd.read_csv("data/tryTweetData.csv",encoding='unicode_escape', error_bad_lines=False, sep=';')
        scorer = Scorer()
        examples = []
        kvkTweet = []
        ners =['SECRET','SEXUAL_LIFE', 'MAIL', 'POLITICS', 'WEALTH']

        for item in tweetsData.values.tolist():
            doc = nlp(item[1])
            for ent in doc.ents:
                if ent.label_ in ners:
                    kvkTweet.append({"text": item[1], "label": ent.label_, "word":ent.text})
                    examples.append(Example.from_dict(doc, {'entities': [(int(str(item[1]).index(ent.text)), int(str(item[1]).index(ent.text))+len(ent.text),ent.label_)]}))
                    self.db.insert({"text": item[1], "start_position_1": str(item[1]).index(ent.text), "stop_position_1":str(item[1]).index(ent.text) + len(ent.text), "category": ent.label_ , "score": scorer.score(examples), "start_position_2": 0,"start_position_3": 0,"stop_position_2": 0,"stop_position_3":0   })
                    self.controlData()
        score = scorer.score(examples)
        return {"data": kvkTweet, "score":score}

    def ner(self):
        nlp = spacy.load('en_core_web_lg')
        scorer = Scorer()
        TRAIN_DATA = pd.read_csv("data/trainData.csv",encoding='unicode_escape',  error_bad_lines=False, sep=';')
        #df=TRAIN_DATA.to_numpy()
        ner = nlp.get_pipe('ner')
        ner.add_label('SECRET')
        ner.add_label('SEXUAL_LIFE')
        ner.add_label('MAIL')
        ner.add_label('POLITICS')
        ner.add_label('WEALTH')
        print(ner.move_names)
        optimizer = nlp.create_optimizer()
        #other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        print("o"*30)
        print(other_pipes)
        with nlp.disable_pipes(*other_pipes):
            examples = []
            losses = {}
            counter = 0
            for id, text, category, start_position_1,start_position_2,start_position_3,stop_position_1,stop_position_2,stop_position_3 in TRAIN_DATA.values:
                counter+=1
                print(counter)
                print(text)
                if(counter == 1000):
                    break
                doc = nlp(text)
                split = category.split(',')
                example=[]
                if(len(split) == 1):
                    sentence = str(text)
                    word = sentence[int(start_position_1):].split(' ')[0]
                    example = Example.from_dict(doc, {'entities': [(int(start_position_1), int(start_position_1)+len(word),category)]})
                    print(doc, {'entities': [(int(start_position_1), int(start_position_1)+len(word),category)]})
                elif(len(split) == 2 and start_position_2 >=0 and stop_position_2 >=0):
                    sentence = str(text)
                    word1 = sentence[int(start_position_1):].split(' ')[0]
                    sentence = str(text)
                    word2 = sentence[int(start_position_2):].split(' ')[0]
                    example = Example.from_dict(doc, {'entities': [(int(start_position_1), int(start_position_1)+len(word1),split[0]),(int(start_position_2), int(start_position_2)+len(word2),split[1])]})
                elif(len(split) == 3 and start_position_2 >=0 and stop_position_2 >=0 and start_position_3 >=0 and stop_position_3 >=0):
                    sentence = str(text)
                    word1 = sentence[int(start_position_1):].split(' ')[0]
                    sentence = str(text)
                    word2 = sentence[int(start_position_2):].split(' ')[0]
                    sentence = str(text)
                    word3 = sentence[int(start_position_3):].split(' ')[0]
                    example = Example.from_dict(doc, {'entities': [(int(start_position_1), int(start_position_1)+len(word1),split[0]),(int(start_position_2), int(start_position_2)+len(word2),split[1]),(int(start_position_3), int(start_position_3)+len(word3),split[2])]})
                else:
                    continue
                examples.append(example)
                nlp.update([example], drop=0.3, sgd=optimizer, losses=losses)
            nlp.to_disk("./trainData.spacy")
            print(losses)
            print(scorer.score(examples))
        return scorer.score(examples)

