from factory.database import Database
import spacy
import pandas as pd
from spacy.training.example import Example
from spacy.scorer import Scorer

class Deneme(object):
    def __init__(self):
        self.db = Database()
        self.collection_name = 'deneme'
        self.fields = {
            "title": "string",
            "body": "string",
            "created": "datetime",
            "updated": "datetime",
        }
        self.create_required_fields = ["title", "body"]
        self.create_optional_fields = []
        self.update_required_fields = ["title", "body"]
        self.update_optional_fields = []

    def create(self, deneme):
        nlp = spacy.load('en_core_web_sm')  # load existing spaCy model
        print(nlp)
        #res = self.db.insert(deneme, self.collection_name)
        return nlp

    def find(self, deneme):
        return self.db.find(deneme, self.collection_name)

    def find_by_id(self, id):
        return self.db.find_by_id(id, self.collection_name)

    def update(self, id, deneme):
        return self.db.update(id, deneme,self.collection_name)

    def delete(self, id):
        return self.db.delete(id, self.collection_name)
    def findData(self, nlp, params):
        test_text = params.text
        doc = nlp(test_text)
        print("Entities in '%s'" % test_text)
        response = ""
        for ent in doc.ents:
            response = (ent.label_, " -- ", ent.text)
        return {"data": response}

    def tweetData(self, nlp, params):
        tweetsData = pd.read_csv("models/Kitap1.csv",encoding='unicode_escape', error_bad_lines=False, sep=';')
        
        kvkTweet = []
        for item in tweetsData.values.tolist():
            doc = nlp(item[5])
            for ent in doc.ents:
                print(ent.label_, " -- ", ent.text)
                kvkTweet.append({"text": item[5], "label": ent.label_, "word":ent.text})
        return kvkTweet

    def ner(self, nlp):
        scorer = Scorer()
        TRAIN_DATA = pd.read_csv("models/trainData.csv",encoding='unicode_escape',  error_bad_lines=False, sep=';')
        df=TRAIN_DATA.to_numpy()
        ner = nlp.get_pipe('ner')
        ner.add_label('SECRET')
        ner.add_label('SEXUAL_LIFE')
        ner.add_label('MAIL')
        ner.add_label('POLITICS')
        ner.add_label('WEALTH')
        print(ner.move_names)
        optimizer = nlp.create_optimizer()
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):
            examples = []
            losses = {}
            for id, text, category, start_position_1,start_position_2,start_position_3,stop_position_1,stop_position_2,stop_position_3 in TRAIN_DATA.values:
                print('girdi')
                print(text)
                doc = nlp(text)
                split = category.split(',')
                if(len(split) == 1):
                    example = Example.from_dict(doc, {'entities': [(int(start_position_1), int(stop_position_1),category)]})
                elif(len(split) == 2 and start_position_2 >=0 and stop_position_2 >=0):
                    example = Example.from_dict(doc, {'entities': [(int(start_position_1), int(stop_position_1),split[0]),(int(start_position_2), int(stop_position_2),split[1])]})
                elif(len(split) == 3 and start_position_2 >=0 and stop_position_2 >=0 and start_position_3 >=0 and stop_position_3 >=0):
                    example = Example.from_dict(doc, {'entities': [(int(start_position_1), int(stop_position_1),split[0]),(int(start_position_2), int(stop_position_2),split[1]),(int(start_position_3), int(stop_position_3),split[2])]})
                else:
                    continue
                examples.append(example)
                nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
            print(losses)
            print(scorer.score(examples))
        return scorer.score(examples)

        
