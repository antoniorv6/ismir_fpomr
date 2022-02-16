
"""
Created on Mon Oct 23 19:19:59 2017

@author: Francisco J. Castellanos
@project name: DAMA
"""

import collections
import json
from file_manager import FileManager

class CustomJson:
    
    dictionary = {}
    
    def __init__(self):
        self.dictionary = {}
        
    def addPair(self, key, value):
        self.dictionary[key] = value
        
    def deleteKey(self, key):
        self.dictionary.pop(key, None)
        
    def __getitem__(self, key):
        return self.dictionary.__getitem__(key)

    def __setitem__(self, key, data):
        self.dictionary.__setitem__(key, data)

    def __delitem__(self, key):
        self.dictionary.__delitem__(key)
        
    
    def saveJson(self, path_file):
        assert(type(path_file) is str)
        content = json.dumps(self.dictionary, sort_keys=True, indent=4, separators=(',', ': '))
        FileManager.saveString(content, path_file, True) #close_file
        
        
    
    def convert(self, data):
        try:
            basestring
        except NameError:
            basestring = str
            
        if isinstance(data, basestring):
            return str(data)
        if type(data) is dict:
            return data
        elif isinstance(data, collections.Mapping):
            return dict(map(self.convert, data.iteritems()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(self.convert, data))
        else:
            return data
        
        
    
    def loadJson(self, path_file):
        assert(type(path_file) is str)
        
        content = FileManager.readStringFile(path_file)
        
        self.dictionary = self.convert(json.loads(content))

    def printJson (self):
    
        print(json.dumps(self.dictionary))
    
    
    
    def listToDictionary(lst):
        assert(type(lst) is list)
        
        
        dictionary = {}
        count = 0
        for item in lst:
            if (type(item) is list):
                dictionary[count] = CustomJson.listToDictionary(item)
            elif type(item) is dict:
                dictionary[count] = item
            else:
                assert(type(item) is str)
                dictionary[count] = item
            
            count = count + 1
        
        return dictionary
        
    listToDictionary = staticmethod(listToDictionary)    

    
    def listFromDictionary(dct):
        assert(type(dct) is dict)
        
        lst = []

        for count in range(len(dct)):
            item = dct[str(count)]
            
            if (type(item) is dict):
                lst.append(CustomJson.listFromDictionary(item))
            else:
                assert(type(item) is str or type(item) is unicode)
                lst.append(str(item))
        
        return lst

    listFromDictionary = staticmethod(listFromDictionary)    
    
    def listFromDictionaryWithoutNumerate(dct):
        assert(type(dct) is dict)
        
        lst = []

        for item in dct:
            
            if (type(item) is dict):
                lst.append(CustomJson.listFromDictionary(item))
            else:
                assert(type(item) is str or type(item) is unicode)
                lst.append(str(item))
        
        return lst

    listFromDictionaryWithoutNumerate = staticmethod(listFromDictionaryWithoutNumerate)    
        
  
#==============================================================================
# Code Tests...
#==============================================================================

if __name__ == "__main__":

    json1 = CustomJson()
    
    json1.addPair("person1", {"name": "Pepe", "dni": 66553})
    
    json1.printJson()
    json1.saveJson("tests/custom_json.txt")
    
    
    
    json2 = CustomJson()
    json2.loadJson("tests/custom_json.txt")
    
    json2.printJson()
    
    print(json2["person1"])
    
    json2.saveJson("tests/custom_json2.txt")
    