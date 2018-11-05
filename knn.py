# -*- coding: utf-8 -*-
import os
import numpy
import re
import nltk
import operator
rootDir = "/home/liubuntu/20news-18828-train"
buildDir = "/home/liubuntu/20news-18828-build"
testDir = "/home/liubuntu/20news-18828-test"
nltk.download('stopwords')
#创建新文件夹，存放预处理后的文本数据
def preProcess():
    dirs=os.listdir(rootDir)
    for i in range(len(dirs)):
        fileDir=rootDir+"/"+dirs[i]
        buildFileDir=buildDir+"/"+dirs[i]

        files=os.listdir(fileDir)

        if os.path.exists(buildFileDir)==False:
            os.mkdir(buildFileDir)        

        for j in range(len(files)):#预处理每个训练集里的文件
            preProcessFile(dirs[i],files[j])

#建立目标文件夹，生成目标文件
def preProcessFile(dir,fileName):
    src = rootDir+ "/" + dir + "/" + fileName
    dst = buildDir + "/" + dir + "/" + fileName

    dstFile = open(dst,'w')
    data = open(src,"rb").readlines()

    for line in data:
      #  line = line.decode('utf-8')
        result = processLine(line)
        for word in result:
            dstFile.write('%s\n' % word)

    dstFile.close()



#对每行字符串进行处理，主要是去除非字母字符，转换大写为小写，去除停用词

def processLine(line):
    stopwords = nltk.corpus.stopwords.words('english') #去停用词
    porter = nltk.PorterStemmer()  #词干分析
    splitter = re.compile('[^a-zA-Z]')  #去除非字母字符，形成分隔
    string = ""
    for i in range(len(line)):
        if chr(line[i]).isalpha():
            string+=chr(line[i])
        else:
            string+=" "
    words = [porter.stem(word.lower()) for word in splitter.split(string)\
        if len(word)>0 and\
        word.lower() not in stopwords]
    return words


# 统计每个词的总的出现次数
def countWords():
    wordMap = {}
    wordMap2 = {}

    dirs = os.listdir(buildDir)
    for i in range(len(dirs)):
        filedir = buildDir + "/" + dirs[i]
        files = os.listdir(filedir)
        for j in range(len(files)):
            filePath = filedir + "/" + files[j]
            for line in open(filePath).readlines():
                word=line.strip('\n')
                wordMap[word]=wordMap.get(word,0) + 1
    for key, value in wordMap.items():
        if value > 10:#去掉出现次数小于10的词
            wordMap2[key]=value
        
    return wordMap2
        

#计算idf

def IDF():
    wordFileMap = {}
    IDFMap = {}
    dirs = os.listdir(buildDir)
    fileNumber = 0
    for i in range(len(dirs)):
        filedir = buildDir + "/" + dirs[i]
        files = os.listdir(filedir)
        for j in range(len(files)):
            filePath = filedir + "/" + files[j]
            fileNumber = fileNumber + 1
            for line in open(filePath).readlines():
                word=line.strip('\n')
                if wordFileMap.get(word)== None:#该词在所有文件里第一次出现
                    wordFileMap[word]=[]
                    wordFileMap[word].append(filePath)
                elif filePath not in wordFileMap[word]: #该词在该文件里第一次出现
                    wordFileMap[word].append(filePath)       
    
    for word in wordFileMap:
        IDFMap[word] = fileNumber/len(wordFileMap[word]) #文件总数/该词出现的文件个数
    return IDFMap

#计算一个文件的向量
def Vector(file,wordMap,IDFMap):
    fileVector = {}
    data = open(file,"rb").readlines()
    for line in data:
        words = processLine(line)
        #words=words.strip('\n').split(" ")
        for word in words:
            if word not in wordMap:
                continue
            if word not in fileVector:
                fileVector[word]=1
            else:
                fileVector[word]+=1

    for key in fileVector:
        fileVector[key]=fileVector[key]*IDFMap[key]
    return fileVector

#建立训练集的 map{向量：分类}
def vectorMapToType(wordMap,IDFMap):
    vectorMapToType=[]
    dirs = os.listdir(buildDir)
    for i in range(len(dirs)):
        filedir = buildDir + "/" + dirs[i]
        files = os.listdir(filedir)
        for j in range(len(files)):
            filePath = filedir + "/" + files[j]
            fileVector = Vector(filePath,wordMap,IDFMap)
            vectorMapToType.append((fileVector,dirs[i]))

    return vectorMapToType

#计算两个向量的相似度
def similarity(trainVector,testVector):
    trainlist = []
    testlist = []
    for word in testVector:
        if word in trainVector:
            trainlist.append(trainVector[word])
            testlist.append(testVector[word])


    trainVect = numpy.mat(trainlist)
    testVect = numpy.mat(testlist)  
    
    num = float(testVect * trainVect.T)#向量点乘
    denom = numpy.linalg.norm(testVect) * numpy.linalg.norm(trainVect)
    return float(num)/(1+float(denom))

#根据模型预测测试文件的分类
def testFileSimilarity(testFile,wordMap,IDFMap,trainVectors):
    similarityMap = {}
    testVectoer = Vector(testFile,wordMap,IDFMap)
    resultMap = {}
    for vector in trainVectors:
        sim = similarity(vector[0],testVectoer)
        similarityMap[sim] = vector[1]
    keys = sorted(similarityMap) 

    sortedResult =  [(k,similarityMap[k]) for k in keys] 
    count = 0
   
    for i in sortedResult:
        if i[1] not in resultMap:
            resultMap[i[1]] = 1
        else:
            resultMap[i[1]] += 1
        count = count + 1
        if count > 100: #K取100
            break

    max = 0
    doctype = ""
    for key in resultMap:
        if resultMap[key]>max:
            doctype = key
            max = resultMap[key]
    return doctype

#预测文件的分类并计算正确度
def predictTestFileCorr(wordMap,IDFMap,trainVectors):
    dirs = os.listdir(testDir)
    fileNumber = 0
    correctCount = 0
    for i in range(len(dirs)):
        filedir = testDir + "/" + dirs[i]
        files = os.listdir(filedir)
        for j in range(len(files)):
            filePath = filedir + "/" + files[j]
            fileNumber = fileNumber + 1
            predict = testFileSimilarity(filePath,wordMap,IDFMap,trainVectors)
            if predict == dirs[i]:
                correctCount = correctCount + 1

    return float(correctCount/fileNumber)

def main():
    
    preProcess()
    wordMap = countWords()

    IDFMap= IDF()

    vectors = vectorMapToType(wordMap,IDFMap)

    correctness = predictTestFileCorr(wordMap,IDFMap,vectors)
   
    print("correctness is ",correctness)

if __name__ == "__main__":
    main()