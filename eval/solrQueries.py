import pip
import json
import urllib.parse
import urllib.request
import codecs

qId=[]
qText=[]

#Solr parameters
port='8983'
collection='trec'
fileNumber=10

#Trec parameter
IRModel='DFR'


with codecs.open("queries.txt","r","UTF-8") as file:
    for line in file:
        elements=line.split(":::")
        qId.append(elements[0])
        qText.append(elements[1])

with codecs.open("solrResults.txt","w","UTF-8") as file:
    #file.write("[query_id ] iter [docno] ranking [solr score] IRModel\n")
    for idNum,query in zip(qId,qText):
        print("query file: "+idNum)
        query=urllib.parse.quote(query)
        url='http://localhost:8983/solr/trec/select?fl=docno%2C%20score&q=doctext%3A('+query+')&rows=10&sort=score%20desc'
        data=urllib.request.urlopen(url)
        results = json.load(data)['response']['docs']
        rank = 1
        for result in results:
            file.write(idNum + ' ' + 'Q0' + ' ' + (str(result["docno"])).replace("'","").replace("[","").replace("]","") + ' ' + str(rank) + ' ' + str(result['score']) + ' ' + IRModel + '\n')
            rank += 1
        
    
