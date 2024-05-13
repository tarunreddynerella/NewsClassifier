import requests
from bs4 import BeautifulSoup
import pandas as pd

def GetRssFeed(resources: object) -> list:
    articles = []
    for category in resources:
        if(type(resources[category]) == list):
            for link in resources[category]:
               articles += GetArticles(category,link)
        else:
            articles += GetArticles(category,resources[category])
    return articles
           
        
def GetArticles(category:str, url: str)-> list:
    res = requests.get(url)
    content = BeautifulSoup(res.content, features='xml')
    articles = content.findAll('item')
    articleData = []
    for article in articles:
        try:
            title = article.find('title').text
            description = article.find('description').text
            text = title + description
            text = CleanArticle(text)
            articleData.append({"text":text,"category":category} )
        except:
            pass
    return articleData

def CleanArticle(text: str) -> str:
    cleanText = ""
    i = 0
    while (i in range(len(text))):
        if(text[i] == '<'):
            while(text[i] != '>'):
                i+=1
            i+=1
        else:
           cleanText+= text[i]
           i+=1
    return cleanText



# article  = GetCNNFeed()
# article = GetRssFeed(CNNResources)
# article += GetRssFeed(NewYorkTimeResources)
# article += GetRssFeed(TheGuardianResources)
# data = pd.DataFrame(article, columns=['text','category'])


# with open("test.csv",'w') as file:
#     data.to_csv(file)
