import os
from werkzeug.wrappers import response
from main import app
import unittest
import json

class FlaskrTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()

    def test_ping(self):
        response = self.app.get('/ping')
        res = response.data.decode('utf8').replace("'", '"')
        data = json.loads(res)
        assert (data['ping'] == 'pong')

    def test_predict(self):
        testData = [{'title': 'Elon Musk: The THREE questions investors should ask as Tesla CEO prompts SHIB surge', 'published_date': '2021-10-08 23:09:15', 'link': 'https://www.express.co.uk/finance/city/1503278/Elon-musk-dogecoin-shiba-inu-coin-price-Tesla-CEO-evg', 'clean_url': 'express.co.uk', 'summary': "Elon Musk, the billionaire CEO of electric car manufacturers Tesla, has caused ripples through the cryptocurrency market again. On Monday, he posted a picture of his new Shiba Inu puppy, Floki Funkpuppy, boosting the token of the same name and inspiring a new addition that has already started flying. His inspiration has helped many novice investors try their hand at cryptocurrencies, but also led experts to ask questions.Mr Musk's ongoing influence underlines the inherent volatility behind altco", 'media': 'https://cdn.images.express.co.uk/img/dynamic/22/750x445/1503278.jpg', 'topic': 'finance'}
            ,{'title': 'Elon Musk Posts Puppy Floki In A Tesla, SHIB Coin Surges 30%', 'published_date': '2021-10-04 17:37:35', 'link': 'https://www.ibtimes.com/elon-musk-posts-puppy-floki-tesla-shib-coin-surges-30-3309019', 'clean_url': 'ibtimes.com', 'summary': "A Twitter post of Elon Musk's Shiba Inu dog has caused the cryptocurrency of the same name to soar 30% on Monday.On Sunday night, Musk posted the picture of his pup Floki in the front trunk of a Tesla, causing not only the Shiba Inu coin to jump but also Dogecoin to spike a short-lived 7.5 centsFloki Frunkpuppy pic.twitter.com/xAr8T0Jfdf— Elon Musk (@elonmusk) October 4, 2021 Musk brought home Floki on Sept. 12 in a Twitter post after declaring in June that he was getting a Shiba Inu dog, the ma", 'media': 'https://s1.ibtimes.com/sites/www.ibtimes.com/files/styles/full/public/2021/08/20/tesla-ceo-elon-musk-says-the-company-will.jpg', 'topic': 'news'}
            ,{'title': "Cryptocurrency named after Elon Musk's dog surges 2,400 percent", 'published_date': '2021-10-06 15:28:33', 'link': 'https://nypost.com/2021/10/06/cryptocurrency-named-after-elon-musks-dog-surges-2400-percent/', 'clean_url': 'nypost.com', 'summary': "Fans of Elon Musk made a cryptocurrency named after his dog — and it's up 2,400 percent.\xa0\n\n\n\nFlokinomics, a dogecoin knockoff that started trading on Sunday, was going for $0.000002254 early Wednesday, according to CoinMarketCap data. While that might not sound like a lot, it's a 2,400 percent increase in value over the previous 24 hours.\n\n\n\n'WE $FLOKIN DID IT,' declared a Twitter account for the cryptocurrency.\xa0\n\n\n\nFlokinomics is named after Elon Musk's dog, Floki.\xa0\n\n\n\nFloki is a Shiba Inu — th", 'media': 'https://nypost.com/wp-content/uploads/sites/2/2021/10/flokinomics-musk-hp.jpg?quality=90&strip=all&w=1024', 'topic': 'news'}
            ,{'title': "Grimes Trolls Paparazzi with Communist Manifesto as She Confirms She's Still Living with Elon Musk", 'published_date': '2021-10-03 22:29:11', 'link': 'https://www.yahoo.com/entertainment/grimes-trolls-paparazzi-communist-manifesto-222911122.html', 'clean_url': 'yahoo.com', 'summary': "Grimes, Elon Musks Ex was spotted on the street for the first time reading 'The communist Manefesto' as she sat on the street by herself.\n\nJvshvisions/BACKGRID\n\nGrimes is taking full advantage of the publicity she's getting from her split with Elon Musk.\n\nThe Canadian musician, 33, confirmed that she and the SpaceX mogul, 50, are still living together after she trolled the paparazzi during a casual walk in a perfectly on-brand, post-apocalyptic chic ensemble while reading Karl Marx's Communist M", 'media': 'https://s.yimg.com/ny/api/res/1.2/ASdOxmkt3AL8YiZgIWU7wQ--/YXBwaWQ9aGlnaGxhbmRlcjt3PTEyMDA7aD0xNDQy/https://s.yimg.com/uu/api/res/1.2/cKmqPWIsYG3rqpRcRxLR1A--~B/aD0xNTAwO3c9MTI0ODthcHBpZD15dGFjaHlvbg--/https://media.zenfs.com/en/people_218/c6496780f72211241a5d25835a31ad06', 'topic': 'entertainment'}
            ,{'title': 'Elon Musk says Tesla moving headquarters to Texas', 'published_date': '2021-10-08 02:23:24', 'link': 'https://www.news.com.au/breaking-news/elon-musk-says-tesla-moving-headquarters-to-texas/news-story/10f78c8cecb6f0bfa35a34a9506c94d8', 'clean_url': 'news.com.au', 'summary': 'Tesla chief Elon Musk says the company is moving its headquarters from Silicon Valley to TexasDon\'t miss out on the headlines from Breaking News. Followed categories will be added to My News.Tesla chief Elon Musk told investors on Thursday that the leading electric vehicle maker is moving its headquarters from Silicon Valley to Texas, where it is building a plant."I\'m excited to announce that we\'re moving our headquarters to Austin, Texas," Musk said at an annual shareholders meeting.Originally ', 'media': 'https://content.api.news/v3/images/bin/a41bbaf0978e2e5ace20af72afbdefe8', 'topic': 'news'}
            ,{'title': 'Musk Causes SHIB Coin To Soar With Dog Picture', 'published_date': '2021-10-04 17:37:35', 'link': 'https://www.ibtimes.com/elon-musk-posts-puppy-floki-tesla-shib-coin-surges-30-3309019?ft=b90u9', 'clean_url': 'ibtimes.com', 'summary': "A Twitter post of Elon Musk's Shiba Inu dog has caused the cryptocurrency of the same name to soar 30% on Monday.\nOn Sunday night, Musk posted the picture of his pup Floki in the front trunk of a Tesla, causing not only the Shiba Inu coin to jump but also Dogecoin to spike a short-lived 7.5 cents Floki Frunkpuppy pic.twitter.com/xAr8T0Jfdf\n— Elon Musk (@elonmusk) October 4, 2021\n\nMusk brought home Floki on Sept. 12 in a Twitter post after declaring in June that he was getting a Shiba Inu dog, th", 'media': 'https://s1.ibtimes.com/sites/www.ibtimes.com/files/styles/embed/public/2021/08/20/tesla-ceo-elon-musk-says-the-company-will.jpg', 'topic': 'news'}
            ,{'title': "On Steve Jobsc' 10th death anniversary, Elon Musk wishes he could have spoken to him", 'published_date': '2021-10-04 12:00:00', 'link': 'https://timesofindia.indiatimes.com/gadgets-news/on-steve-jobs-10th-death-anniversary-elon-musk-wishes-he-could-have-spoken-to-him/articleshow/86749110.cms', 'clean_url': 'indiatimes.com', 'summary': '@SawyerMerritt I wish I had had the opportunity to talk to him — Elon Musk (@elonmusk) 1633248514000\n\nTen years ago on this day, Apple co-founder Steve Jobs passed away after a prolonged battle with pancreatic cancer. Jobs is credited with turning around Apple and making it into one of the biggest companies in the world. In recent times, some people have often drawn parallels between Tesla CEO Elon Musk with Jobs. And Musk in a tweet has said that he wishes he could have spoken to Jobs.Replying ', 'media': 'https://static.toiimg.com/thumb/msid-86749110,width-1070,height-580,imgsize-74600,resizemode-75,overlay-toi_sw,pt-32,y_pad-40/photo.jpg', 'topic': 'tech'}
            ,{'title': 'Elon Musk, Grimes: A timeline of their romance', 'published_date': '2021-10-05 22:12:14', 'link': 'https://www.foxbusiness.com/lifestyle/elon-musk-grimes-timeline-their-romance', 'clean_url': 'foxbusiness.com', 'summary': 'Grimes and Tesla CEO Elon Musk\'s relationship has raised eyebrows since they surprised the world in May 2018.\xa0 Musk, 50, confirmed his breakup with the "Alter Ego" judge, 33, in a statement last month."We are semi-separated but still love each other, see each other frequently, and are on great terms," Musk told\xa0Page Six.\xa0The former couple met in April 2018 via Twitter after they traded quips about Rococo Basilisk and artificial intelligence. ELON MUSK\'S EX GRIMES SAYS SHE\'S NOT A COMMUNIST AFTER', 'media': 'https://a57.foxnews.com/static.foxbusiness.com/foxbusiness.com/content/uploads/2021/09/0/0/Musk-Grimes-Split.jpg?ve=1&tl=1', 'topic': 'news'}
            ,{'title': 'Tesla moving to Texas because California lawmaker tweeted "F*** Elon Musk"', 'published_date': '2021-10-09 10:34:19', 'link': 'https://www.newsweek.com/tesla-moving-texas-because-california-lawmaker-tweeted-f-elon-musk-1637281', 'clean_url': 'newsweek.com', 'summary': "Tesla CEO Elon Musk has suggested an insult from a California lawmaker was a key factor in his decision to shift his company's headquarters out of the Golden State to Texas.Musk confirmed on Thursday that Tesla would move from Palo Alto to Austin where it had been building a new factory for just over a year.During an annual shareholder meeting, Musk said Tesla would continue to operate its electric vehicle factory in Fremont, California, where he wants to increase production by 50 percent and in", 'media': 'https://d.newsweek.com/en/full/1910389/tesla-ceo-elon-musk.jpg', 'topic': 'news'}
            ,{'title': "Grimes said she 'trolled' paparazzi by posing with Karl Marx's 'Communist Manifesto' following her breakup with Elon Musk", 'published_date': '2021-10-04 04:11:41', 'link': 'https://uk.news.yahoo.com/grimes-said-she-trolled-paparazzi-041141028.html', 'clean_url': 'yahoo.com', 'summary': 'Grimes said she \'trolled\' paparazzi by posing with Karl Marx\'s \'Communist Manifesto\' following her breakup with Elon Musk\n\nSinger Grimes was spotted flipping through a copy of Karl Marx\'s The Communist Manifesto in full costume following her split from billionaire Elon Musk. Screengrab/Twitter\n\nGrimes was spotted posing with the "Communist Manifesto" on a street corner.\n\nThe singer posted on social media that she posed with the book on purpose to "troll" the paparazzi.\n\nBillionaire Elon Musk rev', 'media': 'https://s.yimg.com/uu/api/res/1.2/7ZPHct0JtiwaNgKXHMd_2g--~B/aD04Nzc7dz0xMTY5O2FwcGlkPXl0YWNoeW9u/https://media.zenfs.com/en/insider_articles_922/58d88db8b9f787397077e74fc6dd7e89', 'topic': 'business'}
        ]
        response = self.app.get('/train')
        res = response.data.decode('utf8').replace("'", '"')
        resdata = json.loads(res)
        print(resdata['Success'])
        if(resdata['Success'] == 'True'):
            for data in testData:
                response = self.app.post('/predict',json = {'searchText': data['summary']})
                res = response.data.decode('utf8').replace("'", '"')
                resdata = json.loads(res)
                assert (resdata['category'] == data['topic'])
        
if __name__ == '__main__':
    unittest.main()