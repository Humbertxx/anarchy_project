import scrapy
from organ.items import OrganItem
import time
from scrapy import signals

class AnarchySpider(scrapy.Spider):
    name = "anarchy"  
    start_urls = [f"https://theanarchistlibrary.org/latest/"]                
    def __init__(self):
        self.item_id_counter = 0
        self.page_num = 1
    
    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def spider_opened(self, spider):
        self._t0 = time.perf_counter()

    def spider_closed(self, spider, reason):
        elapsed = time.perf_counter() - self._t0
        self.logger.info(f"Spider finished in {elapsed:.2f} seconds. Reason: {reason}")
   
    def parse(self, response):
        links = response.css('a.list-group-item.clearfix::attr(href)').getall()
        for link in links:
            yield response.follow(link, callback=self.final_content)
            
        next_page = f'{self.start_urls[0]}{self.page_num}'
        if next_page and links:
        #if self.page_num < 2 and next_page:
            self.page_num += 1
            yield response.follow(next_page, callback=self.parse)
            
    def final_content(self, response): 
        articles = OrganItem()
        self.item_id_counter += 1 
        articles['article_id'] = self.item_id_counter 
        articles['author'] = response.css('h3#text-author ::text').get()
        articles['title'] = response.css('title ::text').get()
        articles['text'] = ''.join(response.css('div#thework ::text').getall())
        #articles['text'] = ''.join([value.replace("\n", " ") for value in response.css('div#thework ::text').getall()])
        yield articles