import scrapy
from organ.items import OrganItem
import time
from scrapy import signals

class AnarchySpider(scrapy.Spider):
    name = "anarchy"  
    start_urls = [f"https://theanarchistlibrary.org/latest/"]                
    
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
        entries = response.css('div.amw-listing-item').get()
        for entry in entries:
            link = entry.css('a::attr(href)').get()
            link_date = entry.css('span.pull-right.clearfix.amw-list-text-pubdate-locale ::text').get()
            
            if not link:
                continue
            
            yield response.follow(
                link, 
                callback=self.final_content, 
                cb_kwargs={'listing_date': link_date}
            )   
        
        next_href = response.css('ul.pagination li.active + li a::attr(href)').get()
        if next_href:
            yield response.follow(next_href, callback=self.parse)
            
    def final_content(self, response, listing_date=None): 
        articles = OrganItem()
        articles['url'] = response.url
        articles['title'] = response.css('title ::text').get()
        articles['author'] = response.css('h3#text-author ::text').get()
        articles['published_at'] = listing_date
        articles['tags'] = response.css('a.text-topics-item ::text').get()
        articles['body'] = ''.join(response.css('div#thework ::text').getall())
    
        yield articles