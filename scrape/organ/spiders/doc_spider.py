import scrapy
from scrape.organ.items import OrganItem
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
        entries = response.css("div.amw-listing-item")
        for entry in entries:
            link = entry.css("a::attr(href)").get()
            link_date = entry.css(
                "span.pull-right.clearfix.amw-list-text-pubdate-locale ::text"
            ).get()
            
            if not link:
                continue
            
            yield response.follow(
                link, 
                callback=self.final_content, 
                cb_kwargs={"listing_date": (link_date or "").strip()},
            )   
        
        next_page = f'{self.start_urls[0]}{self.page_num}'
        if next_page and entries:
            if self.page_num < 5:
                self.page_num += 1
                yield response.follow(next_page, callback=self.parse)
            
    def final_content(self, response, listing_date=""): 
        self.item_id_counter += 1 

        article: OrganItem = {
            "url": response.url,
            "title": (response.css("title::text").get() or "").strip(),
            "author": (response.css("h3#text-author ::text").get() or "").strip(),
            "published_at": listing_date,
            "tags": [
                tag.strip()
                for tag in response.css("a.text-topics-item ::text").getall()
                if tag.strip()
            ],
            "text": "\n".join(
                text.strip()
                for text in response.css("div#thework ::text").getall()
                if text.strip()
            ),
        }

        yield article
