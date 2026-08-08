import time

import scrapy
from scrapy import signals

from scrape.organ.items import OrganItem

SITEMAP_URL = "https://theanarchistlibrary.org/sitemap.txt"


class AnarchySpider(scrapy.Spider):
    name = "anarchy"
    start_urls = [SITEMAP_URL]

    def __init__(self):
        self.item_id_counter = 0

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
        queued = 0
        for line in response.text.splitlines():
            url = line.strip()
            if not self._is_article_url(url):
                continue
            queued += 1
            yield scrapy.Request(url, callback=self.final_content)

        self.logger.info("Queued %d article URLs from sitemap", queued)

    @staticmethod
    def _is_article_url(url: str) -> bool:
        return "/library/" in url and not url.rstrip("/").endswith("/library")

    def final_content(self, response):
        self.item_id_counter += 1

        article: OrganItem = {
            "article_id": self.item_id_counter,
            "url": response.url,
            "title": (response.css("title::text").get() or "").strip(),
            "author": (response.css("h3#text-author ::text").get() or "").strip(),
            "published_at": "".join(
                p_date.strip()
                for p_date in response.css("div#textdate::text").getall()
                if p_date.strip()
            ),
            "tags": [
                tag.strip()
                for tag in response.css("a.text-topics-item ::text").getall()
                if tag.strip()
            ],
            "text": " ".join(
                text.strip()
                for text in response.css("div#thework ::text").getall()
                if text.strip()
            ),
        }

        yield article
