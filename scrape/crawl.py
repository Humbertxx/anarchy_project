import os
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrape.organ.spiders.doc_spider import AnarchySpider

class Scraper:
    def __init__(self):
        settings_file_path = 'scrape.organ.settings' # The path seen from root, ie. from main.py
        os.environ.setdefault('SCRAPY_SETTINGS_MODULE', settings_file_path)
        self.process = CrawlerProcess(get_project_settings())
        self.spider = AnarchySpider # The spider you want to crawl

    def run_spiders(self):
        self.process.crawl(self.spider)
        self.process.start()  # the script will block here until the crawling is finished