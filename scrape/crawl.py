from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from scrape.organ.spiders.doc_spider import AnarchySpider

def main():
    print("Anarchy Library Analysis \n\n")
    anarchyReading()
    
def anarchyReading():
    process = CrawlerProcess(get_project_settings())
    process.crawl(AnarchySpider)
    process.start()

if __name__ == "__main__":
    main()
