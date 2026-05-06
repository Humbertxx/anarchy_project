from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

def main():
    print("Anarchy Library Analysis \n\n")
    anarchyReading()
    
def anarchyReading():
    process = CrawlerProcess(get_project_settings())
    process.crawl("anarchy", domain="scrapy.org")
    process.start()

if __name__ == "__main__":
    main()