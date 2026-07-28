from scrape.crawl import Scraper

def main():
    print("welcome to Anarchy Lib Crawler")
    scraper = Scraper()
    scraper.run_spiders()
    
    
if __name__ == "__main__":
    main() 