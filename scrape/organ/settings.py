from config import RAW_DIR, ensure_dirs

ensure_dirs()

BOT_NAME = "organ"
SPIDER_MODULES = ["scrape.organ.spiders"]
NEWSPIDER_MODULE = "scrape.organ.spiders"

ADDONS = {}
# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = "Mozilla/5.0 (platform; rv:gecko-version) Gecko/gecko-trail Firefox/firefox-version"

ROBOTSTXT_OBEY = True

#ADDED
REACTOR_THREADPOOL_MAXSIZE = 128
DOWNLOAD_TIMEOUT = 30
RANDOMIZE_DOWNLOAD_DELAY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 256
DOWNLOAD_DELAY = 0

# The download delay setting will honor only one of:
CONCURRENT_REQUESTS_PER_DOMAIN = 256
CONCURRENT_REQUESTS_PER_IP = 256

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    "scrape.organ.pipelines.OrganPipeline": 300,
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html

AUTOTHROTTLE_ENABLED = True
# The initial download delay
AUTOTHROTTLE_START_DELAY = 1
# The maximum download delay to be set in case of high latencies
AUTOTHROTTLE_MAX_DELAY = 0.25
# The average number of requests Scrapy should be sending in parallel to
# each remote server
AUTOTHROTTLE_TARGET_CONCURRENCY = 128
# Enable showing throttling stats for every response received:
AUTOTHROTTLE_DEBUG = True

RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 400, 401, 403, 404, 405, 406, 407, 408, 409, 410, 429]

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.spidermiddlewares.referer.RefererMiddleware': 80,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    'scrapy.downloadermiddlewares.cookies.CookiesMiddleware': 130,
    'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
    'scrapy.downloadermiddlewares.redirect.RedirectMiddleware': 900,
}

FEED_EXPORTERS = {'parquet': 'scrape.organ.exporters.ParquetItemExporter'}

FEEDS = {
    str(RAW_DIR / "shard_%(batch_id)05d.pq"): {
        "format": "parquet",
        "overwrite": True,
    }
} 
FEED_EXPORT_BATCH_ITEM_COUNT = 2500

LOG_ENABLED = True
LOG_LEVEL = 'DEBUG'
TELNETCONSOLE_ENABLED = False
