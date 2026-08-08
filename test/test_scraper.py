from pathlib import Path

from scrapy.http import TextResponse

from scrape.organ.pipelines import OrganPipeline
from scrape.organ.spiders.doc_spider import SITEMAP_URL, AnarchySpider

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ARTICLE_URL = "https://theanarchistlibrary.org/library/kingsley-widmer-on-refusing"
ARTICLE_HTML = (FIXTURES / "article_page.html").read_text(encoding="utf-8")


def test_sitemap_parse_queues_one_article_request():
    spider = AnarchySpider()
    sitemap_body = "\n".join(
        [
            "https://theanarchistlibrary.org/listing",
            ARTICLE_URL,
            "https://theanarchistlibrary.org/category/author",
        ]
    )
    response = TextResponse(
        url=SITEMAP_URL,
        body=sitemap_body.encode(),
        encoding="utf-8",
    )

    requests = list(spider.parse(response))

    assert len(requests) == 1
    assert requests[0].url == ARTICLE_URL
    assert requests[0].callback == spider.final_content


def test_scrape_one_article_returns_expected_fields():
    spider = AnarchySpider()
    response = TextResponse(
        url=ARTICLE_URL,
        body=ARTICLE_HTML.encode(),
        encoding="utf-8",
    )

    raw_item = next(spider.final_content(response))
    item = OrganPipeline().process_item(raw_item, spider)

    assert item["url"] == ARTICLE_URL
    assert item["title"] == "On Refusing"
    assert item["author"] == "Kingsley Widmer"
    assert item["published_at"] == ""  # page only provides the year "1969"
    assert "anarchy" in item["tags"]
    assert "Not long ago I spoke at an anti-war rally" in item["text"]
    assert len(item["text"]) > 1000

    print("\n--- scraped article preview ---")
    for key in ("article_id", "url", "title", "author", "published_at", "tags"):
        print(f"{key}: {item[key]}")
    print(f"text: {item['text'][:200]}...")
