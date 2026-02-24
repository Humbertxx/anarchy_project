from scrapy import Item, Field

class OrganItem(Item):
    article_id = Field()
    title = Field()
    author = Field()
    text = Field()