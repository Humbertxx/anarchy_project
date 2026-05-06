from scrapy import Item, Field

class OrganItem(Item):
    url          = Field()
    title        = Field()
    author       = Field()
    published_at = Field()
    tags         = Field()
    body         = Field()