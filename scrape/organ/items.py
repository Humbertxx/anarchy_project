from typing import TypedDict


class OrganItem(TypedDict):
    url: str
    title: str
    author: str
    published_at: str
    tags: list[str]
    text: str
