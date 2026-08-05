import re
from datetime import datetime
from scrape.organ.items import OrganItem

class OrganPipeline:
    
    def process_item(self, item: OrganItem, spider) -> OrganItem:
        item["text"] = self.normalize_text(item.get("text"))
        item["title"] = self.normalize_title(item.get("title"))
        item["published_at"] = self.normalize_date(item.get("published_at"))
        return item
    
    def normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
        text = text.replace("\n", " ")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
    
    def normalize_title(self, title: str) -> str:
        if not title:
            return ""
        
        title = title.partition("|")[0]
        title = title.strip()
        
        return title 
        
    def normalize_date(self, date_value: str) -> str: 
        if not date_value:
            return ""

        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%Y/%m/%d",
            "%m-%d-%Y",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_value, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return ''
