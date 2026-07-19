import re
from datetime import datetime
from scrape.organ.items import OrganItem

class OrganPipeline:
    
    def process_item(self, item: OrganItem, spider) -> OrganItem:
        item["text"] = self.normalize_text(item.get("text"))
        item["published_at"] = self.normalize_date(item.get("published_at"))
        return item
    
    def normalize_text(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return "\n".join(line.strip() for line in text.split("\n")).strip()
            
    
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
