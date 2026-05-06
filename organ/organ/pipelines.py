import re
import datetime

class OrganPipeline:
    
    def process_item(self, item, spider):
        item['text'] = self.normalize_text(item['text'])
        item['published_at'] = self.normalize_date(item['published_at'])
        return item
    
    def normalize_text(self, text):
        try:
            text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
            text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
            text = '\n'.join(line.strip() for line in text.split('\n'))
            return text
        except ValueError as exc:
            return {"error" : str(exc)}
            
    
    def normalize_date(self, date): 
        formats = [
            '%Y-%m-%d',  # 2026-05-04
            '%m/%d/%Y',  # 5/2/2026
            '%d/%m/%Y',
            '%B %d, %Y',  # April 27, 2026
            '%B %d, %Y',
            '%b %d %Y',  # Apr 27 2026
            '%d %B %Y',
            '%d %b %Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return ''