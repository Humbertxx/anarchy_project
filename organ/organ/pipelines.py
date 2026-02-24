#from itemadapter import ItemAdapter
import re

class OrganPipeline:
    def process_item(self, item, spider):
        item['text'] = self.normalize_text(item['text'])
        return item
    def normalize_text(self, text):
        text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text
