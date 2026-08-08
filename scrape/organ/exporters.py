import pyarrow as pa
import pyarrow.parquet as pq
from scrapy.exporters import BaseItemExporter

class ParquetItemExporter(BaseItemExporter):
    def __init__(self, file, *, dont_fail = False, **kwargs):
        super().__init__(dont_fail=dont_fail, **kwargs)
        self.file = file
        self.items = []
        
    def export_item(self, item):
        self.items.append(dict(item))
    
    def finish_exporting(self):
        if not self.items:
            return
        table = pa.Table.from_pylist(self.items)
        pq.write_table(table, self.file)