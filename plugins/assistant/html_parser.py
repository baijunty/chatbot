from html.parser import HTMLParser as Parser
class HtmlParser(Parser):
    def __init__(self):
        super().__init__()
        self.need_embeedings = False
        self.start=False
        self.text=[]

    def handle_starttag(self, tag, attrs):
        self.start|=tag=='body'
        self.need_embeedings = self.start and tag in ['span','p','tr','th','td','table']

    def handle_data(self, data):
        if self.need_embeedings and data.strip():
            self.text.append(data.strip())

