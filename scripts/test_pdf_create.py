import pydf
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="data")
t = templates.get_template("html_template.html")

html = t.render({"description": "TEST TEST 123"})

pdf = pydf.generate_pdf(html)
with open("test_doc.pdf", "wb") as f:
    f.write(pdf)
