from fpdf import FPDF

def create_test_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.cell(0, 10, txt='This is a test document for summarization.', ln=1)
    pdf.cell(0, 10, txt='It contains multiple sentences that should be summarized.', ln=1)
    pdf.output('test.pdf')

if __name__ == '__main__':
    create_test_pdf()
