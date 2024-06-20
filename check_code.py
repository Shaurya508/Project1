import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import pytesseract

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_content = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in article_content])
    return text.encode('ascii', 'ignore').decode('ascii')

def extract_text_from_pdf(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path, 300)

    # Iterate through all the pages and extract text
    extracted_text = ''
    for page_number, page_data in enumerate(pages):
        # Perform OCR on the image
        text = pytesseract.image_to_string(page_data)
        extracted_text += f"Page {page_number + 1}:\n{text}\n"
    return extracted_text

def main():
    # text = extract_text_from_url('https://github.com/facebookexperimental/Robyn/blob/main/demo/demo.R')
    # print(text)
    # URL of the raw content of the R script on GitHub
    raw_url = "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/demo/demo.R"

    # Send a GET request to the raw content URL
    response = requests.get(raw_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the content of the response
        code = response.text

        # Print the scraped code
        print(code)
    else:
        print(f"Failed to retrieve the URL. Status code: {response.status_code}")

if __name__ == "__main__":
    main()