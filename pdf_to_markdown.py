import os
import base64
import json
import requests
import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PDFToMarkdownConverter:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.app_id = os.getenv("MATHPIX_APP_ID")
        self.app_key = os.getenv("MATHPIX_APP_KEY")
        if not self.app_id or not self.app_key:
            raise ValueError("MATHPIX_APP_ID and MATHPIX_APP_KEY must be set in the environment variables")

    def convert_to_markdown(self):
        with open(self.pdf_path, 'rb') as f:
            pdf_data = f.read()
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

        payload = {
            'src': f'data:application/pdf;base64,{pdf_base64}',
            'formats': ['markdown']
        }

        response = requests.post(
            'https://api.mathpix.com/v3/pdf',
            headers={
                'app_id': self.app_id,
                'app_key': self.app_key,
                'Content-Type': 'application/json'
            },
            data=json.dumps(payload)
        )

        if response.status_code == 200:
            result = response.json()
            return result.get('markdown', '')
        else:
            raise Exception(f"Error: {response.status_code} {response.text}")

    def save_markdown_to_file(self, output_path):
        markdown_content = self.convert_to_markdown()
        with open(output_path, 'w') as f:
            f.write(markdown_content)
        print(f"Markdown content saved to {output_path}")

@click.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path to save the markdown content')
def cli(pdf_path, output):
    converter = PDFToMarkdownConverter(pdf_path)
    if output:
        converter.save_markdown_to_file(output)
    else:
        markdown_content = converter.convert_to_markdown()
        print(markdown_content)

if __name__ == '__main__':
    cli()