import requests
from dotenv import load_dotenv
import os
import assemblyai as aai
import assemblyai.api as aapi

load_dotenv()

aai.settings.api_key = os.getenv('aai_api_key')
aai.settings.base_url = 'https://api.assemblyai.com'

client = aai.Client(settings=aai.settings)
http = client.http_client
params = aai.types.ListTranscriptParameters()
params.limit = 200
# params.before_id=''
list = aapi.list_transcripts(client=http, params=params)

# print(list)

for item in list.transcripts:
    itemid = item.id
    print(f"{itemid}\n")
    transcript = aai.Transcript.get_by_id(itemid)
    response = transcript.delete_by_id(itemid)
    print(f"{response.text}\n\n")