# The following file is used to download the google drive data for all models

import os
import io
import argparse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

parser = argparse.ArgumentParser(description="",
                                 epilog="",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--drive-file-id", type=str, default="", required=True,
                    help="")

parser.add_argument("--data-name", type=str, default="", required=True,
                    help="")

parser.add_argument("--google-auth-file", type=str, default="", required=True,
                    help="")

parser.add_argument("--output-dir", type=str, default="", required=True,
                    help="")



args = parser.parse_args()


SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

credentials = service_account.Credentials.from_service_account_file(
    args.google_auth_file, scopes=SCOPES
)

service = build('drive', 'v3', credentials=credentials)

file = service.files().get(fileId=args.drive_file_id).execute()
print(f"File: {file['name']}")

request = service.files().get_media(fileId=args.drive_file_id)

# Create the full path
file_path = os.path.join(args.output_dir, file['name'])


with io.FileIO(file_path, 'wb') as fh:
    downloader = MediaIoBaseDownload(fh, request)
    print(f"Saving the file in {file_path}")
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download progress: {int(status.progress() * 100)}%")

print("Download completed.")