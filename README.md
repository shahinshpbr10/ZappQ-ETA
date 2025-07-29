1.	Go to the Google Cloud Console.
2.	Create a new project.
3.	Enable the Google Drive API for your project.
4.	Create OAuth client credentials for a desktop app.
5.	Set up DVC to use your credential

dvc remote modify myremote gdrive_client_id <your-client-id>
dvc remote modify myremote gdrive_client_secret <your-client-secret>