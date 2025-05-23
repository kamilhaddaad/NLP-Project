Steps to run the application:

1. Add the external model files to the code (Since they are too big)
2. Run this command to run the application:
   $env:FLASK_APP="app.py"; flask run --no-reload
   (or ``FLASK_APP=app.py flask run --no-reload`` on MacOS/Linux)
3. In case you get error related to authentication problems while running the server, perform [huggingface-cli login](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
