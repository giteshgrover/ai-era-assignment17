---
title: Textual Inversion Image Generator with optional center focus(background blur)
emoji: 📚
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

#  Textual Inversion Image Generator with optional center focus(background blur)

## Description
This is a simple gradio app that allows you to generate images using textual inversion. An prompt is eneterd by the user and a concept is selected from the dropdown menu. The image is generated using the entered prompt and the selected concept. Currently, there are 5 concepts to choose from. To read more about the concepts, refer https://huggingface.co/sd-concepts-library.  The user can optionally select if the background should be blurred or not. Selecting that option generates an image that has a blurred background and the main subject is in focus.


## How to use

1. Enter your prompt in the text input field
2. Select the concept from the dropdown menu
3. Click on the generate button
4. The image will be generated and displayed on the screen

## How to setup and run the app

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app.py file
    ```bash
   python app.py
   ```
   This will start the gradio app on http://127.0.0.1:7860. Open that link in your browser to use the app.
