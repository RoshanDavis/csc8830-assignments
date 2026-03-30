# Visual Search Storyboard with Gemini 2.5 Flash

This project demonstrates how to use Google's cutting-edge **Gemini 2.5 Flash** Vision model to perform visual search on video files. By extracting frames from a video and compiling them into a single time-stamped "storyboard" grid, the application enables you to ask natural language questions about the video content and receive accurate timestamps in response.

## ✨ Features

- **Automated Storyboard Generation**: Extracts frames from any input video at regular intervals.
- **Timestamp Overlay**: Automatically stamps each frame with its corresponding time (MM:SS) for easy tracking.
- **AI-Powered Visual Search**: Uses the Gemini 2.5 Flash model to parse the storyboard and find specifically what you're looking for natively.
- **Interactive Prompts**: Allows you to query the video in real-time.

## 🚀 Setup Instructions

1. **Install Dependencies**
   Make sure you have a Python environment set up, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   Create a `.env` file in the root directory of this folder:
   ```ini
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   *You can get a free API key from [Google AI Studio](https://aistudio.google.com/).*

3. **Add a Video Sample**
   Ensure you have a target video file in the directory. By default, the script looks for `sample.mp4`.

4. **Run the Application**
   ```bash
   python demo.py
   ```

## 🧠 How It Works

1. **Extraction**: `OpenCV` is used to load the video (`sample.mp4`) and extract 9 evenly spaced frames (by default a 3x3 grid).
2. **Annotation**: A timestamp is overlaid onto the top-left corner of each frame.
3. **Stitching**: The frames are concatenated horizontally and vertically to form a single "Storyboard" image (`demo_storyboard.jpg`) for the LLM to read.
4. **LLM Analysis**: The resulting storyboard image is sent to the Gemini API alongside your custom search query to accurately identify events in the video.

## 📝 Technologies Used

- `opencv-python` & `Pillow`: Video frame extraction and image processing.
- `google-genai`: AI inference via the new Gemini SDK.
- `python-dotenv`: Environment variable management.
