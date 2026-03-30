import cv2
from PIL import Image
import os
from google import genai

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# TODO: Paste your free Google AI Studio API key here
os.environ["GEMINI_API_KEY"] = "AIzaSyBL2TnMaCJ0VcFrNr3C9jikoxl_SLwDNeU" 
VIDEO_FILE = "sample.mp4" # Make sure you have a test video in the folder

# ==========================================
# 2. COMPUTER VISION: STORYBOARD GENERATOR
# ==========================================
def create_temporal_storyboard(video_path, grid_w=3, grid_h=3):
    print("🎥 Extracting frames and building Storyboard...")
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = grid_w * grid_h
    step = total_frames // num_frames
    
    frames = []
    for i in range(num_frames):
        # Jump to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * step, total_frames - 1))
        ret, frame = cap.read()
        if not ret: break
            
        # Calculate the timestamp for this frame
        seconds = int((i * step) / fps) if fps > 0 else 0
        time_str = f"{seconds//60:02d}:{seconds%60:02d}"
        
        # Draw a black box for contrast, then overlay the green timestamp text
        cv2.rectangle(frame, (0, 0), (250, 70), (0, 0, 0), -1)
        cv2.putText(frame, time_str, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (0, 255, 0), 4, cv2.LINE_AA)
        
        # Resize to make the final grid manageable
        frame = cv2.resize(frame, (400, 300))
        frames.append(frame)
    cap.release()
    
    # Stitch frames horizontally and vertically into a grid
    rows = []
    for r in range(grid_h):
        row_frames = frames[r*grid_w : (r+1)*grid_w]
        rows.append(cv2.hconcat(row_frames))
    grid = cv2.vconcat(rows)
    
    # Save the grid so you can show it during your demo!
    cv2.imwrite("demo_storyboard.jpg", grid)
    print("✅ Storyboard saved as 'demo_storyboard.jpg'")
    
    # Convert BGR (OpenCV) to RGB (Pillow) for the LLM
    return Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))

# ==========================================
# 3. LLM AGENT: VISUAL SEARCH
# ==========================================
def run_demo():
    # 1. Create the grid
    storyboard_img = create_temporal_storyboard(VIDEO_FILE)
    
    # 2. Ask the user what they want to find
    print("\n" + "="*40)
    user_query = input("🔍 What do you want to find in the video? (e.g., 'When does the car appear?'): ")
    
    # 3. Prompt the Vision LLM
    print("\n🤖 AI is analyzing the storyboard...")
    client = genai.Client()
    
    prompt = f"""
    You are a video analysis agent. Look at this image grid representing a video. 
    Each frame has a timestamp in the top left corner.
    
    Task: Answer the user's query: "{user_query}"
    Tell me the exact timestamp where this occurs based on the text written on the frames. 
    Keep your answer to one short sentence.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[storyboard_img, prompt]
    )
    
    print("\n🎯 RESULT:")
    print(response.text)
    print("="*40)

if __name__ == "__main__":
    run_demo()