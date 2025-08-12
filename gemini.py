import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyCVX9GFTV9Mz4YTqd5IjymS49LE5Llyb0M")

audio_file_path1 = "benchaudios/multi_0.70_256.wav"
audio_file_path2 = "benchaudios/multi_0.50_256.wav"

print(f"Uploading {audio_file_path1}...")
audio_file_A = genai.upload_file(path=audio_file_path1)
print("Completed upload.")

print(f"Uploading {audio_file_path2}...")
audio_file_B = genai.upload_file(path=audio_file_path2)
print("Completed upload.")

model = genai.GenerativeModel(model_name="gemini-2.5-pro")

prompt = [
    """
Please act as an impartial judge and evaluate the overall audio quality of the responses provided by two AI assistants. You should choose the assistant that produced the better audio.

Your evaluation should focus only on technical audio quality. Consider factors such as fidelity (is the audio clean and clear?), realism, unwanted glitches, noise, or poor transitions. 

You should start with your evaluation by comparing the two responses and provide a short rationale. After providing your rationale, you should output the final verdict by strictly following this seven-point Likert scale: 3 if assistant A is much better, 2 if assistant A is better, 1 if assistant A is slightly better, 0 if the two responses have roughly the same quality, -1 if assistant B is slightly better, -2 if assistant B is better, and -3 if assistant B is much better.

You should format as follows:

[Rationale]: 
[Score]:  """,
    "Audio File A:",
    audio_file_A,
    "Audio File B:",
    audio_file_B,
]

response = model.generate_content(prompt)
print(response.text)