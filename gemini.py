import os
import glob
import json
import time
import re
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure with API key from environment
api_key = os.getenv('GEMINIKEY')
if not api_key:
    raise ValueError("GEMINIKEY not found in environment variables")

genai.configure(api_key=api_key)

def extract_score(response_text):
    """Extract numerical score from Gemini response"""
    # Look for [Score]: followed by a number
    score_match = re.search(r'\[Score\]:\s*(-?\d+)', response_text)
    if score_match:
        return int(score_match.group(1))
    # Fallback: look for just the number at the end
    fallback_match = re.search(r'(-?\d+)\s*$', response_text.strip())
    if fallback_match:
        return int(fallback_match.group(1))
    return None

def evaluate_audio_pair(model, gen_file, ref_file, config_name, audio_name):
    """Evaluate a single pair of generated vs reference audio"""
    print(f"🎵 Evaluating {config_name}/{audio_name}...")
    
    # Upload files
    print(f"  Uploading {gen_file}...")
    gen_audio = genai.upload_file(path=gen_file)
    
    print(f"  Uploading {ref_file}...")
    ref_audio = genai.upload_file(path=ref_file)
    
    # Create prompt (keeping it EXACTLY the same)
    prompt = [
        """
Please act as an impartial judge and evaluate the overall audio quality of the responses provided by two AI assistants. You should choose the assistant that produced the better audio.

Your evaluation should focus only on technical audio quality. Consider factors such as fidelity (is the audio clean and clear?), realism, unwanted glitches, noise, or poor transitions. 

You should start with your evaluation by comparing the two responses and provide a short rationale. After providing your rationale, you should output the final verdict by strictly following this seven-point Likert scale: 3 if assistant A is much better, 2 if assistant A is better, 1 if assistant A is slightly better, 0 if the two responses have roughly the same quality, -1 if assistant B is slightly better, -2 if assistant B is better, and -3 if assistant B is much better.

You should format as follows:

[Rationale]: 
[Score]:  """,
        "Assistant A's audio:",
        gen_audio,
        "Assistant B's audio':",
        ref_audio,
    ]
    
    # Get response
    response = model.generate_content(prompt)
    
    # Extract score
    score = extract_score(response.text)
    
    return {
        'config': config_name,
        'audio': audio_name,
        'generated_file': gen_file,
        'reference_file': ref_file,
        'response': response.text,
        'score': score
    }

def main():
    print("🎵 Starting Gemini Audio Evaluation for All 9 Configs")
    print("=" * 60)
    
    model = genai.GenerativeModel(model_name="gemini-2.5-pro")
    
    # Get reference files from chunked_ref
    ref_files = sorted(glob.glob("chunked_ref/*.wav"))
    if not ref_files:
        print("❌ No reference files found in chunked_ref/")
        return
    
    print(f"📁 Found {len(ref_files)} reference files")
    
    all_results = []
    config_scores = {}
    
    # Evaluate all 9 configs
    for config_num in range(1, 10):
        config_name = f"config_{config_num}"
        config_dir = f"artifacts/val/novel/{config_name}"
        
        if not os.path.exists(config_dir):
            print(f"⚠️  Directory {config_dir} not found, skipping...")
            continue
            
        print(f"\n🔧 Processing {config_name}...")
        config_results = []
        
        for ref_file in ref_files:
            ref_basename = os.path.basename(ref_file)
            gen_file = os.path.join(config_dir, ref_basename)
            
            if not os.path.exists(gen_file):
                print(f"⚠️  Generated file {gen_file} not found, skipping...")
                continue
            
            try:
                result = evaluate_audio_pair(
                    model, gen_file, ref_file, config_name, ref_basename
                )
                all_results.append(result)
                config_results.append(result)
                
                print(f"  ✅ Score: {result['score']}")
                
                # Small delay to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ Error evaluating {ref_basename}: {e}")
        
        # Calculate config average
        valid_scores = [r['score'] for r in config_results if r['score'] is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            config_scores[config_name] = avg_score
            print(f"📊 {config_name} average score: {avg_score:.2f} ({len(valid_scores)} files)")
        else:
            print(f"📊 {config_name}: No valid scores")
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"gemini_eval_results_{timestamp}.txt"
    
    print(f"\n💾 Saving detailed results to {results_file}...")
    
    with open(results_file, 'w') as f:
        f.write("GEMINI AUDIO EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Total evaluations: {len(all_results)}\n\n")
        
        # Summary by config
        f.write("SUMMARY BY CONFIG:\n")
        f.write("-" * 30 + "\n")
        for config, avg_score in sorted(config_scores.items()):
            f.write(f"{config}: {avg_score:.2f}\n")
        f.write("\n")
        
        # Detailed results
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        for result in all_results:
            f.write(f"\nConfig: {result['config']}\n")
            f.write(f"Audio: {result['audio']}\n")
            f.write(f"Generated: {result['generated_file']}\n")
            f.write(f"Reference: {result['reference_file']}\n")
            f.write(f"Score: {result['score']}\n")
            f.write("Response:\n")
            f.write(result['response'])
            f.write("\n" + "="*50 + "\n")
    
    # Print final summary
    print(f"\n🎯 FINAL SUMMARY:")
    print("=" * 40)
    for config, avg_score in sorted(config_scores.items()):
        print(f"{config}: {avg_score:.2f}")
    
    print(f"\n✅ Evaluation complete! Results saved to {results_file}")
    print(f"Total evaluations: {len(all_results)}")

if __name__ == "__main__":
    main()