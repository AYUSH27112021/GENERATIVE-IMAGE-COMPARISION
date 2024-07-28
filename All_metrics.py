from os.path import dirname, abspath
import sys
import torch
# Add the directory containing the files to the system path
sys.path.append(abspath(dirname(__file__)) + r"\ASTHETIC_SCORE")
# sys.path.append(abspath(dirname(__file__)) + r"\CLIP")
sys.path.append(abspath(dirname(__file__)) + r"\SAM_CLIP")
# sys.path.append(abspath(dirname(__file__)) + r"\Q_ALING")

import argparse
from ASTHETIC_SCORE import asthetic_score
from CLIP import CLIP_single_image_prompt
from SAM_CLIP import Sam_Clip
from Q_ALING import q_aling_hf

def clear_gpu_cache():
    torch.cuda.empty_cache()

def main(image_path, input_string, all_metric_output):
    result_dict=dict()
    result1 = asthetic_score.predict_aesthetic_score(image_path=image_path)
    clear_gpu_cache()
    result2 = CLIP_single_image_prompt.compute_image_text_similarity(image_path=image_path,text_input=input_string)
    clear_gpu_cache()
    result3 = q_aling_hf.get_image_scores(image_path=image_path)
    clear_gpu_cache()
    result4 = Sam_Clip.Sam_Clip(image_path=image_path,prompt=input_string)
    clear_gpu_cache()
    result4_sum = sum(result4) / len(result4)

    if all_metric_output:
        result_dict = {
            "ASTHETIC_SCORE": result1,
            "CLIP": result2,
            "Q_ALING Quality score": result3[0],
            "Q_ALING Asthetic score": result3[1],
            "SAM_CLIP": result4
        }
        print(result_dict)
    else:
        final_score = ((result1/10) + result2 + (result3[0]/5) + (result3[1]/5) + result4_sum) / 5
        print(final_score)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process an Generated image and the prompt string using various metrics and return either individual results or a final score.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("input_string", type=str, help="Input string to process")
    parser.add_argument("all_metric_output", type=bool, help="Boolean flag to print all metrics individualy or just final score")
    
    args = parser.parse_args()
    
    main(args.image_path, args.input_string, args.all_metric_output)


