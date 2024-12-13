import sys
sys.path.append('/kaggle/working/LongVU')

import numpy as np
import pandas as pd
import math
import argparse
import os
import torch
from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval(args):

    model_path = args.model_path
    model_name = args.model_name
    data_path = args.data_path
    version = args.version

    # model_id = "Vision-CAIR/LongVU_Llama3_2_3B"
    # snapshot_download(repo_id=model_id, local_dir=model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )

    df = pd.read_csv("EnTube/EnTube_filtered.csv")
    preds = []
    truths = []
    for i, row in df.iterrows():
        print(f"At video {i}")
        if row['video_id'] == '4RJuIinnP-E':
            continue # skip partially downloaded video
        model.eval()
        label = row['engagement_rate_label']
        video_path = os.path.join(data_path, str(label), row['video_id'] + ".mp4")
        print(f"video_id: {row['video_id']}")
        # qs = "This video is a Youtube video on one of the following categories: Education, Film & Animation, Comedy, Entertainment, Music, Howto & Style, and People & Blogs. The engagement rate defined for each such video is based on the number of potential likes and dislikes only when published on Youtube. The exact formula for the score is (likes-dislikes) / (likes+dislikes) and the final prediction label is either 0 (not engaged), 1 (neutral), or 2 (engaged) based on thresholding this score. Please predict one of the three labels for this video, based on its contents only."
        # qs = "Classify this video into one of three engagement levels by printing out ONLY a character of 0, 1, or 2, corresponding to being not engaged, neutral, or engaged, respectively."
        # qs = "Please print out either 0, 1, or 2 (corresponding to the video being not engaged, neutral, or engaged). After that, please explain why you evaluate the video to be not engaged, neutral, or engaged with the audience (in terms of likes over dislikes when published to social media)."
        # qs = "Please print out either 0, 1, or 2."
        qs = "The video is from Youtube and is meant to be published on this platform for users to later watch and interact. The engagement rate is calculated based on the number of likes and dislikes the video receives. The formula for the engagement rate is (likes-dislikes) / (likes+dislikes) where the score is between -1 and 1, inclusive. The final label is either 0 (not engaged), 1 (neutral), or 2 (engaged) based on the engagement rate. Please predict the label for this video first by printing out either 0, 1, or 2. Then, explain in words why the video is not engaged, neutral, or engaged."

        ###
        # For Llama 3.2 (3B) and 16GB GPU memory: set # frame <= 56 for each video
        ###

        FRAME_CONST = 50
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        # print(f'len(vr) = {len(vr)}')
        fps = float(vr.get_avg_fps())
        # print(f'avg_fps = {fps}')
        coeff = len(vr) / (FRAME_CONST*fps)
        frame_indices = np.array([i for i in range(0, len(vr), math.floor(coeff*fps),)]) # @tcm: for cuda memory limit
        # print(f'# frames: {frame_indices.shape[0]}')
        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)
        # print(f'frame shape: {img.shape}')
        video = np.stack(video)
        image_sizes = [video[0].shape[:2]]
        # print(f'image_sizes = {image_sizes}')
        video = process_images(video, image_processor, model.config)
        video = [item.unsqueeze(0) for item in video]
        # print(video[:2])

        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[version].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            torch.cuda.empty_cache() # @tcm: for cuda memory limit
            output_ids = model.generate(
                input_ids,
                images=video,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f'Output: {pred}')
        # pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        # print(f"Predicted: {pred}, Actual: {label}")
        # if pred[-1].isnumeric():
        #     print(f'pred={int(pred[-1])}, label={label}')
        #     preds.append(int(pred[-1]))
        #     truths.append(label)
        ok = False
        for c in pred:
            if c.isnumeric():
                print(f'pred={int(c)}, label={label}')
                preds.append(int(c))
                truths.append(label)
                ok = True
                break
        if not ok:
            print('@tcm: not found')

    accuracy = accuracy_score(truths, preds)
    print(f"Accuracy: {accuracy:.2f}")

    precision = precision_score(truths, preds, average=None)
    recall = recall_score(truths, preds, average=None)
    f1 = f1_score(truths, preds, average=None)

    for cls in range(len(precision)):
        print(f"Class {cls} - Precision: {precision[cls]:.2f}, Recall: {recall[cls]:.2f}, F1-Score: {f1[cls]:.2f}")

    macro_precision = precision_score(truths, preds, average='macro')
    macro_recall = recall_score(truths, preds, average='macro')
    macro_f1 = f1_score(truths, preds, average='macro')

    print(f"\nMacro-Average Precision: {macro_precision:.2f}")
    print(f"Macro-Average Recall: {macro_recall:.2f}")
    print(f"Macro-Average F1-Score: {macro_f1:.2f}")

    micro_precision = precision_score(truths, preds, average='micro')
    micro_recall = recall_score(truths, preds, average='micro')
    micro_f1 = f1_score(truths, preds, average='micro')

    print(f"\nMicro-Average Precision: {micro_precision:.2f}")
    print(f"Micro-Average Recall: {micro_recall:.2f}")
    print(f"Micro-Average F1-Score: {micro_f1:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="./checkpoints/longvu_llama3_2")
    parser.add_argument('--model_name', default="cambrian_llama")
    parser.add_argument('--version', default="llama3")
    # parser.add_argument('--local-rank', default=0)
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()

    eval(args)

    # if "llama3" in args.version:
    #     args.model_name = "cambrian_llama3"

    # train(args)