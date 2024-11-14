import numpy as np
import pandas as pd
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
from huggingface_hub import snapshot_download

def val(args):

    model_path = args.model_path
    model_name = args.model_name
    data_path = args.data_path

    model_id = "Vision-CAIR/LongVU_Llama3_2_3B"
    snapshot_download(repo_id=model_id, local_dir=model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name,
    )

    df = pd.read_csv("EnTube_filtered.csv")
    for i, row in df.iterrows():    

        model.eval()
        label = row['engagement_rate_label']
        video_path = os.path.join(data_path, str(label), row['video_id'] + ".mp4")
        qs = "This video is a Youtube video on one of the following categories: Education, Film & Animation, Comedy, Entertainment, Music, Howto & Style, and People & Blogs. The engagement rate defined for each such video is based on the number of potential likes and dislikes only when published on Youtube. The exact formula for the score is (likes-dislikes) / (likes+dislikes) and the final prediction label is either 0 (not engaged), 1 (neutral), or 2 (engaged) based on thresholding this score. Please predict one of the three labels for this video, based on its contents only."

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())
        frame_indices = np.array([i for i in range(0, len(vr), round(fps),)])
        video = []
        for frame_index in frame_indices:
            img = vr[frame_index].asnumpy()
            video.append(img)
        video = np.stack(video)
        image_sizes = [video[0].shape[:2]]
        video = process_images(video, image_processor, model.config)
        video = [item.unsqueeze(0) for item in video]

        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
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
        print(f"Predicted: {pred}, Actual: {label}")

        if i == 2:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="../checkpoints/longvu_llama3_2")
    parser.add_argument('--model_name', default="cambrian_llama")
    parser.add_argument('--version', default="llama3")
    parser.add_argument('--local-rank', default=0)
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()

    eval(args)

    # if "llama3" in args.version:
    #     args.model_name = "cambrian_llama3"

    # train(args)