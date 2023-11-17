import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from MultiMEDal_multimodal_medical.src.datasets.dataset_init import get_datasets, get_combined_datasets
from MultiMEDal_multimodal_medical.src.datasets.custom_concat_dataset import CustomConcatDataset
from MultiMEDal_multimodal_medical.src.datasets.data_loader import get_dataloaders
from MultiMEDal_multimodal_medical.src.datasets.preprocessing.prompt_factory import tab2prompt_breast_lesion

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def preprocess_llava(model_config, image_processor, tokenizer, image, text):
    qs = text
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_tensor = process_images([image], image_processor, model_config)[0]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    return image_tensor, input_ids


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    from MultiMEDal_multimodal_medical.src.datasets.data_transform import build_transform_dict_llava

    transform_dict, _ = build_transform_dict_llava(model_path, args.model_base)

    tokenizer, model, _, _ = load_pretrained_model(model_path, args.model_base, model_name)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    # Load Dataset
    dataset_name = [
      ["CBIS-DDSM-tfds-with-tabular-2classes"],
      ["CBIS-DDSM-tfds-with-tabular-2classes"],
      ["CBIS-DDSM-tfds-with-tabular-2classes"],
    ]
    data_dir = [[None], [None], [None]]

    combined_datasets = get_combined_datasets(
        dataset_name[0],
        dataset_name[1],
        dataset_name[2],
        transform_dict,
        data_dir[0],
        data_dir[1],
        data_dir[2],
    )
    all_train_datasets, all_val_datasets, all_test_datasets = combined_datasets
    train_dataset = CustomConcatDataset(all_train_datasets)
    val_dataset = CustomConcatDataset(all_val_datasets)
    test_dataset = CustomConcatDataset(all_test_datasets)


    # Get Dataloader
    train_sampler, val_sampler = None, None
    batch_size = 1
    njobs = 4
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset,
        test_dataset,
        val_dataset,
        train_sampler,
        val_sampler,
        batch_size,
        njobs,
    )

    print("Total #samples:", len(test_dataloader))

    
    chunk_idx_list = get_chunk(np.arange(len(test_dataloader)), args.num_chunks, args.chunk_idx)

    for idx, batch_data in tqdm(enumerate(test_dataloader)):
        if idx < chunk_idx_list[0] or idx > chunk_idx_list[-1]:
            continue
        image_tensor, labels = batch_data["image"], batch_data["label"]
    
        processed_text_samples, text_samples = tab2prompt_breast_lesion("llava", "test",
                                                            batch_data, tokenizer, model.config.mm_use_im_start_end, 
                                                            args.conv_mode, answer_mode=args.answer_mode)
        input_ids = processed_text_samples[0].unsqueeze(0)
        cur_prompt = text_samples[0]

        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)
      
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)
        

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    parser.add_argument("--answer-mode", type=str, default="short")
    
    args = parser.parse_args()

    

    eval_model(args)
