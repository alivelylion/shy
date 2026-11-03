from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def get_img_feats(img, model=None, processor=None, device='cuda', pretrained_name_or_path='openai/clip-vit-large-patch14'):
    if model is None:
        global clip_model
        global clip_processor
        if clip_model is None:
            clip_processor = AutoProcessor.from_pretrained(pretrained_name_or_path)
            clip_model = CLIPModel.from_pretrained(pretrained_name_or_path).to(device)
        model = clip_model
        processor = clip_processor

    img_feats = processor(images=img, return_tensors="pt").to(device)
    img_feats = model.get_image_features(**img_feats)
    return img_feats


def get_text_feats(text: str, model, tokenizer, device='cuda'):
    text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)

    # Convert the tokenized text to a tensor and move it to the device
    text_input = text_tokens["input_ids"].to(device)

    text_features = model.get_text_features(text_input)

    return text_features


clip_processor = clip_model = clip_tokenizer = None


@torch.no_grad()
def img_clip_score(img: Image.Image, text: str, text_feats=None, img_feats=None, pretrained_name_or_path='openai/clip-vit-large-patch14', device='cuda'):
    global clip_processor, clip_model, clip_tokenizer
    if clip_processor is None:
        clip_processor = AutoProcessor.from_pretrained(pretrained_name_or_path)
        clip_model = CLIPModel.from_pretrained(pretrained_name_or_path).to(device)
        clip_tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path)

    if text_feats is None:
        text_feats = get_text_feats(text, clip_model, clip_processor, device=device)[0]
    if img_feats is None:
        img_feats = get_img_feats(img, clip_model, clip_processor, device=device)[0]
    score = torch.nn.functional.cosine_similarity(text_feats, img_feats, dim=0).item()

    return score, img_feats, text_feats


@torch.no_grad()
def avg_clip_score(prompt_list: List[str], tgt_list: List[str], device='cuda', pretrained_name_or_path='openai/clip-vit-large-patch14', progress_bar: bool = True):

    metric_name = 'avg_clip_score'

    assert len(prompt_list) == len(tgt_list)
    ncompare = len(prompt_list)

    processor = AutoProcessor.from_pretrained(pretrained_name_or_path)
    model = CLIPModel.from_pretrained(pretrained_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name_or_path)

    iters = range(ncompare)

    if progress_bar:
        iters = tqdm(iters, desc=metric_name)

    score_total = 0

    cossim = nn.CosineSimilarity(dim=0)

    for ii in iters:
        ref, tgt_lst = prompt_list[ii], tgt_list[ii]
        if isinstance(tgt_lst, str) and file_is_video(tgt_lst):
            tgt_lst = video_to_frame_list(tgt_lst, output_type='pil')

        if len(tgt_list) < 2:
            continue

        text_feats = get_text_feats(ref, model, tokenizer, device=device)[0]
        score_seq = 0
        for tgt in tgt_lst:
            img_feats = get_img_feats(tgt, model, processor, device=device)[0]
            score = cossim(text_feats, img_feats).item()
            score_seq += score

        score_total += score_seq / len(tgt_lst)

        if progress_bar:
            score_tmp = float(score_total / (ii + 1))
            iters.set_description(f'{metric_name}: {score_tmp:.3f}')

    return float(score_total / ncompare)