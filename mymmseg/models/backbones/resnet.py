# gt=None → trả về logits thô, bạn tự tính loss
logits = model(img)

# gt có giá trị → trả về loss dict như mmseg
losses = model(img, gt_semantic_seg=gt)