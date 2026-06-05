import numpy as np
from PIL import Image
from datasets import Dataset


def mask_to_img(mask: np.ndarray) -> Image.Image:
    """
    把 numpy mask 转成 PIL Image
    支持 bool/float/uint8 mask，自动转换到 0-255 uint8 灰度图。
    """
    # 1) 标准化到 0 - 255 区间
    if mask.dtype == bool:
        # bool mask → 0/255
        arr = (mask.astype(np.uint8) * 255)
    elif np.issubdtype(mask.dtype, np.floating):
        # float mask 假设范围 0~1 → 0~255
        arr = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        # 整数类型 → clip 保证 0~255
        arr = np.clip(mask, 0, 255).astype(np.uint8)

    # 2) Pillow 转为灰度图
    # 如果是单通道 mask（h, w）
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        # 如果意外是多通道（h, w, c），直接交给 Pillow 自动识别
        img = Image.fromarray(arr)
    return img


def img_to_mask(img: Image.Image) -> np.ndarray:
    """
    把 PIL Image 转回 numpy mask（二值 0/1）。
    - 如果 image 是灰度/单通道 → 0/1
    - 如果 image 是彩色 → 按灰度阈值转成二值 mask
    """
    # 1) 转成灰度
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    # 2) 二值化：大于 0 视为前景
    # 如果原来就是 0/255 mask，它就会正确映射成 0/1
    mask = (arr > 0).astype(np.uint8)
    return mask


def overlay_mask_on_image(
    image_path: str, 
    mask: Image.Image, 
    color=(255, 255, 0), 
    alpha=128,
):
    """
    Args:
        image: PIL Image, RGB or RGBA
        mask: PIL Image, binary (white=1 region, black=0)
        color: tuple (R,G,B) highlight color
        alpha: int 0-255 transparency (0=transparent, 255=opaque)
    Returns:
        PIL Image with mask overlay
    """
    # Convert input images to RGBA so we can blend
    image = Image.open(image_path).convert("RGBA")
    mask = mask.convert("L")  # grayscale mask: white=255, black=0

    # Create a colored overlay the same size as the image
    overlay = Image.new("RGBA", image.size, color + (0,))  # start fully transparent

    # Create a mask with the given alpha where mask is white
    # mask pixel==255 → alpha (semi-transparent) at that area
    overlay_mask = Image.new("L", image.size, 0)
    overlay_mask.paste(alpha, (0, 0), mask)

    # Apply the overlay with transparency only where the mask is white
    overlay.putalpha(overlay_mask)

    # Composite the overlay onto the original image
    result = Image.alpha_composite(image, overlay)

    return result


def save_datalist_to_disk(data_list, features, save_path):
    ds = Dataset.from_list(data_list, features=features)
    ds.save_to_disk(save_path)
