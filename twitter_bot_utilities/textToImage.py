
import math
import os
from PIL import Image, ImageFont, ImageDraw

def textToImage(toImage: str):
    # font_path = os.path.join('twitter_bot_utilities', 'Share_Tech_Mono', 'ShareTechMono-Regular.ttf')
    # font_ratio = 1.66
    font_path = os.path.join('twitter_bot_utilities', 'VT323', 'VT323-Regular.ttf')
    font_ratio = 1.9
    
    font_size = 40
    num_chars = len(toImage)
    image_width = font_size*int(math.sqrt(num_chars))
    chars_per_line = int(font_ratio * math.sqrt(num_chars))
    padding = chars_per_line*font_ratio
    font = ImageFont.truetype(font_path, font_size)

    # Split string by words and create lines
    words = toImage.split(" ")
    lines = []
    curr_line_len = 0
    curr_line = ""
    for word in words:
        if len(word) < chars_per_line:  # If the word can fit in a single line
            if curr_line_len + len(word)+1 < chars_per_line: # if the word can fit in current line
                if curr_line_len == 0:  # First word of line, no space
                    curr_line = word
                else:
                    curr_line += " " + word # Words after first need space
                curr_line_len = len(curr_line)
            else:   # Start new line
                lines.append(curr_line)
                curr_line = word
                curr_line_len = len(curr_line)
        else:
            full_line = word[:chars_per_line-1] + "-"
            rest_word = word[chars_per_line-1:]
            while len(full_line) == chars_per_line:
                lines.append(full_line)
                full_line = rest_word[:chars_per_line-1] + "-"
                rest_word = rest_word[chars_per_line-1:]
            curr_line = full_line[:-1]
            curr_line_len = len(curr_line)
    if len(curr_line) > 0:
        lines.append(curr_line)

    img = Image.new('RGB', (image_width, font_size*(len(lines) + 2)), (255,255,255))
    editable = ImageDraw.Draw(img)

    for line_idx in range(len(lines)):
        line = lines[line_idx]
        line = line.center(chars_per_line)
        editable.text((padding, (1+line_idx)*font_size), line, (0,0,0), font=font)

    return img

