from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"

img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

frequencies, _, _ = plt.hist(
    img.ravel(),
    bins=256,
    range=[0, 256],
)

peak_intensity = int(np.argmax(frequencies))
peak_frequency = int(frequencies[peak_intensity])

print(f"Shape:           {img.shape}")
print(f"Total pixels:    {img.size:,}")
print(f"Minimum:         {int(img.min())}")
print(f"Maximum:         {int(img.max())}")
print(f"Mean:            {img.mean():.2f}")
print(f"Median:          {np.median(img):.1f}")
print(f"Peak intensity:  {peak_intensity}")
print(f"Peak frequency:  {peak_frequency}")
print(f"Histogram total: {int(frequencies.sum()):,}")

plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.xlim([0, 256])
plt.show()
