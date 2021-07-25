# league-healthbar-detection
attempt at using python/openCV to detect health bars from an in-game screenshot

## How it works
I use OpenCV to apply masks and transform the image and then I use contour detecting methods to find all rectangles of a particular size that resemble a health bar.

## Example screens:
Input:
![screen1](https://i.imgur.com/3HxivOv.png)
Output: 
HEALTH BARS DETECTED ON SCREEN: 4
![screen2](https://i.imgur.com/wk6bIth.png)
