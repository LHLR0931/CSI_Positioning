# AR reference image

`CSI_Reference.png` is the image registered in the Unity AR reference library. `CSI_Reference.svg` is its vector source, and `index.html` provides a simple full-screen display page.

## Display or print

The complete image must remain visible and unchanged. Do not crop the border, title or top-direction mark.

For a screen display:

1. Open `index.html` in a browser.
2. Adjust the display size and keep browser zoom fixed.
3. Measure the complete rendered image width with a ruler.
4. Convert the measurement to metres and enter that value in the app.

For a print:

1. Print at a fixed scale without “fit to page”.
2. Measure the printed image itself, excluding the paper margin.
3. Mount it flat, vertical and immobile, with the marked top edge upward.
4. Record the final image width, centre map coordinate, centre height and front-facing direction.

The measured physical width controls the AR image scale. Pixel dimensions, browser CSS size and paper size cannot replace this measurement.

## Use in the Unity client

1. Open **Positioning → AR camera**.
2. Enter the measured image pose and width in **Image measurements**.
3. Select **Scan image** and point the camera at the entire image.
4. Wait for image tracking, then select **Lock alignment**.
5. Compare rendered floor points 0, 4 and 10 with the physical grid before confirming alignment.

Check the overlay from several positions. A successful image detection only establishes an image anchor; it does not prove that the map pose, floor height, scale or CSI position is correct.

If the image is moved, reprinted at another scale, or the saved measurements change, repeat the measurement and alignment. If `CSI_Reference.png` is replaced, the copy under `unity-app/Assets/Resources/AR/` and the Unity reference library must also be updated and the app rebuilt.

## Coordinate convention

- Grid point 0 is the map origin.
- Map `+x` follows points 0 → 4.
- Map `+y` follows points 0 → 10.
- Adjacent grid points are 0.8 m apart.
- The facing angle describes the printed front normal in map coordinates; the image top remains physically upward.

Use [`../FRONTEND_BACKEND_USER_GUIDE.md`](../FRONTEND_BACKEND_USER_GUIDE.md) for the complete positioning and recovery procedure.
