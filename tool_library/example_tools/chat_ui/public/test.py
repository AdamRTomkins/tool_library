from PIL import Image


def convert_png_to_favicon(png_file_path, output_path="favicon.ico"):
    """
    Convert a PNG image to a favicon.ico file.

    Parameters:
    - png_file_path: Path to the input PNG file.
    - output_path: Path where the favicon.ico file will be saved.
    """
    # Standard sizes for favicon files
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]

    # Open the original image
    image = Image.open(png_file_path)

    # Create a list to hold the resized images
    icon_images = []

    # Resize image to each of the specified sizes and append to list
    for size in sizes:
        resized_image = image.resize(size, Image.Resampling.LANCZOS)
        icon_images.append(resized_image)

    # Save the icon image with multiple sizes
    icon_images[0].save(
        output_path, format="ICO", sizes=[(s.width, s.height) for s in icon_images]
    )


# Example usage
convert_png_to_favicon("logo.png")
