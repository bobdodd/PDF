import os
import shutil
import tempfile
import random
import string
import cv2
import click
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class DataCollector:
    """UI for collecting and labeling text samples for the italic detector."""
    
    def __init__(self, 
                 source_dir: str,
                 italic_dir: str = "data/italic",
                 regular_dir: str = "data/spare_regular",
                 temp_dir: str = None):
        """Initialize the data collector."""
        self.source_dir = source_dir
        self.italic_dir = italic_dir
        self.regular_dir = regular_dir
        
        # Create temp directory if not provided
        if temp_dir:
            self.temp_dir = temp_dir
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="italic_detector_")
            
        # Ensure directories exist
        os.makedirs(self.italic_dir, exist_ok=True)
        os.makedirs(self.regular_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Counter for labeled images
        self.italic_count = 0
        self.regular_count = 0
        
    def _generate_unique_filename(self, prefix: str, extension: str = ".png") -> str:
        """Generate a unique filename with a prefix."""
        timestamp = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"{prefix}_{timestamp}{extension}"
    
    def _display_image(self, image_path: str) -> None:
        """Display an image for labeling."""
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Create figure and axis
        fig = Figure(figsize=(8, 2))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Display image
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        fig.tight_layout()
        
        # Save to a temporary file
        temp_display = os.path.join(self.temp_dir, "current_image_display.png")
        canvas.print_figure(temp_display)
        
        # Display using PIL
        try:
            display_img = Image.open(temp_display)
            display_img.show()
        except Exception as e:
            print(f"Error displaying image: {e}")
            print(f"Image path: {image_path}")
            
    def _process_image(self, image_path: str, label: int) -> None:
        """Process an image based on the label (1=italic, 2=regular)."""
        try:
            if label == 1:  # Italic
                # Generate unique filename
                target_filename = self._generate_unique_filename("italic")
                target_path = os.path.join(self.italic_dir, target_filename)
                
                # Copy to italic directory
                shutil.copy2(image_path, target_path)
                self.italic_count += 1
                
                print(f"Added to italic collection ({self.italic_count})")
                
            elif label == 2:  # Regular
                # Generate unique filename
                target_filename = self._generate_unique_filename("regular")
                target_path = os.path.join(self.regular_dir, target_filename)
                
                # Copy to regular directory
                shutil.copy2(image_path, target_path)
                self.regular_count += 1
                
                print(f"Added to regular collection ({self.regular_count})")
                
            else:
                print("Skipped")
                
        except Exception as e:
            print(f"Error processing image: {e}")
    
    def collect_labels_from_directory(self, directory: str) -> None:
        """Collect labels for all images in a directory."""
        # Get all image files
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"No images found in {directory}")
            return
            
        print(f"Found {len(image_files)} images to label")
        print("Press '1' for italic, '2' for regular, any other key to skip")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            print(f"\nImage {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Display image
            self._display_image(image_path)
            
            # Get label
            label = input("Enter label (1=italic, 2=regular, other=skip): ").strip()
            
            try:
                label = int(label)
            except ValueError:
                label = 0
                
            # Process image
            self._process_image(image_path, label)
            
        print(f"\nLabeling complete! Added {self.italic_count} italic and {self.regular_count} regular images.")

def collect_labels(source_dir: str, italic_dir: str, regular_dir: str, temp_dir: str = None) -> None:
    """Collect labels for text images in a directory."""
    collector = DataCollector(
        source_dir=source_dir,
        italic_dir=italic_dir,
        regular_dir=regular_dir,
        temp_dir=temp_dir
    )
    
    collector.collect_labels_from_directory(source_dir)