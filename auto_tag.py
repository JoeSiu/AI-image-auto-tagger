#!/usr/bin/env python
"""
Automated tagging script that runs without browser UI
All options can be controlled via command-line arguments
"""
import argparse
from wdv3tagger import tag_images

def str_to_bool(value):
    """Convert string to boolean"""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Image Captioning and Tagging Tool (No Browser UI)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required argument
    parser.add_argument("image_folder", type=str, help="Path to the image directory")
    
    # Optional arguments with defaults
    parser.add_argument("--recursive", type=str_to_bool, default=False, 
                        help="Process subdirectories (default: False)")
    parser.add_argument("--general-thresh", type=float, default=0.35, 
                        help="General tags threshold (default: 0.35)")
    parser.add_argument("--character-thresh", type=float, default=0.85, 
                        help="Character tags threshold (default: 0.85)")
    parser.add_argument("--hide-rating-tags", type=str_to_bool, default=True, 
                        help="Hide rating tags (default: True)")
    parser.add_argument("--character-tags-first", type=str_to_bool, default=False, 
                        help="Character tags first (default: False)")
    parser.add_argument("--remove-separator", type=str_to_bool, default=False, 
                        help="Remove separator (replace _ with spaces) (default: False)")
    parser.add_argument("--overwrite-tags", type=str_to_bool, default=False, 
                        help="Overwrite existing metadata tags (default: False)")
    parser.add_argument("--skip-if-tagged", type=str_to_bool, default=False, 
                        help="Skip images that already have metadata tags (default: False)")
    parser.add_argument("--output-to", type=str, choices=["Text File", "Metadata"], default="Metadata", 
                        help="Output to Text File or Metadata (default: Metadata)")
    parser.add_argument("--sort-order", type=str, 
                        choices=["Newest First", "Oldest First", "Name (A-Z)", "Name (Z-A)", "None"], 
                        default="Newest First", 
                        help="Sort files by (default: Newest First)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AI Image Auto-Tagger (No Browser Mode)")
    print("="*60)
    print(f"\nTarget folder: {args.image_folder}")
    print("\nSettings:")
    print(f"  - Output to: {args.output_to}")
    print(f"  - Sort order: {args.sort_order}")
    print(f"  - Recursive: {args.recursive}")
    print(f"  - General threshold: {args.general_thresh}")
    print(f"  - Character threshold: {args.character_thresh}")
    print(f"  - Hide rating tags: {args.hide_rating_tags}")
    print(f"  - Character tags first: {args.character_tags_first}")
    print(f"  - Remove separator: {args.remove_separator}")
    print(f"  - Overwrite tags: {args.overwrite_tags}")
    print(f"  - Skip if tagged: {args.skip_if_tagged}")
    print("\nProcessing...\n")
    
    # Call tag_images with provided parameters
    status, processed, skipped = tag_images(
        image_folder=args.image_folder,
        recursive=args.recursive,
        general_thresh=args.general_thresh,
        character_thresh=args.character_thresh,
        hide_rating_tags=args.hide_rating_tags,
        character_tags_first=args.character_tags_first,
        remove_separator=args.remove_separator,
        overwrite_tags=args.overwrite_tags,
        skip_if_tagged=args.skip_if_tagged,
        output_to=args.output_to,
        sort_order=args.sort_order
    )
    
    print("\n" + "="*60)
    print(f"{status}")
    print("="*60)
    if processed:
        print(f"\nProcessed files:\n{processed}")
    if skipped:
        print(f"\nSkipped files:\n{skipped}")

