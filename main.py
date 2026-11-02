import argparse
import sys
from video_processing import process_video

def main():
    parser = argparse.ArgumentParser(
        description="Local BJJ pose analyzer with persistent athlete IDs"
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--json", default="output.json", help="Output JSON path")
    parser.add_argument("--model", required=True, help="Path to MediaPipe .task model")
    parser.add_argument(
        "--sample-rate", type=float, default=10.0,
        help="Pose analysis rate in FPS"
    )
    parser.add_argument(
        "--max-poses", type=int, default=2,
        help="Maximum number of detected people"
    )
    args = parser.parse_args()

    try:
        process_video(
            video_path=args.input,
            output_path=args.output,
            json_path=args.json,
            model_path=args.model,
            sample_rate=args.sample_rate,
            max_poses=args.max_poses,
        )
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
