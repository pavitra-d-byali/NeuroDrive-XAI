import os
import urllib.request

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {filepath} from {url}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                out_file.write(response.read())
            print("Downloaded.")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
    else:
        print(f"{filepath} already exists.")

def main():
    os.makedirs("demo", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    
    # Download a real demo video instead of synthetic
    video_url = "https://github.com/datvuthanh/HybridNets/raw/main/demo/video/1.mp4"
    download_file(video_url, "demo/sample_drive.mp4")

    # Download HybridNets pretrained weights
    weights_url = "https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth"
    download_file(weights_url, "weights/hybridnets.pth")

if __name__ == "__main__":
    main()
