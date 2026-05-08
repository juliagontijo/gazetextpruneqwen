---
license: apache-2.0
task_categories:
- visual-question-answering
language:
- en
size_categories:
- 10K<n<100K
---

# EgoGazeVQA Dataset
Paper: In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting
## Dataset Description

We introduce **EgoGazeVQA**, an egocentric gaze-guided video question answering benchmark that leverages gaze information to improve the understanding of daily-life videos.


## Dataset Structure

### File Organization

```
EgoGazeVQA/
├── metadata.jsonl          
├── videos/                 
│   ├── ego4d/
│   ├── egoexo/
│   └── egtea/
└── README.md              
```

### Data Format

The dataset is organized with the following structure:

**Main Files:**
- `metadata.csv`: Contains QA pairs with video clip references
- `ego4d.json`, `egoexo.json`, `egtea.json`: Narrations with gaze information for each dataset
- `videos/{dataset}/{video_id}/{start_frame}_{end_frame}.mp4`: Video clips organized by dataset

**metadata.csv fields:**
- `file_name`: Video clip path (e.g., `ego4d/b70b1c34-46b6-49af-bddb-505dc7738f96/123_1205.mp4`)
- `video_id`: Source video identifier
- `dataset`: Dataset name (`ego4d`, `egoexo`, or `egtea`)
- `qa_type`: Question type (`causal`, `spatial`, or `temporal`)
- `question`: Question text
- `answer_options`: Multiple choice options (pipe-separated string)
- `correct_answer`: Correct answer label

**Narration JSON structure** (`{dataset}.json`):
```json
{
  "video_id": {
    "narrations": [
      {
        "timestamp_sec": 4.124,
        "timestamp_frame": 123,
        "narration_text": "#C C Operates a television",
        "annotation_uid": "uuid",
        "gaze_info": {
          "gaze_x": 0.76,
          "gaze_y": 0.44,
          "confidence": 1.0
        },
        "image_path": "video_id/123.jpg"
      }
    ]
  }
}
```

**Video file naming convention:**
- Format: `{start_frame}_{end_frame}.mp4`
- Example: `123_1205.mp4` contains frames 123 to 1205

## Usage

### Quick Start

```python
import os
import json
from datasets import load_dataset

# Load dataset
dataset = load_dataset("taiyi09/EgoGazeVQA", split="train")

from huggingface_hub import hf_hub_download
for ds_name in ["ego4d", "egoexo", "egtea"]:
    hf_hub_download(
        repo_id="taiyi09/EgoGazeVQA",
        filename=f"{ds_name}.json",
        repo_type="dataset",
        local_dir="./narrations"
    )

def parse_video_path(video_path):
    """Extract dataset, video_id, start_frame, end_frame from video path"""
    parts = video_path.split("/")
    dataset = parts[0] if parts[0] in ["ego4d", "egoexo", "egtea"] else parts[1]
    video_id = parts[1] if parts[0] in ["ego4d", "egoexo", "egtea"] else parts[2]
    filename = parts[-1].replace(".mp4", "")
    start_frame, end_frame = map(int, filename.split("_"))
    return dataset, video_id, start_frame, end_frame

def load_gaze_sequence(dataset_name, video_id, start_frame, end_frame, narrations_dir="./narrations"):
    """Load gaze sequence for a video clip"""
    json_path = os.path.join(narrations_dir, f"{dataset_name}.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    
    if video_id not in data:
        return []
    
    narrations = data[video_id].get("narrations", [])
    gaze_sequence = []
    
    for narr in narrations:
        frame = narr.get("timestamp_frame")
        gaze_info = narr.get("gaze_info", {})
        if start_frame <= frame <= end_frame and gaze_info:
            gaze_sequence.append({
                "frame": frame,
                "x": gaze_info.get("gaze_x"),
                "y": gaze_info.get("gaze_y")
            })
    
    return sorted(gaze_sequence, key=lambda x: x["frame"])

sample = dataset[0]

video_path = sample["video"]
question = sample["question"]
answer_options = sample["answer_options"].split("|")  # Convert to list
correct_answer = sample["correct_answer"]

# Get gaze sequence
dataset_name, video_id, start_frame, end_frame = parse_video_path(video_path)
gaze_sequence = load_gaze_sequence(dataset_name, video_id, start_frame, end_frame)

```

### Batch Processing

```python
for i in range(10):
    sample = dataset[i]
    
    video_path = sample["video"]
    question = sample["question"]
    correct_answer = sample["correct_answer"]
    
    dataset_name, video_id, start_frame, end_frame = parse_video_path(video_path)
    gaze_sequence = load_gaze_sequence(dataset_name, video_id, start_frame, end_frame)
    
```

### Accessing Video Files

Videos are automatically downloaded when accessing the `video` field. If you need the local path:

```python
sample = dataset[0]
video = sample["video"]  # This will download the video if not cached

# Video is either a path string or a dict with 'path' key
if isinstance(video, dict):
    video_path = video.get("path")
else:
    video_path = video
```


## Citation
```bibtex
@misc{peng2025eyemllmbenchmarkingegocentric,
      title={In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting}, 
      author={Taiying Peng and Jiacheng Hua and Miao Liu and Feng Lu},
      year={2025},
      eprint={2509.07447},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.07447}, 
}
```
## License

Apache 2.0
