# ⚽ Football Object Tracking & Re-Identification

A modular video analysis pipeline that tracks players and the ball in football matches using YOLOv8 + ByteTrack + TorchReID. Goalkeepers and referees are normalized as players to maintain consistent re-identification.

---

## 🚀 Features

- 🎯 **YOLOv8-based detection** trained on custom football classes
- 🧠 **Re-Identification (ReID)** using `osnet_ain_x0_25` via TorchReID
- 🔁 **Short-term memory gallery** for cosine similarity-based ID assignment
- 🔍 **ByteTrack tracking** integrated via Supervision
- 🔄 **Class Normalization**: goalkeeper & referee → player
- 💾 **Stub caching** for faster debugging
- 🔧 **Modular and easily extendable**

---

```markdown
## 📁 Directory Structure


.                # Custom YOLOv8 model (football classes)
├── stubs/
│   ├── gallery.pkl             # Pickled embeddings (ReID gallery)
│   └── tracks\_with\_global\_ids.pkl  # Cached tracking results
├── videos/
│   ├── input/                  # Input video files
│   └── output/                 # Output frames with drawn tracks
├── main.py                    # Entry point
├── tracker.py                 # Core tracking + ReID logic
├── utils.py                   # I/O utilities
├── requirements.txt           # Dependencies
└── README.md                  # You’re looking at it

```

---

## 🧪 YOLOv11(Custom Trained) Class Map

````
```python
{0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
````

> ⚠️ All detections of classes `1` (goalkeeper) and `3` (referee) are **remapped to `2` (player)** to simplify tracking and identity consistency.

---

## 🔧 Installation

1. Clone the repo:

```bash
git clone https://github.com/yourname/football-reid.git
cd football-reid
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your input videos in `videos/input/`
4. Ensure YOLO weights are in `model/best.pt`

---

## ▶️ Run the Pipeline

```bash
python main.py
```

* Detects objects using YOLOv8
* Normalizes and tracks valid classes (`ball`, `player`)
* Extracts embeddings using TorchReID
* Assigns persistent global IDs
* Saves annotated frames to `videos/output/`

---

## ⚙️ Configuration (tracker.py)

| Parameter          | Purpose                              | Default    |
| ------------------ | ------------------------------------ | ---------- |
| `conf`             | YOLO confidence threshold            | `0.1`      |
| `reid_gallery`     | ReID embedding memory (deque length) | `30`       |
| `cosine threshold` | Similarity threshold for ReID        | `0.75–0.8` |
| `batch_size`       | Inference batch size                 | `20`       |

---

## 📦 Dependencies
* [TorchReID](https://github.com/KaiyangZhou/deep-person-reid)
* `OpenCV`, `scikit-learn`, `numpy`, `tqdm`, `pickle`

Install them via:

```bash
pip install -r requirements.txt
```

---

## 📸 Output

Annotated frames and videos are saved to:

```
videos/output/
```

Each tracked object is labeled with:

* `GID: <global_id>` – consistent identity across frames
* Optionally `LID: <local_track_id>` – ByteTrack ID (can be toggled)

---

## 🧠 ReID Strategy

* For each tracked object, crop the image and extract an embedding
* Match it against recent embeddings using cosine similarity
* Assign the closest match if similarity ≥ threshold
* If no match, assign a new global ID
* Maintain a sliding window of recent embeddings (`deque(maxlen=30)`)

---

## 📌 Roadmap

* [ ] Integrate **ByteTrack** for improved appearance + motion tracking
* [ ] ReID **OSNET_AIN_X0_25** model used for similarity matching

---

## 📝 License

This project is intended for academic and research purposes only.
Please ensure you respect the licenses of YOLOv8, TorchReID, and other frameworks used.

---

**Author**: \[Arcane-WD]
Feel free to fork, star, and contribute!
