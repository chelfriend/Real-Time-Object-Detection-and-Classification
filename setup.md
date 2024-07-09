# Setup Instructions

1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Download the COCO dataset and place it in the `data/` directory.

3. Train the model by running:
    ```bash
    python src/train.py
    ```

4. Convert and optimize the model by running:
    ```bash
    python src/convert_to_onnx.py
    ```

5. Run the real-time object detection application:
    ```bash
    python src/inference.py
    ```

6. Press 'q' to quit the real-time application.
