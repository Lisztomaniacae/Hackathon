
# Hackathon 2024 - Submission of Group // **whoami** //

## Team Members:
- **Lou Rainier Gaano**  
- **Saba Nagervadze**  
- **Zhaneta Gasparyan**  
- **Maria Grigoryan**  
- **Mykyta Balandin**  
- **Selin Ilayda Özdemir**

---

## Project Description
This project focuses on solving a robotic manipulation task using **convolutional neural networks (CNNs)** integrated with **attention mechanisms** and **U-Net architecture**. The model is designed to make precise alignment predictions for **manufacturing automation**.  

We have optimized the model for both **accuracy** and **runtime efficiency** to ensure practical applicability in real-world automation scenarios.

---

## How to Run

### Option 1: Direct Execution
1. Update the paths in the following files to match your machined parts, grippers, and masks:
   - `evaluate/task.csv`
   - `evaluate/ground_truth.csv`
2. Run the evaluation script:
   ```bash
   python3 evaluate/eval.py
   ```
   or
   ```bash
   python evaluate/eval.py
   ```

### Option 2: Alternative Script
Run:
```bash
python solution/main.py evaluate/task.csv evaluate/ground_truth.csv
```

### Option 3: Docker (Without Visualization)
1. Build the Docker image:
   ```bash
   docker build . -t my-group
   ```
2. Run the container:
   ```bash
   docker run -it my-group python evaluate/eval.py
   ```

---

## Design Decisions
- **Architecture**:  
  We adopted a **multi-stage architecture** combining **CNN layers**, **attention mechanisms**, and **U-Net-style upsampling** for high-precision predictions. This design allows for more accurate feature extraction and alignment adjustments.

---

## Challenges Faced
1. **Device Compatibility**:  
   We encountered issues with **MPS devices** and dataset variability during training.
   
2. **Resource Limitations**:  
   Fine-tuning model performance and optimizing runtime were crucial due to resource constraints.

3. **Multiprocessing Issues**:  
   Troubleshooting **PyTorch DataLoader multiprocessing** (e.g., adjusting the number of workers) was a key challenge. We overcame this by modifying global environment variables and settings for shared memory.

4. **Custom Loss Functions**:  
   We explored custom loss functions to fine-tune the model's response to outliers and out-of-bound predictions, improving overall stability.

---

## Features
- **Gripper Overlay**:  
  Adjustable parameters for X, Y, and rotation angle.  

- **Dataset Creation**:  
  Tools for generating datasets tailored for machine learning.

---

## Additional Notes
- This project demonstrates a practical solution for **manufacturing automation**, emphasizing both **precision** and **efficiency**.
