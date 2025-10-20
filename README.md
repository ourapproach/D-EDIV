# Replication package for paper "Decoupled Edge Data Integrity via Federated Adaptation and Cryptographic Verification"

![ZNAX Framework](./D-EDIV_1.png)
> Figure: A Conceptual Overview 

This repository provides the code and resources for the research paper **"Decoupled Edge Data Integrity via Federated Adaptation and Cryptographic Verification."** The proposed framework provides the efficient and fine-grained data integrity verification in **Mobile Edge Computing (MEC)** environments.

---

## Framework Overview  
The **D-EDIV (Decoupled Edge Data Integrity Verification)** framework introduces an efficient approach for ensuring the integrity of cached data in heterogeneous **MEC** environments. It separates the EDIV process into two stages — **corruption detection** and **corruption localization** — to improve detection accuracy and localize corrupted data with greater precision. In this framework, **intrusion detection models** operate at **edge nodes** to detect possible data corruption using **personalized federated training**, allowing adaptation to each node’s **computational capacity** and **local data distribution**. Once potential anomalies are flagged, the **Application Vendor (AV)** performs **cryptographic verification** using a **digital signature** scheme and a **Merkle tree structure** to authenticate data integrity and precisely locate corrupted blocks. **D-EDIV** achieves **efficient** and **fine-grained integrity verification** for edge-cached data while maintaining **low computational and communication costs**, without requiring an external verifier such as a Third-Party Auditor (TPA).

## Pipeline Stages

### 1. Corruption Detection
A personalized federated training approach enables the adaptation of intrusion detection models across heterogeneous edge nodes. 
- Central server initializes training with an overparameterized reference model on a proxy dataset.
- Generates a global pruning mask to identify and preserve globally important feature channels.
- Disseminates the reference model and global pruning mask to all edge nodes.

Each edge node:
- Uses a network adapter to derive a local pruning mask based on its computational capacity.
- Prunes the reference model while respecting globally important channels.
- Fine-tunes the pruned model using local data.
- Edge nodes transmit fine-tuned models back to the central server.
- Server performs channel-wise aggregation to update the global reference model.
The process iterates for several rounds until each edge node obtains an optimized intrusion detection model adapted to local conditions.

### 2. Corruption Localization
Upon corruption detection, the Application Vendor (AV) initiates a cryptographic verification process.
- The digital signature on the Merkle root is validated to ensure authenticity and prevent replay/forgery attacks.
- Once authenticated, the hierarchical Merkle tree structure is utilized to efficiently verify received proofs.
- Verification is performed against the authenticated root to ensure data integrity to enable fine-grained, block-level localization of corrupted data segments.

---

## Datasets
The **D-EDIV framework** is evaluated using three well-established intrusion detection datasets: **NSL-KDD**, **UNSW-NB15**, and **CIC-IDS2017**.
The experimental setup proceeds as follows:
- The **reference model** is initially trained on the **NSL-KDD** dataset to obtain a foundational understanding of network flow patterns and attack behaviors.
- Following the global pruning stage, local fine-tuning is conducted on the **UNSW-NB15** and **CIC-IDS2017** datasets under various experimental configurations to ensure adaptability across heterogeneous network environments. These datasets can be downloaded from their official repositories.
-  **NSL-KDD:** https://ieee-dataport.org/documents/nsl-kdd-0#files
-  **UNSW-NB15:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
-  **CIC-IDS2017:** https://www.unb.ca/cic/datasets/ids-2017.html 

---

## Dependencies
Install the required packages using:

```bash
pip install -r requirements.txt
```
This project is built with **Python ≥ 3.10** and **PyTorch 2.1.0**. Offline computations are performed on **NVIDIA A100 GPU**.

## Important: Update File Paths
In the source code, look for lines marked with:

```python
# Replace with your actual path
```
Please replace these placeholder paths with the actual paths to data or files. For example:

```python
data_path = "/path/to/data"  # Replace with your actual path
```

---

## How to run D-EDIV?
### Corruption Detection
Run the following script to realize the corruption detection stage:

```python 
Corruption_Detection.py
```
This script implements the **corruption detection** stage of the **D-EDIV** framework, which performs **personalized federated adaptation** for **intrusion detection models** across distributed edge nodes. It trains a global reference model on proxy data, then enables each edge node to prune and fine-tune a local variant using its own data under device-specific latency constraints, producing optimized, resource-aware detection models. Update any lines marked with `# Replace with your actual path` in the script to point to the correct data location on your system.

### Corruption Localization
This script implements the corruption localization stage of the D-EDIV framework, which performs cryptographic verification to identify and isolate corrupted data after a detection alert. It validates replica commitments through digital signatures and Merkle tree proofs, enabling fine-grained, block-level localization of tampered data across distributed edge nodes. To run:

```python
Localization.py
```

Again, make sure to replace the file path with the actual path to your test dataset in the code.
