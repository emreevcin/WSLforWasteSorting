# Illicit Waste Detection: Weakly Supervised Localization

## Project Structure & Execution Order

The original monolithic notebook has been modularized into five sequential stages. To reproduce the results, the notebooks must be executed in order, as they generate intermediate state files (`.pth` and `.pkl`).

1.  **`01_data_audit.ipynb`**
    *   *Purpose:* Handles data loading, preprocessing, and exploratory data analysis. Applies targeted augmentations (Random Erasing, Color Jitter) designed to break spurious correlations.
2.  **`02_model_training.ipynb`**
    *   *Purpose:* Constructs the LSE-ResNet-50 model and executes the training loop. 
    *   *Outputs:* `best_waste_model.pth` (Persisted model weights).
3.  **`03_localization_comparison.ipynb`**
    *   *Purpose:* Restores the trained model and performs a side-by-side qualitative visual comparison of three localization techniques: Manual LSE-Aware CAM, standard Grad-CAM, and Grad-CAM++.
    *   *Requires:* `best_waste_model.pth`
4.  **`04_quantitative_metrics.ipynb`**
    *   *Purpose:* Runs inference on the entire test set. Calculates Classification Accuracy, F1-Score, False Positive rates on clean images, and 3-Way Mean Intersection over Union (mIoU) for the generated masks.
    *   *Requires:* `best_waste_model.pth`
    *   *Outputs:* `df_diag.pkl` (A serialized DataFrame containing confidence and IoU metrics for every test image).
5.  **`05_diagnostics_and_exports.ipynb`**
    *   *Purpose:* Conducts a deep failure audit. Categorizes model behavior into specific failure modes ("The Arrogant", "The Clumsy", "The Scattered") and generates high-resolution evaluation grids saved to the `assets/` directory.
    *   *Requires:* `best_waste_model.pth` and `df_diag.pkl`

*Note: Once `01` and `02` have been run to generate the model weights, and `04` has been run to generate the diagnostic dataset, notebooks `03`, `04`, and `05` can be executed entirely independently.*