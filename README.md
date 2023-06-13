# AI-based-Smart-Attendance-System-
Face recognition systems can be used to identify people in photos,videos or in real-time . The attendance system will track employee location , time of clocking/out, and the report recorded. The attendance system will then process the data, to produce timesheet reports, absence reports, task and request reports

## Support Vector Machine - SVM Algorithm
- Support vector machines(SVMs) are powerful yet flexiblle supervised machine learning algorithms which are used for both classification and regression.
- The goal of SVM is to divide the datasets into classes to find a maximum marginal hyperplane(MMH)
- Support Vectors - Datapoints that are closet to the hyperplane is called support vectors.Separating line will be defined with the help of these data points.
- Hyperplane - It is a decision plane or space which is divided between a set of objects having different classes.
- Margin - It may be defined as the gap between two lines on the closet data points of different classes. It can be calculated as the perpendicular distance from the line to the support vectors. Large margin is considered as a good margin and small margin is considered as a bad margin.

## Workflow of the Attendance System 
1. Dataset Creation with CSV name and roll no
2. Pre-Processing Face detection
3. Pre-Processing 128-0 embedding for ML
4. Training- ML-SVM
5. Load Model, Label Encoder & CSV
6. Pre-Process frame from Camera
7. Classification