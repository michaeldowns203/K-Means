import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class EditedKNN {
    private List<List<Double>> trainingData; // Features of the training data
    private List<String> trainingLabels; // Labels of the training data
    private int k; // Number of neighbors to consider
    private double bandwidth; // Bandwidth for the Gaussian kernel
    private double errorThreshold; // Acceptable error threshold for regression

    public EditedKNN(int k, double bandwidth, double errorThreshold) {
        this.k = k;
        this.bandwidth = bandwidth;
        this.errorThreshold = errorThreshold;
        this.trainingData = new ArrayList<>();
        this.trainingLabels = new ArrayList<>();
    }

    // Fit the model with training data and labels
    public void fit(List<List<Double>> data, List<String> labels) {
        this.trainingData = data;
        this.trainingLabels = labels;
    }

    public void edit() {
        List<List<Double>> editedTrainingData = new ArrayList<>();
        List<String> editedTrainingLabels = new ArrayList<>();

        // Loop over each point in the training set
        for (int i = 0; i < trainingData.size(); i++) {
            List<Double> currentPoint = trainingData.get(i);
            String currentLabel = trainingLabels.get(i);

            // Temporarily remove the current point from the training set
            List<List<Double>> tempTrainingData = new ArrayList<>(trainingData);
            List<String> tempTrainingLabels = new ArrayList<>(trainingLabels);
            tempTrainingData.remove(i);
            tempTrainingLabels.remove(i);

            // Create a temporary KNN model with the remaining data
            KNN tempKNN = new KNN(k, bandwidth, errorThreshold);
            tempKNN.fit(tempTrainingData, tempTrainingLabels);

            // Predict the label for the current point using the remaining data
            String predictedLabel = tempKNN.predict(currentPoint);

            // For classification: if the predicted label matches the current label, keep the point
            if (predictedLabel.equals(currentLabel)) {
                editedTrainingData.add(currentPoint);
                editedTrainingLabels.add(currentLabel);
            }
        }

        // Update the training data and labels to the edited set
        this.trainingData = editedTrainingData;
        this.trainingLabels = editedTrainingLabels;
    }

    public void editR() {
        List<List<Double>> editedTrainingData = new ArrayList<>();
        List<String> editedTrainingLabels = new ArrayList<>();

        // Loop over each point in the training set
        for (int i = 0; i < trainingData.size(); i++) {
            List<Double> currentPoint = trainingData.get(i);
            double currentValue = Double.parseDouble(trainingLabels.get(i)); // Assume labels are numeric for regression

            // Temporarily remove the current point from the training set
            List<List<Double>> tempTrainingData = new ArrayList<>(trainingData);
            List<String> tempTrainingLabels = new ArrayList<>(trainingLabels);
            tempTrainingData.remove(i);
            tempTrainingLabels.remove(i);

            // Create a temporary KNN model with the remaining data
            KNN tempKNN = new KNN(k, bandwidth, errorThreshold);
            tempKNN.fit(tempTrainingData, tempTrainingLabels);

            // Predict the value for the current point using the remaining data
            double predictedValue = tempKNN.predictValue(currentPoint);

            // If the predicted value is within the acceptable error threshold of the actual value, keep the point
            if (isPredictionCorrect(currentValue, predictedValue)) {
                editedTrainingData.add(currentPoint);
                editedTrainingLabels.add(trainingLabels.get(i)); // Keep the original label
            }
        }

        // Update the training data and labels to the edited set
        this.trainingData = editedTrainingData;
        this.trainingLabels = editedTrainingLabels;
    }



    // Predict the label for a single instance (classification)
    public String predict(List<Double> instance) {
        List<Neighbor> neighbors = new ArrayList<>();

        // Calculate distances from the instance to all training instances
        for (int i = 0; i < trainingData.size(); i++) {
            double distance = euclideanDistance(instance, trainingData.get(i));
            neighbors.add(new Neighbor(distance, trainingLabels.get(i)));
        }

        // Sort neighbors by distance
        Collections.sort(neighbors);

        // Get the k nearest neighbors
        List<String> kNearestLabels = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            kNearestLabels.add(neighbors.get(i).label);
        }

        // Return the most common label among the k nearest neighbors
        return mostCommonLabel(kNearestLabels);
    }

    // Predict the value for a single instance (regression)
    public double predictValue(List<Double> instance) {
        List<Neighbor> neighbors = new ArrayList<>();

        // Calculate distances from the instance to all training instances
        for (int i = 0; i < trainingData.size(); i++) {
            double distance = euclideanDistance(instance, trainingData.get(i));
            neighbors.add(new Neighbor(distance, trainingLabels.get(i)));
        }

        // Sort neighbors by distance
        Collections.sort(neighbors);

        // Get the k nearest neighbors
        List<Neighbor> kNearestNeighbors = neighbors.subList(0, Math.min(k, neighbors.size()));

        double weightedSum = 0.0;
        double totalWeight = 0.0;

        // Calculate weighted contributions from the k nearest neighbors
        for (Neighbor neighbor : kNearestNeighbors) {
            double weight = gaussianKernel(neighbor.distance);
            double labelValue = Double.parseDouble(neighbor.label); // Ensure labels are numeric
            weightedSum += weight * labelValue;
            totalWeight += weight;
        }

        // Return weighted average for regression
        return totalWeight > 0 ? weightedSum / totalWeight : 0.0;
    }

    // Check if the predicted value is within the acceptable error threshold of the actual value
    public boolean isPredictionCorrect(double actualValue, double predictedValue) {
        return Math.abs(actualValue - predictedValue) <= errorThreshold;
    }

    // Calculate Euclidean distance between two instances
    private double euclideanDistance(List<Double> instance1, List<Double> instance2) {
        double sum = 0;
        for (int i = 0; i < instance1.size(); i++) {
            sum += Math.pow(instance1.get(i) - instance2.get(i), 2);
        }
        return Math.sqrt(sum);
    }

    // Gaussian kernel function
    private double gaussianKernel(double distance) {
        return Math.exp(-Math.pow(distance, 2) / (2 * Math.pow(bandwidth, 2)));
    }

    // Find the most common label in a list (for classification)
    private String mostCommonLabel(List<String> labels) {
        // Count occurrences of each label
        java.util.Map<String, Integer> labelCount = new java.util.HashMap<>();
        for (String label : labels) {
            labelCount.put(label, labelCount.getOrDefault(label, 0) + 1);
        }

        // Find the label with the highest count
        String mostCommon = null;
        int maxCount = 0;
        for (java.util.Map.Entry<String, Integer> entry : labelCount.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                mostCommon = entry.getKey();
            }
        }
        return mostCommon;
    }

    // Method to calculate mean squared error
    public double calculateMSE(List<Double> actualValues, List<Double> predictedValues) {
        double sumSquaredError = 0.0;
        int n = actualValues.size();

        // Calculate the sum of squared errors
        for (int i = 0; i < n; i++) {
            double error = actualValues.get(i) - predictedValues.get(i);
            sumSquaredError += error * error;
        }

        // Return the mean squared error
        return sumSquaredError / n;
    }

    // Inner class to represent a neighbor
    private static class Neighbor implements Comparable<Neighbor> {
        double distance;
        String label;

        Neighbor(double distance, String label) {
            this.distance = distance;
            this.label = label;
        }

        @Override
        public int compareTo(Neighbor other) {
            return Double.compare(this.distance, other.distance);
        }
    }
}
