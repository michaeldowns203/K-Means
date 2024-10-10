import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class KNNPrint {
    private List<List<Double>> trainingData; // Features of the training data
    private List<String> trainingLabels; // Labels of the training data
    private int k; // Number of neighbors to consider
    private double bandwidth; // Bandwidth for the Gaussian kernel
    private double errorThreshold; // Acceptable error threshold for regression

    public KNNPrint(int k, double bandwidth, double errorThreshold) {
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

    // Predict the label for a single instance (classification)
    public String predict(List<Double> instance) {
        List<Neighbor> neighbors = new ArrayList<>();

        // Calculate distances from the instance to all training instances
        for (int i = 0; i < trainingData.size(); i++) {
            double distance = euclideanDistance(instance, trainingData.get(i));
            neighbors.add(new Neighbor(distance, trainingLabels.get(i), trainingData.get(i)));
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
            neighbors.add(new Neighbor(distance, trainingLabels.get(i), trainingData.get(i)));
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
        //System.out.println(instance1 + " " + instance2);
        //System.out.println("Euclidean distance: " + Math.sqrt(sum));
        return Math.sqrt(sum);
    }

    // Gaussian kernel function
    private double gaussianKernel(double distance) {
        System.out.println("Euclidean distance: " + distance + " Gaussian kernel: " + Math.exp(-Math.pow(distance, 2) / (2 * Math.pow(bandwidth, 2))));
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

    // Print method to demonstrate k-NN classification
    public void demonstrateClassification(List<Double> instance) {
        List<Neighbor> neighbors = new ArrayList<>();

        // Calculate distances from the instance to all training instances
        for (int i = 0; i < trainingData.size(); i++) {
            double distance = euclideanDistance(instance, trainingData.get(i));
            neighbors.add(new Neighbor(distance, trainingLabels.get(i), trainingData.get(i)));
        }

        // Sort neighbors by distance
        Collections.sort(neighbors);

        // Get the k nearest neighbors
        List<Neighbor> kNearestNeighbors = neighbors.subList(0, Math.min(k, neighbors.size()));

        // Print neighbors
        System.out.println("Neighbors for classification:");
        for (Neighbor neighbor : kNearestNeighbors) {
            System.out.println("Point: " + neighbor.point + ", Distance: " + neighbor.distance + ", Label: " + neighbor.label);
        }

    }

    // Print method to demonstrate k-NN regression
    public void demonstrateRegression(List<Double> instance) {
        List<Neighbor> neighbors = new ArrayList<>();

        // Calculate distances from the instance to all training instances
        for (int i = 0; i < trainingData.size(); i++) {
            double distance = euclideanDistance(instance, trainingData.get(i));
            neighbors.add(new Neighbor(distance, trainingLabels.get(i), trainingData.get(i)));
        }

        // Sort neighbors by distance
        Collections.sort(neighbors);

        // Get the k nearest neighbors
        List<Neighbor> kNearestNeighbors = neighbors.subList(0, Math.min(k, neighbors.size()));

        // Print neighbors
        System.out.println("Neighbors for regression:");
        for (Neighbor neighbor : kNearestNeighbors) {
            System.out.println("Point: " + neighbor.point + ", Distance: " + neighbor.distance + ", Label: " + neighbor.label);
        }
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
            System.out.println(trainingData.get(i));

            // Create a temporary KNN model with the remaining data
            KNN tempKNN = new KNN(k, bandwidth, errorThreshold);
            tempKNN.fit(tempTrainingData, tempTrainingLabels);

            // Predict the label for the current point using the remaining data
            String predictedLabel = tempKNN.predict(currentPoint);
            System.out.println("Predicted: " + predictedLabel);
            System.out.println("Actual: " + currentLabel);

            // For classification: if the predicted label matches the current label, keep the point
            if (predictedLabel.equals(currentLabel)) {
                editedTrainingData.add(currentPoint);
                editedTrainingLabels.add(currentLabel);
                System.out.println("Correct; Kept");
            }
            else {
                System.out.println("Incorrect; Removed");
            }
        }

        // Update the training data and labels to the edited set
        this.trainingData = editedTrainingData;
        this.trainingLabels = editedTrainingLabels;
        System.out.println("Edited dataset size: " + editedTrainingData.size());
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
        System.out.println(editedTrainingData.size());
    }

    // K-means clustering method
    public void kMeansAndReduce(int numClusters, int maxIterations) {
        // Step 1: Initialize centroids randomly
        List<List<Double>> centroids = initializeCentroids(numClusters);
        List<Integer> clusterAssignments = new ArrayList<>(Collections.nCopies(trainingData.size(), -1));

        boolean converged = false;
        int iterations = 0;

        while (!converged && iterations < maxIterations) {
            converged = true;

            // Step 2: Assign each point to the nearest centroid
            for (int i = 0; i < trainingData.size(); i++) {
                int nearestCluster = findNearestCluster(trainingData.get(i), centroids);
                if (clusterAssignments.get(i) != nearestCluster) {
                    clusterAssignments.set(i, nearestCluster);
                    converged = false;
                }
                System.out.println(trainingData.get(i));
                System.out.println("Cluster assignment: " + clusterAssignments.get(i));
            }

            // Step 3: Update centroids by calculating the mean of points in each cluster
            centroids = updateCentroids(clusterAssignments, numClusters);

            iterations++;
        }

        // Step 4: Assign labels to the centroids based on majority voting from original labels
        List<String> centroidLabels = assignLabelsToCentroids(centroids, clusterAssignments, numClusters);

        // Replace original training data with the centroids
        this.trainingData = centroids;
        this.trainingLabels = centroidLabels;
    }

    // K-means clustering for regression data
    public void kMeansAndReduceRegression(int numClusters, int maxIterations) {
        // Step 1: Initialize centroids randomly
        List<List<Double>> centroids = initializeCentroids(numClusters);
        List<Integer> clusterAssignments = new ArrayList<>(Collections.nCopies(trainingData.size(), -1));

        boolean converged = false;
        int iterations = 0;

        while (!converged && iterations < maxIterations) {
            converged = true;

            // Step 2: Assign each point to the nearest centroid
            for (int i = 0; i < trainingData.size(); i++) {
                int nearestCluster = findNearestCluster(trainingData.get(i), centroids);
                if (clusterAssignments.get(i) != nearestCluster) {
                    clusterAssignments.set(i, nearestCluster);
                    converged = false;
                }
            }

            // Step 3: Update centroids by calculating the mean of points in each cluster
            centroids = updateCentroids(clusterAssignments, numClusters);

            iterations++;
        }

        // Step 4: Assign average values to the centroids based on the original target values
        List<Double> centroidValues = assignValuesToCentroids(clusterAssignments, numClusters);

        // Replace original training data and labels with the centroids and their values
        this.trainingData = centroids;
        this.trainingLabels = new ArrayList<>();
        for (Double value : centroidValues) {
            this.trainingLabels.add(value.toString());
        }
    }

    // Assign average values to centroids based on points in each cluster
    private List<Double> assignValuesToCentroids(List<Integer> clusterAssignments, int numClusters) {
        List<Double> centroidValues = new ArrayList<>(Collections.nCopies(numClusters, 0.0));
        List<Integer> pointsInCluster = new ArrayList<>(Collections.nCopies(numClusters, 0));

        // Sum up the values for points in each cluster
        for (int i = 0; i < trainingData.size(); i++) {
            int cluster = clusterAssignments.get(i);
            double labelValue = Double.parseDouble(trainingLabels.get(i)); // Convert label to double
            centroidValues.set(cluster, centroidValues.get(cluster) + labelValue);
            pointsInCluster.set(cluster, pointsInCluster.get(cluster) + 1);
        }

        // Calculate the average value for each centroid
        for (int i = 0; i < numClusters; i++) {
            int pointCount = pointsInCluster.get(i);
            if (pointCount > 0) {
                centroidValues.set(i, centroidValues.get(i) / pointCount);
            }
        }

        return centroidValues;
    }

    // Initialize centroids randomly from the training data
    private List<List<Double>> initializeCentroids(int numClusters) {
        List<List<Double>> centroids = new ArrayList<>();
        Random rand = new Random();

        for (int i = 0; i < numClusters; i++) {
            centroids.add(new ArrayList<>(trainingData.get(rand.nextInt(trainingData.size()))));
        }

        return centroids;
    }

    // Find the nearest centroid to a given instance
    private int findNearestCluster(List<Double> instance, List<List<Double>> centroids) {
        int nearestCluster = -1;
        double minDistance = Double.MAX_VALUE;

        for (int i = 0; i < centroids.size(); i++) {
            double distance = euclideanDistance(instance, centroids.get(i));
            if (distance < minDistance) {
                minDistance = distance;
                nearestCluster = i;
            }
        }

        return nearestCluster;
    }

    // Update centroids by calculating the mean of points in each cluster
    private List<List<Double>> updateCentroids(List<Integer> clusterAssignments, int numClusters) {
        List<List<Double>> newCentroids = new ArrayList<>();
        List<Integer> pointsInCluster = new ArrayList<>(Collections.nCopies(numClusters, 0));

        // Initialize centroid sums
        for (int i = 0; i < numClusters; i++) {
            newCentroids.add(new ArrayList<>(Collections.nCopies(trainingData.get(0).size(), 0.0)));
        }

        // Sum up all points in each cluster
        for (int i = 0; i < trainingData.size(); i++) {
            int cluster = clusterAssignments.get(i);
            List<Double> point = trainingData.get(i);
            List<Double> centroidSum = newCentroids.get(cluster);

            for (int j = 0; j < point.size(); j++) {
                centroidSum.set(j, centroidSum.get(j) + point.get(j));
            }

            pointsInCluster.set(cluster, pointsInCluster.get(cluster) + 1);
        }

        // Calculate the mean for each cluster
        for (int i = 0; i < numClusters; i++) {
            List<Double> centroidSum = newCentroids.get(i);
            int pointCount = pointsInCluster.get(i);

            if (pointCount > 0) {
                for (int j = 0; j < centroidSum.size(); j++) {
                    centroidSum.set(j, centroidSum.get(j) / pointCount);
                }
            }
        }

        return newCentroids;
    }

    // Assign labels to centroids based on majority voting from the original labels
    private List<String> assignLabelsToCentroids(List<List<Double>> centroids, List<Integer> clusterAssignments, int numClusters) {
        List<String> centroidLabels = new ArrayList<>(Collections.nCopies(numClusters, ""));
        List<List<String>> labelsInCluster = new ArrayList<>();

        // Initialize label lists for each cluster
        for (int i = 0; i < numClusters; i++) {
            labelsInCluster.add(new ArrayList<>());
        }

        // Assign labels to clusters
        for (int i = 0; i < trainingData.size(); i++) {
            int cluster = clusterAssignments.get(i);
            labelsInCluster.get(cluster).add(trainingLabels.get(i));
        }

        // Determine the most common label for each centroid
        for (int i = 0; i < numClusters; i++) {
            centroidLabels.set(i, mostCommonLabel(labelsInCluster.get(i)));
        }

        return centroidLabels;
    }

    // Inner class to represent a neighbor
    private static class Neighbor implements Comparable<Neighbor> {
        double distance;
        String label;
        List<Double> point; // Store the point associated with the neighbor

        Neighbor(double distance, String label, List<Double> point) {
            this.distance = distance;
            this.label = label;
            this.point = point;
        }

        @Override
        public int compareTo(Neighbor other) {
            return Double.compare(this.distance, other.distance);
        }
    }
}
