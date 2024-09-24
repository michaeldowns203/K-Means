import java.util.*;

public class KNN {
    private int k; // Number of neighbors
    private List<List<Double>> trainingData; // Training data (numeric)
    private List<String> trainingLabels; // Labels for training data (for classification)
    private List<Double> trainingTargets; // Target values (for regression)
    private boolean isRegression; // If true, perform regression instead of classification

    // Constructor for k-NN (classification)
    public KNN(int k, boolean isRegression) {
        this.k = k;
        this.isRegression = isRegression;
    }

    // Method to fit the model for classification
    public void fitClassification(List<List<Double>> trainingData, List<String> trainingLabels) {
        this.trainingData = trainingData;
        this.trainingLabels = trainingLabels;
    }

    // Method to fit the model for regression
    public void fitRegression(List<List<Double>> trainingData, List<Double> trainingTargets) {
        this.trainingData = trainingData;
        this.trainingTargets = trainingTargets;
    }

    // Method to classify or regress a new instance
    public String classify(List<Double> testInstance) {
        List<Neighbor> neighbors = getNeighbors(testInstance);
        if (isRegression) {
            return String.valueOf(getRegressionPrediction(neighbors));
        } else {
            return getMajorityVote(neighbors);
        }
    }

    // Method to calculate the Euclidean distance (or modify to use other distance metrics)
    private double euclideanDistance(List<Double> instance1, List<Double> instance2) {
        double distance = 0.0;
        for (int i = 0; i < instance1.size(); i++) {
            distance += Math.pow(instance1.get(i) - instance2.get(i), 2);
        }
        return Math.sqrt(distance);
    }

    // Method to find the k nearest neighbors
    private List<Neighbor> getNeighbors(List<Double> testInstance) {
        PriorityQueue<Neighbor> neighbors = new PriorityQueue<>(Comparator.comparingDouble(n -> n.distance));

        for (int i = 0; i < trainingData.size(); i++) {
            double distance = euclideanDistance(trainingData.get(i), testInstance);
            if (isRegression) {
                neighbors.add(new Neighbor(String.valueOf(trainingTargets.get(i)), distance));
            } else {
                neighbors.add(new Neighbor(trainingLabels.get(i), distance));
            }

            // Maintain only k neighbors
            if (neighbors.size() > k) {
                neighbors.poll();
            }
        }

        return new ArrayList<>(neighbors);
    }

    // Method to get the majority vote (for classification)
    private String getMajorityVote(List<Neighbor> neighbors) {
        Map<String, Integer> frequencyMap = new HashMap<>();
        for (Neighbor neighbor : neighbors) {
            frequencyMap.put(neighbor.label, frequencyMap.getOrDefault(neighbor.label, 0) + 1);
        }

        // Find the label with the maximum frequency
        return Collections.max(frequencyMap.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    // Method to compute the average for regression (or weighted average)
    private double getRegressionPrediction(List<Neighbor> neighbors) {
        double total = 0.0;
        double totalWeight = 0.0;

        for (Neighbor neighbor : neighbors) {
            double distance = neighbor.distance == 0.0 ? 1e-9 : neighbor.distance; // Avoid division by 0
            double weight = 1.0 / distance; // Inverse distance weighting
            total += Double.parseDouble(neighbor.label) * weight;
            totalWeight += weight;
        }

        return total / totalWeight; // Weighted average
    }

    // Class to hold the neighbor information
    private static class Neighbor {
        String label;
        double distance;

        Neighbor(String label, double distance) {
            this.label = label;
            this.distance = distance;
        }
    }
}
