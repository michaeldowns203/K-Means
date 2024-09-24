import java.util.*;

public class EditedKNN {

    private List<DataPoint> dataset;   // The training dataset
    private int kn;                    // Number of neighbors in k-NN
    private double sigma;              // Bandwidth for Gaussian kernel

    public EditedKNN(List<DataPoint> dataset, int kn, double sigma) {
        this.dataset = new ArrayList<>(dataset);  // Clone the dataset
        this.kn = kn;
        this.sigma = sigma;                      // Set the Gaussian kernel bandwidth
    }

    // Method to edit the dataset based on misclassified points
    public List<DataPoint> editDataset() {
        List<DataPoint> editedDataset = new ArrayList<>(dataset);
        boolean changesMade;

        do {
            changesMade = false;
            List<DataPoint> pointsToRemove = new ArrayList<>();

            for (DataPoint point : editedDataset) {
                int predictedLabel = predictLabel(point, editedDataset);
                if (predictedLabel != point.label) {
                    pointsToRemove.add(point);  // Mark for removal if misclassified
                }
            }

            if (!pointsToRemove.isEmpty()) {
                editedDataset.removeAll(pointsToRemove);
                changesMade = true;
            }

        } while (changesMade);  // Keep editing until no more points are removed

        return editedDataset; // Return the edited dataset
    }

    // Method to predict the label of a given point using KNN
    private int predictLabel(DataPoint point, List<DataPoint> dataset) {
        PriorityQueue<Neighbor> neighbors = new PriorityQueue<>((a, b) ->
                Double.compare(a.distance, b.distance)
        );

        for (DataPoint p : dataset) {
            if (p != point) {
                double dist = euclideanDistance(p.features, point.features);
                double weight = gaussianKernel(dist);
                neighbors.add(new Neighbor(p.label, dist, weight));
            }
        }

        // Find the `kn` nearest neighbors and calculate weighted votes
        Map<Integer, Double> classVotes = new HashMap<>();
        for (int i = 0; i < kn && !neighbors.isEmpty(); i++) {
            Neighbor neighbor = neighbors.poll();
            classVotes.put(neighbor.label, classVotes.getOrDefault(neighbor.label, 0.0) + neighbor.weight);
        }

        // Return the class with the highest weighted vote
        return classVotes.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
    }

    // Calculate the Euclidean distance between two points
    private double euclideanDistance(double[] features1, double[] features2) {
        double sum = 0;
        for (int i = 0; i < features1.length; i++) {
            sum += Math.pow(features1[i] - features2[i], 2);
        }
        return Math.sqrt(sum);
    }

    // Gaussian kernel function to calculate the weight based on distance
    private double gaussianKernel(double distance) {
        return Math.exp(-Math.pow(distance, 2) / (2 * Math.pow(sigma, 2)));
    }

    // Inner class to represent a neighbor with distance and weight
    private static class Neighbor {
        int label;
        double distance;
        double weight;

        Neighbor(int label, double distance, double weight) {
            this.label = label;
            this.distance = distance;
            this.weight = weight;
        }
    }

    // Method to get the edited dataset for KNN
    public List<DataPoint> getEditedDataset() {
        return editDataset(); // Calculate and return the edited dataset
    }
}
