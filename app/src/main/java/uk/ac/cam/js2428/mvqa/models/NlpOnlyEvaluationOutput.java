package uk.ac.cam.js2428.mvqa.models;

public class NlpOnlyEvaluationOutput {
    private int packetSize;
    private final int[] matches;
    private final long[] elapsedTime;

    /**
     * Gets a list of accuracies of each mini test set
     * @return float[] of accuracies
     */
    public float[] getAccuracies() {
        float[] accuracies = new float[matches.length];
        for (int i = 0; i < matches.length; i++) {
            accuracies[i] = ((float)matches[i])/((float)packetSize);
        }
        return accuracies;
    }

    /**
     * Returns the array of elapsed time to execute each mini test set
     * @return long[] of elapsed time in ms
     */
    public long[] getElapsedTime() {
        return elapsedTime;
    }

    public NlpOnlyEvaluationOutput(int packetSize, int[] matches, long[] elapsedTime) {
        this.packetSize = packetSize;
        this.matches = matches;
        this.elapsedTime = elapsedTime;
    }
}
