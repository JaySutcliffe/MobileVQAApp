package uk.ac.cam.js2428.mvqa;

public class EvaluationOutput {
    private int packetSize;
    private final int[] matches;
    private final long[] elapsedTime;

    public float[] getAccuracies() {
        float[] accuracies = new float[matches.length];
        for (int i = 0; i < matches.length; i++) {
            accuracies[i] = ((float)matches[i])/((float)packetSize);
        }
        return accuracies;
    }

    public long[] getElapsedTime() {
        return elapsedTime;
    }

    public EvaluationOutput(int packetSize, int[] matches, long[] elapsedTime) {
        this.packetSize = packetSize;
        this.matches = matches;
        this.elapsedTime = elapsedTime;
    }
}
