package uk.ac.cam.js2428.mvqa;

public class EvaluationOutput {
    private int matches;
    private final int[] elapsedCnnTime;
    private final int[] elapsedNlpTime;

    /**
     * Returns the classifier accuracy tested on elapsedNlpTime.length questions
     * @return accuracy as a float in the range [0,1]
     */
    public float getAccuracy() {
        return ((float)matches/(float)elapsedNlpTime.length);
    }

    /**
     * Returns the classifier's average elapsed time to generate image features
     * @return time as a float in ms
     */
    public float getMeanCnnTime() {
        long total = 0;
        for (int i = 0; i < elapsedCnnTime.length; i++) {
            total += elapsedCnnTime[i];
        }
        return (float)((double)total/(double)elapsedCnnTime.length);
    }

    /**
     * Returns the classifier's average elapsed time to generate an answer
     * from image features and processing a question
     * @return time as a float in ms
     */
    public float getMeanNlpTime() {
        long total = 0;
        for (int i = 0; i < elapsedNlpTime.length; i++) {
            total += elapsedNlpTime[i];
        }
        return (float)((double)total/(double)elapsedNlpTime.length);
    }

    public EvaluationOutput(int matches, int[] elapsedCnnTime, int[] elapsedNlpTime) {
        this.matches = matches;
        this.elapsedCnnTime = elapsedCnnTime;
        this.elapsedNlpTime = elapsedNlpTime;
    }
}
