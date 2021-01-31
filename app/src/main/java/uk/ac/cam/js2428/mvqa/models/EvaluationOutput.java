package uk.ac.cam.js2428.mvqa.models;

import android.os.Environment;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class EvaluationOutput {
    private int matches;
    private final int[] elapsedCnnTime;
    private final int[] elapsedNlpTime;
    private final List<List<Integer>> cpuTime;

    /**
     * Returns the classifier accuracy tested on elapsedNlpTime.length questions
     * @return accuracy as a float in the range [0,1]
     */
    public float getAccuracy() {
        return ((float)matches/(float)elapsedNlpTime.length);
    }

    /**
     * Returns the classifier's average elapsed time to generate image features.
     * @return time as a float in ms.
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
     * from image features and processing a question.
     * @return time as a float in ms.
     */
    public float getMeanNlpTime() {
        long total = 0;
        for (int i = 0; i < elapsedNlpTime.length; i++) {
            total += elapsedNlpTime[i];
        }
        return (float)((double)total/(double)elapsedNlpTime.length);
    }

    /**
     * Returns the classifier's mean CPU utilisation across the all inferences.
     * @return CPU utilisation as a float
     */
    public float getMeanCpuUtilisation() {
        long total = 0; // Long will be enough should not have overflow
        int size = 0;
        for (List<Integer> cpuList : cpuTime) {
            size += cpuList.size();
            for (int i : cpuList) {
                total += i;
            }
        }
        return (float)((double)total/(double)size);
    }

    /**
     * Gets the list of CPU readings, useful to store on disk for visualisations.
     * @return A list of lists of integer CPU readings.
     */
    public List<List<Integer>> getCpuTime() {
        return cpuTime;
    }

    public EvaluationOutput(int matches, int[] elapsedCnnTime, int[] elapsedNlpTime,
                            List<List<Integer>> cpuTime) {
        this.matches = matches;
        this.elapsedCnnTime = elapsedCnnTime;
        this.elapsedNlpTime = elapsedNlpTime;
        this.cpuTime = cpuTime;
    }
}
