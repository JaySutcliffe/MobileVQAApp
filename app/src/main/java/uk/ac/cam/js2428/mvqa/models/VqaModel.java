package uk.ac.cam.js2428.mvqa.models;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;

import souch.androidcpu.CpuInfo;
import uk.ac.cam.js2428.mvqa.EvaluationOutput;
import uk.ac.cam.js2428.mvqa.questions.QuestionException;
import uk.ac.cam.js2428.mvqa.questions.QuestionTooLongException;
import uk.ac.cam.js2428.mvqa.questions.UnknownAnswerException;

public abstract class VqaModel {
    protected final int maxQuestionLength = 26;
    protected final Context context;

    private final Map<String, Integer> wordToIx = new HashMap<>();
    private final Map<Integer, String> ixToAnswer = new HashMap<>();
    private float unknownWord;

    private static final int NUMBER_OF_READINGS = 400;

    /**
     * Sets the image to perform the inference on. In the case where the image
     * feature model is separate from the question processing part of the model,
     * the image features will be calculated by this function.
     * @param bitmap the bitmap image to be entered.
     */
    public abstract void setImage(Bitmap bitmap);

    /**
     * Runs the VQA model inference on the question entered. The image must be
     * set beforehand. A QuestionException is thrown if the question is too long.
     * Currently the maximum length of the question is 26 words which covers all
     * possible questions in the VQA dataset.
     * @param question a String of words
     * @return float array of word indices.
     * @throws QuestionTooLongException if the question length is longer than
     * the maximum number of words for the model.
     */
    public abstract String runInference(String question) throws QuestionException;

    public VqaModel(Context context) {
        this.context = context;
        // Opening the JSON file

        try {
            InputStream is = context.getAssets().open("data_prepro.json");
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            String fileString = new String(buffer, "UTF-8");
            JSONObject dataPrePro = new JSONObject(fileString);
            JSONObject ixToWordJSON = dataPrePro.getJSONObject("ix_to_word");
            JSONObject ixToAnswerJSON = dataPrePro.getJSONObject("ix_to_ans");

            // Creating lookup table word -> index
            Iterator<String> wordKeys = ixToWordJSON.keys();
            while(wordKeys.hasNext()) {
                String key = wordKeys.next();
                int ix = Integer.parseInt(key) - 1;
                String s = ixToWordJSON.getString(key);
                wordToIx.put(s, ix);
            }

            // Creating lookup table index -> answer
            Iterator<String> answerKeys = ixToAnswerJSON.keys();
            while (answerKeys.hasNext()) {
                String key = answerKeys.next();
                int ix = Integer.parseInt(key) - 1;
                String s = ixToAnswerJSON.getString(key);
                ixToAnswer.put(ix, s);
            }
            unknownWord = (float)wordToIx.get("UNK");
        } catch (IOException | JSONException e) {
            System.err.println("Problem loading JSON file containing word indices");
            e.printStackTrace();
        }
    }

    /**
     * Converts the question to a suitable format to pass to the VQA model.
     * @param question a String of words
     * @return float array of word indices.
     * @throws QuestionTooLongException if the question length is longer than
     * the maximum number of words for the model.
     */
    protected float[] parseQuestion(String question) throws QuestionException {
        question = question.toLowerCase();
        String reg = "[-.\"',;? !$#@~()*&^%\\[\\]/\\\\+<>\\n=]";
        String[] wordArray = question.split(reg);

        if (wordArray.length >= maxQuestionLength) {
            throw new QuestionTooLongException(maxQuestionLength);
        }

        float[] result = new float[maxQuestionLength];
        int i = 0;
        for (String s : wordArray) {
            Integer ix = wordToIx.get(s);
            if (ix != null) {
                result[i] = (float)ix + 1;
            } else {
                result[i] = unknownWord + 1;
            }
            i++;
        }
        return result;
    }

    /**
     * Converts the question to a suitable format to pass to the VQA model.
     * This is for VQA models when the question is left-padded
     * @param question a String of words
     * @return float array of word indices.
     * @throws QuestionTooLongException if the question length is longer than
     * the maximum number of words for the model.
     */
    protected float[] parseQuestionLeftPadded(String question) throws QuestionException {
        question = question.toLowerCase();
        String reg = "[-.\"',;? !$#@~()*&^%\\[\\]/\\\\+<>\\n=]";
        String[] wordArray = question.split(reg);

        if (wordArray.length >= maxQuestionLength) {
            throw new QuestionTooLongException(maxQuestionLength);
        }

        float[] result = new float[maxQuestionLength];
        int i = maxQuestionLength - wordArray.length - 1;
        for (String s : wordArray) {
            Integer ix = wordToIx.get(s);
            if (ix != null) {
                result[i] = (float)ix;
            } else {
                result[i] = unknownWord;
            }
            i++;
        }
        return result;
    }

    /**
     * Converts the input index into an answer.
     * @param answer an int representing the answer.
     * @return the answer as a String of characters.
     * @throws UnknownAnswerException if index has no
     * corresponding answer.
     */
    protected String decodeAnswer(int answer) throws UnknownAnswerException {
        String s = ixToAnswer.get(answer);
        if (s == null) {
            throw new UnknownAnswerException(answer);
        }
        return s;
    }

    /**
     * Runs the network on a subset of the test data generating a object
     * containing information about performance metrics and accuracy
     * @return an EvaluationOutput object
     */
    public EvaluationOutput evaluate() {
        try {
            JSONArray jsonArray;
            {
                InputStream is = context.getAssets().open("vqa_device_test.json");
                int size = is.available();
                byte[] buffer = new byte[size];
                is.read(buffer);
                is.close();
                String fileString = new String(buffer, "UTF-8");
                jsonArray = new JSONArray(fileString);
            }
            int match = 0;
            int[] elapsedCnnTime = new int[NUMBER_OF_READINGS];
            int[] elapsedNlpTime = new int[NUMBER_OF_READINGS];
            List<List<Integer>> cpuUsages = new LinkedList<>();
            String[] imageNames = new String[NUMBER_OF_READINGS];
            String[] questions = new String[NUMBER_OF_READINGS];
            String[] answers = new String[NUMBER_OF_READINGS];
            // Retrieving the questions, answers and image details separately
            // so not covered in by the performance analysis.
            for (int i = 0; i < NUMBER_OF_READINGS; i++) {
                JSONObject jsonObject = jsonArray.getJSONObject(i);
                imageNames[i] = jsonObject.getString("img_name");
                questions[i] = jsonObject.getString("question");
                answers[i] = jsonObject.getString("ans");
            }
            for (int i = 0; i < NUMBER_OF_READINGS; i++) {
                InputStream is = context.getAssets().open("images/" + imageNames[i]);
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                is.close();

                // Sets a timer to run and store an approximation of the CPU
                // around every 10ms until the answer has been generated.
                List<Integer> cpuUsage = new LinkedList<>();
                TimerTask cpuTimerTask = new TimerTask() {
                    public void run() {
                        cpuUsage.add(CpuInfo.getCpuUsage());
                    }
                };
                Timer cpuTimer = new Timer("Cpu timer");
                cpuTimer.schedule(cpuTimerTask, 0, 10);

                long startTime = System.currentTimeMillis();
                setImage(bitmap);
                elapsedCnnTime[i] = (int)(System.currentTimeMillis() - startTime);
                startTime = System.currentTimeMillis();
                String guess = runInference(questions[i]);
                elapsedNlpTime[i] = (int)(System.currentTimeMillis() - startTime);
                if (answers[i].equals(guess)) {
                    match++;
                }
                cpuTimerTask.cancel();
                cpuTimer.cancel();
                cpuUsages.add(cpuUsage);
            }
            return new EvaluationOutput(match, elapsedCnnTime, elapsedNlpTime, cpuUsages);
        } catch (IOException | JSONException | QuestionException e) {
            e.printStackTrace();
        }
        return null;
    }
}