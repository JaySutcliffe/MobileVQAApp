package uk.ac.cam.js2428.mvqa;

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
import java.util.Map;

public abstract class VqaModel {
    protected final int maxQuestionLength = 26;
    protected final Context context;

    private final Map<String, Integer> wordToIx = new HashMap<>();
    private final Map<Integer, String> ixToAnswer = new HashMap<>();
    private float unknownWord;

    abstract void setImage(Bitmap bitmap);
    abstract String runInference(String question) throws QuestionException;
    abstract NlpOnlyEvaluationOutput evaluateQuestionOnly();

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
     * @throws UnknownWordException if a word occurs that the model is not
     * trained on.
     */
    protected float[] parseQuestion(String question) throws QuestionException {
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
                i++;
            } else {
                result[i] = unknownWord;
            }
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
            int[] elapsedCnnTime = new int[100];
            int[] elapsedNlpTime = new int[100];
            for (int i = 0; i < 100; i++) {
                JSONObject jsonObject = jsonArray.getJSONObject(i);
                String imageName = jsonObject.getString("img_name");
                String question = jsonObject.getString("question");
                String answer = jsonObject.getString("ans");

                InputStream is = context.getAssets().open("images/" + imageName);
                Bitmap bitmap = BitmapFactory.decodeStream(is);

                long startTime = System.currentTimeMillis();
                setImage(bitmap);
                elapsedCnnTime[i] = (int)(System.currentTimeMillis() - startTime);
                startTime = System.currentTimeMillis();
                String guess = runInference(question);
                elapsedNlpTime[i] = (int)(System.currentTimeMillis() - startTime);
                if (answer.equals(guess)) {
                    match++;
                }
            }
            return new EvaluationOutput(match, elapsedCnnTime, elapsedNlpTime);
        } catch (IOException | JSONException | QuestionException e) {
            e.printStackTrace();
        }
        return null;
    }

}
