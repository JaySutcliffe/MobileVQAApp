package uk.ac.cam.js2428.mvqa;

import android.content.Context;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public abstract class VqaModel {
    private final int maxQuestionLength = 26;
    private final Map<String, Integer> wordToIx = new HashMap<>();
    private final Map<Integer, String> ixToAnswer = new HashMap<>();

    abstract void setImage(String imageLocation) throws IOException;
    abstract String runInference(String question) throws QuestionException;

    public VqaModel(Context context) {
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

        if (wordArray.length > maxQuestionLength) {
            throw new QuestionTooLongException(maxQuestionLength);
        }

        float[] result = new float[maxQuestionLength];
        int i = maxQuestionLength - 1;
        for (String s : wordArray) {
            Integer ix = wordToIx.get(s);
            if (ix != null) {
                result[i] = (float)ix;
                i--;
            } else {
                throw new UnknownWordException(s);
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

}
