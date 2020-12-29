package uk.ac.cam.js2428.mvqa;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import uk.ac.cam.js2428.mvqa.ml.Vgg19Quant8;
import uk.ac.cam.js2428.mvqa.ml.Vqa;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    private final int maxQuestionLength = 26;
    private final Map<String, Integer> wordToIx = new HashMap<>();
    private final Map<Integer, String> ixToAnswer = new HashMap<>();
    private Vgg19Quant8 cnn;
    private Vqa model;
    private TensorBuffer imageFeature;
    private TensorBuffer questionFeature;
    private TensorBuffer cnnImageFeature;


    /**
     * Converts the question to a suitable format to pass to the VQA model.
     * @param question a String of words
     * @return float array of word indices.
     * @throws QuestionTooLongException if the question length is longer than
     * the maximum number of words for the model.
     * @throws UnknownWordException if a word occurs that the model is not
     * trained on.
     */
    private float[] parseQuestion(String question) throws QuestionException {
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
    private String decodeAnswer(int answer) throws UnknownAnswerException {
        String s = ixToAnswer.get(answer);
        if (s == null) {
            throw new UnknownAnswerException(answer);
        }
        return s;
    }

    /**
     * Attempts to read the JSON file containing the index against word
     * and answer lookup table. Also attempts to initialise the VQA model.
     */
    private void initialiseClassifier() {
        // Opening the JSON file
        try {
            InputStream is = getAssets().open("data_prepro.json");
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

        try {
            cnn = Vgg19Quant8.newInstance(this);

            // Runs model inference and gets result.
            Vgg19Quant8.Outputs outputs = cnn.process(imageFeature);
            cnnImageFeature = outputs.getOutputFeature0AsTensorBuffer();
        } catch (IOException e) {
            System.err.println("Problem initialising TensorFlow VQA model");
            e.printStackTrace();
        }

        // Initialising the VQA model
        try {
            model = Vqa.newInstance(this);
            questionFeature =
                    TensorBuffer.createFixedSize(new int[]{1, 26}, DataType.FLOAT32);
            //cnnImageFeature =
            //        TensorBuffer.createFixedSize(new int[]{1, 1000}, DataType.FLOAT32);
        } catch (IOException e) {
            System.err.println("Problem initialising TensorFlow VQA model");
            e.printStackTrace();
        }
    }

    /**
     * Function called when the submit question button is pressed to
     * call the model on the question and image entered.
     * @param view View of the onclick listener
     */
    public void submitQuestion(View view) {
        initialiseClassifier(); // To be moved to on start here for debugging
        EditText editText = findViewById(R.id.questionEditText);
        String question = editText.getText().toString();
        String answer;
        try {
            questionFeature.loadArray(parseQuestion(question));
            Vqa.Outputs outputs = model.process(questionFeature, cnnImageFeature);
            TensorBuffer answerFeature = outputs.getOutputFeature0AsTensorBuffer();
            answer = decodeAnswer(answerFeature.getIntArray()[0]);
        } catch (QuestionException e) {
            answer = e.getMessage();
        }
        TextView textView = findViewById(R.id.answerTextView);
        textView.setText(answer);
    }
}