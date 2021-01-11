package uk.ac.cam.js2428.mvqa;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import uk.ac.cam.js2428.mvqa.ml.Cnn;
import uk.ac.cam.js2428.mvqa.ml.Vqa;

public class MainActivity extends AppCompatActivity {
    private VqaModel vqa;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        vqa = new CnnLstmModel(this);
        setContentView(R.layout.activity_main);
    }

    public void submitImage(View view) {
        try {
            vqa.setImage("images/COCO_val2014_000000000042.jpg");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }




    /**
     * Function called when the submit question button is pressed to
     * call the model on the question and image entered.
     * @param view View of the onclick listener
     */
    public void submitQuestion(View view) {
        EditText editText = findViewById(R.id.questionEditText);
        String question = editText.getText().toString();
        String answer;
        try {
            //vqa.setImage("COCO_val2014_000000000042.jpg");
            answer = vqa.runInference(question);
        } catch (QuestionException e) {//| IOException e) {
            answer = e.getMessage();
        }
        TextView textView = findViewById(R.id.answerTextView);
        textView.setText(answer);
    }

    /**
     * Function called when the test button is pressed to
     * evaluate the model on device on the test set
     * @param view View of the onclick listener
     */
    public void evaluateTestSet(View view) {
        EvaluationOutput eo = vqa.evaluateQuestionOnly();
        TextView accuracyTextView = findViewById(R.id.accuracyTextView);
        accuracyTextView.setText(new StringBuilder().append("Accuracies: ").
                append(Arrays.toString(eo.getAccuracies())).toString());
        TextView executionTimeTextView = findViewById(R.id.ExecutionTimeTextView);
        executionTimeTextView.setText(new StringBuilder().append("Execution times(ms): ").
                append(Arrays.toString(eo.getElapsedTime())).toString());
    }
}