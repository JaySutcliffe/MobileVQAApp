package uk.ac.cam.js2428.mvqa;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private VqaModel vqa;
    private EvaluationOutput eo;
    private String EVALUATION_OUTPUT_FILE_NAME = "cpu_usage.txt";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        vqa = new CnnLstmModel(this);
        setContentView(R.layout.activity_main);
    }

    public void submitImage(View view) {
        // https://www.youtube.com/watch?v=TXjf3aK3GNo
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*"); // Pick any type of image
        startActivityForResult(Intent.createChooser(intent, "Choose image"), 1);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == 1) {
            ImageView imageView = findViewById(R.id.imageView);

            try {
                InputStream is = getContentResolver().openInputStream(data.getData());
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                imageView.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
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
            answer = vqa.runInference(question);
        } catch (QuestionException e) {
            answer = e.getMessage();
        }
        TextView textView = findViewById(R.id.answerTextView);
        textView.setText(answer);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        if (requestCode == 1) {
            if(grantResults[0] == PackageManager.PERMISSION_GRANTED){
                File file = new File(Environment.getExternalStorageDirectory(), EVALUATION_OUTPUT_FILE_NAME);
                try (FileOutputStream fos = new FileOutputStream(file)){
                    JSONObject json = new JSONObject();
                    JSONArray cpuTimeJSON = new JSONArray();
                    for (List<Integer> list : eo.getCpuTime()) {
                        JSONArray cpuTimeRecord = new JSONArray();
                        for (int i : list) {
                            cpuTimeRecord.put(i);
                        }
                        cpuTimeJSON.put(cpuTimeRecord);
                    }
                    json.put("cpu_usage", cpuTimeJSON);
                    fos.write(json.toString().getBytes());
                } catch (IOException | JSONException e) {
                    e.printStackTrace();
                }
            }
        }
    }


    /**
     * Function called when the test button is pressed to
     * evaluate the model on device on the test set
     * @param view View of the onclick listener
     */
    public void evaluateTestSet(View view) {
        eo = vqa.evaluate();

        TextView accuracyTextView = findViewById(R.id.accuracyTextView);
        accuracyTextView.setText(new StringBuilder().append("Accuracy = ").
                append(eo.getAccuracy()));

        TextView executionTimeTextView = findViewById(R.id.executionTimeTextView);
        executionTimeTextView.setText(new StringBuilder().append("Execution times(ms): cnn = ").
                append(eo.getMeanCnnTime()).
                append(", nlp = ").
                append(eo.getMeanNlpTime()));

        TextView cpuTimeTextView = findViewById(R.id.cpuTimeTextView);
        cpuTimeTextView.setText(new StringBuilder().append("Average CPU time = ").
                append(eo.getMeanCpuUtilisation()));


        String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE};
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(permissions, 1);
        }
    }
}