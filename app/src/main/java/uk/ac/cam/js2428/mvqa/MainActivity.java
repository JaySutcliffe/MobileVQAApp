package uk.ac.cam.js2428.mvqa;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.PowerManager;
import android.view.View;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import uk.ac.cam.js2428.mvqa.ml.FullAttentionVqaF16;
import uk.ac.cam.js2428.mvqa.ml.SoftAttentionVqa;
import uk.ac.cam.js2428.mvqa.ml.Vqa;
import uk.ac.cam.js2428.mvqa.models.CnnF16LstmModel;
import uk.ac.cam.js2428.mvqa.models.CnnLstmDyModel;
import uk.ac.cam.js2428.mvqa.models.CnnLstmF16Model;
import uk.ac.cam.js2428.mvqa.models.CnnLstmModel;
import uk.ac.cam.js2428.mvqa.models.FullAttentionModel;
import uk.ac.cam.js2428.mvqa.models.SoftAttentionModel;
import uk.ac.cam.js2428.mvqa.models.VqaModel;
import uk.ac.cam.js2428.mvqa.questions.QuestionException;

public class MainActivity extends AppCompatActivity {
    private FullAttentionModel vqa;
    private EvaluationOutput eo;
    private String EVALUATION_OUTPUT_FILE_NAME = "evaluation_output_full2.json";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        vqa = new FullAttentionModel(this);
        setContentView(R.layout.activity_main);
    }

    /**
     * Generates an intent to get content. This is then used to pick an image to perform inference
     * on.
     */
    public void submitImage(View view) {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(Intent.createChooser(intent, "Choose an image"), 1);
    }


    /**
     * Handles the result to requesting to access the camera roll. This is used to select
     * and retrieve the chosen bitmap image to perform inference on.
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == 1) {
            ImageView imageView = findViewById(R.id.imageView);

            try {
                //InputStream is = getContentResolver().openInputStream(data.getData());
                InputStream is = getAssets().open("images/COCO_val2014_000000107656.jpg");
                Bitmap bitmap = BitmapFactory.decodeStream(is);
                vqa.setImage(bitmap);
                imageView.setImageBitmap(Bitmap.createScaledBitmap(bitmap, 800, 800, false));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
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


    /**
     * Handles the request permission to access external storage. Then the CPU readings
     * output from the evaluation code is written to a json file in the external storage.
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        if (requestCode == 1) {
            if(grantResults[0] == PackageManager.PERMISSION_GRANTED){
                File file = new File(Environment.getExternalStorageDirectory(), EVALUATION_OUTPUT_FILE_NAME);
                try (FileOutputStream fos = new FileOutputStream(file)){
                    JSONObject json = new JSONObject();

                    // Put in CPU times
                    JSONArray cpuTimeJSON = new JSONArray();
                    for (List<Integer> list : eo.getCpuTime()) {
                        JSONArray cpuTimeRecord = new JSONArray();
                        for (int i : list) {
                            cpuTimeRecord.put(i);
                        }
                        cpuTimeJSON.put(cpuTimeRecord);
                    }
                    json.put("cpu_usage", cpuTimeJSON);

                    // Put in CNN times
                    JSONArray cnnElapsedTimeJSON = new JSONArray();
                    for (int cnnTime : eo.getElapsedCnnTime()) {
                        cnnElapsedTimeJSON.put(cnnTime);
                    }
                    json.put("cnn_inference_time", cnnElapsedTimeJSON);

                    // Put in NLP times
                    JSONArray nlpElapsedTimeJSON = new JSONArray();
                    for (int nlpTime : eo.getElapsedNlpTime()) {
                        nlpElapsedTimeJSON.put(nlpTime);
                    }
                    json.put("nlp_inference_time", nlpElapsedTimeJSON);

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
        PowerManager powerManager = (PowerManager) getSystemService(POWER_SERVICE);
        PowerManager.WakeLock wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK,
                "VQA::EvaluationWakeLock");
        wakeLock.acquire(10 * 60 * 1000L); // Lock for 10 minutes

        // Stops screen turning off as appears to effect behaviour
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

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
        cpuTimeTextView.setText(new StringBuilder().append("Average CPU usage = ").
                append(eo.getMeanCpuUtilisation()));


        String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE};
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(permissions, 1);
        }

        wakeLock.release();
    }
}