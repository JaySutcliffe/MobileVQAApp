package uk.ac.cam.js2428.mvqa;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private VqaModel vqa;

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
        EvaluationOutput eo = vqa.evaluate();
        TextView accuracyTextView = findViewById(R.id.accuracyTextView);
        accuracyTextView.setText(new StringBuilder().append("Accuracy = ").
                append(eo.getAccuracy()));
        TextView executionTimeTextView = findViewById(R.id.ExecutionTimeTextView);
        executionTimeTextView.setText(new StringBuilder().append("Execution times(ms): cnn = ").
                append(eo.getMeanCnnTime()).
                append(", nlp = ").
                append(eo.getMeanNlpTime()));
        vqa.evaluate();
    }
}