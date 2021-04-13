package uk.ac.cam.js2428.mvqa.models;

import android.content.Context;
import android.graphics.Bitmap;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;

import uk.ac.cam.js2428.mvqa.questions.QuestionException;

public abstract class CnnLstmModelBase extends VqaModel {

    protected ImageProcessor imageProcessor;

    protected TensorBuffer imageFeature;
    protected TensorBuffer questionFeature;
    protected TensorBuffer cnnImageFeature;

    private static final int PACKET_COUNT = 20;
    private static final int PACKET_SIZE = 20;

    protected abstract void setCnnImageFeature();
    protected abstract float[] getAnswer();

    @Override
    public void setImage(Bitmap bitmap) {
        TensorImage tImage = new TensorImage(DataType.FLOAT32);
        tImage.load(bitmap);
        imageFeature = imageProcessor.process(tImage).getTensorBuffer();
        setCnnImageFeature();
    }

    @Override
    public String runInference(String question) throws QuestionException {
        float[] q = parseQuestion(question);
        questionFeature.loadArray(parseQuestion(question));
        float[] probs = getAnswer();
        float maxProb = 0;
        int answer = -1;
        for (int k = 0; k < probs.length; k++) {
            if (maxProb < probs[k]) {
                answer = k;
                maxProb = probs[k];
            }
        }
        return decodeAnswer(answer);
    }

    public NlpOnlyEvaluationOutput evaluateQuestionOnly() {
        try {
            long[] elapsedTime = new long[PACKET_COUNT];
            int[] matches = new int[PACKET_COUNT];
            for (int i = 0; i < PACKET_COUNT; i++) {
                float[][] questions = new float[PACKET_SIZE][maxQuestionLength];
                float[][] imageFeatures = new float[PACKET_SIZE][4096];
                int[] answers = new int[PACKET_SIZE];
                InputStream is = context.getAssets().open(
                        "packets/test_packet" + i + ".bin");
                DataInputStream dis = new DataInputStream(is);
                for (int n = 0; n < PACKET_SIZE; n++) {
                    for (int m = 0; m < maxQuestionLength; m++) {
                        questions[n][m] = dis.readFloat();
                    }
                }
                for (int n = 0; n < PACKET_SIZE; n++) {
                    for (int m = 0; m < 4096; m++) {
                        imageFeatures[n][m] = dis.readFloat();
                    }
                }
                for (int n = 0; n < PACKET_SIZE; n++) {
                    answers[n] = (int)dis.readFloat();
                }
                dis.close();

                long startTime = System.currentTimeMillis();
                for (int j = 0; j < PACKET_SIZE; j++) {
                    cnnImageFeature.loadArray(imageFeatures[j]);
                    questionFeature.loadArray(questions[j]);
                    float[] probs = getAnswer();
                    float maxProb = 0;
                    int answer = -1;
                    for (int k = 0; k < probs.length; k++) {
                       if (maxProb < probs[k]) {
                           answer = k;
                           maxProb = probs[k];
                       }
                    }
                    if (answers[j] == answer) {
                        matches[i]++;
                    }
                }
                elapsedTime[i] = System.currentTimeMillis() - startTime;
            }
            return new NlpOnlyEvaluationOutput(PACKET_SIZE, matches, elapsedTime);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public CnnLstmModelBase(Context context) {
        super(context);

        imageProcessor = new ImageProcessor.Builder()
                            .add(new ResizeOp(224, 224,
                                     ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                            .add(new NormalizeOp(127.5f, 127.5f))
                            .build();
    }
}