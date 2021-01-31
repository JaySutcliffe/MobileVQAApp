package uk.ac.cam.js2428.mvqa.models;

import android.content.Context;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;

import uk.ac.cam.js2428.mvqa.ml.Cnn;
import uk.ac.cam.js2428.mvqa.ml.Vqa;

public class CnnLstmModel extends CnnLstmModelBase {
    private Cnn cnn;
    private Vqa model;

    @Override
    protected void setCnnImageFeature() {
        cnnImageFeature = cnn.process(imageFeature).getOutputFeature0AsTensorBuffer();
    }

    @Override
    protected float[] getAnswer() {
        Vqa.Outputs vqaOutputs = model.process(questionFeature, cnnImageFeature);
        return vqaOutputs.getOutputFeature0AsTensorBuffer().getFloatArray();
    }


    public CnnLstmModel(Context context) {
        super(context);

        Model.Options options1;
        Model.Options options2;
        CompatibilityList compatList = new CompatibilityList();

        if (compatList.isDelegateSupportedOnThisDevice()){
            options1 = new Model.Options.Builder().setDevice(Model.Device.GPU).build();
        } else {
            options1 = new Model.Options.Builder().setNumThreads(4).build();
        }

        // CnnLstmModel does not appear to work on the GPU due to incompatibilities
        options2 = new Model.Options.Builder().setNumThreads(4).build();

        try {
            cnn = Cnn.newInstance(context, options1);
        } catch (IOException e) {
            System.err.println("Problem initialising TensorFlow VQA model");
            e.printStackTrace();
        }

        // Initialising the VQA model
        try {
            model = Vqa.newInstance(context, options2);
            questionFeature =
                    TensorBuffer.createFixedSize(new int[]{1, 26}, DataType.FLOAT32);
            cnnImageFeature =
                    TensorBuffer.createFixedSize(new int[]{1, 1280}, DataType.FLOAT32);
        } catch (IOException e) {
            System.err.println("Problem initialising TensorFlow VQA model");
            e.printStackTrace();
        }
    }
}
