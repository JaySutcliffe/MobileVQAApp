package uk.ac.cam.js2428.mvqa.models;

import android.content.Context;

import uk.ac.cam.js2428.mvqa.ml.FullAttentionVqaF16;
import uk.ac.cam.js2428.mvqa.ml.Mobilenet3by3;
import uk.ac.cam.js2428.mvqa.ml.MobilenetF16;
import uk.ac.cam.js2428.mvqa.ml.Vqa;
import uk.ac.cam.js2428.mvqa.ml.VqaF16;

public class CnnF16LstmModel extends CnnLstmModelBase {
    private MobilenetF16 cnn;
    private VqaF16 model;

    @Override
    protected void setCnnImageFeature() {
        cnnImageFeature = cnn.process(imageFeature).getOutputFeature0AsTensorBuffer();
    }

    @Override
    protected float[] getAnswer() {
        VqaF16.Outputs vqaOutputs = model.process(questionFeature, cnnImageFeature);
        return vqaOutputs.getOutputFeature0AsTensorBuffer().getFloatArray();
    }


    public CnnF16LstmModel(Context context) {
        super(context);
    }
}
