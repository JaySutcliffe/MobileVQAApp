package uk.ac.cam.js2428.mvqa.models;

import android.content.Context;

public class CnnLstmI8Model extends CnnLstmModelBase {

    @Override
    protected void setCnnImageFeature() {

    }

    @Override
    protected float[] getAnswer() {
        return new float[0];
    }

    public CnnLstmI8Model(Context context) {
        super(context);
    }
}
