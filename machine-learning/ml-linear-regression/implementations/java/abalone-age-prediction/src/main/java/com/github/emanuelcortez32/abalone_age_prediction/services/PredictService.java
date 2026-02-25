package com.github.emanuelcortez32.abalone_age_prediction.services;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import org.springframework.stereotype.Service;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

@Service
public class PredictService {

    public PredictService() {

    }

    public float predictAbalonAge(double[] features) throws Exception {
        float predict = 0.0f;

        try (OrtEnvironment environment = OrtEnvironment.getEnvironment();
             OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
             OrtSession session = environment.createSession("/home/emanuel/Developer/Personal/learning-ia/ml-linear-regression/out/lin_reg_model.onnx", sessionOptions)) {

            long[] shape = new long[]{1,8};
            float[] floatArray = new float[features.length];

            for (int i = 0; i < features.length; i++) {
                floatArray[i] = (float) features[i];
            }

            FloatBuffer fb = FloatBuffer.wrap(floatArray);

            OnnxTensor tensor = OnnxTensor.createTensor(environment, fb, shape);

            Map<String, OnnxTensor> inputs = new HashMap<>();

            inputs.put("input", tensor);

            try (OrtSession.Result result = session.run(inputs)) {
                OnnxValue out = result.get(0);

                float[][] predictions = (float[][]) out.getValue();

                predict = predictions[0][0];
            }
        }

        return predict;
    }
}
