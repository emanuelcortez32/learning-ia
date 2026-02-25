package com.github.emanuelcortez32.abalone_age_prediction.controllers;

import com.github.emanuelcortez32.abalone_age_prediction.dto.PredictRequest;
import com.github.emanuelcortez32.abalone_age_prediction.dto.PredictResponse;
import com.github.emanuelcortez32.abalone_age_prediction.services.PredictService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class PredictController {

    private final PredictService predictService;

    public PredictController(final PredictService predictService) {
        this.predictService = predictService;
    }

    @PostMapping(value = "/predict", produces = {MediaType.APPLICATION_JSON_VALUE})
    @ResponseBody
    public ResponseEntity<PredictResponse> POSTPredict(@RequestBody PredictRequest request) throws Exception {
        double[] features = new double[]{
                request.getSex(),
                request.getLength(),
                request.getDiameter(),
                request.getHeight(),
                request.getWholeWeight(),
                request.getShuckedWeight(),
                request.getVisceraWeight(),
                request.getShellWeight()
        };

        float prediction = predictService.predictAbalonAge(features);

        return ResponseEntity.ok(new PredictResponse(prediction));
    }
}
