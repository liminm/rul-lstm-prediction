import onnxruntime as ort
import numpy as np
import pandas as pd


def run_onnx_rul_inference(request, payload):

    max_rul = 542  # use the same value you trained with

    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)] 
    col_names = index_names + setting_names + sensor_names
    df = pd.read_csv("data/test_FD001.txt", sep="\s+", header=None, names=col_names)
    df = df[df.unit_nr == payload.unit_nr]

    df = df.drop(columns=["unit_nr", "time_cycles"])  # Drop non-feature columns for scaling
    scaler = request.app.state.scaler
    scaled = scaler.transform(df)
    df = pd.DataFrame(scaled, columns=df.columns, index=df.index)

    features_to_drop = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]
    df = df.drop(columns=features_to_drop)

    input_np = df.to_numpy().astype(np.float32)
    input_np = np.expand_dims(input_np, axis=0)  # Add batch dimension  

    # Load ONNX model
    ort_session = ort.InferenceSession("models/lstm_model.onnx", providers=["CPUExecutionProvider"])

    # Perform prediction
    inputs = {
        "input": input_np,
        "lengths": np.array([input_np.shape[1]], dtype=np.int64)
    }
    outputs = ort_session.run(None, inputs)

    prediction = outputs[0].item()

    print(f"Raw ONNX model prediction: {prediction}")
    scaled_prediction = prediction * max_rul
    return scaled_prediction, prediction

