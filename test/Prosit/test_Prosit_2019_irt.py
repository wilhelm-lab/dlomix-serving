from server_config import SERVER_GRPC, SERVER_HTTP
import tritonclient.grpc as grpcclient
import numpy as np
import requests

MODEL_NAME = "Prosit_2019_irt"


def test_available_http():
    req = requests.get(f"{SERVER_HTTP}/v2/models/{MODEL_NAME}", timeout=1)
    assert req.status_code == 200


def test_available_grpc():
    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)
    assert triton_client.is_model_ready(MODEL_NAME)


def test_inference():
    seq = np.load(f"test/Prosit/arr_Prosit_2019_intensity_seq.npy")

    triton_client = grpcclient.InferenceServerClient(url=SERVER_GRPC)

    in_pep_seq = grpcclient.InferInput("sequence_integer", seq.shape, "INT32")
    in_pep_seq.set_data_from_numpy(seq)

    result = triton_client.infer(
        MODEL_NAME,
        inputs=[in_pep_seq],
        outputs=[
            grpcclient.InferRequestedOutput("prediction/BiasAdd:0"),
        ],
    )

    intensities = result.as_numpy("prediction/BiasAdd:0")

    assert intensities.shape == (5, 1)

    assert np.allclose(
        intensities,
        np.load(f"test/Prosit/arr_{MODEL_NAME}_raw.npy"),
        rtol=0,
        atol=1e-5,
    )