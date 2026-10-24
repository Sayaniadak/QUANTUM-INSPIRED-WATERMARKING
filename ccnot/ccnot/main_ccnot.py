import os
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import KEY, LOGISTIC_KEY, MAP_CHOICE as CONFIG_MAP, WM_SIZE, DCT_BASE_STEP, default_repeat

# Ensure script directory and parent directory are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in [str(SCRIPT_DIR), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Locate images folder (check beside script, in cwd, or in project root)
if (SCRIPT_DIR / "images").is_dir():
    IMAGES_DIR = SCRIPT_DIR / "images"
elif (Path.cwd() / "images").is_dir():
    IMAGES_DIR = Path.cwd() / "images"
elif (PROJECT_ROOT / "images").is_dir():
    IMAGES_DIR = PROJECT_ROOT / "images"
else:
    IMAGES_DIR = SCRIPT_DIR / "images"

# ============================================================
# STEP 1: LOAD CARRIER IMAGE DIRECTLY AS GRAYSCALE
# ============================================================

carrier_path = str(IMAGES_DIR / "carrier.png")
gray_carrier = cv2.imread(carrier_path, cv2.IMREAD_GRAYSCALE)

# Check whether carrier image loaded successfully
if gray_carrier is None:
    print(f"ERROR: Carrier image not found at '{carrier_path}'!")
    print("Make sure carrier.png is inside the images folder.")
    exit()

print("Carrier image loaded successfully.")
print("Grayscale carrier shape:", gray_carrier.shape)


# ============================================================
# DISPLAY GRAYSCALE CARRIER
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    gray_carrier,
    cmap="gray"
)

plt.title("Grayscale Carrier Image")
plt.axis("off")
plt.show()


# ============================================================
# STEP 3: PREPROCESS CARRIER
# ============================================================

# Convert pixel values from:
#
#       0 - 255
#
# to:
#
#       0 - 1
#
# This makes calculations such as variance easier to handle.

normalized_carrier = (
    gray_carrier.astype(np.float32) / 255.0
)

print("Carrier preprocessing completed.")
print(
    "Normalized carrier range:",
    normalized_carrier.min(),
    "to",
    normalized_carrier.max()
)


# ============================================================
# STEP 4: REGION SCORER
# ============================================================
#
# The carrier image is divided into:
#
#       10 × 10 = 100 regions
#
# For every region we calculate:
#
# 1. Pixel variance
# 2. Edge strength
#
# These are combined to produce a region score.
#
# Higher score = more detailed/complex region
#
# Lower score = smoother region
# ============================================================

def calculate_region_scores(image, rows=10, cols=10):

    height, width = image.shape

    regions = []

    # --------------------------------------------------------
    # Calculate region boundaries
    #
    # Using linspace ensures that ALL pixels are included,
    # even when image dimensions are not perfectly divisible
    # by the number of regions.
    # --------------------------------------------------------

    y_boundaries = np.linspace(
        0,
        height,
        rows + 1,
        dtype=int
    )

    x_boundaries = np.linspace(
        0,
        width,
        cols + 1,
        dtype=int
    )

    # --------------------------------------------------------
    # Process every region
    # --------------------------------------------------------

    for r in range(rows):

        for c in range(cols):

            # Region boundaries
            y1 = y_boundaries[r]
            y2 = y_boundaries[r + 1]

            x1 = x_boundaries[c]
            x2 = x_boundaries[c + 1]

            # Extract region
            region = image[y1:y2, x1:x2]

            # ------------------------------------------------
            # 1. PIXEL VARIANCE
            # ------------------------------------------------
            #
            # Variance tells us how much pixel values vary
            # inside the region.
            #
            # Smooth region:
            #       low variance
            #
            # Detailed region:
            #       high variance
            # ------------------------------------------------

            variance = np.var(region)

            # ------------------------------------------------
            # 2. EDGE STRENGTH
            # ------------------------------------------------
            #
            # Canny detects edges.
            #
            # More edges = more image detail.
            # ------------------------------------------------

            region_uint8 = (
                region * 255
            ).astype(np.uint8)

            edges = cv2.Canny(
                region_uint8,
                50,
                150
            )

            # Percentage of pixels that are edges
            edge_strength = np.mean(
                edges > 0
            )

            # ------------------------------------------------
            # 3. COMBINED REGION SCORE
            # ------------------------------------------------
            #
            # For now:
            #
            #       score =
            #       variance + edge strength
            #
            # We can later experiment with different
            # weights if required.
            # ------------------------------------------------

            score = (
                variance +
                edge_strength
            )

            # Store information about region
            regions.append({

                "row": r,
                "col": c,

                "x": x1,
                "y": y1,

                "width": x2 - x1,
                "height": y2 - y1,

                "variance": variance,

                "edge_strength": edge_strength,

                "score": score
            })

    return regions


# ============================================================
# CALCULATE REGION SCORES
# ============================================================

rows = 10
cols = 10

regions = calculate_region_scores(
    normalized_carrier,
    rows=rows,
    cols=cols
)

print("\nCarrier divided into:")
print(rows, "×", cols, "regions")

print(
    "Total regions:",
    len(regions)
)


# ============================================================
# STEP 5: SORT REGIONS
# ============================================================
#
# Sort regions from highest score to lowest score.
#
# Highest score:
#       most detailed/complex region
#
# Lowest score:
#       smoothest region
# ============================================================

regions_sorted = sorted(
    regions,
    key=lambda region: region["score"],
    reverse=True
)


# ============================================================
# DISPLAY TOP 10 REGIONS
# ============================================================

print("\n")
print("====================================================")
print("TOP 10 CARRIER REGIONS")
print("====================================================")

for i, region in enumerate(
    regions_sorted[:10]
):

    print(
        f"Rank {i + 1}: "
        f"Region ({region['row'] + 1}, "
        f"{region['col'] + 1}) | "
        f"Variance = {region['variance']:.4f} | "
        f"Edge = {region['edge_strength']:.4f} | "
        f"Score = {region['score']:.4f}"
    )


# ============================================================
# DISPLAY ALL 100 REGION SCORES
# ============================================================

plt.figure(figsize=(12, 10))

plt.imshow(
    gray_carrier,
    cmap="gray"
)

for region in regions:

    x = region["x"]
    y = region["y"]

    region_width = region["width"]
    region_height = region["height"]

    # --------------------------------------------------------
    # Draw region boundary
    # --------------------------------------------------------

    rectangle = plt.Rectangle(
        (x, y),
        region_width,
        region_height,

        fill=False,

        edgecolor="red",

        linewidth=0.7
    )

    plt.gca().add_patch(
        rectangle
    )

    # --------------------------------------------------------
    # Display region score
    # --------------------------------------------------------

    plt.text(
        x + region_width / 2,
        y + region_height / 2,

        f"{region['score']:.2f}",

        color="yellow",

        ha="center",
        va="center",

        fontsize=6
    )


plt.title(
    "10 × 10 Carrier Region Scores"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# STEP 6: LOAD WATERMARK IMAGE
# ============================================================

watermark_path = str(IMAGES_DIR / "watermark.png")
watermark = cv2.imread(
    watermark_path,
    cv2.IMREAD_GRAYSCALE
)

# Check whether watermark loaded successfully
if watermark is None:

    print(f"\nERROR: Watermark image not found at '{watermark_path}'!")

    print(
        "Make sure watermark.png is inside the images folder."
    )

    exit()


print("\nWatermark image loaded successfully.")

print(
    "Grayscale watermark shape:",
    watermark.shape
)


# ============================================================
# STEP 8: RESIZE WATERMARK TO 128 × 128
# ============================================================

watermark_128 = cv2.resize(
    watermark,

    (WM_SIZE, WM_SIZE),

    interpolation=cv2.INTER_AREA
)

print(
    "Resized watermark shape:",
    watermark_128.shape
)


# ============================================================
# DISPLAY RESIZED WATERMARK
# ============================================================

plt.figure(figsize=(6, 6))

plt.imshow(
    watermark_128,
    cmap="gray"
)

plt.title(
    "128 × 128 Grayscale Watermark"
)

plt.axis("off")

plt.show()


# ============================================================
# STEP 9: CONVERT WATERMARK TO BINARY
# ============================================================
#
# The grayscale watermark contains values:
#
#       0 → 255
#
# We convert it to:
#
#       0 → black
#       1 → white
#
# Threshold:
#
#       < 127 → 0
#       >= 127 → 1
# ============================================================

_, binary_watermark = cv2.threshold(
    watermark_128,

    127,

    1,

    cv2.THRESH_BINARY
)


# Make sure the datatype is uint8
binary_watermark = binary_watermark.astype(
    np.uint8
)


print(
    "Binary watermark shape:",
    binary_watermark.shape
)

print(
    "Unique binary values:",
    np.unique(binary_watermark)
)


# ============================================================
# DISPLAY BINARY WATERMARK
# ============================================================

plt.figure(figsize=(6, 6))

plt.imshow(
    binary_watermark,
    cmap="gray"
)

plt.title(
    "128 × 128 Binary Watermark"
)

plt.axis("off")

plt.show()


# ============================================================
# STEP 10: CHAOTIC SCRAMBLING
# ============================================================
#
# We use a Logistic Chaotic Map.
#
# Formula:
#
#       x(n+1) = r × x(n) × (1 - x(n))
#
# where:
#
#       r ≈ 4
#
# The secret key determines the initial value.
#
# The chaotic sequence is used for:
#
#       1. Permutation
#       2. XOR substitution
#
# This produces a scrambled watermark.
# ============================================================

def chaotic_scramble(
    binary_image,
    key=0.7345
):

    # --------------------------------------------------------
    # Check key
    # --------------------------------------------------------

    if not 0 < key < 1:

        raise ValueError(
            "Chaotic key must be between 0 and 1."
        )


    # --------------------------------------------------------
    # Flatten 2D watermark
    #
    # 128 × 128
    #
    # becomes
    #
    # 16384 × 1
    # --------------------------------------------------------

    data = binary_image.flatten()

    N = len(data)


    # --------------------------------------------------------
    # Generate chaotic sequence
    # --------------------------------------------------------

    x = key

    chaotic_sequence = []

    for i in range(N):

        # Logistic map
        x = (
            3.999 *
            x *
            (1 - x)
        )

        chaotic_sequence.append(
            x
        )


    chaotic_sequence = np.array(
        chaotic_sequence
    )


    # --------------------------------------------------------
    # Generate permutation
    # --------------------------------------------------------
    #
    # Sorting the chaotic sequence gives us a deterministic
    # permutation.
    # --------------------------------------------------------

    permutation = np.argsort(
        chaotic_sequence
    )


    # --------------------------------------------------------
    # Permute watermark pixels
    # --------------------------------------------------------

    scrambled = data[
        permutation
    ]


    # --------------------------------------------------------
    # Generate chaotic binary sequence
    #
    # Values > 0.5 become 1
    # Values <= 0.5 become 0
    # --------------------------------------------------------

    chaotic_bits = (
        chaotic_sequence > 0.5
    ).astype(np.uint8)


    # --------------------------------------------------------
    # XOR substitution
    # --------------------------------------------------------

    scrambled = np.bitwise_xor(
        scrambled,
        chaotic_bits
    )


    # --------------------------------------------------------
    # Convert back to 128 × 128
    # --------------------------------------------------------

    scrambled = scrambled.reshape(
        binary_image.shape
    )


    return (
        scrambled,
        permutation,
        chaotic_bits
    )


# ============================================================
# PERFORM CHAOTIC SCRAMBLING
# ============================================================

from chaotic_lscm import chaotic_scramble as chaotic_scramble_lscm

MAP_CHOICE = "lscm"   # "logistic" = old baseline, "lscm" = new

if MAP_CHOICE == CONFIG_MAP:
    (
        scrambled_watermark,
        permutation,
        chaotic_bits
    ) = chaotic_scramble_lscm(
        binary_watermark,
        key=KEY
    )
else:
    (
        scrambled_watermark,
        permutation,
        chaotic_bits
    ) = chaotic_scramble(
        binary_watermark,
        key=0.7345
    )


print("\nChaotic scrambling completed.")

print(
    "Scrambled watermark shape:",
    scrambled_watermark.shape
)


# ============================================================
# DISPLAY SCRAMBLED WATERMARK
# ============================================================

plt.figure(figsize=(6, 6))

plt.imshow(
    scrambled_watermark,
    cmap="gray"
)

plt.title(
    "Chaotically Scrambled Watermark"
)

plt.axis("off")

plt.show()


# ============================================================
# CURRENT IMPLEMENTATION STATUS
# ============================================================

print("\n")
print("====================================================")
print("CURRENT IMPLEMENTATION STATUS")
print("====================================================")

print("1. Load carrier image             : DONE")
print("2. Convert carrier to grayscale   : DONE")
print("3. Preprocess carrier             : DONE")
print("4. 10×10 region scoring          : DONE")
print("5. Region ranking                 : DONE")
print("6. Load watermark                 : DONE")
print("7. Grayscale watermark            : DONE")
print("8. Resize watermark to 128×128    : DONE")
print("9. Convert watermark to binary    : DONE")
print("10. Chaotic scrambling            : DONE")


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# ------------------------------------------------------------
# 1. SE ATTENTION MODULE
# ------------------------------------------------------------

class SEAttention(nn.Module):

    def __init__(self, feature_size=64, reduction=16):
        super().__init__()

        self.fc1 = nn.Linear(
            feature_size,
            feature_size // reduction
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            feature_size // reduction,
            feature_size
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, features):

        # Generate importance weights
        weights = self.fc1(features)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = self.sigmoid(weights)

        # Reweight the original features
        attended_features = features * weights

        return attended_features, weights


# ------------------------------------------------------------
# 2. COMPLETE FEATURE PROCESSING PIPELINE
# ------------------------------------------------------------

class WatermarkFeaturePipeline(nn.Module):

    def __init__(self, pretrained=True):

        super().__init__()

        # Load pretrained ResNet18
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        resnet = models.resnet18(weights=weights)

        # Keep convolutional layers, remove classifier
        self.cnn = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # Freeze pretrained CNN parameters
        for parameter in self.cnn.parameters():
            parameter.requires_grad = False

        # 512 CNN features -> 64 features
        self.projection = nn.Linear(512, 64)

        # SE attention: 64 -> 64
        self.attention = SEAttention(
            feature_size=64,
            reduction=16
        )

        # Dense compression: 64 -> 8
        self.compressor = nn.Sequential(
            nn.Linear(64, 8),
            nn.Sigmoid()
        )

    def forward(self, image):

        # ----------------------------------------------------
        # CNN FEATURE EXTRACTION
        # ----------------------------------------------------

        # CNN is frozen; gradients are unnecessary here
        with torch.no_grad():
            feature_maps = self.cnn(image)

            # Convert feature maps to 512 values
            pooled_features = F.adaptive_avg_pool2d(
                feature_maps,
                (1, 1)
            )

            pooled_features = pooled_features.flatten(1)

        # Project 512 features into 64 features
        cnn_features = F.relu(
            self.projection(pooled_features)
        )

        # ----------------------------------------------------
        # SE ATTENTION
        # ----------------------------------------------------

        attended_features, attention_weights = (
            self.attention(cnn_features)
        )

        # ----------------------------------------------------
        # DENSE COMPRESSION
        # ----------------------------------------------------

        compressed_features = self.compressor(
            attended_features
        )

        # ----------------------------------------------------
        # ANGLE ENCODING
        # ----------------------------------------------------

        # Map feature values from [0, 1] to [0, pi]
        angles = compressed_features * math.pi

        return (
            cnn_features,
            attended_features,
            attention_weights,
            compressed_features,
            angles
        )


# ------------------------------------------------------------
# 3. PREPARE THE SCRAMBLED WATERMARK FOR THE CNN
# ------------------------------------------------------------

# Use the scrambled watermark generated by your earlier code.
# Its shape should be 128 x 128 and its values should be 0 or 1.

watermark_input = np.asarray(
    scrambled_watermark,
    dtype=np.float32
)

assert watermark_input.shape == (WM_SIZE, WM_SIZE), (
    "Expected a 128x128 scrambled watermark."
)

# Convert NumPy array to PyTorch tensor:
# (128, 128) -> (1, 1, 128, 128)
watermark_tensor = torch.from_numpy(
    watermark_input
).unsqueeze(0).unsqueeze(0)

# Convert grayscale input to 3 channels for ResNet18
watermark_tensor = watermark_tensor.repeat(
    1, 3, 1, 1
)

# ResNet18 pretrained weights expect 224 x 224 input
watermark_tensor = F.interpolate(
    watermark_tensor,
    size=(224, 224),
    mode="bilinear",
    align_corners=False
)

# Apply ImageNet normalization
mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)

watermark_tensor = (
    watermark_tensor - mean
) / std


# ------------------------------------------------------------
# 4. INITIALIZE THE PIPELINE
# ------------------------------------------------------------

torch.manual_seed(42)

# Downloads pretrained weights the first time if necessary.
model = WatermarkFeaturePipeline(
    pretrained=True
)

model.eval()


# ------------------------------------------------------------
# 5. RUN ALL FOUR STAGES
# ------------------------------------------------------------

with torch.no_grad():

    (
        cnn_features,
        attended_features,
        attention_weights,
        compressed_features,
        angles
    ) = model(watermark_tensor)


# ------------------------------------------------------------
# 6. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n==========================================")
print("CNN + SE + COMPRESSION + ANGLE ENCODING")
print("==========================================")

print(
    "CNN feature shape:",
    tuple(cnn_features.shape)
)

print(
    "SE-attended feature shape:",
    tuple(attended_features.shape)
)

print(
    "Attention weight shape:",
    tuple(attention_weights.shape)
)

print(
    "Compressed feature shape:",
    tuple(compressed_features.shape)
)

print(
    "Angle encoding shape:",
    tuple(angles.shape)
)

print("\nEight compressed features:")

print(
    compressed_features[0].cpu().numpy()
)

print("\nEight rotation angles (radians):")

print(
    angles[0].cpu().numpy()
)

print("\nEight rotation angles (degrees):")

print(
    torch.rad2deg(angles[0]).cpu().numpy()
)

print("\nAll four processing stages executed.")




# ============================================================
# STEP 16: DNA ENCODING
# ============================================================

print("\n==========================================")
print("STEP 16: DNA ENCODING")
print("==========================================")

# Convert the 8 compressed features (range 0-1) into 8-bit values.
compressed_uint8 = torch.round(
    compressed_features * 255
).to(torch.uint8)

compressed_values = compressed_uint8[0].cpu().numpy()

print("\n8-bit compressed values:")
print(compressed_values)

# Convert every value into an 8-bit binary string.
binary_string = "".join(
    format(int(value), "08b")
    for value in compressed_values
)

print("\nBinary representation:")
print(binary_string)
print("Total binary bits:", len(binary_string))

# Map each pair of bits to one DNA base: 00=A, 01=C, 10=G, 11=T.
DNA_MAP = {
    "00": "A",
    "01": "C",
    "10": "G",
    "11": "T"
}

dna_sequence = "".join(
    DNA_MAP[binary_string[i:i + 2]]
    for i in range(0, len(binary_string), 2)
)

print("\nDNA sequence:")
print(dna_sequence)
print("DNA sequence length:", len(dna_sequence))


# ============================================================
# STEP 16B: DNA DECODING TEST
# ============================================================

print("\n==========================================")
print("STEP 16B: DNA DECODING TEST")
print("==========================================")

REVERSE_DNA_MAP = {
    "A": "00",
    "C": "01",
    "G": "10",
    "T": "11"
}

decoded_binary = "".join(
    REVERSE_DNA_MAP[base]
    for base in dna_sequence
)

print("\nDecoded binary:")
print(decoded_binary)

if decoded_binary == binary_string:
    print("\nDNA round-trip test: SUCCESS")
else:
    print("\nDNA round-trip test: FAILED")

# Convert decoded bits back into 8-bit integer values.
decoded_values = np.array(
    [
        int(decoded_binary[i:i + 8], 2)
        for i in range(0, len(decoded_binary), 8)
    ],
    dtype=np.uint8
)

print("\nDecoded 8-bit values:")
print(decoded_values)

if np.array_equal(compressed_values, decoded_values):
    print("Binary -> DNA -> Binary test: SUCCESS")
else:
    print("Binary -> DNA -> Binary test: FAILED")



# ============================================================
# STEP 17: QEC-INSPIRED BACKUP (3-BIT REPETITION CODE)
# ============================================================
#
# This is a classical simulation of a simple error-correction idea.
# Each bit is stored three times: 0 -> 000, 1 -> 111.
# Majority voting can recover the original bit if at most one bit
# in a triplet is flipped.
#
# This demonstrates a repetition-code concept; it is NOT a full
# quantum circuit or a complete quantum error-correction system.
# ============================================================

print("\n==========================================")
print("STEP 17: QEC-INSPIRED BACKUP")
print("==========================================")

# Use the binary data produced by the DNA encoding stage.
qec_original_bits = binary_string

print("\nOriginal bit count:", len(qec_original_bits))
print("Original bits:")
print(qec_original_bits)

# 1. Encode each bit three times.
qec_encoded_bits = "".join(bit * 3 for bit in qec_original_bits)

print("\nRepetition-coded bit count:", len(qec_encoded_bits))
print("First 48 encoded bits:")
print(qec_encoded_bits[:48])

# 2. Simulate one bit-flip error in the first triplet.
qec_noisy_bits = list(qec_encoded_bits)
if qec_noisy_bits:
    qec_noisy_bits[0] = "1" if qec_noisy_bits[0] == "0" else "0"
qec_noisy_bits = "".join(qec_noisy_bits)

print("\nSimulated one bit-flip error in the first triplet.")
print("First 48 noisy bits:")
print(qec_noisy_bits[:48])

# 3. Decode each triplet by majority vote.
qec_recovered_list = []
for i in range(0, len(qec_noisy_bits), 3):
    triplet = qec_noisy_bits[i:i + 3]
    if len(triplet) != 3:
        raise ValueError("QEC data length is not divisible by 3.")
    qec_recovered_list.append("1" if triplet.count("1") >= 2 else "0")
qec_recovered_bits = "".join(qec_recovered_list)

print("\nRecovered bits:")
print(qec_recovered_bits)

# 4. Verify that majority voting restored the original data.
if qec_recovered_bits == qec_original_bits:
    print("\nQEC repetition-code test: SUCCESS")
    print("The simulated single-bit error was corrected.")
else:
    print("\nQEC repetition-code test: FAILED")
    print("The recovered data does not match the original.")

# 5. Convert the recovered bits back to DNA and verify the sequence.
qec_recovered_dna = "".join(
    DNA_MAP[qec_recovered_bits[i:i + 2]]
    for i in range(0, len(qec_recovered_bits), 2)
)
print("\nRecovered DNA sequence:")
print(qec_recovered_dna)

if qec_recovered_dna == dna_sequence:
    print("DNA + QEC backup verification: SUCCESS")
else:
    print("DNA + QEC backup verification: FAILED")


# ============================================================
# STEP 17B: REAL QUANTUM CIRCUIT DEMO (QISKIT CCNOT / TOFFOLI)
# ============================================================
# Classical bits are prepared as computational-basis qubits:
# bit 0 -> |0> (leave qubit at its initial state)
# bit 1 -> |1> (apply an X gate)
# Then apply a real CCNOT/Toffoli gate to three qubits.
# This is a small demonstration only; it does not mean the whole
# watermark is quantum-encoded or quantum-error-corrected.
# ============================================================

print("\n==========================================")
print("STEP 17B: QISKIT QUANTUM CCNOT DEMONSTRATION")
print("==========================================")

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    # Use the first three available classical data bits as an example.
    # Pad with zeros if the input has fewer than three bits.
    ccnot_input_bits = (qec_original_bits[:3] + "000")[:3]
    print("Classical input bits (q0, q1, q2):", ccnot_input_bits)

    # Create three qubits and three classical readout bits.
    ccnot_circuit = QuantumCircuit(3, 3)

    # Encode classical 0/1 values into computational-basis states.
    for qubit_index, bit in enumerate(ccnot_input_bits):
        if bit == "1":
            ccnot_circuit.x(qubit_index)

    ccnot_circuit.barrier()

    # CCNOT: q0 and q1 are controls; q2 is the target.
    # q2 flips only when q0 = 1 AND q1 = 1.
    ccnot_circuit.ccx(0, 1, 2)
    ccnot_circuit.barrier()

    ccnot_circuit.measure([0, 1, 2], [0, 1, 2])

    print("\nCCNOT circuit:")
    print(ccnot_circuit.draw(output="text"))

    simulator = AerSimulator()
    compiled_circuit = transpile(ccnot_circuit, simulator)
    simulation_result = simulator.run(
        compiled_circuit,
        shots=1,
        memory=True
    ).result()

    measured_qiskit_order = simulation_result.get_memory()[0]
    # Qiskit displays classical bit strings in reverse index order.
    measured_bits = measured_qiskit_order[::-1]

    expected_bits = list(ccnot_input_bits)
    if ccnot_input_bits[0] == "1" and ccnot_input_bits[1] == "1":
        expected_bits[2] = "1" if expected_bits[2] == "0" else "0"
    expected_bits = "".join(expected_bits)

    print("\nMeasured output bits (q0, q1, q2):", measured_bits)
    print("Expected CCNOT output bits:          ", expected_bits)

    if measured_bits == expected_bits:
        print("CCNOT simulation test: SUCCESS")
    else:
        print("CCNOT simulation test: FAILED")

except ImportError:
    print("Qiskit is not installed, so the real quantum-circuit demo was skipped.")
    print("Install the required packages in your active virtual environment with:")
    print("python -m pip install qiskit qiskit-aer")
    print("Then run this file again to execute the CCNOT circuit.")


# ============================================================
# STEP 18: WATERMARK EMBEDDING (DCT-QIM)   <-- replaces the old LSB Step 18
# ============================================================
# The carrier is cut into 8x8 blocks and every block gets a DCT (same transform
# JPEG uses). Each bit is hidden in a low-frequency coefficient with QIM, and every
# bit is stored REPEAT times. The step size is bigger in detailed regions (bolder,
# hidden by texture) and smaller in smooth regions (gentle), using the region scorer.
# ============================================================

print("\n==========================================")
print("STEP 18: WATERMARK EMBEDDING (DCT-QIM)")
print("==========================================")

from embed_common import build_payload, decode_payload, watermark_hash_bits
from dct_embed import step_map, embed_stream, extract_stream

# 8 angle values (8-bit each) = 64 bits
angle_u8 = torch.round(angles[0] / math.pi * 255).to(torch.uint8).cpu().numpy()

# Scrambled watermark bits (4096 for 64x64)
wm_bits = (np.asarray(scrambled_watermark) > 0).astype(np.uint8).flatten()

# Integrity hash: 32-bit fingerprint of the scrambled watermark bits
hash_bits = watermark_hash_bits(wm_bits)

# Payload: [64 angle bits + 32 hash bits] -> CCNOT -> DNA -> QEC (3 copies) = 288 bits
payload = build_payload(angle_u8, extra_bits=hash_bits)
payload_bits = np.array([int(b) for b in payload["qec_bits"]], dtype=np.uint8)
n_p = payload_bits.size

# Stream layout: [ QEC payload | scrambled watermark ]
stream = np.concatenate([payload_bits, wm_bits])

REPEAT = default_repeat(WM_SIZE)
steps = step_map(gray_carrier, DCT_BASE_STEP, adaptive=True)

watermarked_image, passes, residual_errors = embed_stream(
    gray_carrier, stream, REPEAT, steps
)

# ---- self-checks (no attack, so everything must read back exactly) ----
assert residual_errors == 0, (
    f"{residual_errors} bits did not read back after embedding; try a larger DCT_BASE_STEP"
)
read_bits, _ = extract_stream(watermarked_image, stream.size, REPEAT, steps)
assert np.array_equal(read_bits, stream), "stream read-back mismatch"
assert decode_payload("".join(map(str, read_bits[:n_p])), payload["pad_len"]) == payload["bits"]
print("Embedding self-checks: SUCCESS")
print("Payload bits (QEC):", n_p, "| watermark bits:", wm_bits.size,
      "| copies of each bit:", REPEAT, "| embedding passes:", passes)
print("Integrity hash (32 bits):", hash_bits)

# ---- save what extraction needs ----
np.savez(
    str(IMAGES_DIR / "embed_meta.npz"),
    method="dct",
    map_choice=MAP_CHOICE,
    pad_len=payload["pad_len"],
    payload_bits=payload["bits"],          # 64 angle bits + 32 hash bits (before QEC)
    n_payload=n_p,
    n_wm=wm_bits.size,
    n_hash=len(hash_bits),
    repeat=REPEAT,
    base_step=DCT_BASE_STEP,
    adaptive=1,
    steps=steps,
    orig_watermark=binary_watermark,
    rows=10, cols=10, top_k=50,
)
torch.save(model.state_dict(), str(IMAGES_DIR / "pipeline_weights.pt"))

watermark_bits_2d = (np.asarray(scrambled_watermark) > 0).astype(np.uint8)
watermark_bits = wm_bits

# Save the watermarked carrier (PNG is lossless, so nothing changes on saving).
output_path = str(IMAGES_DIR / "watermarked.png")
if not cv2.imwrite(output_path, watermarked_image):
    raise OSError(f"Could not save {output_path}. Check that the images folder exists.")

# Image-quality metrics against the original grayscale carrier.
difference = gray_carrier.astype(np.float64) - watermarked_image.astype(np.float64)
mse_embedding = float(np.mean(difference ** 2))
psnr_embedding = float("inf") if mse_embedding == 0 else float(10 * np.log10(255.0 ** 2 / mse_embedding))

print("Original carrier shape:", gray_carrier.shape)
print("Scrambled watermark shape:", watermark_bits_2d.shape)
print("Embedded watermark bits:", int(watermark_bits.size))
print("Watermarked image saved to:", output_path)
print("Embedding MSE:", mse_embedding)
print("Embedding PSNR (dB):", psnr_embedding)
try:
    from skimage.metrics import structural_similarity
    print("Embedding SSIM:", structural_similarity(gray_carrier, watermarked_image, data_range=255))
except Exception:
    pass
print("Embedding method: DCT-QIM (8x8 blocks, base step", DCT_BASE_STEP, ", repeat", REPEAT, ")")

# Display original carrier and watermarked output for visual comparison.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_carrier, cmap="gray", vmin=0, vmax=255)
plt.title("Original Grayscale Carrier")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(watermarked_image, cmap="gray", vmin=0, vmax=255)
plt.title("Watermarked Carrier (DCT-QIM)")
plt.axis("off")
plt.tight_layout()
plt.show()


# ============================================================
# UPDATED IMPLEMENTATION STATUS
# ============================================================

print("\nNext stages:")
print("16. DNA encoding and decoding      : DONE")
print("17. QEC-inspired repetition backup : DONE (classical simulation)")
print("17B. Qiskit CCNOT circuit          : RUN if Qiskit is installed")
print("18. Watermark embedding            : DONE (DCT-QIM)")
print("19. Extraction                     : run extract.py")