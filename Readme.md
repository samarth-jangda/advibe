MODULE: SKETCH TO VIDEO

I) Sketchy COCO Dataset
    GT - Ground Truth
    trainInTrain - Train images of SketchyCOCO dataset from the train images of the COCO-Stuff dataset
    valInTrain - Val images of SketchyCOCO dataset from the train images of the COCO-Stuff dataset
    val - Val images of SketchyCOCO dataset from the val images of the COCO-Stuff dataset
    Sketch - Sketch scene of GT (a sketch scene has the same name with the corresponding GT)
    Annotation - Annotations for sketch scene segmentation

II) Algorithm Steps

    STEP:1
    2️⃣ Object Detection using DETR
    If the input is a sketch with multiple objects, DETR detects bounding boxes & class labels for each object.
    This helps identify objects and their rough spatial placement in the sketch.

    ✅ Output of DETR → A list of detected objects & their positions (bounding boxes).


    STEP:2
    2️⃣ Prompt Engineering (Preparing text prompt)
    1) Uses the input sketch image and the detected objects along with their positions
        to prepare a text prompt for the input sketch image
    2) There will be a list of criteria being given by user to improve text prompt.  


    STEP:3
    2️⃣ Edge Detection using Canny
    The Canny edge detector extracts sharp edges from the input image/sketch.
    This ensures that the key contours and structure of the objects are preserved.

    ✅ Output of Canny → A binary edge map (white edges on a black background).

    
    STEP:4
    2️⃣ CLIP Encodes the Text Prompt (Semantic Understanding)
    The user provides a text prompt, such as “A futuristic city with flying cars.”
    CLIP converts this prompt into a semantic vector (text embedding).
    This embedding guides Stable Diffusion on what kind of image to generate.

    ✅ Output of CLIP → A semantic embedding of the text prompt.


    STEP:5
    3️⃣ ControlNet Uses the Canny Edge Map for Structural Guidance
    The edge map from Canny is passed into ControlNet to ensure that the final generated image follows the sketch structure.
    ControlNet forces the output image to match the object placements from the sketch.
    It aligns objects correctly rather than letting Stable Diffusion generate random placements.

    ✅ Output of ControlNet → A refined control-conditioned latent space.


    STEP:6
    4️⃣ Stable Diffusion Generates the Final Image
    Stable Diffusion combines the text embedding from CLIP and the structural constraints from ControlNet.
    The image is generated in a latent space and then decoded back to a full-resolution image.
    The generated image respects the edges (from ControlNet) and the semantic meaning (from CLIP).

    ✅ Final Output → A high-quality image that maintains object structure while matching the text prompt.