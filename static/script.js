// ----------------------
// PPE Compliance Monitor
// ----------------------

const uploadInput = document.getElementById("videoInput");
const processButton = document.getElementById("processButton");
const loading = document.getElementById("loading");
const resultVideo = document.getElementById("resultVideo");
const downloadReportButton = document.getElementById("downloadReportButton");
document
.getElementById("downloadReportButton")
.addEventListener("click", () => {

    window.location.href = "/download-report";

});

let processedVideoURL = null;


// ----------------------
// Process button
// ----------------------

processButton.addEventListener("click", async () => {

    // Make sure a file is selected
    if (uploadInput.files.length === 0) {
        alert("Please select a video first.");
        return;
    }

    const file = uploadInput.files[0];

    // Create multipart form data
    const formData = new FormData();
    formData.append("video", file);

    // Disable button while processing
    processButton.disabled = true;
    processButton.textContent = "Processing...";

    loading.style.display = "block";

    // Hide previous result
    resultVideo.style.display = "none";
    downloadReportButton.style.display = "none";

    // Release old object URL
    if (processedVideoURL) {
        URL.revokeObjectURL(processedVideoURL);
        processedVideoURL = null;
    }

    try {

        const response = await fetch("/process-video", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {

            const error = await response.text();

            throw new Error(error);

        }

        const blob = await response.blob();

        console.log(blob);
        console.log(blob.size);
        console.log(blob.type);

        processedVideoURL = URL.createObjectURL(blob);
        window.open(processedVideoURL);

        // Display video
        resultVideo.src = processedVideoURL;

        resultVideo.onloadedmetadata = () => {
        console.log("Metadata loaded");
        console.log("Duration:", resultVideo.duration);
        };

        resultVideo.oncanplay = () => {
            console.log("Can play");
        };

        resultVideo.onplay = () => {
            console.log("Playing");
        };

        resultVideo.onerror = () => {
            console.log("Video error");
            console.log(resultVideo.error);
        };

        resultVideo.load();

        resultVideo.load();

        resultVideo.style.display = "block";

        // Enable download button
        downloadReportButton.style.display = "inline-block";

    }

    catch (error) {

        console.error(error);

        alert("Video processing failed.\n\n" + error.message);

    }

    finally {

        loading.style.display = "none";

        processButton.disabled = false;
        processButton.textContent = "Process Video";

    }

});

// ----------------------
// Download Complete Report
// ----------------------

downloadReportButton.addEventListener("click", () => {

    window.location.href = "/download-report";

});