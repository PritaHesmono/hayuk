document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
    const resultBoxes = document.querySelectorAll(".result-box");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const fileInput = document.getElementById("xlsFile");
        const file = fileInput.files[0];
        
        if (!file) {
            alert("Please select a file.");
            return;
        }

        const formData = new FormData();
        formData.append("xlsFile", file);

        try {
            // Send file to server for prediction
            const response = await fetch("http://localhost:8501", { // Ganti dengan URL Streamlit jika diperlukan
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error("Prediction failed. Please try again.");

            const data = await response.json();

            // Update result boxes with response
            resultBoxes[0].innerText = data.result1 || "No data for Result Box 1";
            resultBoxes[1].innerText = data.result2 || "No data for Result Box 2";
            resultBoxes[2].innerText = data.result3 || "No data for Result Box 3";
        } catch (error) {
            console.error("Error:", error.message);
            alert("An error occurred while predicting.");
        }
    });
});