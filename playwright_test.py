import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("Navigating to frontend...")
        await page.goto("http://localhost:3000", timeout=60000)
        
        # Wait for the page to load by looking for the header
        await page.wait_for_selector("text=Facial Expression Recognition")
        print("Page loaded successfully.")
        
        # Test image path
        image_path = os.path.abspath(r"C:\Facial-Expression-Recognition\mock_photo.jpg")
        
        # Find the file input and upload the image
        print(f"Uploading image: {image_path}")
        await page.set_input_files("input[type='file']", image_path)
        
        # Wait for preview
        await page.wait_for_selector("img[alt='Preview']")
        print("Image preview loaded.")
        
        # Click analyze
        print("Clicking Analyze Expression...")
        await page.click("button:has-text('Analyze Expression')")
        
        # Wait for the result text 'Confidence'
        await page.wait_for_selector("h2", timeout=30000)
        print("Analysis complete.")
        
        # 5. prediction appears
        predicted_class_el = await page.query_selector("h2")
        predicted_class = await predicted_class_el.inner_text()
        assert predicted_class, "Prediction missing."
            
        # 6. confidence appears
        # 7. probability distribution appears
        # 8. bbox information appears
        body_el = await page.query_selector("body")
        text_content = await body_el.inner_text()
        
        required_texts = [
            "Confidence:",
            "Probability Distribution",
            "Detection Details",
            "Face Bounding Box:",
            "Using the largest detected face.",
            "Expression prediction is an AI estimate and may be inaccurate."
        ]
        
        for text in required_texts:
            assert text in text_content, f"Expected text not found on page: {text}"
            
        print("Verified all required text present.")
        
        # Verify bounding box overlay element is rendered
        overlay = await page.query_selector(".border-green-500")
        assert overlay is not None, "Face bounding box overlay was not rendered."
        print("Verified: Face bounding box overlay is rendered.")
        
        print("=============================")
        print(f"Predicted Expression: {predicted_class}")
        print("=============================")
        
        # Click try another image
        print("Testing reset...")
        await page.click("button:has-text('Try another image')")
        await page.wait_for_selector("text=Upload Image")
        print("Reset successful.")
        
        # Click analyze without image should not be possible as button is hidden, verify
        analyze_btn = await page.query_selector("button:has-text('Analyze Expression')")
        if analyze_btn:
            print("ERROR: Analyze button is visible without image!")
        else:
            print("Verified: Analyze button is hidden in upload state.")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
