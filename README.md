# Safran × AIonOS Aircraft Engine Acoustic Diagnostics Demo

This is a Streamlit demo repo adapted from the HVAC prototype into an aircraft engine use case focused on:

- acoustic diagnostics
- AI diagnostic copilot
- parametric identification
- quality check agents
- production management

## Included assets

- `assets/videos/engine_test_cell.mp4` — inline looping engine test-cell video
- `assets/images/solution_overview.png` — concept image used as the visual banner

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Upload to GitHub and deploy

1. Create a new GitHub repository on the web.
2. Upload all extracted files from this zip into the repo root.
3. In Streamlit Community Cloud or your own environment, point the app to `app.py`.
4. Ensure the video stays under `assets/videos/engine_test_cell.mp4`.

## Important note on autoplay sound

The app is coded to play the video inline and loop it. Modern browsers often block **autoplay with sound** until the user interacts once with the page. So loop will work inline, but sound may require one click depending on browser policy.
