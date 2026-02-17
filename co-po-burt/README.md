# CO-PO Mapping Application

A Streamlit application for CO-PO/PSO attainment calculation and NLP-based mapping.

## Deployment

**Important:** Netlify doesn't natively support Streamlit apps (Streamlit requires a persistent Python server). 

### Recommended: Streamlit Cloud (Free & Easy)
1. Push your code to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" and select this repository
5. Set main file path: `co-po-burt/app.py`
6. Deploy!

### Alternative: Heroku
1. Install Heroku CLI
2. Run: `heroku create your-app-name`
3. Run: `git push heroku main`

### Netlify (Limited Support)
Netlify is designed for static sites. For Streamlit, you'd need to use serverless functions or consider the alternatives above.




