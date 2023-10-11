import uvicorn
import dotenv

dotenv.load_dotenv()

if __name__ == "__main__":
    uvicorn.run("app.app:helios_app", host="0.0.0.0", port=8000, reload=True)