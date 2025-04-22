# SalesGPT run script.
import uvicorn
if __name__ == "__main__":
	# Start the Uvicorn server
	uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
