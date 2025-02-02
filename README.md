to run api 1 is input -> detailed prompt -> summarised version of prompt to display use this 

uvicorn promptAPI.server1:app --host 0.0.0.0 --port 8001 --reload

to run api 2 is selected prompt -> content generation, use this

uvicorn contentAPI.server2:app --host 0.0.0.0 --port 8001 --reload


