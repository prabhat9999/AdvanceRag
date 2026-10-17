from dotenv import load_dotenv
from graph.graph import app
load_dotenv()

if __name__ == "__main__":
   print("Advanced Rag")
   res=app.invoke( input={"question": "What is  pizza?"})
   print("Final Output: ",res["generation"])