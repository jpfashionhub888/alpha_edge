class VetoAgent:
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY', '')