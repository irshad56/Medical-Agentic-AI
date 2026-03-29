class ChatMemory:
    def __init__(self):
        self.messages = []
        self.metadata = {
            "last_analysis": None,
            "last_viz_path": None
        }

    def add_chat(self, role, content):
        self.messages.append({"role": role, "content": content})

    def update_metadata(self, key, value):
        self.metadata[key] = value

    def get_full_context(self):
        """Builds a string of the recent conversation for LLM context."""
        if not self.messages:
            return "No previous conversation history."

        history_lines = []
        for msg in self.messages[-6:]:  # Last 3 exchanges
            history_lines.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(history_lines)


# Global instance for the application
memory_store = ChatMemory()