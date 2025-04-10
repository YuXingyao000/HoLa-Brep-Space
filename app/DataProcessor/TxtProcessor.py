from app.DataProcessor.DataProcessor import DataProcessor

class TextProcessor(DataProcessor):
    def process_input_data(self, text: str):
        return { "txt" : [text[0]] * self.NUM_PROPOSALS}