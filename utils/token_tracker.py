# token_tracker.py

class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0
        self.model_usage = {}

    def calculate_cost(self, usage, model):
        """
        Calculates the cost based on token usage and model pricing.
        Handles both embedding and chat completion usage formats.
        """
        # Pricing per 1M tokens in USD
        model_pricing = {
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'text-embedding-3-small': {'input': 0.02, 'output': 0},
            'text-embedding-3-large': {'input': 0.13, 'output': 0},
            'default': {'input': 30, 'output': 60}
        }
        pricing = model_pricing.get(model, model_pricing['default'])

        # Initialize input and output tokens
        if isinstance(usage, dict):
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
        elif isinstance(usage, int):
            input_tokens = usage
            output_tokens = 0
        else:
            raise ValueError("Invalid usage format")

        # Calculate cost (price is per 1M tokens, so divide by 1M)
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']
        total_cost = input_cost + output_cost

        return total_cost

    def update(self, usage, model):
        """
        Updates the token counts and calculates the cost based on usage.
        """
        if isinstance(usage, dict):
            total_tokens = usage.get('total_tokens', 0)
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
        elif isinstance(usage, int):
            total_tokens = usage
            input_tokens = usage
            output_tokens = 0
        else:
            raise ValueError("Invalid usage format")

        self.total_tokens += total_tokens
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        cost = self.calculate_cost(usage, model)
        self.cost += cost

        if model not in self.model_usage:
            self.model_usage[model] = {
                'total_tokens': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0
            }
        self.model_usage[model]['total_tokens'] += total_tokens
        self.model_usage[model]['input_tokens'] += input_tokens
        self.model_usage[model]['output_tokens'] += output_tokens
        self.model_usage[model]['cost'] += cost

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cost': self.cost,
            'model_usage': self.model_usage
        }