import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, DPOTrainer
from langchain.agents import AgentExecutor, Tool
import tooluniverse as tu  # Tool management framework
import pyprimekg  # PrimeKG knowledge graph
import requests  # For API-based tools

# Configuration
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize model with QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# 2. ToolUniverse + PrimeKG Integration
class PrimeKGTool(tu.Tool):
    def __init__(self):
        super().__init__(
            name="PrimeKG_Query",
            description="Query PrimeKG knowledge graph for drug-disease relationships",
            params={"disease": "str", "drug": "str"}
        )
        self.kg = pyprimekg.load_primekg()  # Load knowledge graph

    def execute(self, disease: str, drug: str = None):
        """Check drug-disease relationships with evidence"""
        if drug:
            return self.kg.query(f"""
                MATCH (d:Disease {{name: '{disease}'}})<-[r:TREATS]-(m:Drug {{name: '{drug}'}})
                RETURN r.evidence, r.source, r.confidence
            """)
        else:
            return self.kg.query(f"""
                MATCH (d:Disease {{name: '{disease}'}})<-[r:TREATS]-(m:Drug)
                RETURN m.name, r.evidence, r.confidence
                ORDER BY r.confidence DESC LIMIT 5
            """)

class ClinicalDataTool(tu.Tool):
    def __init__(self):
        super().__init__(
            name="Clinical_Trial_Lookup",
            description="Fetch clinical trial results for specific drugs/diseases",
            params={"drug": "str", "disease": "str", "phase": "int"}
        )

    def execute(self, drug: str, disease: str, phase: int = 3):
        """Mock clinical trial data lookup - would connect to real API"""
        return {
            "drug": drug,
            "disease": disease,
            "phase": phase,
            "adverse_effects": {
                "headache": {"percentage": 15.2, "group": "treatment"},
                "nausea": {"percentage": 8.7, "group": "treatment"}
            }
        }

# 3. Initialize ToolUniverse
tool_manager = tu.ToolUniverse()
tool_manager.register_tool(PrimeKGTool())
tool_manager.register_tool(ClinicalDataTool())

# 4. Format dataset according to your structure
def format_sample(sample):
    """Convert dataset sample to agent-readable format"""
    return f"""
<|system|>
You are a therapeutic AI agent. Use tools to answer clinical questions.
Access to: {tool_manager.list_tools()}
</s>
<|user|>
Question: {sample['question']}
Options: {', '.join([f"{k}: {v}" for k,v in sample['options'].items()])}
</s>
<|assistant|>
Correct answer is {sample['correct_answer']}. Reasoning:
"""

# 5. Supervised Fine-Tuning with Tool Integration
def train_sft(model, dataset):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset.map(format_sample),
        max_seq_length=2048,
        dataset_text_field="text",
        peft_config=peft_config,
        args=TrainingArguments(
            output_dir="./sft_model",
            per_device_train_batch_size=2,  # Reduced for 8B model
            gradient_accumulation_steps=8,   # Compensate for smaller batch
            optim="paged_adamw_8bit",
            learning_rate=1e-5,              # Lower LR for larger model
            num_train_epochs=2,
            fp16=True,
            logging_steps=10
        )
    )
    trainer.train()
    return model

# 6. Therapeutic Reasoning Agent
class TherapeuticAgent:
    def __init__(self, model, tokenizer, tool_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.tool_manager = tool_manager
        
    def run(self, question: str, options: dict):
        # Generate tool-augmented prompt
        prompt = f"""
<|system|>
You are a clinical reasoning agent. Use available tools to answer the question.
Available tools: {self.tool_manager.list_tools()}
</s>
<|user|>
Question: {question}
Options: {options}
</s>
<|assistant|>
        """
        
        # Generate tool-using response
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Parse tool calls from response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tool_calls = self._parse_tool_calls(response)
        
        # Execute tools and generate final answer
        if tool_calls:
            tool_results = []
            for call in tool_calls:
                result = self.tool_manager.execute_tool(
                    call['name'], 
                    **call['params']
                )
                tool_results.append(f"Tool {call['name']} result: {str(result)}")
            
            # Augment prompt with tool results
            augmented_prompt = f"{prompt}{response}\nObservation: {'; '.join(tool_results)}\nFinal Answer:"
            inputs = tokenizer(augmented_prompt, return_tensors="pt").to(DEVICE)
            outputs = model.generate(**inputs, max_new_tokens=100)
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _parse_tool_calls(self, response: str):
        """Extract tool calls from model response (simplified)"""
        # In practice: Use more robust parsing (e.g., regex or JSON)
        tool_calls = []
        if "Action:" in response:
            parts = response.split("Action:")[1].split("\n")[0]
            if "PrimeKG_Query" in parts:
                # Extract disease/drug from response
                tool_calls.append({
                    "name": "PrimeKG_Query",
                    "params": {"disease": "headache"}  # Simplified
                })
            elif "Clinical_Trial_Lookup" in parts:
                tool_calls.append({
                    "name": "Clinical_Trial_Lookup",
                    "params": {"drug": "GOPRELTO", "disease": "headache"}
                })
        return tool_calls

# 7. DPO Training with Clinical Preferences
def train_dpo(model, preference_data):
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=TrainingArguments(
            output_dir="./dpo_model",
            per_device_train_batch_size=1,  # Small batch for 8B model
            gradient_accumulation_steps=16,
            learning_rate=5e-6,
            max_length=1024,
            gradient_checkpointing=True
        ),
        beta=0.1,
        train_dataset=preference_data,
    )
    dpo_trainer.train()
    return model

# Example usage
if __name__ == "__main__":
    # Sample from your dataset structure
    sample = {
        "id": "ep0KXYKj2lYJ",
        "question_type": "multi_choice",
        "question": "Which group had the highest percentage of patients reporting headache?",
        "correct_answer": "A",
        "options": {
            "A": "GOPRELTO 4% solution group",
            "B": "Cocaine Hydrochloride 8% solution group",
            "C": "Placebo group",
            "D": "None of the groups"
        }
    }
    
    # Initialize agent
    agent = TherapeuticAgent(model, tokenizer, tool_manager)
    
    # Run inference
    result = agent.run(
        question=sample["question"],
        options=sample["options"]
    )
    print("Agent Response:", result)