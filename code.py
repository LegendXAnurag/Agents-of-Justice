from __future__ import annotations
import os
os.environ["GROQ_API_KEY"]="your api key"
import time
from typing import List, Dict
import groq

class SharedMemory:
    def __init__(self):
        """
        Initialize the SharedMemory instance.

        This constructor initializes an empty list to store messages.
        """
        self.messages: List[Dict[str, str]] = []

    def add_message(self, speaker: str, content: str):
        """
        Add a new message to the messages list.

        Args:
            speaker (str): The speaker of the message.
            content (str): The content of the message.
        """
        self.messages.append({"speaker": speaker, "content": content})

    def get_transcript(self) -> str:
        """
        Generate a transcript of all messages stored in shared memory.

        Returns:
            str: A string representation of all messages, formatted as "speaker: content".
        """
        return "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in self.messages])

class LegalAgent:
    def __init__(self, name: str, system_prompt: str, shared_memory: SharedMemory, 
                 model: str = "llama3-8b-8192"):
        """
        Initialize the LegalAgent instance.

        Args:
            name (str): The name of the agent.
            system_prompt (str): The system prompt for the agent.
            shared_memory (SharedMemory): The shared memory instance for storing messages.
            model (str): The model to be used by the agent. Default is "llama3-8b-8192".
        """
        self.name = name
        self.system_prompt = system_prompt.strip()
        self.shared_memory = shared_memory
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def _format_messages(self, user_input: str) -> List[Dict[str, str]]:
        """
        Format the messages for the agent's response.

        Args:
            user_input (str): The user's input message.

        Returns:
            List[Dict[str, str]]: A list of formatted messages.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Current trial transcript:\n{self.shared_memory.get_transcript()}\n\n{user_input}"}
        ]
        return messages

    def respond(self, user_input: str, **gen_kwargs) -> str:
        """
        Generate a response from the agent based on user input.

        Args:
            user_input (str): The user's input message.
            **gen_kwargs: Additional keyword arguments for the response generation.

        Returns:
            str: The agent's response.
        """
        messages = self._format_messages(user_input)
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                **gen_kwargs
            )
            response = completion.choices[0].message.content.strip()
            self.shared_memory.add_message(self.name, response)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

# System Prompts
JUDGE_SYSTEM = """You are Judge Eleanor Roberts. Maintain judicial decorum, ensure fair proceedings, 
and ultimately deliver a reasoned verdict. Focus on legal standards and evidence reliability."""

PROSECUTION_SYSTEM = """You are ADA Michael Torres. Present compelling arguments using available evidence. 
Challenge defense claims and protect public interest. Remain professional but assertive."""

DEFENSE_SYSTEM = """You are Defense Attorney Sarah Lin. Vigorously defend your client using legal precedents. 
Identify reasonable doubt and challenge prosecution's evidence chain."""

PLAINTIFF_SYSTEM = """You are the Plaintiff. Clearly articulate damages suffered and maintain consistency 
in your testimony. Avoid speculation while emphasizing factual impacts."""

DEFENDANT_SYSTEM = """You are the Defendant. Maintain presumption of innocence. Provide plausible explanations 
while avoiding self-incrimination. Stay consistent with documented facts."""

class TrialManagement:
    def __init__(self, case_background: str):
        """
        Initialize the TrialManagement instance.

        Args:
            case_background (str): The background information of the case.
        """
        self.case_background = case_background
        self.shared_memory = SharedMemory()
        
        # Initialize agents
        self.judge = LegalAgent("Judge", JUDGE_SYSTEM, self.shared_memory)
        self.prosecution = LegalAgent("Prosecution", PROSECUTION_SYSTEM, self.shared_memory)
        self.defense = LegalAgent("Defense", DEFENSE_SYSTEM, self.shared_memory)
        self.plaintiff = LegalAgent("Plaintiff", PLAINTIFF_SYSTEM, self.shared_memory)
        self.defendant = LegalAgent("Defendant", DEFENDANT_SYSTEM, self.shared_memory)

    def _print_phase_header(self, title: str):
        """
        Print a header for each phase of the trial.

        Args:
            title (str): The title of the phase.
        """
        print(f"\n{'='*50}")
        print(f"==== {title.upper()} PHASE ====")
        print(f"{'='*50}\n")

    def opening_statements_phase(self):
        """
        Conduct the opening statements phase of the trial.
        """
        self._print_phase_header("Opening Statements")
        
        # Judge's introductory remarks
        judge_intro = self.judge.respond(
            f"Begin trial for case: {self.case_background}. Provide opening instructions."
        )
        print(f"JUDGE: {judge_intro}\n")
        time.sleep(1)

        # Prosecution opening
        prosecution_open = self.prosecution.respond(
            "Present opening statement outlining the prosecution's case."
        )
        print(f"PROSECUTION: {prosecution_open}\n")
        time.sleep(1)

        # Defense opening
        defense_open = self.defense.respond(
            "Present opening statement responding to prosecution's claims."
        )
        print(f"DEFENSE: {defense_open}\n")
        time.sleep(1)

    def argumentation_phase(self):
        """
        Conduct the argumentation phase of the trial.
        """
        self._print_phase_header("Argumentation")
        
        # Plaintiff testimony
        plaintiff_stmt = self.plaintiff.respond("Provide detailed testimony about the alleged harm.")
        print(f"PLAINTIFF: {plaintiff_stmt}\n")
        time.sleep(1)

        # Prosecution arguments
        pros_args = self.prosecution.respond(
            "Strengthen plaintiff's testimony with legal arguments and evidence analysis."
        )
        print(f"PROSECUTION: {pros_args}\n")
        time.sleep(1)

        # Defense cross-examination
        defense_cross = self.defense.respond(
            "Challenge plaintiff's testimony and prosecution's arguments."
        )
        print(f"DEFENSE: {defense_cross}\n")
        time.sleep(1)

        # Defendant testimony
        defendant_stmt = self.defendant.respond("Present your version of events and defense.")
        print(f"DEFENDANT: {defendant_stmt}\n")
        time.sleep(1)

    def closing_statements_phase(self):
        """
        Conduct the closing statements phase of the trial.
        """
        self._print_phase_header("Closing Statements")
        
        # Prosecution closing
        pros_close = self.prosecution.respond(
            "Deliver compelling closing arguments summarizing the prosecution's case."
        )
        print(f"PROSECUTION: {pros_close}\n")
        time.sleep(1)

        # Defense closing
        defense_close = self.defense.respond(
            "Present final arguments emphasizing reasonable doubt and defense position."
        )
        print(f"DEFENSE: {defense_close}\n")
        time.sleep(1)

    def judges_ruling_phase(self):
        """
        Conduct the judge's ruling phase of the trial.
        """
        self._print_phase_header("Judgment")
        
        # Judge's deliberation
        judgment = self.judge.respond(
            "After considering all evidence and arguments, deliver final verdict with legal reasoning."
        )
        print(f"JUDGE: {judgment}\n")
        verdict = self.judge.respond("Based on your verdict just answer 0 for DENIED or 1 for Granted. Answer only 0 or 1")
        print("VERDICT:",verdict)
        return verdict

    def run_trial(self):
        """
        Run the entire trial process through all phases.

        Returns:
            str: The final verdict of the trial.
        """
        phases = [
            self.argumentation_phase,
            self.closing_statements_phase,
            self.judges_ruling_phase
        ]
        
        print("\n🚨 TRIAL COMMENCING 🚨\n")
        for phase in phases:
            output = phase()
        print("\n⚖️ TRIAL CONCLUDED ⚖️\n")
        return output

import pandas as pd

if __name__ == "__main__":
    # Read the CSV file containing case details
    data = pd.read_csv(r'cases.csv')
    output_file = 'outputs.csv'

    # Check if the output file exists, if not create it with the specified columns
    if not os.path.exists(output_file):
        pd.DataFrame(columns=["id", "verdict"]).to_csv(output_file, index=False)

    # Iterate over each row in the data
    for index, row in data.iterrows():
        case_details = row.text
        # Initialize a TrialManagement instance with the case details
        trial = TrialManagement(case_details)
        # Run the trial and get the verdict
        verdict = trial.run_trial()

        # Create a DataFrame row with the case ID and verdict
        df_row = pd.DataFrame([{
            "id": row.id,
            "verdict": verdict[0] #First character of the verdict string is the ouput, others contain newline characters
        }])
        # Append the row to the output CSV file
        df_row.to_csv(output_file, mode='a', header=False, index=False)
