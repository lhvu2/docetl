from docetl.api import Pipeline, Dataset, MapOp, UnnestOp, ResolveOp, ReduceOp, PipelineStep, PipelineOutput

from os.path import join
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
print(current_dir)

# Define the dataset - JSON file with medical transcripts
dataset = Dataset(
    type="file",
    path=join(current_dir, "medical_transcripts_v2.json")
)

# Define operations
operations = [
    # Extract medications from each transcript
    MapOp(
        name="extract_medications",
        type="map",
        prompt="""
        Note that, focus only on the female patients from East coast.

        Analyze the following transcript of a conversation between a doctor and a patient:
        {{ input.src }}
        Extract and list all medications mentioned in the transcript.
        If no medications are mentioned, return an empty list.
        """,
        output={
            "schema": {
                "medication": "list[str]"
            }
        },
        validate=None,
        limit=None,
        num_calibration_docs=10
    ),
    # Unnest to create separate items for each medication
    UnnestOp(
        name="unnest_medications",
        type="unnest",
        unnest_key="medication"
    ),
    # Resolve similar medication names
    ResolveOp(
        name="resolve_medications",
        type="resolve",
        blocking_keys=["medication"],
        blocking_threshold=0.6162,
        comparison_prompt="""
        Compare the following two medication entries:
        Entry 1: {{ input1.medication }}
        Entry 2: {{ input2.medication }}
        Are these medications likely to be the same or closely related?
        """,
        #embedding_model="text-embedding-3-small",
        #embedding_model="all-MiniLM-L6-v2",
        embedding_model="all-mpnet-base-v2",
        #embedding_model="BAAI/bge-base-en-v1.5",
        #embedding_model="intfloat/multilingual-e5-large-instruct",
        #embedding_model="models/e5-large-v2",
        output={
            "schema": {
                "medication": "str"
            }
        },
        resolution_prompt="""
        Given the following matched medication entries:
        {% for entry in inputs %}
        Entry {{ loop.index }}: {{ entry.medication }}
        {% endfor %}
        Determine the best resolved medication name for this group of entries. The resolved
        name should be a standardized, widely recognized medication name that best represents
        all matched entries.
        """,
        blocking_target_recall=None,
        embedding_batch_size=None,
        compare_batch_size=None,
        limit_comparisons=None,
        timeout=None
    )
    ,
    # Summarize side effects and uses for each medication
    ReduceOp(
        name="summarize_prescriptions",
        type="reduce",
        reduce_key=["medication"],
        prompt="""
        Here are some transcripts of conversations between a doctor and a patient:

        {% for value in inputs %}
        Transcript {{ loop.index }}:
        {{ value.src }}
        {% endfor %}

        For the medication {{ reduce_key }}, please provide the following information based on all the transcripts above:

        1. Side Effects: Summarize all mentioned side effects of {{ reduce_key }}.
        2. Therapeutic Uses: Explain the medical conditions or symptoms for which {{ reduce_key }} was prescribed or recommended.
        3. The number of the patients.

        Ensure your summary:
        - Is based solely on information from the provided transcripts
        - Focuses only on {{ reduce_key }}, not other medications
        - Includes relevant details from all transcripts
        - Is clear and concise
        - Includes quotes from the transcripts
        """,
        output={
            "schema": {
                "side_effects": "str",
                "uses": "str"
            }
        },
        fold_batch_size=None,
        merge_batch_size=None,
        limit=None
    )
]

# Define pipeline step
step = PipelineStep(
    name="medical_info_extraction",
    input="transcripts",
    operations=[
        "extract_medications",
        "unnest_medications",
        "resolve_medications",
        "summarize_prescriptions"
    ]
)

# Define output
output = PipelineOutput(
    type="file",
    path=join(current_dir, "medication_summaries_v2.json"),
    intermediate_dir=join(current_dir, "intermediate_results_v2")
)

# Define system prompt (optional but recommended)
system_prompt = {
    "dataset_description": "a collection of transcripts of doctor visits",
    "persona": "a medical practitioner analyzing patient symptoms and reactions to medications"
}

# Create the pipeline
pipeline = Pipeline(
    name="medical_transcript_analysis",
    datasets={"transcripts": dataset},
    operations=operations,
    steps=[step],
    output=output,
    default_model="ibm/openai/aws/claude-haiku-4-5",
    system_prompt=system_prompt
)

# Run the pipeline
cost = pipeline.run()
print(f"Pipeline execution completed. Total cost: ${cost:.2f}")

pass