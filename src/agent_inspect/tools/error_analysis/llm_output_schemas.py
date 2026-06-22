ERROR_SUMMARIZATION_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "error_summarization_output_schema",
        "schema": {
            "type": "object",
            "properties": {
                "error_type": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["error_type", "explanation"],
            "additionalProperties": False,
            "strict": True,
        },
    },
}


MAJORITY_VOTE_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "majority_vote_output_schema",
        "schema": {
            "type": "object",
            "properties": {"most_probable_error_type": {"type": "string"}},
            "required": ["most_probable_error_type"],
            "additionalProperties": False,
            "strict": True,
        },
    },
}


UNSUPERVISED_CLUSTERING_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "unsupervised_clustering_output_schema",
        "schema": {
            "type": "object",
            "properties": {
                "clusters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cluster_label": {"type": "string"},
                            "error_types": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "error_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["cluster_label", "error_types", "error_ids"],
                        "additionalProperties": False,
                        "strict": True,
                    },
                },
            },
            "required": ["clusters"],
            "additionalProperties": False,
            "strict": True,
        },
    },
}


SEMI_SUPERVISED_CLUSTERING_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "semi_supervised_clustering_output_schema",
        "schema": {
            "type": "object",
            "properties": {
                "clusters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cluster_label": {"type": "string"},
                            "error_types": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "error_ids": {"type": "array", "items": {"type": "string"}},
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "cluster_label",
                            "error_types",
                            "error_ids",
                            "rationale",
                        ],
                        "additionalProperties": False,
                        "strict": True,
                    },
                },
            },
            "required": ["clusters"],
            "additionalProperties": False,
            "strict": True,
        },
    },
}


TOOL_VALIDATION_CLASSIFICATION_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "tool_validation_classification_output_schema",
        "schema": {
            "type": "object",
            "properties": {
                "cluster_label": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["cluster_label", "rationale"],
            "additionalProperties": False,
            "strict": True,
        },
    },
}
