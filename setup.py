from setuptools import setup, find_packages

setup(
    name="llm_eval_audit",
    version="0.1.0",
    description="LLM Evaluation & Contamination Audit Framework",
    packages=find_packages(include=["llm_eval_audit", "llm_eval_audit.*"]),
    python_requires=">=3.10",
)
