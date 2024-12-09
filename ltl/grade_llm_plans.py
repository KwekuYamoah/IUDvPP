import re
import json
from collections import defaultdict

def evaluate_and_save_results(ground_truth_file, generated_file, results_file, averages_file):
    """
    Evaluate generated task plans against ground truth and save evaluation results.

    Args:
        ground_truth_file (str): Path to JSON file containing ground truth task plans
        generated_file (str): Path to JSON file containing generated task plans
        results_file (str): Path to save detailed evaluation results
        averages_file (str): Path to save average scores
    """
    # Load the ground truth and generated task plans
    with open(ground_truth_file) as f:
        ground_truth = json.load(f)

    with open(generated_file) as f:
        generated_plans = json.load(f)

    # Function to normalize and grade a single task plan
    def grade_task_plan(gt_plan, gen_plan):
        """
        Grade a generated task plan against the ground truth plan with normalization.
        
        Args:
            gt_plan (list): Ground truth task plan steps
            gen_plan (list): Generated task plan steps to evaluate
            
        Returns:
            tuple: (number of correct steps, total number of steps in ground truth)
        """
        def normalize_step(step):
            return step.lower().strip()

        # Normalize both plans
        normalized_gt = [normalize_step(step) for step in gt_plan]
        normalized_gen = [normalize_step(step) for step in gen_plan]

        print(f"Normalized GT: {normalized_gt}")
        print(f"Normalized Gen: {normalized_gen}")

        # Count matches
        correct = sum(1 for step in normalized_gen if step in normalized_gt)
        total = len(normalized_gt)
        return correct, total

    # Helper to extract task plan
    def extract_task_plan(entry):
        if "TaskPlan" in entry:
            return entry["TaskPlan"]
        if "llm_output" in entry:
            # Remove empty lines and numbering from llm_output
            return [re.sub(r"^\d+\.\s*", "", line.strip()) for line in entry["llm_output"] if line.strip()]
        return []


    # Initialize data structures for results
    results = defaultdict(list)
    instruction_scores = defaultdict(list)
    global_correct = 0
    global_total = 0

    # Grade each interpretation and collect results
    for instruction_id, gt_data in ground_truth.items():
        if instruction_id not in generated_plans:
            continue
        for gt_interpretation in gt_data["Interpretations"]:
            gt_interpretation_id = gt_interpretation["interpretation_id"]
            gt_task_plan = gt_interpretation["TaskPlan"]
            for gen_entry in generated_plans[instruction_id]:
                if gen_entry["interpretation_id"] == gt_interpretation_id:
                    gen_task_plan = extract_task_plan(gen_entry)
                    correct, total = grade_task_plan(gt_task_plan, gen_task_plan)
                    instruction_scores[instruction_id].append((correct, total))
                    global_correct += correct
                    global_total += total

                    # Append results for this interpretation
                    results[instruction_id].append({
                        "speaker": gen_entry.get("speaker", "unknown"),
                        "instruction": gen_entry.get("instruction", ""),
                        "interpretation_id": gen_entry["interpretation_id"],
                        "baseline_choice": gen_entry.get("baseline_choice", "unknown"),
                        "task_plan_grade": f"{correct}/{total}"
                    })

    # Calculate averages
    instruction_averages = {
        instruction_id: f"{sum(score[0] for score in scores)}/{sum(score[1] for score in scores)}"
        for instruction_id, scores in instruction_scores.items()
    }
    global_average = f"{global_correct}/{global_total}" if global_total else "0/0"

    # Save results and averages to JSON files
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    with open(averages_file, "w") as f:
        json.dump({
            "instruction_averages": instruction_averages,
            "global_average": global_average
        }, f, indent=4)

    print(f"Results saved to {results_file}")
    print(f"Averages saved to {averages_file}")

# Call the function with file paths
evaluate_and_save_results(
    ground_truth_file="../ltl/llm_generated_plans/test_instruction_task_plans.json",
    generated_file="../ltl/llm_generated_plans/prosody_vector_only_llm_output.json",
    results_file="../ltl/llm_generated_plans/evaluation_results/prosody_vector_only_llm_output_evaluation.json",
    averages_file="../ltl/llm_generated_plans/evaluation_results/prosody_vector_only_llm_output_per_instruction_results.json"
)
