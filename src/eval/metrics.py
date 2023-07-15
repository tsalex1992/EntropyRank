from tqdm import tqdm
import evaluate

import evaluate


def evaluate_metrics(
    eval_set: list[str],
    gt: list[list[str]],
    extraction_callback: callable,
    k_values: list[int],
    use_partial_match_values: list[bool],
) -> dict:
    """Evaluate the metrics for a given eval set and ground truth."""
    # extract the results
    results = []
    for eval_case in tqdm(eval_set):
        results.append(extraction_callback(eval_case))
    return_dict = {}
    # compute the metrics
    for k in k_values:
        for use_partial_match in use_partial_match_values:
            match_type = "_partial" if use_partial_match else ""
            precision, recall, f1, rouge = compute_avg_metrics_at_k(
                results, gt, k, use_partial_match=use_partial_match
            )
            return_dict[f"precision@{k}{match_type}"] = precision
            return_dict[f"recall@{k}{match_type}"] = recall
            return_dict[f"f1@{k}{match_type}"] = f1
            return_dict[f"rouge1@{k}"] = rouge
    return return_dict


def compute_avg_metrics_at_k(
    results: list[list[str]], ground_truth: list[list[str]], k: int, use_partial_match=False
) -> tuple[float, float, float, float]:
    """Compute the average precision, recall and f1 at k for a list of results and ground truth."""
    precision = compute_avg_precision_at_k(
        results, ground_truth, k, use_partial_match=use_partial_match
    )
    recall = compute_avg_recall_at_k(results, ground_truth, k, use_partial_match=use_partial_match)
    f1 = compute_avg_f1_at_k(results, ground_truth, k, use_partial_match=use_partial_match)
    rouge = compute_rouge1_at_k(results, ground_truth, k)
    return precision, recall, f1, rouge


def compute_rouge1_at_k(results: list[list[str]], ground_truth: list[list[str]], k: int) -> float:
    # for each result and ground truth take the first k elements and join them into a string with spaces
    rouge_metric = evaluate.load("rouge")
    results = [" ".join(result[:k]) for result in results]
    ground_truth = [" ".join(gt[:k]) for gt in ground_truth]
    return rouge_metric.compute(predictions=results, references=ground_truth)["rouge1"]


def compute_avg_precision_at_k(
    results: list[list[str]], ground_truth: list[list[str]], k: int, use_partial_match=False
) -> float:
    """Compute the average precision at k for a list of results and ground truth."""
    return sum(
        compute_precision_at_k(result, gt, k, use_partial_match=use_partial_match)
        for result, gt in zip(results, ground_truth)
    ) / len(results)


def compute_avg_recall_at_k(
    results: list[list[str]], ground_truth: list[list[str]], k: int, use_partial_match=False
) -> float:
    """Compute the average recall at k for a list of results and ground truth."""
    return sum(
        compute_recall_at_k(result, gt, k, use_partial_match=use_partial_match)
        for result, gt in zip(results, ground_truth)
    ) / len(results)


def compute_avg_f1_at_k(
    results: list[list[str]], ground_truth: list[list[str]], k: int, use_partial_match=False
) -> float:
    """Compute the average f1 at k for a list of results and ground truth."""
    return sum(
        compute_f1_at_k(result, gt, k, use_partial_match=use_partial_match)
        for result, gt in zip(results, ground_truth)
    ) / len(results)


def compute_precision_at_k(
    results: list[str], ground_truth: list[str], k: int, use_partial_match=False
) -> float:
    """Compute the precision at k for a list of results and ground truth."""
    # lower case all the results and ground truth
    results, ground_truth = _lc_gt_and_results(results, ground_truth)
    if not use_partial_match:
        return len(set(results[:k]).intersection(set(ground_truth))) / k

    # partial match
    return sum(_is_partial_match_of_any_result(result, ground_truth) for result in results[:k]) / k


def compute_recall_at_k(
    results: list[str], ground_truth: list[str], k: int, use_partial_match=False
) -> float:
    """Compute the recall at k for a list of results and ground truth."""
    # lower case all the results and ground truth
    results, ground_truth = _lc_gt_and_results(results, ground_truth)
    if not use_partial_match:
        return len(set(results[:k]).intersection(set(ground_truth))) / len(ground_truth)

    # partial match
    return sum(
        _is_partial_match_of_any_result(result, ground_truth) for result in results[:k]
    ) / len(ground_truth)


def compute_f1_at_k(
    results: list[str], ground_truth: list[str], k: int, use_partial_match=False
) -> float:
    """Compute the f1 at k for a list of results and ground truth."""
    precision = compute_precision_at_k(
        results, ground_truth, k, use_partial_match=use_partial_match
    )
    recall = compute_recall_at_k(results, ground_truth, k, use_partial_match=use_partial_match)

    if precision + recall != 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0


def compute_average_intersection_over_union_at_k(
    results: list[list[str]], ground_truth: list[list[str]], k: int
) -> float:
    """Compute the average intersection over union at k for a list of results and ground truth."""
    return sum(
        compute_intersection_over_union_at_k(result, gt, k)
        for result, gt in zip(results, ground_truth)
    ) / len(results)


def compute_intersection_over_union_at_k(
    results: list[str], ground_truth: list[str], k: int
) -> float:
    """Compute the intersection over union at k for a list of results and ground truth."""
    # lower case all the results and ground truth
    results, ground_truth = _lc_gt_and_results(results, ground_truth)
    return len(set(results[:k]).intersection(set(ground_truth))) / len(
        set(results[:k]).union(set(ground_truth))
    )


def _lc_gt_and_results(results: list[str], ground_truth: list[str]) -> tuple[list[str], list[str]]:
    """Lower case the ground truth and results."""
    return [result.lower() for result in results], [truth.lower() for truth in ground_truth]


def _is_partial_match(result: str, ground_truth: str) -> bool:
    """Check if the result is a partial match of the ground truth."""
    # lower case the result and ground truth
    result, ground_truth = result.lower(), ground_truth.lower()
    return result in ground_truth or ground_truth in result


def _is_partial_match_of_any_result(result: str, ground_truths: list[str]) -> bool:
    """Check if the result is a partial match of any of the ground truths."""
    return any(_is_partial_match(result, ground_truth) for ground_truth in ground_truths)
