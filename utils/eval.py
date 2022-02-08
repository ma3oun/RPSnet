__all__ = ["batch_accuracy"]


def batch_accuracy(output, target) -> float:
    """Batch accuracy in percent"""

    batch_size = target.size(0)
    predictedLabel = output.argmax(dim=1, keepdim=True)
    correct = predictedLabel.eq(target.view_as(predictedLabel)).sum()
    accuracy = 100.0 * correct / batch_size

    return accuracy
