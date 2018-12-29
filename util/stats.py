import torch


class StatsTracker(object):
    """
    Container for tracking the statistics associated with an epoch. For each of
    a training and validation pass, a new StatsTracker should be instantiated.
    The common use pattern of the class looks as follows::

        for e in range(num_epochs):
            stats = StatsTracker()

            # Add some loss stats
            stats.update_loss(loss)

            # Add accuracy metrics
            stats.update_accuracies(decoded_output, labels, true_labels, mask)

            # Get current average stats
            a, b, c = stats.averages()
    """

    def __init__(self):
        self.loss = 0.

        # Number of correct samples from the view of reconstruction-accuracy.
        self.num_reconstruction_match = 0

        # Number of correct samples from the view of overall-accuracy.
        self.num_overall_match = 0

        # Hold different counters for the number of loss and accuracy attempts.
        # Losses are added in the unit of the average for a minibatch, while
        # accuracy metrics are added for individual samples.
        self.num_loss_attempts = 0
        self.num_match_attempts = 0

    def averages(self):
        """
        Returns average loss, reconstruction-accuracy, and overall-accuracy
        since this ``StatsTracker`` was instantiated.
        """
        avg_loss = self.loss / self.num_loss_attempts
        avg_recon_acc = self.num_reconstruction_match / self.num_match_attempts
        avg_overall_acc = self.num_overall_match / self.num_match_attempts
        return avg_loss, avg_recon_acc, avg_overall_acc

    def update_accuracies(self, decoded, base_model_outputs, true_labels, mask):
        """
        Calculates the number of decoded outputs that match (1) the outputs
        from the base model and (2) the true labels associated with the decoded
        sample. These results are maintained for later aggregate statistics.
        """
        self.num_match_attempts += decoded.size(0)
        max_decoded = torch.max(decoded, dim=2)[1]
        max_outputs = torch.max(base_model_outputs, dim=2)[1]

        self.num_reconstruction_match += torch.sum(
            (max_decoded == max_outputs) * mask).item()
        self.num_overall_match += torch.sum(
            (max_decoded == true_labels) * mask).item()

    def update_loss(self, loss):
        """
        Adds ``loss`` to the current aggregate loss for this epoch.
        """
        self.loss += loss
        self.num_loss_attempts += 1
