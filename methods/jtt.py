from .standard import StandardMethod


class JTTMethod(StandardMethod):
    """
    Just Train Twice (JTT) Method.

    From a loss perspective, JTT is identical to Standard training (ERM)
    in both stages. It uses Binary Cross Entropy.

    We create this class to maintain consistency with the `methods` factory pattern.
    """

    def __init__(self, config):
        super().__init__(config)

    def compute_loss(self, model_output, targets, extra_info=None, weight=None):
        logits, _ = model_output
        bce_loss, wbce_loss = self.compute_bce_terms(logits, targets, weight=weight)

        # During training (requires_grad=True), JTT uses the weighted BCE
        # to mathematically simulate oversampling the error set.
        # During validation (requires_grad=False), it falls back to unweighted BCE
        # to ensure train/val curves are comparable.
        if weight is not None and logits.requires_grad:
            loss = wbce_loss
        else:
            loss = bce_loss

        return loss, {"bce": bce_loss.item(), "wbce": wbce_loss.item()}
