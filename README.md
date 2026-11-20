# Strict-unary two-block causal Transformer

A small decoder-only Transformer for a structured next-token task, not a general-purpose language model. Give the model scrambled pointer-edge tokens, payload-label tokens and a final request token. It freely generates an answer glyph followed by EOS. All fixtures here are generated synthetic examples, not external text or personal data.

```sh
python3 -B fit.py
python3 -B test.py
python3 -B test.py | cmp result.json -
sha256sum -c SHA256SUMS
```

**Model.** Two causal Q/K/V/output-projection hard-attention blocks, each followed by a trained two-projection ReLU feed-forward residual update. Positive-only hardmax uses a fixed null option, no softmax. Each query reads a strictly earlier position; global attention is not replaced by a per-example action plan. Learned grouped scalar weights define the token embedding gain, each block's projections, a value-channel remapping, and answer/EOS readouts. Raw embeddings have user-designed pointer/label/query channels and frozen symbolic vocabulary features. The weights and arithmetic values (including attention scores, hidden coordinates, logits and objective sums) are literal signed unary mark sequences, using append/cancel and repeated addition. Host-side loops, dictionary storage, boolean control and fixture generation are ordinary Python; this is not binary-packed model arithmetic.

**Fitting.** Six short prompts, each with two trainable next-token targets (reply glyph and EOS). Prompt tokens are masked from training loss. Exact discrete 0–1 token loss is a tally of mistakes; no float, gradients or target intermediate routing. The zero initializer misses all 12 labels, the joint final checkpoint misses none. Most weights are nonnegative, tied zero-or-unit gains. A mandatory projection/gain at zero disconnects the sole answer path or EOS path, so such hypotheses can be ruled out analytically. For surviving hypotheses the solver searches both blocks and the value palette together; two full candidates were evaluated prior to the first exact fit. Signed Q/K choices have paired sign symmetry and should not be interpreted as independently identified. The value mapping is learned from labels (identity, swap, or both), but the role of each layer and the allowed sparse connections are designed. This is not unrestricted, fully learned end-to-end Transformer architecture search or broad corpus pretraining.

**Test.** Four equal 96-row cohorts: context lengths 9/17/65 with novel first- to second-hop key combinations, and 129 with observed link types but novel arrangements and values. Longer contexts contain filler; they do not have 129 informative links. Tests check exact free generation including learned EOS, teacher-forced token accuracy, counterfactual pointer and label modifications, and causal future-suffix isolation. Combined index is 100 times the equally weighted free-generation exact rate over the four disjoint cohorts. Disabling attention block 1, attention block 2, or either feed-forward block uses the same generator and exact checkpoint, but removes the tested component. Identity value mapping and a constant-answer model are additional controls. No inference speedup or general-language performance is claimed.

**Research direction:** credited to [@Marxist-Leninist](https://github.com/Marxist-Leninist), who specified that the model weights and arithmetic themselves should be unary, not just the input text. This is a CPU research prototype and uses a standalone Python implementation, not the Hugging Face Transformers loading API.
