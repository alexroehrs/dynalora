def doc_to_text(doc) -> str:
    """Prompt of the harness task `pubmedqa`, for either field naming of the dataset."""
    if 'CONTEXTS' in doc:
        ctxs, question = doc['CONTEXTS'], doc['QUESTION']
    else:
        ctx = doc['context']
        ctxs = ctx['contexts'] if isinstance(ctx, dict) else [c for d in ctx for c in d['contexts']]
        question = doc['question']
    return "Abstract: {}\nQuestion: {}\nAnswer:".format("\n".join(ctxs), question)
