"""BFCL (Berkeley Function Calling Leaderboard) evaluation against an OpenAI-compatible llama-server, scored with the
official BFCL AST checker (loaded from the gorilla repo with its Java/JS and model-registry imports stubbed out;
Python categories only).

Categories: simple_python, multiple, parallel, parallel_multiple, live_simple, live_multiple (AST check) and
irrelevance (correct = no function call)."""
import json, math, os, random, re, enum

AST_CATS = {'simple_python': 'simple', 'multiple': 'multiple', 'parallel': 'parallel',
            'parallel_multiple': 'parallel_multiple', 'live_simple': 'simple', 'live_multiple': 'multiple'}
ALL_CATS = list(AST_CATS) + ['irrelevance']


def load_checker(bfcl_root):
    """Exec the official ast_checker.py with its bfcl_eval imports replaced by stubs; returns ast_checker()."""
    src = open(os.path.join(bfcl_root, 'bfcl_eval/eval_checker/ast_eval/ast_checker.py')).read()
    src = re.sub(r'^from bfcl_eval[^\n]*\([^)]*\)\n', '', src, flags=re.M)
    src = re.sub(r'^from bfcl_eval[^\n]*\n', '', src, flags=re.M)
    assert 'from bfcl_eval' not in src, 'unexpected import layout in ast_checker.py'

    class Language(enum.Enum):
        PYTHON = 'python'; JAVA = 'java'; JAVASCRIPT = 'javascript'

    def _no(*a, **k):
        raise NotImplementedError('Java/JS categories are not evaluated here')
    ns = dict(Language=Language, MODEL_CONFIG_MAPPING={}, JAVA_TYPE_CONVERSION={}, JS_TYPE_CONVERSION={},
              java_type_converter=_no, js_type_converter=_no)
    exec(compile(src, 'ast_checker.py', 'exec'), ns)
    ns['convert_func_name'] = lambda name, model_name: name     # we map names back to the dotted originals ourselves
    return lambda funcs, output, answer, cat: ns['ast_checker'](funcs, output, answer, Language.PYTHON, AST_CATS[cat], 'local')


def load_category(bfcl_root, cat, n=None, seed=0):
    d = os.path.join(bfcl_root, 'bfcl_eval/data')
    rows = [json.loads(l) for l in open(os.path.join(d, f'BFCL_v4_{cat}.json')) if l.strip()]
    if cat in AST_CATS:
        ans = {r['id']: r['ground_truth'] for r in (json.loads(l) for l in open(os.path.join(d, 'possible_answer', f'BFCL_v4_{cat}.json')) if l.strip())}
        rows = [dict(r, answer=ans[r['id']]) for r in rows if r['id'] in ans]
    if n and len(rows) > n:
        rows = random.Random(seed).sample(rows, n)
    return rows


TYPE_MAP = {'dict': 'object', 'float': 'number', 'tuple': 'array', 'any': 'string', 'integer': 'integer',
            'string': 'string', 'boolean': 'boolean', 'array': 'array', 'object': 'object', 'number': 'number'}


def to_json_schema(p):
    """BFCL parameter schemas use Python-ish types ('dict', 'float', 'tuple', 'any'); convert to JSON Schema."""
    if not isinstance(p, dict):
        return p
    out = {}
    for k, v in p.items():
        if k == 'type' and isinstance(v, str):
            out[k] = TYPE_MAP.get(v, 'string')
        elif k == 'properties' and isinstance(v, dict):
            out[k] = {pk: to_json_schema(pv) for pk, pv in v.items()}
        elif k == 'items':
            out[k] = to_json_schema(v)
        else:
            out[k] = v
    return out


def safe_name(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:64]


def tools_for(row):
    """OpenAI tool list + map from the safe name sent to the model back to BFCL's original (often dotted) name."""
    tools, back = [], {}
    for f in row['function']:
        s = safe_name(f['name']); back[s] = f['name']
        tools.append({'type': 'function', 'function': {'name': s, 'description': f.get('description', ''),
                                                       'parameters': to_json_schema(f.get('parameters', {'type': 'dict', 'properties': {}}))}})
    return tools, back


def coerce(v):
    if isinstance(v, str):
        t = v.strip()
        try:
            return json.loads(t)
        except Exception:
            return v
    return v


def parse_calls(message, back):
    """Tool calls from a chat-completion message: the server's parsed tool_calls if present, else <tool_call> blocks in
    the text (JSON {"name", "arguments"} or Qwen XML <function=..><parameter=..>). Returns BFCL's output format:
    [{original_name: {param: value}}]."""
    calls = []
    for tc in (message.get('tool_calls') or []):
        fn = tc.get('function', {})
        args = fn.get('arguments', {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except Exception:
                args = {}
        calls.append((fn.get('name', ''), args))
    if not calls:
        text = message.get('content') or ''
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
        for blk in re.findall(r'<tool_call>(.*?)(?:</tool_call>|$)', text, flags=re.S):
            blk = blk.strip()
            m = re.search(r'<function=([^>\s]+)>(.*?)(?:</function>|$)', blk, flags=re.S)
            if m:
                params = {p: coerce(val.strip()) for p, val in re.findall(r'<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)', m.group(2), flags=re.S)}
                calls.append((m.group(1), params))
                continue
            try:
                j = json.loads(blk)
                calls.append((j.get('name', ''), coerce(j.get('arguments', j.get('parameters', {}))) or {}))
            except Exception:
                pass
    out = []
    for name, args in calls:
        if not isinstance(args, dict):
            args = {}
        out.append({back.get(name, name): dict(args)})            # server-parsed arguments are already typed JSON
    return out


def score(checker, row, cat, calls):
    if cat == 'irrelevance':
        return len(calls) == 0
    if not calls:
        return False
    try:
        return bool(checker(row['function'], calls, row['answer'], cat)['valid'])
    except Exception:
        return False


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant pairs (b: only A right, c: only B right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
