#!/usr/bin/env python3
"""Generate src/generated/tables.rs from lm15-contract tables/providers.json.

The contract file is the reference's provider tables as data (its
tables/README.md): registry rows with their whole access policies, the
managed-login declared providers, the compat presets with their base-URL and
alias tables, the router's built-in rules and litellm prefixes, and the
managed-login service labels. This crate reads them from the generated module
as `const` data; nothing here re-derives a value (playbooks/port.md rule 2).

The crate's own types decide the Rust spelling: the generator reads each
enum's `as_str` arms (and the `vocab!` macro bodies) for the string → variant
map, and each compat struct's fields for their types, from this checkout's
source. A table value the crate cannot express (a new enum word, a new knob)
stops the generator with a message instead of guessing. Output is formatted by
`rustfmt` (on PATH; the CI toolchain's).

    python3 tools/gen_tables.py [--contract ../lm15-contract] [--check]

`--check` writes nothing and exits 1 when the committed file is stale (CI runs
it against the contract checkout at CONTRACT_PIN). Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "generated" / "tables.rs"
SCHEMA = 1


class Crate:
    """What the crate's source says about its own types."""

    def __init__(self, root: Path) -> None:
        sources = {p: p.read_text(encoding="utf-8") for p in (root / "src").rglob("*.rs")
                   if "generated" not in p.parts}
        # word -> every variant some match arm maps to it; more than one is a
        # conflict (an arm outside `as_str`), never resolved by guessing.
        seen: dict[str, dict[str, set[str]]] = {}
        for text in sources.values():
            for enum, variant, word in re.findall(r"\b([A-Z]\w*)::([A-Z]\w*)\s*=>\s*\"([^\"]*)\"", text):
                seen.setdefault(enum, {}).setdefault(word, set()).add(variant)
            for word, enum, variant in re.findall(r"\"([^\"]*)\"\s*=>\s*(?:Some\(|Ok\()?([A-Z]\w*)::([A-Z]\w*)\b", text):  # parse arms
                seen.setdefault(enum, {}).setdefault(word, set()).add(variant)
            for enum, body in re.findall(r"\n\s*([A-Z]\w*),\s*\"[^\"]*\",\s*\{([^}]*)\}", text):  # vocab! { V => "s" }
                for variant, word in re.findall(r"([A-Z]\w*)\s*=>\s*\"([^\"]*)\"", body):
                    seen.setdefault(enum, {}).setdefault(word, set()).add(variant)
        self.enums = {enum: {w: next(iter(v)) for w, v in words.items() if len(v) == 1} for enum, words in seen.items()}
        self.conflicts = {enum: {w: sorted(v) for w, v in words.items() if len(v) > 1} for enum, words in seen.items()}
        self.aliases = dict(re.findall(r"pub type (\w+) = (\w+);", "\n".join(sources.values())))
        # Declared variants, for enums whose source maps no string to them.
        self.variants: dict[str, set[str]] = {}
        for text in sources.values():
            for enum, body in re.findall(r"pub enum (\w+)(?:<[^>]*>)? \{\n(.*?)\n\}", text, re.S):
                self.variants[enum] = set(re.findall(r"^\s*([A-Z]\w*)\s*[,(]", body, re.M))
        self.fields: dict[str, dict[str, str]] = {}
        for text in sources.values():
            for name, body in re.findall(r"pub struct (\w+) \{\n(.*?)\n\}", text, re.S):
                self.fields[name] = dict(re.findall(r"^\s*pub (\w+): (.+?),\s*(?://.*)?$", body, re.M))

    def variant(self, enum: str, word: str) -> str:
        enum = self.aliases.get(enum, enum)
        if word in self.conflicts.get(enum, {}):
            raise SystemExit(f"{enum}: match arms map {word!r} to {self.conflicts[enum][word]}; cannot tell which is the wire word")
        table = self.enums.get(enum)
        if table is not None and word in table:
            return f"{enum}::{table[word]}"
        # No arm maps this word: the variant is the word in PascalCase, if
        # the enum declares it (the crate's naming rule for wire words).
        guess = "".join(part[:1].upper() + part[1:] for part in re.split(r"[_\-]", word))
        if table is None and guess in self.variants.get(enum, set()):
            return f"{enum}::{guess}"
        raise SystemExit(f"{enum} has no variant for {word!r} in this crate (add it, then regenerate)")

    def field_type(self, struct: str, field: str) -> str:
        try:
            return self.fields[struct][field]
        except KeyError:
            raise SystemExit(f"{struct} has no field {field!r} in this crate (add it, then regenerate)") from None


def s(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # a valid Rust string literal for this table's text


def opt(value: str | None) -> str:
    return "None" if value is None else f"Some({s(value)})"


def strs(values: list[str]) -> str:
    return "&[" + ", ".join(s(v) for v in values) + "]"


def pairs(values) -> str:
    items = values.items() if isinstance(values, dict) else values
    return "&[" + ", ".join(f"({s(a)}, {s(b)})" for a, b in items) + "]"


def screaming(provider: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", provider).upper()


class Emitter:
    def __init__(self, crate: Crate) -> None:
        self.c = crate

    def setting(self, x: dict) -> str:
        return f"HostSetting {{ name: {s(x['name'])}, env: {strs(x['env'])}, default: {opt(x['default'])} }}"

    def supports(self, x: dict) -> str:
        if x["extra"]:
            raise SystemExit(f"EndpointSupport.extra {x['extra']!r}: this crate's EndpointSupport has no extra field")
        return "EndpointSupport { " + ", ".join(f"{k}: {'true' if v else 'false'}" for k, v in x.items() if k != "extra") + " }"

    def host(self, h: dict | None) -> str:
        if h is None:
            return "None"
        version = h["anthropic_version_in"]
        if version == "header":
            version_lit = "AnthropicVersionIn::Header"
        elif version.startswith("body:"):
            version_lit = f"AnthropicVersionIn::Body({s(version[5:])})"
        else:
            raise SystemExit(f"anthropic_version_in {version!r}: not header or body:<value>")
        return ("Some(HostSpec { "
                f"base_url: {s(h['base_url'])}, endpoint_env: {strs(h['endpoint_env'])}, "
                f"settings: &[{', '.join(self.setting(x) for x in h['settings'])}], paths: {pairs(h['paths'])}, "
                f"model_in: {self.c.variant('ModelPlacement', h['model_in'])}, anthropic_version_in: {version_lit}, "
                f"stream_framing: {self.c.variant('StreamFraming', h['stream_framing'])}, "
                f"required_headers: {pairs(h['required_headers'])}, sigv4_service: {opt(h['sigv4_service'])} }})")

    def access(self, a: dict, placeholder_key: str | None) -> str:
        return ("AccessPolicy { "
                f"provider: {s(a['provider'])}, supports: {self.supports(a['supports'])}, "
                f"credential_policy: {self.c.variant('CredentialPolicy', a['credential_policy'])}, "
                f"auth_modes: {strs(a['auth_modes'])}, enterprise_variants: {strs(a['enterprise_variants'])}, "
                f"env_keys: {strs(a['env_keys'])}, "
                f"auth_scheme: &[{', '.join(self.c.variant('AuthScheme', x) for x in a['auth_scheme'])}], "
                f"headers: {pairs(a['headers'])}, host: {self.host(a['host'])}, login_hint: {opt(a['login_hint'])}, "
                f"backend: {s(a['backend'])}, backend_options: {pairs(a['backend_options'])}, "
                f"system_prefix: {opt(a['system_prefix'])}, base_url: {opt(a['base_url'])}, "
                f"placeholder_key: {opt(placeholder_key)}, "
                f"backend_settings: &[{', '.join(self.setting(x) for x in a['backend_settings'])}] }}")

    def knob(self, struct: str, field: str, value) -> str:
        type_ = self.c.field_type(struct, field)
        m = re.fullmatch(r"Option<Knob<(\w+)>>", type_)
        if m:
            return "Some(Knob::Auto)" if value == "auto" else f"Some(Knob::Set({self.c.variant(m.group(1), value)}))"
        if type_ == "Option<ReasoningEfforts>":
            return "Some(&[" + ", ".join(self.c.variant("ReasoningEffort", w) for w in value) + "])"
        if type_ == "Option<&'static [&'static str]>":
            return f"Some({strs(value)})"
        raise SystemExit(f"{struct}.{field}: type {type_} has no table form (extend tools/gen_tables.py)")

    def compat(self, struct: str, c: dict) -> str:
        fields = []
        for field, value in c.items():
            if field == "model_overrides":
                items = ", ".join(f"({s(prefix)}, {self.override(knobs)})" for prefix, knobs in value)
                fields.append(f"model_overrides: &[{items}]")
            else:
                fields.append(f"{field}: {self.knob(struct, field, value)}")
        return f"{struct} {{ " + "".join(f + ", " for f in fields) + f"..{struct}::EMPTY }}"

    def override(self, knobs: dict) -> str:
        fields = "".join(f"{k}: {self.knob('ChatModelOverride', k, v)}, " for k, v in knobs.items())
        return f"ChatModelOverride {{ {fields}..ChatModelOverride::NONE }}"

    def row(self, r: dict) -> str:
        kind = {"adapter-owned": "AdapterOwned", "bound": "Bound", "hosted": "Hosted"}[r["kind"]]
        compat = r["compat"]
        if compat is not None and not isinstance(compat, str):
            raise SystemExit(f"{r['id']}: a registry row with a compat object has no ProviderDefinition form")
        return ("ProviderDefinition { "
                f"id: {s(r['id'])}, dialect: {self.c.variant('DialectId', r['dialect'])}, kind: EntryKind::{kind}, "
                f"compat: {opt(compat)}, placeholder_key: {opt(r['placeholder_key'])}, note: {s(r['note'])} }}")


def render(tables: dict, digest: str, crate: Crate) -> str:
    if tables.get("schema") != SCHEMA:
        raise SystemExit(f"tables/providers.json schema {tables.get('schema')!r}; this generator reads {SCHEMA}")
    e, c = Emitter(crate), tables["compat"]
    out = [
        "// Generated by tools/gen_tables.py from lm15-contract tables/providers.json — do not edit.",
        f"// Contract tables sha256 {digest}. Regenerate: python3 tools/gen_tables.py",
        "// The receipts behind each value are cited at the reference's own table",
        "// (lm15-python lm15/registry.py, access.py, compat.py, router.py).",
        "",
        "use crate::auth::{",
        "    AccessPolicy, AnthropicVersionIn, AuthScheme, CredentialPolicy, EndpointSupport, HostSetting, HostSpec,",
        "    ModelPlacement, StreamFraming,",
        "};",
        "use crate::compat::*;",
        "use crate::registry::{DialectId, EntryKind, ProviderDefinition};",
        "use crate::router::RouteRule;",
        "use crate::types::ReasoningEffort;",
        "",
    ]
    declared_ids = {r["id"] for r in tables["declared_login"]}
    for r in tables["providers"] + tables["declared_login"]:
        out.append(f"/// `{r['id']}`{' (managed-login declared provider; not in ACCESS_POLICIES)' if r['id'] in declared_ids else ''}.")
        out.append(f"pub const {screaming(r['id'])}: AccessPolicy = {e.access(r['access'], r['placeholder_key'])};")
        out.append("")
    out.append("/// Every registry provider's access policy, in registry order.")
    out.append("pub const ACCESS_POLICIES: &[AccessPolicy] = &[" + ", ".join(screaming(r["id"]) for r in tables["providers"]) + "];")
    out.append("")
    out.append("/// Registry rows in declaration (presentation) order.")
    out.append("pub const PROVIDERS: &[ProviderDefinition] = &[" + ", ".join(e.row(r) for r in tables["providers"]) + "];")
    out.append("")
    for r in tables["declared_login"]:
        struct = {"anthropic": "AnthropicCompat", "openai-chat": "OpenAIChatCompat"}.get(r["dialect"])
        if struct is None or not isinstance(r["compat"], dict):
            raise SystemExit(f"{r['id']}: a declared-login row needs a chat or anthropic compat object")
        out.append(f"/// `{r['id']}` declared-login compat: {r['note']}")
        out.append(f"pub const {screaming(r['id'])}_COMPAT: {struct} = {e.compat(struct, r['compat'])};")
        out.append("")
    for name, struct, key in (("OPENAI_CHAT_PRESETS", "OpenAIChatCompat", "chat"),
                              ("OPENAI_RESPONSES_PRESETS", "OpenAIResponsesCompat", "responses"),
                              ("ANTHROPIC_PRESETS", "AnthropicCompat", "anthropic")):
        out.append(f"pub const {name}: &[(&str, {struct})] = &[" + ", ".join(
            f"({s(k)}, {e.compat(struct, v)})" for k, v in c[key].items()) + "];")
        out.append(f"pub const {name.replace('PRESETS', 'PRESET_BASE_URLS')}: &[(&str, &str)] = {pairs(c[key + '_base_urls'])};")
        out.append("")
    deepinfra = [prefix for prefix, knobs in c["chat"].get("deepinfra", {}).get("model_overrides", [])
                 if knobs.get("forced_tool_choice") == "send"]
    out.append("/// The deepinfra preset's model overrides that send a forced tool choice, as one list")
    out.append("/// (the crate's public `compat::DEEPINFRA_FORCED_TOOL_CHOICE`).")
    out.append(f"pub const DEEPINFRA_FORCED_TOOL_CHOICE: &[&str] = {strs(deepinfra)};")
    out.append("")
    out.append("/// Preset spelling aliases, read after lowercasing and mapping `-`, `.` and spaces to `_`.")
    out.append(f"pub const PRESET_ALIASES: &[(&str, &str)] = {pairs(c['preset_aliases'])};")
    out.append("")
    out.append("/// The router's built-in prefix rules; first match wins.")
    out.append("pub const DEFAULT_RULES: &[RouteRule] = &[" + ", ".join(
        f"RouteRule {{ prefix: {s(r['prefix'])}, provider: {s(r['provider'])}, note: {s(r['note'])} }}"
        for r in tables["routing"]["default_rules"]) + "];")
    out.append("")
    out.append("/// litellm's `provider/` spellings for the OpenAI-SDK/litellm door.")
    out.append(f"pub const LITELLM_PROVIDER_PREFIXES: &[(&str, &str)] = {pairs(tables['routing']['litellm_prefixes'])};")
    out.append("")
    out.append("/// AUTH-12 service labels.")
    out.append(f"pub const SERVICE_LABELS: &[(&str, &str)] = {pairs(tables['login']['service_labels'])};")
    source = "\n".join(out) + "\n"
    proc = subprocess.run(["rustfmt", "--edition", "2021", "--emit", "stdout"], input=source,
                          capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"rustfmt rejected the generated source:\n{proc.stderr}")
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--contract", type=Path, default=ROOT.parent / "lm15-contract")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    source = (args.contract / "tables" / "providers.json").read_bytes()
    text = render(json.loads(source.decode("utf-8")), hashlib.sha256(source).hexdigest(), Crate(ROOT))
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else None
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is stale for this contract checkout: run python3 tools/gen_tables.py")
            return 1
        print(f"{OUT.relative_to(ROOT)}: current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
