"""Конвертация markdown от LLM в HTML, который понимает Telegram."""
import html
import re


def clean_think(text: str) -> str:
    """Убирает блоки <think>…</think> у reasoning-моделей (в том числе незакрытые)."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
    i = text.find("<think>")
    return text[:i] if i != -1 else text


def close_fences(text: str) -> tuple[str, str]:
    """Если в куске открыт блок кода — закрывает его и возвращает префикс для следующего куска."""
    if text.count("```") % 2 == 1:
        return text.rstrip() + "\n```", "```\n"
    return text, ""


def pick_cut(text: str, offset: int, chunk: int) -> int:
    """Где аккуратно разрезать длинный ответ на сообщения."""
    limit = offset + chunk
    for sep in ("\n\n", "\n", " "):
        i = text.rfind(sep, offset + chunk // 2, limit)
        if i != -1:
            return i + len(sep)
    return limit


def md_to_html(text: str) -> str:
    blocks: list[str] = []

    def stash(s: str) -> str:
        blocks.append(s)
        return f"\x00{len(blocks) - 1}\x00"

    def fence(m: re.Match) -> str:
        lang = m.group(1) or ""
        code = html.escape(m.group(2).strip("\n"))
        cls = f' class="language-{html.escape(lang)}"' if lang else ""
        return stash(f"<pre><code{cls}>{code}</code></pre>")

    # блоки кода и инлайн-код прячем, чтобы разметка внутри них не ломалась
    text = re.sub(r"```([\w+#.-]*)[ \t]*\n?(.*?)```", fence, text, flags=re.S)
    text = re.sub(r"`([^`\n]+)`", lambda m: stash(f"<code>{html.escape(m.group(1))}</code>"), text)

    # markdown-таблицы Telegram не умеет — показываем моноширинным блоком
    def table(m: re.Match) -> str:
        return stash(f"<pre>{html.escape(m.group(0).strip(chr(10)))}</pre>") + "\n"

    text = re.sub(r"(?:^[ \t]*\|.*\|[ \t]*(?:\n|$))+", table, text, flags=re.M)

    text = html.escape(text, quote=False)

    text = re.sub(r"^[ \t]*[-*+][ \t]+", "• ", text, flags=re.M)
    text = re.sub(r"^#{1,6}[ \t]+(.+)$", r"<b>\1</b>", text, flags=re.M)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.S)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    text = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"^&gt;[ \t]?(.*)$", r"<blockquote>\1</blockquote>", text, flags=re.M)
    text = text.replace("</blockquote>\n<blockquote>", "\n")
    text = re.sub(r"^-{3,}$", "──────────", text, flags=re.M)

    return re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], text)
