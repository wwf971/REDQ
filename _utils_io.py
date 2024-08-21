from __future__ import annotations

class _PipeOutIncreaseIndent:
    def __init__(self, Parent: PipeOut):
        self.Parent = Parent
    def __enter__(self):
        self.Parent.increase_indent()
    def __exit__(self, type, value, trace):
        self.Parent.decrease_indent()

from collections import defaultdict
class PipeOut:
    def __init__(self, pipe=None):
        if pipe is None:
            import sys
            pipe = sys.stdout
        self.pipe = pipe
        self.indent = 0
        self.print_every_counter = defaultdict(lambda: 0)
        return
    def increased_indent(self):
        return _PipeOutIncreaseIndent(self)
    def increase_indent(self):
        self.indent += 1
        return self
    def decrease_indent(self):
        if self.indent > 0:
            self.indent -= 1
        return self
    def print_with_increased_indent(self, *args, **kwargs):
        result = self.print(*args, indent=self.indent + 1, **kwargs)
        return result
    def print(self, *args, indent=None, **kwargs):
        if indent is None:
            indent = self.indent
        result = print_to_pipe(self.pipe, *args, indent=indent, **kwargs)
        return result
    def print_every(self, interval, *args, indent=None, **kwargs):
        if indent is None:
            indent = self.indent
        count = self.print_every_counter[interval]
        if count == 0:
            result = self.print(*args, indent=indent, **kwargs)
        else:
            result = None
        count += 1
        if count >= interval:
            count = 0
        self.print_every_counter[interval] = count
        return result
    def SetPrintCounterToZero(self):
        self.PrintCounterDict = defaultdict(lambda: 0)
    def write(self, Str: str):
        # self.PrintWithoutIndent(Str)
        self.Print(Str, end="")
    ResetPrintCounter = SetPrintCounterToZero

def print_to_pipe(pipe, *list, indent=None, **kwargs):
    _str = print_to_str(*list, **kwargs)
    if indent is None:
        indent = 0
    print_str_to_pipe(pipe, _str, indent=indent)

from io import StringIO
def print_to_str(*args, **kwargs):
    # faster than reusing.
    # https://stackoverflow.com/questions/4330812/how-do-i-clear-a-stringio-object
    print_buf = StringIO()
    print_buf.seek(0)
    print_buf.truncate(0)
    print(*args, **kwargs, file=print_buf)
    str_ = print_buf.getvalue()
    print_buf.flush()
    del print_buf
    return str_

def print_str_to_pipe(
        pipe,
        _str: str,
        indent=None,
        flush=True,
        indent_str="    " # "\t"
    ):
    if indent is None:
        indent = 0
    if indent > 0:
        if _str.endswith("\n"):
            IsEndWithNewLine = True
            _str = _str.rstrip("\n")
        else:
            IsEndWithNewLine = False
        str_list = _str.split("\n")
        for index, str_line in enumerate(str_list):
            str_list[index] = "".join([indent_str for _ in range(indent)] + [str_line])
        _str = "\n".join(str_list)
        if IsEndWithNewLine:
            _str = _str + "\n"
    if hasattr(pipe, "buffer"):
        pipe.buffer.write(_str.encode("utf-8"))
    else:
        pipe.write(_str)
    if flush and hasattr(pipe, "flush"):
        pipe.flush()

if __name__ == "__main__":
    pipe = PipeOut()
    pipe.print("aaa")
    with pipe.increased_indent():
        pipe.print("bbb")
    pipe.print("ccc")

    for index in range(100):
        pipe.print_every(10, "%d"%index)