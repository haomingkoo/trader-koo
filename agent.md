# Kiro Agent Best Practices & Lessons Learned

## Critical Issue: 50-Line Limit on fsAppend

### The Problem

- `fsAppend` has a 50-line limit that causes errors when trying to append large blocks of text
- Error message: "Either the text arg was not provided or text content provided exceeded the write file limit of 50 lines"
- This is a known issue in Kiro (GitHub issue #798)

### The Solution: Use `strReplace` Instead

Instead of using `fsAppend` to add content, use `strReplace` to replace small sections with larger sections.

**Strategy:**

1. **Read the file** to find a small anchor point (a line or section that exists)
2. **Use strReplace** with:
   - `oldStr`: The small anchor text (can be just one line)
   - `newStr`: The anchor text PLUS all the new content you want to add
3. **No line limit** on strReplace - you can replace with hundreds of lines

**Example:**

```javascript
// Instead of this (FAILS with 50+ lines):
fsAppend({
  path: "tasks.md",
  text: "... 200 lines of content ...",
});

// Do this (WORKS):
strReplace({
  path: "tasks.md",
  oldStr: "- [ ] 2. Implement LLM output validation",
  newStr: `- [ ] 2. Implement LLM output validation
  - [ ] 2.1 First subtask
  - [ ] 2.2 Second subtask
  ... 200 more lines ...`,
});
```

**Key Points:**

1. **strReplace has NO line limit** - use it for large content additions
2. **Find a unique anchor** - the oldStr must exist exactly once in the file
3. **Include the anchor in newStr** - you're replacing it with itself + new content
4. **Work in chunks** - if file is very large, replace multiple sections sequentially
5. **Never use fsAppend for >50 lines** - it will always fail

---

## Best Practices for Kiro Agents

### 1. File Operations

**Reading Files:**

- Use `readCode` for code files - it intelligently handles large files with AST parsing
- Use `readFile` only when you need specific line ranges or non-code files
- Use `readMultipleFiles` to read several files at once (more efficient than multiple single reads)
- Always provide meaningful `explanation` parameters

**Writing Files:**

- For new files: Use `fsWrite` for initial creation (keep under 50 lines)
- For large files: Use `fsWrite` for first chunk, then `strReplace` to add more content
- For updates: Use `strReplace` or `editCode` (AST-based, preferred for code)
- Never use `fsAppend` for more than 50 lines

**Code Editing:**

- Prefer `editCode` over `strReplace` for code files (AST-based, handles indentation)
- Use `semanticRename` for renaming symbols (updates all references automatically)
- Use `smartRelocate` for moving/renaming files (updates imports automatically)

### 2. Security Best Practices

**Never expose secrets:**

- Don't log API keys, passwords, tokens, or credentials
- Use placeholder values in examples: `<your-api-key>`, `[REDACTED]`, `***`
- Sanitize any data structures before logging
- Check files for accidental secret exposure before committing

**Secret patterns to watch for:**

- API keys: `OPENAI_API_KEY`, `AWS_SECRET_ACCESS_KEY`, etc.
- Tokens: JWT tokens, session tokens, auth tokens
- Passwords: Any password fields or hashed passwords
- Connection strings: Database URLs with credentials

### 3. Task Execution

**When working on specs:**

- Read requirements.md, design.md, and tasks.md first
- Update task status to "in_progress" before starting
- Update task status to "completed" when done
- Mark optional tasks with `*` after checkbox: `- [ ]* Optional task`

**When delegating to subagents:**

- Use `invokeSubAgent` for complex tasks requiring multiple steps
- Provide clear, specific prompts with all necessary context
- Don't delegate simple tasks that you can do directly

### 4. Testing

**Always test your changes:**

- Use `getDiagnostics` to check for syntax/type errors (don't use bash for this)
- Run unit tests after code changes
- For property-based tests, use hypothesis with min 100 iterations
- Tag tests with feature name and property number

### 5. Error Handling

**When errors occur:**

- Read the full error message carefully
- Check file paths are correct (relative to workspace root)
- Verify tool parameters match the schema
- If a tool fails repeatedly, try a different approach
- Don't retry the same failing operation more than 2-3 times

**Common mistakes:**

- Using absolute paths instead of relative paths
- Forgetting to read files before trying to modify them
- Using `cd` command in bash (not supported - use `cwd` parameter instead)
- Trying to append >50 lines with fsAppend

### 6. Communication

**With users:**

- Be concise and direct
- Don't repeat yourself
- Provide actionable information
- Use minimal summaries (2-3 sentences max)
- Don't create unnecessary documentation files

**In code:**

- Write clear commit messages
- Add comments for complex logic
- Use type hints in Python
- Follow language-specific style guides (PEP 8 for Python)

### 7. Performance

**Optimize your workflow:**

- Read multiple files at once when possible
- Use parallel tool calls for independent operations
- Cache frequently accessed data
- Avoid unnecessary file reads

**Don't:**

- Read the same file multiple times
- Make sequential calls when parallel is possible
- Process files one at a time when batch processing is available

### 8. Spec Workflow

**Creating specs:**

- Always ask user for workflow type first (requirements-first vs design-first)
- Never create spec files before user confirms workflow choice
- Use strReplace for large spec documents (not fsAppend)
- Include property-based tests for all correctness properties

**Updating specs:**

- Read .config.kiro to get workflow type
- Delegate to appropriate subagent based on workflow
- Don't re-ask user for choices already in config

### 9. Long-Running Commands

**Never use these in executeBash:**

- Development servers: `npm run dev`, `yarn start`
- Build watchers: `webpack --watch`, `jest --watch`
- Interactive commands: `vim`, `nano`

**Instead:**

- Use `controlBashProcess` with action="start" for long-running processes
- Or recommend user runs them manually in their terminal

### 10. Context Management

**Stay focused:**

- Only read files relevant to current task
- Use grepSearch to find specific code before reading entire files
- Use context-gatherer subagent for repository exploration
- Don't load unnecessary context

---

## Common Pitfalls to Avoid

1. **Token waste**: Don't get stuck in loops, don't repeat failed operations
2. **Over-explaining**: Keep summaries minimal, don't create unnecessary docs
3. **Ignoring errors**: Always read and understand error messages
4. **Wrong tools**: Use the right tool for the job (editCode for code, strReplace for large text)
5. **Missing context**: Read files before trying to modify them
6. **Security lapses**: Never expose secrets in logs, files, or responses
7. **Incomplete work**: Always complete tasks fully before marking as done
8. **Poor testing**: Test your changes before claiming completion

---

## Quick Reference

**File Operations:**

- New file: `fsWrite` (< 50 lines)
- Large file: `fsWrite` + `strReplace`
- Update code: `editCode` (preferred) or `strReplace`
- Read code: `readCode` (preferred) or `readFile`
- Move/rename: `smartRelocate`
- Rename symbol: `semanticRename`

**Process Management:**

- Long-running: `controlBashProcess` action="start"
- Stop process: `controlBashProcess` action="stop"
- Check output: `getProcessOutput`
- List processes: `listProcesses`

**Testing:**

- Check errors: `getDiagnostics`
- Run tests: `executeBash` with test command
- Never use watch mode in tests

**Search:**

- Find files: `fileSearch`
- Search content: `grepSearch`
- Explore repo: `invokeSubAgent` name="context-gatherer"
