General AI and GitHub Copilot coding guidelines for Python and for the Spinguin software package.
    
Spinguin is a software package for spin physics and spin dynamics simulations of nuclear magnetic resonance (NMR) in the liquid state, developed by Joni Eronen and Perttu Hilla from the NMR Research Unit at the University of Oulu. The software is written mostly in Python, containing some Cython and C code as well as example Jupyter notebooks. 

1. General Coding Guidelines

- Follow the existing coding style and structure of the codebase you are working on. This is the most important guideline. Always read the existing code carefully and understand its style before making any edits or additions.

- Do not create overly complicated functions. Always break down complex series of operations into smaller, well-defined functions. However, do not create functions that are too small and trivial. Do not create functions
that are only wrappers around a single Python or Python library functionality. 

- Use descriptive variable and function names that reflect their purpose and content. Names should be concise but informative. Use lowercase with underscores, except for conventional symbols used in mathematics or physics. Avoid vague names. The only exceptions are simple loop indices and one-letter variables commonly used in scientific literature.

- Every conceptually distinct operation in the code must be preceded by a one-line comment explaining its purpose. If the operation is complex, additional comments may be added to clarify specific steps, but the main comment should be a single line that summarizes the overall purpose of the operation. Never place comments on the same line as code.

- Every function definition must follow the same structure as existing functions in the codebase. This includes input and output data structures and a documentation header at the top of the function that describes the function’s purpose, its usage syntax, input parameters, and outputs. For function definitions that are too long to fit on a single line, place each argument on its own line so that the list of arguments is easy to read.

- For indentation, use tab, not spaces. 

- In actual code, include spaces around operators except for the operators *, / and @. In function definitions with optional arguments, do not include spaces around the equals sign (=) in the argument list. 

- Length of code lines should not exceed the standard limit of 80 characters. If a line of code, comment, or documentation exceeds this limit, it should be continued on the next line to maintain readability.

- Always use British spelling in variable names, function names, comments, and documentation unless a specific term or variable in the codebase you are working with already uses a different spelling. In such cases, follow the existing spelling for consistency. Prefer using of-genetive instead of s-genetive, for example "the function of the variable" instead of "the variable's function".

- Always make sure that the functions you are calling actually exist in the programming language or libraries you are using. Never call functions that do not exist. If you are unsure about the existence of a function, you should check the documentation or source code of the language or library to confirm its existence before using it in the code.

- Documentation is generated using Sphinx. When writing docstrings for functions, you must follow the Sphinx docstring format and guidelines. This includes using the appropriate sections (e.g., Parameters, Returns, Raises) and formatting for types and descriptions. Do not write the Returns section if nothing is returned, unless None is deliberately returned by the function or method. Do not write the Raises section for standard errors caused by basic type or function signature violations.

- Class docstrings describe what the class is and its overall behaviour. Do not document the public attributes. The __init__ docstring describes the parameters required to instantiate the class.

- Class properties should be documented only in the getters. Do not add documentation to the setters.

- No hallucinations, lies, or errors. You must not fabricate information, code, or documentation that is not accurate or supported by the codebase or user instructions. If you are unsure about something, refer to the existing code or ask the user for clarification. 
    
- Do not omit any part of the user instructions or stop generating output before all tasks are completed. Always verify that you have completed all tasks and that the output is of high quality before finishing. 

- After generating the output, you must evaluate its quality against the user instructions and these guidelines. If the output is incomplete, of low quality, or does not follow the guidelines, you must continue improving it until it meets the standards. You must also evaluate the quality of the code in terms of academic, software engineering, general Python coding, and numerical efficiency standards. If the code does not meet these standards, you must continue improving it until it does.

- Docstrings must always start with """ followed by a line break, then the actual docstring content, and end with a line break followed by """. 

- Do not make edits just for the sake of making edits. Only make edits that improve the code in terms of quality, readability, documentation, or structure. If there are no such improvements to be made, do not make any edits. 

- Do not use shared aliases for function input and output data types because this does not work well with Pylance function documentation window.

- Do not add error handling where it is not absolutely necessary or specifically requested by the user. Assume that the user of the code has a basic understanding of programming and does not try to use the code in an obviously incorrect way. If you are unsure about whether error handling is necessary in a specific case, you should ask the user for clarification.

- If you are asked to remove a piece of code, check the entire codebase to see if that piece of code is used anywhere else in addition to the file you are editing to avoid breaking dependencies. If the code to be removed is used elsewhere, you must ask the user for further instructions on how to proceed.

- When you are asked to go through a piece of code, for example using the "Tidy up" command, and update and/or improve it in terms of code quality, documentation, readability, commenting, or code structure, you are only allowed to edit the existing code and suggest improvements to it, which the user can then choose to accept or reject. You must not suggest to replace the entire piece of code with a new code that only contains the part that needs to be changed. Always go through the entire code and suggest improvements to all of it, including the parts that do not necessarily need to be changed, if this improves the overall quality and readability of the code. 

2. Specific commands for GitHub Copilot

- "Tidy up": Go through the code and tidy it up in terms of code structure and quality, readability, documentation, and commenting. The purpose is not to change the functionality of the code, but to improve its quality and documentation. In particular, focus on improving the docstrings and comments, making sure they are clear, informative, follow the Sphinx format and are written in good British English. Use scientific text as the general style. If the given code file starts with a general descriptive docstring, also improve that docstring, and if such a docstring is missing, add one. Make sure you go through the entire code, including all functions, classes, and any other code, and improve the quality of all of it when necessary. If the code is Jupyter notebook, also improve the markdown cells and their formatting. Apply these improvements to the code as suggested edits.