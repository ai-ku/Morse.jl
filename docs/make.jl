using Documenter, Morse

makedocs(

    modules = [Morse],
    clean = false,              # do we clean build dir
    format = :html,
    sitename = "Morse.jl",
    authors = "Ekin Akyürek, Erenay Dayanık, Deniz Yuret",
    doctest = true,
    pages = Any[ # Compat: `Any` for 0.4 compat
        "Home" => "index.md",
        "Function Documentation" => Any[
            "reference.md",
        ],
    ],

)

deploydocs(
    repo = "github.com/ekinakyurek/Morse.jl.git",
    julia = "1.0",
    osname = "linux",
    target = "build",
    make = nothing,
    deps = nothing,
    #deps   = Deps.pip("mkdocs", "python-markdown-math"),
)
