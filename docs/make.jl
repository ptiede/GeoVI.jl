using GeoVI
using Documenter

DocMeta.setdocmeta!(GeoVI, :DocTestSetup, :(using GeoVI); recursive=true)

makedocs(;
    modules=[GeoVI],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    sitename="GeoVI.jl",
    format=Documenter.HTML(;
        canonical="https://ptiede.github.io/GeoVI.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/GeoVI.jl",
    devbranch="main",
)
