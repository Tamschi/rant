<p align="center">
<img src="https://i.imgur.com/s9jMDhI.png" alt="Rant Logo" ></img>
</p>

[![Crates.io](https://img.shields.io/crates/v/rant)](https://crates.io/crates/rant)
[![Docs.rs](https://docs.rs/rant/badge.svg)](https://docs.rs/rant)
[![Discord](https://img.shields.io/discord/332251452334669834?color=6C8BD5&label=discord&logo=discord&logoColor=%23fff)](https://discord.gg/U8Bj6gSshJ)



**Rant** is a high-level procedural templating language with a diverse toolset for easily creating dynamic code templates, game dialogue, stories, name generators, test data, and much more.

***

> ## **Prerelease notice**
>
> **This project is in alpha.**
> **This means that the API is unstable, functionality may be broken/missing, and everything is subject to change.**
>
> **Please do try it out and give feedback; however, _do not_ use in production environments until a stable version is released.**
>
> **Features may appear/disappear at any time for any reason. Assume that every alpha release will have breaking changes.**


## Introducing: Rant 4

Rant 4 is a complete redesign and reimplementation of the Rant language, written in Rust. 
It's designed with usability in mind for everyone-- programmers, writers, and everyone in-between!


Rant is basically the opposite of [Regex](https://en.wikipedia.org/wiki/Regular_expression): 
just as a regular expression compares inputs to a pattern, Rant generates matching outputs from a pattern!

## Why Rant?

🧰 **Pain-Free API** <br/> 
Rant has a straightforward, no-nonsense API that is designed to be easy to use.
No getting lost in configuration hell. Integrating Rant into your project only takes a few lines of code.

💻 **Cross-Platform** <br/> 
Write once, run anywhere-- Rant code is fully portable across Windows, Mac, and Linux.

🎲 **Built with ♥ for RNG** <br/>
Rant is made with random generation in mind as a major use-case.

Enjoy a wide array of built-in utilities for generating random numbers, strings, booleans, lists, list subsets, and much more for all your randomization needs. The internal RNG can be manually seeded to produce repeatable outputs.

🎨 **It's a Scripting Language** <br/> 
Inspired by other popular scripting languages, Rant is a fully-fledged imperative programming language with a dynamic type system. Enjoy support for common primitives like strings, numbers, collections, closures, and more.

🖨 **It's a Templating Language** <br/> 
Rant is all about "printing": each lexical scope has an output to print (append values) to, which then prints itself to the parent output, and so on. 
This enables you to intuitively build strings, collections, and more in a familiar templating structure.

🔱 **Advanced Branching** <br/> 
Fine-tune branching behavior with a multitude of configuration options and branch selection modes for iterative, random, and weighted selection.

🧬 **Delightful Combinatorics** <br/> 
Perform nested mappings, filters, zips, combinations, and more with Rant's intuitive piping syntax.

🎛 **Automatic Formatting** <br/> 
Passively format text output with automatic capitalization, whitespace normalization, and number formatting.

🧩 **Simple Module System** <br/> 
Creating a Rant module and depending on it in another Rant program is as trivial as writing the script and importing it... that's it.

You can add your own package manager on top if you want, but it's not required.

📚 **Rant Standard Library** <br/> 
A comprehensive standard library provides the tools needed to rapidly iterate on your ideas.

## Getting started

### CLI

Rant's CLI can run Rant code from files or the command line.
Install it from Cargo with:

```sh
$ cargo install rant --version 4.0.0-alpha.23 --features cli
```

Then run it:

```sh
$ rant
```

### Library

Add Rant to your `Cargo.toml`:

```toml
[dependencies]
rant = "*"
```

You can run a Rant program with just a few lines of code:

```rust
use rant::Rant;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
  // Create a default Rant context
  let mut rant = Rant::new();

  // Compile a simple program
  let program = rant.compile_quiet(r#"
  [$greet:name] {
    {Hello|Hi|Hey} <name>!
  }
  [greet:world]
  "#)?;

  // Run the program and print the output
  let output = rant.run(&program)?;
  println!("{}", output);

  Ok(())
}
```

## [Examples](./examples/rant/)

This repository contains a directory of example programs written in Rant for you to learn from. Check them out!

## Documentation

The latest reference documentation can be found at **[docs.rant-lang.org](https://docs.rant-lang.org)**.

Since Rant 4 is early in development, some documentation may be outdated/incomplete, but it is actively updated to ensure that it reflects current features with reasonable accuracy.

## [Changelog](https://github.com/rant-lang/rant/blob/master/CHANGELOG.md)

The changelog is updated constantly throughout the development process, providing a complete summary of upcoming changes at a glance even before the next release.

## MSRV

The minimum supported Rust version is `1.57.0`.

## License

Rant is distributed under the GNU Affero General Public License v3. See [LICENSE](./LICENSE) for more details.

An alternative, commercial license is planned for the future.
