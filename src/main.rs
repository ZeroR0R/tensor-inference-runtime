use std::io::Write;

use clap::Parser;

use tensor_inference_runtime::{
    generate, load_model, print_metadata, print_tensors, reader::read_gguf, test_forward,
};

#[derive(Parser, Debug)]
struct Args {
    /// Path to a GGUF model, Llama family with Q8_0 weights only for now
    #[arg(short, long, default_value = "./tinyllama-1.1b-chat-v1.0.Q8_0.gguf")]
    model: String,

    /// Inspect loaded GGUF
    #[arg(short, long)]
    inspect: bool,

    /// Forward Test Token
    #[arg(short, long)]
    forward_test: bool,

    /// Prompt to complete
    #[arg(short, long, visible_alias = "generate")]
    prompt: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let map = read_gguf(&args.model)?;

    let (llama, metadata_info, tensor_info, tensor_data) = load_model(&map)?;

    if args.inspect {
        println!("Metadata:");
        print_metadata(metadata_info.clone());

        println!("Tensors:");
        print_tensors(tensor_info.clone());

        println!("Tensors data:");
        println!("{:?}", tensor_data.len());
    }

    if args.forward_test {
        let id = test_forward(&llama, &tensor_data);
        println!("{id:?}");
    }

    if let Some(prompt) = args.prompt.as_deref() {
        let embed_tensor = tensor_data
            .get("token_embd.weight")
            .ok_or("model has no token_embd.weight tensor")?;

        // Generate streams tokens as they land, so just echo the prompt first
        print!("{prompt}");
        std::io::stdout().flush()?;

        generate(prompt, &llama, &metadata_info, &tensor_data, embed_tensor)?;
        println!();
    }

    Ok(())
}
