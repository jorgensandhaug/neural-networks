import requests
import os
import openai

def generate_readme():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    tree =  os.popen("tree -I '__pycache__'")
    print(tree.read())

    # Get all folders in the current directory except the pycache folder and . files
    folders = [f for f in os.listdir(".") if os.path.isdir(f) and not f.startswith(".") and not f == "__pycache__"]


    # Add the current directory to the list of folders
    folders.append(".")

    print(folders)
    

    # Get all python files in the folders
    outer_summaries = []
    for folder in folders:
        print("Now in folder: ", folder)
        files = [f"{folder}/{f}" for f in os.listdir(folder) if f.endswith(".py")]
        summaries = []
        # Loop through all the files, but only get the first 5
        for file in files[:3]:
            print("Now in file: ", file)
            # Send a request to the GPT-3 API to summarize the code in the file
            prompt = f"This is a very short summary of the {file} python code, that is part of the Neural Network repository, inside the folder {folder}. The summary contains only text, and no code." + open(file).read() + "\nThe following summarizes the code in one sentences:"
            response = openai.Completion.create(
                engine="code-cushman-001",
                # Get text from the file
                prompt=prompt,
                # Return the summary
                temperature=0.5,
                max_tokens=80,
                frequency_penalty=0,
                presence_penalty=0,
                n= 1,
            )
            # Get the summary from the response
            summary = response["choices"][0]["text"]
            # Add a header to the summary
            summary = f"## {file}\n{summary}"
            # Add the summary to the list of summaries
            summaries.append(summary)
            print("Summary: ", summary)
        # Join all the summaries together
        readme = "\n".join(summaries)

        prompt = f"Given the folder: {folder}\n\n" + readme + "\n\nThe following summarizes the content:"
        # Then ask GPT-3 to generate a summary of the summaries
        response = openai.Completion.create(
            engine="davinci",
            prompt=readme,
            temperature=0.5,
            max_tokens=500,
            frequency_penalty=0,
            presence_penalty=0,
            n= 1,
        )

        # Get the summary from the response
        summary = response["choices"][0]["text"]
        # Add a header to the summary
        summary = f"## {folder}\n{summary}"
        # Add the summary to the list of summaries
        outer_summaries.append(summary)

        print("Summary folder: ", summary)

    # Join all the summaries together
    readme = "\n".join(outer_summaries)

    # Now get the tree structure of the current directory excluding the pycache folder and . files using the tree command
    tree =  os.popen("tree -I '__pycache__'").read()

    print("Tree", tree)
    # Ask the GPT-3 API to generate a README.md file from all of the summaries
    # First add a prompt to the README.md file to tell GPT-3 what to do
    prompt = "The following is a summary of the code in this repository.\n\n"
    prompt2 = "The code is organized in the following way:\n\n"+tree+"\n\n"
    prompt3 = "The following is a README.md file for this repository:"

    readme = prompt + readme + prompt2 + prompt3
    
    response = openai.Completion.create(
        engine="code-davinci-002",
        prompt=readme,
        temperature=0.9,
        # Set max_tokens to maximum possible
        max_tokens=2048,
        frequency_penalty=0,
        presence_penalty=0.6,   
        n= 1,
    )

    print(response.choices[0].text)
  
    # Parse the response and return the generated README.md file
    return response.choices[0].text


if __name__ == "__main__":
    # Generate the README.md file
    generated_readme = generate_readme()
    
    # Write the generated README.md file to disk
    with open("README.md", "w") as f:
        f.write(generated_readme)