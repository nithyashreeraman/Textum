import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline


#model_path= "../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"
text_summ = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
#text_summ = pipeline("summarization", model=model_path, tokenizer=model_path, device=-1)


# text="Flowers, also known as blooms and blossoms, are the reproductive structures of flowering plants. Typically, they are structured in four circular levels around the end of a stalk. These include: sepals, which are modified leaves that support the flower; petals, often designed to attract pollinators; male parts, where pollen is presented; and female parts, where pollen is received and its movement is facilitated to the egg. When flowers are arranged in a group, they are known collectively as an inflorescence.The development of flowers is a complex and important part in the life cycles of flowering plants. In most plants, flowers are able to produce sex cells of both sexes. Pollen, which can produce the male sex cells, is transported between the male and female parts of flowers in pollination. Pollination can occur between different plants, as in cross-pollination, or between flowers on the same plant or even the same flower, as in self-pollination. Pollen movement may be caused by animals, such as birds and insects, or non-living things like wind and water. The colour and structure of flowers assist in the pollination process."

def summary (input):
    output = text_summ(input)
    return output[0]['summary_text']

gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text",outputs="text")
demo = gr.Interface(fn=summary,
                    inputs=[gr.Textbox(label="Input text to summarize",lines=6)],
                    outputs=[gr.Textbox(label="Summarized text",lines=4)],
                    title="Textum")

demo.launch()
