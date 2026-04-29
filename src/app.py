import gradio as gr
from rag import answer_question

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(label="Ερώτηση"),
    outputs=gr.Textbox(label="Απάντηση"),
    title="HRV Thesis assistant",
    description="Make any question on Stefanos Botsaris' Thesis (both in greek and english)"
)

demo.launch()