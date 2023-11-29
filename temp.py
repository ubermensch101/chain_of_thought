import streamlit as st
from predict import predict_one  

def main():
    st.title("Math Word Problem Solver")
    user_query = st.text_area("Enter a math word problem:")

    if st.button("Submit"):
        if user_query:
            answer = predict_one(user_query)
            st.subheader("Answer:")
            formatted_answer = format_response(answer)
            st.code(formatted_answer, language='python')

def format_response(response):
    answer, explanation = response
    formatted_response = f"Answer: {answer}\n\nExplanation:\n{explanation}"
    return formatted_response

if __name__ == "__main__":
    main()
