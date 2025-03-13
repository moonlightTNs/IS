import streamlit as st

st.title("THIS IS MAIN PAGE")

# ใช้ Sidebar เพื่อเปลี่ยนไปยังหน้าอื่น
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Dataset Preview", "Model Training", "Evaluation"])

# เพิ่มเนื้อหาเพิ่มเติมใน Sidebar
st.sidebar.markdown("## Additional Information")
st.sidebar.info("This is an example of additional information in the sidebar.")

if page == "Dataset Preview":
    st.experimental_set_query_params(page="Dataset Preview")
    st.write("##    # Dataset Preview Page")
    st.write("This is the Dataset Preview page.")
elif page == "Model Training":
    st.experimental_set_query_params(page="Model Training")
    st.write("### Model Training Page")
    st.write("This is the Model Training page.")
elif page == "Evaluation":
    st.experimental_set_query_params(page="Evaluation")
    st.write("### Evaluation Page")
    st.write("This is the Evaluation page.")

