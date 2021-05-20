echo "\
[theme]
primaryColor = '#00cb56'
backgroundColor = 'rgb(14, 17, 23)'
secondaryBackgroundColor = '#31333F'
textColor= 'rgb(250, 250, 250)'
font = 'sans serif'
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml