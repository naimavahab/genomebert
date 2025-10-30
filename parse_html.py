from bs4 import BeautifulSoup

text = 'interpret_small/output.html'
S = BeautifulSoup(text, 'lxml') 
  
# Using the prettify method 
print(S.prettify())
