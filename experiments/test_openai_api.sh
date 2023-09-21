#!/usr/bin/env sh

# Echo test:
# 
# curl https://api.openai.com/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#      "model": "gpt-3.5-turbo",
#      "messages": [{"role": "user", "content": "Say this is a test!"}],
#      "temperature": 0.7
#    }'

# Story-telling test:

# a stop sign with a red and white arrow
SCENARIO="araffe sitting in a chair with a bear on his lap"
# Output "gpt-3.5-turbo":
# "Once upon a time, in a cozy cabin, a giraffe named Gerald sat in a chair, cuddling with his best friend, a bear named Benny. They shared love, warmth, and endless adventures."

curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "You are a story teller.\nYou can tell a short story based on a simple narrative.\nThe story should be no longer than 25 words and a few sentences.\nCONTEXT: araffe sitting in a chair with a bear on his lap.\nSTORY:"}],
     "temperature": 0.3
   }'

# General output:
# "Once upon a time, in a cozy cabin, a giraffe named Gerald sat in a chair, cuddling with his best friend, a bear named Benny. They shared love, warmth, and endless adventures."

# curl https://api.openai.com/v1/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#      "engine": "text-davinci-001",
#      "prompt": "You are a story teller.\nYou can tell a short story based on a simple narrative.\nThe story should be no longer than 25 words and a few sentences.\nCONTEXT: araffe sitting in a chair with a bear on his lap.\nSTORY:",
#    }'
