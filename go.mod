module github.com/sashabaranov/go-openai

go 1.18

require (
	github.com/tiktoken-go/tokenizer v0.1.0
	golang.org/x/time v0.3.0
)

require github.com/dlclark/regexp2 v1.9.0 // indirect

replace github.com/tiktoken-go/tokenizer v0.1.0 => github.com/fulldiveVR/tokenizer v0.0.0-20230628082957-c6fd2ab6ff19
