module github.com/sashabaranov/go-openai

go 1.18

require (
	github.com/tiktoken-go/tokenizer v0.1.0
	golang.org/x/time v0.5.0
)

require github.com/dlclark/regexp2 v1.10.0 // indirect

replace github.com/tiktoken-go/tokenizer v0.1.0 => github.com/fulldiveVR/tokenizer v0.0.9
