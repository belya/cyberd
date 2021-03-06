package main

import (
	"bufio"
	"os"
	"regexp"
	"strings"
	"unicode"
)

func Index() {

	startArticleId := int64(1)

	sendLinks := InitAddLink()

	f, err := os.OpenFile("enwiki-latest-all-titles", 0, 0)
	if err != nil {
		panic(err)
	}
	br := bufio.NewReader(f)
	defer f.Close()

	reg, err := regexp.Compile("[^a-zA-Z0-9]+")

	counter := int64(0)
	links := make([]Link, 0, 100)
	for {

		line, err := br.ReadString('\n')

		if err != nil {
			break
		}

		if counter < startArticleId {
			counter++
			continue
		}

		split := strings.Split(strings.TrimSuffix(line, "\n"), "\t")
		ids := strings.Split(split[1], "_")

		for _, id := range ids {

			id = reg.ReplaceAllString(id, "")
			id = strings.ToLower(id)

			if len(id) == 0 || id == "" {
				continue
			}

			if len(id) == 1 && unicode.IsSymbol(rune(id[0])) {
				continue
			}

			page := ".wiki/wiki/" + split[1] + ".html"
			links = append(links, Link{from: id, to: page})
			counter++

			if len(links) == 1000 {
				println(split[1])
				println(counter)
				sendLinks(links)
				links = make([]Link, 0, 100)
			}
		}
	}
}
