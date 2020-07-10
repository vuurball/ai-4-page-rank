import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000
CONVERGENCE = 0.001

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # no link on page = page has links to all pages
    links = corpus[page]  if len(corpus[page]) > 0 else corpus.keys()
    # probability to chose random link on current page
    pr_link = damping_factor / len(links)
    # probability to chose random page out of existing pages in corpus
    pr_page = (1 - damping_factor) / len(corpus)
    res = {}
    for i in corpus:
        res[i] = pr_page + pr_link if i in links else pr_page
    return res


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    result = {}
    # first page chosen randomely 
    page = random.choice(list(corpus))
    for i in corpus:
        result[i] = 1 if page == i else 0
    for i in range(n-1):
        prob_dist = transition_model(corpus, page, damping_factor)
        page = random.choices(list(prob_dist.keys()), weights=list(prob_dist.values()))[0]
        result[page] += 1
    for i in result:
        result[i] /= n
    return result


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pr_res = {}
    d = damping_factor
    N = len(corpus)
    
    # start by assuming the PageRank of every page is 1 / N
    for p in corpus:
        pr_res[p] = 1 / N
        # a page with no links is interpreted as having 1 link for every page in the corpus
        if len(corpus[p]) == 0:
            corpus[p] = corpus.keys()

    # then use the Iterative Algorithm formula to calculate new PageRank values for each page,
    # based on the previous PageRank values (pr_res)
    is_converged = False
    while is_converged == False:
        for p in corpus:
            old_pr_p = pr_res[p]
            pr_res[p] = ((1 - d) / N)
            pr_i_sum = 0
            for i in corpus:
                if p in corpus[i]:
                    pr_i_sum += pr_res[i] / len(corpus[i])
            pr_res[p] += (d * pr_i_sum)    
            diff = abs(pr_res[p]  - old_pr_p)
            if diff <= CONVERGENCE:
                is_converged = True
           
    return pr_res   


if __name__ == "__main__":
    main()
