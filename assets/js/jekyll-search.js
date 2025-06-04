(function () {
  function search(query, data) {
    return data.filter(function (entry) {
      return entry.title.toLowerCase().includes(query) ||
             entry.content.toLowerCase().includes(query);
    });
  }

  function displayResults(results) {
    const resultsContainer = document.getElementById('search-results');
    if (!resultsContainer) return;

    if (results.length === 0) {
      resultsContainer.innerHTML = "<p>No results found.</p>";
      return;
    }

    const html = results.map(result => `
      <article>
        <h2><a href="${result.url}">${result.title}</a></h2>
        <p>${result.content}</p>
        <small>${result.date}</small>
        <hr>
      </article>
    `).join('');

    resultsContainer.innerHTML = html;
  }

  const params = new URLSearchParams(window.location.search);
  const query = params.get("q");

  if (query) {
    fetch("/AIlearn.github.io/search.json")
      .then(response => response.json())
      .then(data => {
        const results = search(query.toLowerCase(), data);
        displayResults(results);
      });
  }
})();
