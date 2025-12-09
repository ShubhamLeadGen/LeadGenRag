document.getElementById('startChat').addEventListener('click', () => {
  chrome.tabs.create({ url: 'http://localhost:8501' });
});
