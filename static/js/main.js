const form = document.getElementById('myForm');
let username = document.getElementById('username');
let submit_btn = document.getElementById('btnSubmit');

const usernamePattern = /^@[a-zA-Z0-9_]{1,15}$/;
const progressBar = document.getElementById("progress-bar");
const progressBarContainer = document.getElementById("progress-bar-container");

function startProgress() {
  let progress = 0;
  const interval = setInterval(function() {
    progress += 10;
    progressBar.style.width = progress + "%";
    if (progress >= 100) {
      clearInterval(interval);
      progressBarContainer.style.display = "none";
    }
  }, 1000);
}

function fetchData() {
  // Show progress bar
  progressBarContainer.style.display = "block";
  startProgress();

  // Fetch data from API
  fetch("https://api.alquran.cloud/v1/surah/2/ar.alafasy")
    .then(response => response.json())
    .then(data => {
      // Hide progress bar
      progressBarContainer.style.display = "none";
      // Do something with the data
      console.log(data);
    })
    .catch(error => {
      console.error(error);
      // Hide progress bar
      progressBarContainer.style.display = "none";
    });
}

function validateForm() {
  if (username.value === '') {
    username.style.border = '3px solid #ED2B2A'
    return false;
  }
  else{
      username.style.border = 'none'
      if (usernamePattern.test(username.value))
          fetchData();
      else
          console.log("Invalid Twitter username!");
  }

  return true;
}

form.addEventListener('submit', function(event) {
  event.preventDefault();

  if (validateForm()) {
    form.submit();
  }
});