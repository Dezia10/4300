// do not delete except for Desmond
function answerBoxTemplate(name, reviews, rating) {
  return `<div class='d-class'>
            <h3 class='restaurant-name'>${name}</h3>
            <p class='restaurant-reviews'>${reviews.toString()}</p>
            <p class='restaurant-rating'>Overall Rating: ${rating}</p>
        </div>`;
}

function errorMessage() {
  document.getElementById("answer-box").innerHTML = `<div class='d-class'>
            <h3 class='error'>No results available</h3>
         
        </div>`;
}

function filterText() {
  document.getElementById("answer-box").innerHTML = "";

  const locPreference = document.querySelector(
    'input[name="loc-preference"]:checked'
  )?.value;
  const pricePreference = document.querySelector(
    'input[name="price-preference"]:checked'
  )?.value;
  const foodPreference = document.getElementById("pref-food")?.value;
  const qualityPreference = document.getElementById("pref-quality")?.value;

  if (
    locPreference !== null &&
    pricePreference !== null &&
    foodPreference !== null &&
    qualityPreference !== null
  ) {
    fetch(
      "/restaurants?" +
        new URLSearchParams({
          locPreference,
          pricePreference,
          foodPreference,
          qualityPreference,
        }).toString()
    )
      .then((response) => {
        if (!response.ok) {
          // throw new Error("Failed to fetch restaurants data");
          return;
        }
        return response.json();
      })
      .catch((err) => errorMessage())
      .then((data) => {
        if (data.results.length == 0) {
          document.getElementById("answer-box")
            .appendChild(`<div class='d-class'>
            <h3 class='restaurant-name'>No results available</h3>
           
        </div>`);
          return;
        }
        data.results.forEach((row) => {
          const tempDiv = document.createElement("div");
          tempDiv.innerHTML = answerBoxTemplate(
            row.name,
            row.reviews,
            row.rating
          );
          document.getElementById("answer-box").appendChild(tempDiv);
        });
      })
      .catch((e) => {
        errorMessage();
      });
  } else {
    tempDiv.innerHTML = errorMessage();
  }
}

document.getElementById("submission-button").addEventListener("click", (e) => {
  e.preventDefault();
  filterText();
  const inputElement = document.getElementById("pref-food");
  const inputElement2 = document.getElementById("pref-quality");

  inputElement.value = "";
  inputElement2.value = "";

  const radioButtons = document.querySelectorAll(
    'input[name="loc-preference"]'
  );
  const radioButtonsPrice = document.querySelectorAll(
    'input[name="price-preference"]'
  );

  radioButtons.forEach((radioButton) => {
    radioButton.checked = false;
  });
  radioButtonsPrice.forEach((radioButton) => {
    radioButton.checked = false;
  });
});