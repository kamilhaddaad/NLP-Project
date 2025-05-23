$(document).ready(function() {
    // Hide loading and result sections initially
    $('.loading-recommendations, .loading-sentiment, .loading-analysis, .loading-blurb, .loading-title').hide();
    $('.recommendations-result, .sentiment-result, .analysis-result, .blurb-result, .title-result').hide();

    // Book Recommendation Form Handler
    $('#recommendForm').on('submit', function(e) {
        e.preventDefault();
        
        const bookTitle = $('#bookTitle').val();
        if (!bookTitle) {
            alert('Please enter a book title');
            return;
        }

        // Show loading spinner
        $('.loading-recommendations').show();
        $('.recommendations-result').hide();

        // Make API call
        $.ajax({
            url: '/recommend',
            method: 'POST',
            data: { book_title: bookTitle },
            success: function(response) {
                $('.loading-recommendations').hide();
                
                // Clear previous results
                $('#recommendationsTable').empty();
                
                // Add new results
                response.recommendations.forEach(function(book) {
                    $('#recommendationsTable').append(`
                        <tr>
                            <td>${book.Title}</td>
                            <td>${book.Author}</td>
                            <td>${book.Year}</td>
                        </tr>
                    `);
                });
                
                // Show results
                $('.recommendations-result').fadeIn();
            },
            error: function(xhr) {
                $('.loading-recommendations').hide();
                alert('Error getting recommendations. Please try again.');
            }
        });
    });

    // Book Analysis Form Handler
    $('#bookAnalysisForm').on('submit', function(e) {
        e.preventDefault();
        
        const description = $('#bookDescription').val();
        if (!description) {
            alert('Please enter a book description');
            return;
        }

        // Show loading spinner
        $('.loading-analysis').show();
        $('.analysis-result').hide();

        // Make API call
        $.ajax({
            url: '/analyze-book',
            method: 'POST',
            data: { description: description },
            success: function(response) {
                $('.loading-analysis').hide();
                
                // Clear previous results
                $('#genresList, #themesList, #entitiesList').empty();
                
                // Add genres
                response.genres.forEach(function(genre) {
                    $('#genresList').append(`
                        <span class="badge bg-primary">
                            ${genre.genre} (${(genre.confidence * 100).toFixed(1)}%)
                        </span>
                    `);
                });
                
                // Add themes
                response.themes.forEach(function(theme) {
                    $('#themesList').append(`
                        <span class="badge bg-info text-dark">
                            ${theme}
                        </span>
                    `);
                });
                
                // Add entities
                response.entities.forEach(function(entity) {
                    $('#entitiesList').append(`
                        <span class="badge bg-secondary">
                            ${entity}
                        </span>
                    `);
                });
                
                // Update sentiment
                const sentimentScore = response.sentiment.score;
                const sentimentLabel = response.sentiment.label;
                
                $('.sentiment-fill').css('width', `${(sentimentScore + 1) * 50}%`);
                $('.sentiment-label').text(sentimentLabel);
                
                // Show results
                $('.analysis-result').fadeIn();
            },
            error: function(xhr) {
                $('.loading-analysis').hide();
                alert('Error analyzing book. Please try again.');
            }
        });
    });

    // Review Sentiment Analysis Form Handler
    $('#reviewForm').on('submit', function(e) {
        e.preventDefault();
        
        const review = $('#reviewInput').val();
        if (!review) {
            alert('Please enter a review');
            return;
        }

        // Show loading spinner
        $('.loading-sentiment').show();
        $('.sentiment-result').hide();

        // Make API call
        $.ajax({
            url: '/analyze',
            method: 'POST',
            data: { review: review },
            success: function(response) {
                $('.loading-sentiment').hide();

                const sentiment = response.sentiment;
                const confidence = response.confidence;

                // Update sentiment text and icon
                $('.sentiment-text').text(sentiment);
                if (sentiment === 'positive') {
                    $('.fa-thumbs-up').show();
                    $('.fa-thumbs-down').hide();
                    $('.sentiment-result').removeClass('negative').addClass('positive');
                } else {
                    $('.fa-thumbs-up').hide();
                    $('.fa-thumbs-down').show();
                    $('.sentiment-result').removeClass('positive').addClass('negative');
                }

                // Update confidence bar
                $('.confidence-value').text(`${(confidence * 100).toFixed(1)}%`);
                $('.confidence-fill').css('width', `${confidence * 100}%`);

                // Show results
                $('.sentiment-result').fadeIn();
            },
            error: function(xhr) {
                $('.loading-sentiment').hide();
                alert('Error analyzing review. Please try again.');
            }
        });
    });

    // Blurb Generator Form Handler
    $('#blurbForm').on('submit', function(e) {
        e.preventDefault();
        const title = $('#blurbTitle').val();
        const genre = $('#blurbGenre').val();
        if (!title || !genre) {
            alert('Please enter both topic and genre');
            return;
        }
        $('.loading-blurb').show();
        $('.blurb-result').hide();
        $.ajax({
            url: '/generate-blurb',
            method: 'POST',
            data: { title: title, genre: genre },
            success: function(response) {
                $('.loading-blurb').hide();
                if (response.blurb) {
                    $('#generatedBlurb').text(response.blurb);
                    $('.blurb-result').fadeIn();
                } else {
                    alert('No blurb generated.');
                }
            },
            error: function(xhr) {
                $('.loading-blurb').hide();
                alert('Error generating blurb. Please try again.');
            }
        });
    });
    
    // Book Title Generation Form Handler
    $('#titleForm').on('submit', function(e) {
        e.preventDefault();

        const summary = $('#summaryInput').val();
        const generatedTitlesDiv = $('#generatedTitles');
        const loadingTitleDiv = $('.loading-title');
        const titleResultDiv = $('.title-result');

        if (!summary) {
            generatedTitlesDiv.html('<p style="color: red;">Please enter a summary.</p>');
            titleResultDiv.show();
            return;
        }

        loadingTitleDiv.show();
        titleResultDiv.hide();
        generatedTitlesDiv.empty();

        $.ajax({
            url: '/generate_title',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ summary: summary }),
            success: function(response) {
                loadingTitleDiv.hide();

                if (response.error) {
                    generatedTitlesDiv.html(`<p style="color: red;">Error: ${response.error}</p>`);
                } else if (response.titles && response.titles.length > 0) {
                    let titlesHtml = '<h6>Generated Titles</h6><div id="generatedTitles">';
                    response.titles.forEach(function(title) {
                        titlesHtml += `<p>${title}</p>`;
                    });
                    titlesHtml += '</div>';
                    titleResultDiv.html(titlesHtml);
                } else {
                     titleResultDiv.html('<h6>Generated Titles</h6><div id="generatedTitles"><p>No titles generated.</p></div>');
                }

                titleResultDiv.fadeIn();
            },
            error: function(xhr) {
                loadingTitleDiv.hide();
                titleResultDiv.hide();
                alert('Error generating titles. Please try again.');
                console.error('Error:', xhr);
            }
        });
    });    

    // Summary Generator Form Handler
    $('#summaryForm').on('submit', function(e) {
        e.preventDefault();
        const title = $('#bookTitleSummary').val();
        const additionalInfo = $('#additionalInfoSummary').val();
        if (!title || !additionalInfo) {
            alert('Please enter both title and additional informations that should be included in the summary');
            return;
        }
        $('.loading-summary').show();
        $('.summary-result').hide();
        $.ajax({
            url: '/generate-summary',
            method: 'POST',
            data: { title: title, additionalInfo: additionalInfo },
            success: function(response) {
                $('.loading-summary').hide();
                if (response.summary) {
                    $('#generatedSummary').text(response.summary);
                    $('.summary-result').fadeIn();
                } else {
                    alert('No summary generated.');
                }
            },
            error: function(xhr) {
                $('.loading-summary').hide();
                alert('Error generating summary. Please try again.');
            }
        });
    });
});


function generateTitle() {
    const summary = document.getElementById('summaryInput').value;
    const generatedTitlesDiv = document.getElementById('generatedTitles');

    if (!summary) {
        generatedTitlesDiv.innerHTML = '<p style="color: red;">Please enter a summary.</p>';
        return;
    }

    generatedTitlesDiv.innerHTML = '<p>Generating titles...</p>';

    fetch('/generate_title', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ summary: summary })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            generatedTitlesDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else if (data.titles && data.titles.length > 0) {
            let titlesHtml = '<h3>Generated Titles:</h3><ul>';
            data.titles.forEach(title => {
                titlesHtml += `<li>${title}</li>`;
            });
            titlesHtml += '</ul>';
            generatedTitlesDiv.innerHTML = titlesHtml;
        } else {
             generatedTitlesDiv.innerHTML = '<p>No titles generated.</p>';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        generatedTitlesDiv.innerHTML = `<p style="color: red;">An error occurred: ${error.message}</p>`;
    });
    }