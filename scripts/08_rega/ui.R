# Script cleaned for public release. Edit /path/to/... inputs before running.
library(shiny)
library(DT)
library(plotly)

fluidPage(
  tags$head(
    tags$title("REGA")
  ),
  titlePanel(
    div(
      "REGA: REsilience Gene Analyzer",
      style = "font-size: 36px; font-weight: bold;"
    )
  ),

  div(
    p("This application allows you to analyze selected gene's contribution to resilience score prediction. Select a method and gene symbols to visualize the data.",
      style = "font-size: 18px;")
  ),

  sidebarLayout(
    sidebarPanel(
      p("Please select a method and gene symbols."),
      selectInput("method", "Select ML Method:",
                  choices = c("SVM", "ElasticNet", "RandomForest", "XGBoost"),
                  selected = "SVM"),
      selectizeInput("gene", "Select Gene(s):", choices = NULL, multiple = TRUE,
                     options = list(maxItems = 30)),
      width = 3
    ),

    mainPanel(
      tabsetPanel(type = "tabs",
                  tabPanel("Top Genes",
                           br(),
                           plotlyOutput("topGeneBar", height = "600px")
                  ),
                  tabPanel("Plot", 
                           br(),
                           uiOutput("multiGenePlots")  
                  ),
                  tabPanel("Table", 
                           br(),
                           DTOutput("geneTable", width="70%"), 
                           downloadButton("downloadTable", "Download Table")
                  ),
                  tabPanel("Correlation",
                           br(),
                           uiOutput("correlationPlots")
                  )
      ),
      width = 8
    )
  )
)
