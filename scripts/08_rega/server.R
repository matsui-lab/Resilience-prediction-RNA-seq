# Script cleaned for public release. Edit /path/to/... inputs before running.
library(shiny)
library(DT)
library(plotly)
library(dplyr)

methods <- c("SVM" = "svm", 
             "ElasticNet" = "elasticnet", 
             "RandomForest" = "randomforest", 
             "XGBoost" = "xgboost")

feature_data <- lapply(methods, function(m) {
  read.csv(paste0("feature_importance_", m, "_label.csv"), row.names = 1)
})
names(feature_data) <- names(methods)

expr <- read.csv("exp_withClin4shiny.csv", row.names = 1)

shinyServer(function(input, output, session) {

  selected_data <- reactive({
    req(input$method)
    df <- feature_data[[input$method]]
    df %>% dplyr::rename(Gene = label, Importance = importance)
  })

  observe({
    choices <- unique(selected_data()$Gene)
    updateSelectizeInput(session, "gene", choices = as.list(choices), server = TRUE)
  })

  output$geneTable <- renderDT({
    datatable(selected_data(), options = list(pageLength = 10))
  })

  output$topGeneBar <- renderPlotly({
    df <- selected_data()
    tmp <- df %>% arrange(desc(Importance))
    top_pos <- head(tmp, 10)
    top_neg <- head(arrange(tmp, Importance), 10)
    top <- rbind(top_pos, top_neg)
    top$Type <- rep(c("Top Positive", "Top Negative"), each=10)
    top$Gene <- factor(top$Gene, levels = top$Gene[order(top$Importance)])
    plot_ly(top, x = ~Importance, y = ~Gene, color = ~Type,
            type = "bar", orientation = "h", colors = c("blue", "red")) %>%
      layout(title = paste(input$method, "Feature Importance: Top10 Positive & Negative Genes"),
             yaxis = list(title = ""), barmode = 'relative')
  })

  output$multiGenePlots <- renderUI({
    req(input$gene)
    plot_output_list <- lapply(input$gene, function(gene) {
      plotlyOutput(paste0("plot_", gene), height = "400px")
    })
    do.call(tagList, plot_output_list)
  })

  observe({
    req(input$gene)
    data <- selected_data()
    for (gene in input$gene) {
      local({
        g <- gene
        output[[paste0("plot_", g)]] <- renderPlotly({
          gene_data <- data %>% filter(Gene == g)
          gene_value <- gene_data$Importance
          if (length(gene_value) == 0) {
            return(
              plot_ly(type = "scatter", mode = "markers", x = c(), y = c()) %>%
                layout(
                  xaxis = list(visible = FALSE),
                  yaxis = list(visible = FALSE),
                  annotations = list(list(
                    x = 0.5, y = 0.5,
                    text = paste0("The gene '", g, "' is not available in the current dataset."),
                    showarrow = FALSE,
                    xref = 'paper', yref = 'paper',
                    font = list(size = 18, color = 'red'),
                    align = 'center'
                  ))
                )
            )
          }
          h <- hist(data$Importance, breaks = 50, plot = FALSE)
          max_count <- max(h$counts)
          p <- plot_ly() %>%
            add_trace(x = data$Importance, type = "histogram",
                      marker = list(color = "gray"), nbinsx = 50, name = "All Genes") %>%
            add_trace(x = c(gene_value, gene_value), y = c(0, 50),
                      type = "scatter", mode = "lines",
                      line = list(color = "red", width = 4, dash = "dash"),
                      name = paste(g, "selected"),
                      showlegend = FALSE) %>%
            layout(title = paste0(g, " Feature Importance Distribution (", input$method, ")"),
                   xaxis = list(title = "Feature Importance"),
                   yaxis = list(title = "Count"),
                   barmode = "overlay")
          p
        })
      })
    }
  })

  output$downloadTable <- downloadHandler(
    filename = function() {
      paste("table-", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(selected_data(), file, row.names = FALSE)
    }
  )

  output$correlationPlots <- renderUI({
    req(input$gene)
    data <- expr
    plotlist <- lapply(input$gene, function(gene) {
      if (!(gene %in% colnames(data))) {
        return(NULL)
      }
      p2 <- plot_ly(data, x = ~get(gene), y = ~resilience_score,
                    type = "scatter", mode = "markers",
                    marker = list(size = 8)) %>%
        layout(xaxis = list(title = gene),
               yaxis = list(title = "Resilience Score"))
      fluidRow(column(6, p2))
    })
    do.call(tagList, plotlist)
  })

})
