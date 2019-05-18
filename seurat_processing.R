library(Seurat)
pbmc.counts <- Read10X(data.dir = "~/Downloads/filtered_gene_bc_matrices/hg19/")
pbmc <- CreateSeuratObject(counts = pbmc.counts, project = "pbmc3k", min.cells = 3, min.features = 200)
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")

# Filter out contaminated cells and doublets
pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

# Normalize
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)


# Feature selection
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# CLuster
all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))
pbmc <- JackStraw(pbmc, num.replicate = 100)
pbmc <- ScoreJackStraw(pbmc, dims = 1:20)
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.5)
length(Idents(pbmc))
new.cluster.ids <- c("Naive CD4 T", "Memory CD4 T", "CD14+ Mono", "B", "CD8 T", "FCGR3A+ Mono", 
                     "NK", "DC", "Platelet")
names(new.cluster.ids) <- levels(pbmc)
pbmc <- RenameIdents(pbmc, new.cluster.ids)

######################################################################
######################################################################
# Write cluster labels and preprocessed data to csv
library(data.table)
labels <- Idents(pbmc)
labels <- as.data.frame(labels)
labels2 <- pbmc@meta.data$seurat_clusters
labels2 <- as.data.frame(labels2)

data <- GetAssayData(pbmc, slot = 'scale.data')
head(data[, 1:10])
dim(data)
data <- as.data.frame(as.matrix(data))

fwrite(x = data, row.names = TRUE, file = "scPMBC.csv")
fwrite(x = labels, row.names = TRUE, file = "pbmc_labels_str.csv")
fwrite(x = labels2, row.names = TRUE, file = "pbmc_labels_int.csv")

pbmc@assays$RNA
