# simulation
R = fread("./Rall_gene.txt") %>%
  as.data.frame(check.names=F)
Tsg=fread("./Tsg.all_gene..txt")

Tsg$V1 = paste0("layer",Tsg$V1)
Tsg=as.data.frame(Tsg,check.names=F)
rownames(Tsg)=Tsg$V1
Tsg = data.frame(t(Tsg[,-1]),check.names = F)
identical(rownames(Tsg),R$V1)

max_fold_all = c()
for(j in 1:ncol(Tsg)){
  max_fold_change=rep(0,nrow(Tsg))
  for(i in 2:ncol(R)){
    Tsg_cell = log(1+exp(Tsg+R[,i]))+1e-10
    max_fold_change_cell=rep(1,nrow(Tsg_cell))
    #for(j in 1:nrow(combine0)){
    max_fold_change_temp = Tsg_cell[,1]/Tsg_cell[,2]
    max_fold_change_temp = log2(max_fold_change_temp)
    #  max_fold_change_cell = max_two_vector(max_fold_change_cell,max_fold_change_temp)
    #}
    max_fold_change=max_two_vector(max_fold_change_temp,max_fold_change)
  }
  max_fold_all=cbind(max_fold_all,max_fold_change)
}


colnames(max_fold_all)=paste0("layer",1:2)
max_fold_all = max_fold_all %>% as.data.frame()
max_fold_all$gene = rownames(Tsg)
max_fold_all = max_fold_all %>%select(layer1,gene)
max_fold_all_melt=melt(max_fold_all,id.vars="gene")
max_fold_all_melt$variable="layer1 / layer2"
df_col=data.frame(layer="layer1 / layer2",y=0)
k=ggplot(max_fold_all_melt,aes(x=variable,y=value))+
  geom_jitter(position = position_jitter(seed=1),
              color=ifelse(max_fold_all_melt$value>1 | max_fold_all_melt$value< -1,"pink","grey"))+theme_article()+
  ylab("log2 Fold Change")+xlab("Layer")+
  #geom_text_repel(aes(label=ifelse(value>1 | value< -1,gene,"")),force=1.2)+
  geom_label_repel(position = position_jitter(seed=1),
                   aes(label=ifelse(value>2 | value< -2,gene,"")),
                   max.overlaps = 20000,force_pull=0,size=2.5,
                   min.segment.length = unit(0, 'lines'))+
  geom_tile(aes(x=variable,y=0),height=0.5,color="black",fill="pink",alpha=0.3)+
  geom_text(aes(x=layer,y=0,label=layer),df_col,fontface=2)+
  theme(legend.position = "none",
        axis.text.x = element_blank(),
        axis.ticks.x=element_blank(),
        panel.grid.major.x = element_blank(),
        #panel.border = element_blank(),
        #axis.line.y = element_line(colour="black")
  )+geom_hline(yintercept=1,color="grey",lty=2)+geom_hline(yintercept=-1,color="grey",lty=2)+
  ggtitle("Predicted")+xlab(NULL)