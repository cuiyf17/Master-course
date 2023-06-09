import Bio.SeqIO as so

file = so.parse("./sequence2.fastq", "fastq")
so.write(file, "./seq2fastq.fa", "fasta")