# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:56:50 2023

@author: Xinrui
"""


from Bio.Blast import NCBIWWW, NCBIXML
import pandas as pd

def match_sequence_to_uniprot(sequences):
    uniprot_id_list, match_index = [],[]
    for k in range(len(sequences)):
        try:
            # Perform a BLAST search using the sequence
            result_handle = NCBIWWW.qblast("blastp", "swissprot", sequences[k])
        
            # Parse the BLAST search results
            blast_record = NCBIXML.read(result_handle)
        
            # Check if any alignments were found
            if len(blast_record.alignments) > 0:
                # Retrieve the UniProt ID for the best matching sequence
                alignment = blast_record.alignments[0]
                uniprot_id = alignment.accession
            else:
                uniprot_id = 'null'
            
            match_index += [k]
            uniprot_id_list += [uniprot_id]
        except:
            pass
    
    match_sequences = [sequences[i] for i in match_index]
    df = pd.DataFrame(list(zip(match_sequences, uniprot_id_list)))
    
    return df



if __name__ == '__main__':   
    # Example usage
    sequences = [
        'PQITLWKRPIVTVKIGGQLREALLDTGADDTVLEDINLPGKWKPKMIVGIGGFVKVKQYEQVPIEICGKKAIGTVLVGPTPANIIGRNMLTQIGCTLNF',
        '',  # Empty sequence
        'INVALIDSEQUENCE'  # Sequence that doesn't match a UniProt ID
    ]
    
    df = match_sequence_to_uniprot(sequences)