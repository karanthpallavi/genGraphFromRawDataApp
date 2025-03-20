from flask import Flask, request, render_template, jsonify, current_app
from werkzeug.utils import secure_filename
import pandas as pd
import os
import time
import logging
from os import listdir
from os.path import isfile, join
import rdflib
from rdflib import *
import re
import urllib.parse
import sys
import requests
from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE
import warnings

from urllib.parse import urlparse


def generate_output_filename(input_filename):
    """
    Generate a dynamic output file name for the processed file.
    The output file name will be based on the original file name and a timestamp.
    """
    base_name, ext = os.path.splitext(input_filename)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ext = ".csv"
    return f"{base_name}_processed_{timestamp}{ext}"

def generate_output_filename_ttl(input_filename):
    """
    Generate a dynamic output file name for the processed file.
    The output file name will be based on the original file name and a timestamp.
    """
    base_name, ext = os.path.splitext(input_filename)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    #ext = ".csv"
    return f"{base_name}_processed_{timestamp}"

def readIndivSheetsTransformToCSV(all_sheets,templateMeasurementCSVFile):
    sheetIndex = 0
    for sheet_name, df in all_sheets.items():
        if sheetIndex > 0:
            #print(f"Sheet name: {sheet_name}")
            ##print(df.head())  # Print the first few rows of each DataFrame
            ##print("-" * 40)

            #Transform each sheet from 2nd sheet in SSBR_Requests.xlsx

            #Drop irrelevant columns
            df.drop(['SecondColumn', 'Comments'], axis=1, inplace=True)

            #Rename
            df_indivRecipeSheet_csvMeasurements = pd.read_csv(templateMeasurementCSVFile)

            colNames = df.columns.values
            #print("Names of columns in this sheet: " ,colNames)
            lenTotalColumns = len(colNames)
            recipeNames = colNames[2:lenTotalColumns]
            numRecipes = len(recipeNames)
            #print("Number of recipes here in this sheet: ",numRecipes)

            currentNumRowsInDf = df.shape[0]
            #print("number of rows: ",currentNumRowsInDf)


            df_sheet1_csvMeasurements = pd.read_csv(templateMeasurementCSVFile)

            idxForRow = 0
            totalRows = currentNumRowsInDf * numRecipes
            #print("Total rows in new df is ",totalRows)
            #print("-" * 40)

            # To transform each sheet into templateMeasurementCSVFile format
            incrementValRow = 0

            for index,row in df.iterrows():

                recipeNamesIdx = 0
                for incrementVal in range(0,numRecipes):
                    # check the number of recipes before
                    if recipeNamesIdx < numRecipes:
                        colNameIsAbout = recipeNames[recipeNamesIdx]
                        #print("Is about: ",recipeNames[recipeNamesIdx])
                        #print("Qual meas of ",row['FirstColumn'])
                        #print("specified numeric value ",row[colNameIsAbout])
                        #print("Unit: ",row['Units'])
                        #print("new row number where we write ",incrementValRow)
                        df_sheet1_csvMeasurements.loc[incrementValRow,'is_quality_measurement_of'] = row['FirstColumn']
                        df_sheet1_csvMeasurements.loc[incrementValRow,'has_specified_numeric_value'] = row[colNameIsAbout]
                        df_sheet1_csvMeasurements.loc[incrementValRow,'is_about'] = colNameIsAbout
                        df_sheet1_csvMeasurements.loc[incrementValRow,'has_measurement_unit_label'] = row['Units']
                        incrementValRow = incrementValRow + 1
                        recipeNamesIdx = recipeNamesIdx + 1
                    #incrementValRow = incrementValRow + currentNumRowsInDf
            #Write to CSV
            fileNameStr = sheet_name

            output_filename = generate_output_filename(fileNameStr)
            output_filepath = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
            #outputCSVFilePath = "output\\"+fileNameStr+"transformedCSV.csv"
            df_sheet1_csvMeasurements.to_csv(output_filepath,sep=',', index=False, encoding='utf-8')
            #df
        else:
            sheetIndex = sheetIndex + 1


def preprocessSSBRRequestsFile(dataFileExcel,templateMeasurementCSVFile):
    #print("Preprocessing file: ",dataFileExcel)
    all_sheets = pd.read_excel(dataFileExcel, sheet_name=None)
    ##print(all_sheets[0])

    df_sheet1 = all_sheets['Summary']
    ##print("Sheet names: ",all_sheets.sheet_names)
    #print(df_sheet1)



    #Drop irrelevant columns
    df_sheet1.drop(['Mooney_Poly', 'MSR_Poly','Mooney_Step1', 'MSR_Step1', 'Mooney_End','MSR_End','Tg'], axis=1, inplace=True)

    #print(df_sheet1)

    #Rename columns
    dictColNamesRename = {'Request_No': 'Compound Number', 'Polymer_Names': 'Polymer Name', 'Mooney_Stripped': 'Mooney Viscosity', 'MSR_Stripped': 'Mooney Stress Relaxation', 'TgStr': 'Glass Transition Temperature','Mn': 'Number Averaged Molecular Weight', 'Mw': 'Weight Averaged Molecular Weight'}
    df_sheet1.rename(columns=dictColNamesRename,inplace=True)
    df_sheet1['Polymer Name'] = df_sheet1['Polymer Name'].str.replace(r'\s+', '', regex=True)
    df_sheet1['SampleName'] = df_sheet1['Polymer Name'].astype(str) + '_'+df_sheet1['Compound Number'].astype(str)

    #Remove duplicate rows based on column - Polymer Name
    ##df_sheet1.drop_duplicates(subset=['Polymer Name'], keep="first")

    #print(df_sheet1)

    #Add functionalization code from file - fx_SSBR
    #Commenting below lines as fx info included in current SSBR input file
    #df_sheet1_fx = pd.read_excel(fxFile, sheet_name='Sheet1')
    #df_sheet1_fx['Polymer Name'] = df_sheet1_fx['Polymer']
    #Get functionalization codes from fx_SBBR excel file
    #key_list = df_sheet1_fx['Polymer Name']
    #val_list = df_sheet1_fx['Functionalization (fx)']
    #Prepare dict to map functionalization codes to df_sheet1
    #dict_fx = dict(zip(key_list,val_list ))
    #print(dict_fx)
    #df_sheet1['Functionalization (fx)'] = df_sheet1['Polymer Name'].map(dict_fx)
    #print(df_sheet1)
    df_sheet1.loc[df_sheet1["fx"] == "0", "fx"] = ""
    df_sheet1.loc[df_sheet1["fx"] == "Empty", "fx"] = ""
    #df_sheet1.to_csv('output/outputPreprocessing/df_sheet1.csv',sep=',', index=False, encoding='utf-8')

    # Fill df_sheet1 in template format for measurement

    # Read each column from df_sheet1 - make DQ checks
    # Fill in template.csv - Prepare df_sheet1_template_filled and save in template.csv

    #Check if Compound Number is empty in any cell
    numEmptyValues = df_sheet1['Compound Number'].isnull().sum()

    #Get unique Compound Numbers in a list
    listCompoundNumbers = df_sheet1['SampleName'].tolist()
    #Column names from D to L represent qualities
    colNames = df_sheet1.columns.values
    #print(colNames)
    lenTotalColumns = len(colNames)
    qualityNames = colNames[4:lenTotalColumns-1]
    #print(qualityNames)
    lenQualityNames = len(qualityNames)
    #print(lenQualityNames)
    lenCompoundNumbers = df_sheet1.shape[0]
    #print("number of rows: ",lenCompoundNumbers)
    #print(numEmptyValues)
    if numEmptyValues == 0:
        #Compound Number is non empty
        df_sheet1_csvMeasurements = pd.read_csv(templateMeasurementCSVFile)

        #Compound Number is the Object (pmd:Object)
        #for j in range(0,lenCompoundNumbers):
        idxForRow = 0
        totalRows = lenCompoundNumbers * lenQualityNames
        #print("Total rows ", totalRows)

        #print("________________ Before choosing columns __________________")
        #print(df_sheet1)
        #print("___________________________________")

        df_sheet1 = df_sheet1[['Mooney Viscosity','Mooney Stress Relaxation','Styrene_Cont','Cis_Cont','Trans_Cont','Vinyl_Cont','Glass Transition Temperature','Number Averaged Molecular Weight','Weight Averaged Molecular Weight','SampleName']]
        #print("___________ After choosing columns ________________________")
        #print(df_sheet1)
        #print("___________________________________")

        colListOfQualityMeasurements = ['Mooney Viscosity','Mooney Stress Relaxation','Styrene_Cont','Cis_Cont','Trans_Cont','Vinyl_Cont','Glass Transition Temperature','Number Averaged Molecular Weight','Weight Averaged Molecular Weight']
        incrementValue = 0
        for index,row in df_sheet1.iterrows():
            #Debug statements for print below
            #if incrementValue == 0:
            #print("Each row : ",row)
            #print("index: ",index)
            #print("incrementValue value before writing ",incrementValue)
            #print("Mooney Viscosity: ",row["Mooney Viscosity"])


            ##compPoly = df_sheet1.loc[incrementValue,'SampleName']
            compPoly = row['SampleName']
            ##k = incrementValue+lenQualityNames-1
            #isAbout = df_sheet1_csvMeasurements.loc[incrementValue,'is_about']
            #isAbout = df_sheet1_csvMeasurements['is_about'].loc[df_sheet1_csvMeasurements.index[incrementValue]]
            #print("compPoly is ",compPoly)
            #print("isAbout is ",isAbout)
            #if(compPoly==isAbout):
            #print("Correct row to write")
            df_sheet1_csvMeasurements.loc[incrementValue,'has_specified_numeric_value'] = row["Mooney Viscosity"]
            df_sheet1_csvMeasurements.loc[incrementValue,'is_quality_measurement_of'] = "Mooney Viscosity"
            df_sheet1_csvMeasurements.loc[incrementValue,'is_about'] = row['SampleName']
            df_sheet1_csvMeasurements.loc[incrementValue+1,'has_specified_numeric_value'] = row["Mooney Stress Relaxation"]
            df_sheet1_csvMeasurements.loc[incrementValue+1,'is_quality_measurement_of'] = "Mooney Stress Relaxation"
            df_sheet1_csvMeasurements.loc[incrementValue+1,'is_about'] = row['SampleName']
            df_sheet1_csvMeasurements.loc[incrementValue+2,'has_specified_numeric_value'] = row["Styrene_Cont"]
            df_sheet1_csvMeasurements.loc[incrementValue+2,'is_quality_measurement_of'] = "Styrene_Cont"
            df_sheet1_csvMeasurements.loc[incrementValue+2,'is_about'] = row['SampleName']
            df_sheet1_csvMeasurements.loc[incrementValue+3,'has_specified_numeric_value'] = row["Cis_Cont"]
            df_sheet1_csvMeasurements.loc[incrementValue+3,'is_quality_measurement_of'] = "Cis_Cont"
            df_sheet1_csvMeasurements.loc[incrementValue+3,'is_about'] = row['SampleName']
            df_sheet1_csvMeasurements.loc[incrementValue+4,'has_specified_numeric_value'] = row["Trans_Cont"]
            df_sheet1_csvMeasurements.loc[incrementValue+4,'is_quality_measurement_of'] = "Trans_Cont"
            df_sheet1_csvMeasurements.loc[incrementValue+4,'is_about'] = row['SampleName']
            df_sheet1_csvMeasurements.loc[incrementValue+5,'has_specified_numeric_value'] = row["Vinyl_Cont"]
            df_sheet1_csvMeasurements.loc[incrementValue+5,'is_quality_measurement_of'] = "Vinyl_Cont"
            df_sheet1_csvMeasurements.loc[incrementValue+5,'is_about'] = row['SampleName']
            df_sheet1_csvMeasurements.loc[incrementValue+6,'has_specified_numeric_value'] = row["Glass Transition Temperature"]
            df_sheet1_csvMeasurements.loc[incrementValue+6,'is_quality_measurement_of'] = "Glass Transition Temperature"
            df_sheet1_csvMeasurements.loc[incrementValue+6,'is_about'] = row['SampleName']
            df_sheet1_csvMeasurements.loc[incrementValue+7,'has_specified_numeric_value'] = row["Number Averaged Molecular Weight"]
            df_sheet1_csvMeasurements.loc[incrementValue+7,'is_quality_measurement_of'] = "Number Averaged Molecular Weight"
            df_sheet1_csvMeasurements.loc[incrementValue+7,'is_about'] = row['SampleName']
            df_sheet1_csvMeasurements.loc[incrementValue+8,'has_specified_numeric_value'] = row["Weight Averaged Molecular Weight"]
            df_sheet1_csvMeasurements.loc[incrementValue+8,'is_quality_measurement_of'] = "Weight Averaged Molecular Weight"
            df_sheet1_csvMeasurements.loc[incrementValue+8,'is_about'] = row['SampleName']
            incrementValue = incrementValue+lenQualityNames
            ##else:
            ##print("incorrect row")
            ##print("Value of incrementValue before increment is ",incrementValue)
            ##incrementValue = incrementValue+lenQualityNames

        #print("--------------------------------------------")
        #print("Before wrting to csv file")
        #print(df_sheet1_csvMeasurements)
        #print("--------------------------------------------")


        # Get filename and extension separately
        inputFileName, ext = os.path.splitext(os.path.basename(dataFileExcel))

        # Get filename without extension
        inputFileNameOnly = inputFileName.split('.')[0]
        output_filename = generate_output_filename(inputFileNameOnly)
        output_filepath = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)

        df_sheet1_csvMeasurements.to_csv(output_filepath,sep=',', index=False, encoding='utf-8')

        #Rest of the sheets - transformation
        readIndivSheetsTransformToCSV(all_sheets,templateMeasurementCSVFile)

def validate_iri(iri):
    """
    Validate if the given IRI is correctly formatted.

    Args:
        iri (str): The IRI to validate.

    Returns:
        bool: True if the IRI is valid, False otherwise.
    """
    try:
        # Attempt to parse the IRI using urlparse
        result = urlparse(iri)

        # Check if the result has a valid scheme and netloc
        if result.scheme and result.netloc:
            return True
        else:
            return False
    except Exception as e:
        # If an exception occurs, return False as the IRI is invalid
        print(f"Invalid IRI: {iri} ({e})")
        return False

def validate_float(value):
    try:
        return float(value)
    except ValueError:
        warnings.warn(f"Warning: '{value}' is not a valid float. Returning None.")
        return None

# Regex for valid IRI
IRI_REGEX = re.compile(
    r"^[a-zA-Z][a-zA-Z0-9+.-]*:"  # Scheme
    r"(?://[^/?#]*)?"             # Authority
    r"[^?#]*"                     # Path
    r"(?:\?[^#]*)?"               # Query
    r"(?:#.*)?$"                  # Fragment
)

def is_valid_iri(iri):
    """
    Checks if a given IRI is valid using a regex.
    """
    if isinstance(iri, str):
        return bool(IRI_REGEX.match(iri))
    return False

def clean_and_validate_iri(iri):
    """
    Cleans and validates an IRI:
    - Removes leading/trailing spaces.
    - Replaces CRLF (\r\n) with LF (\n).
    - Returns None if invalid.
    """
    if not isinstance(iri, URIRef):
        return iri

    # Convert IRI to string and normalize line endings
    iri_str = str(iri).strip().replace("\r\n", "\n").replace("\r", "")

    # Validate the cleaned IRI
    if is_valid_iri(iri_str):
        return URIRef(iri_str)  # Return the cleaned IRI
    else:
        print(f"Invalid IRI detected and skipped: {iri}")
        return None  # Return None for invalid IRIs

from urllib.parse import urlparse, quote

def clean_and_validate_iri_safeIRI(iri):
    """
    Cleans and validates an IRI:
    - Strips whitespace.
    - Encodes special characters.
    - Ensures valid URIRef.
    """
    if not isinstance(iri, URIRef):
        return iri  # Skip non-IRI values

    iri_str = str(iri).strip()
    if not iri_str:
        return None

    parsed = urlparse(iri_str)
    if not all([parsed.scheme, parsed.netloc]):  # Check for valid scheme and authority
        print(f"Invalid IRI skipped: {iri}")
        return None

    # Encode special characters
    safe_iri = quote(iri_str, safe=':/#?&=@')
    return URIRef(safe_iri)  # Return as a valid URIRef



def process_ttl_file(input_path, output_path):
    """
    Processes a single RDF file to fix invalid IRIs.
    """
    graph = Graph()
    graph.parse(input_path, format="turtle")

    updated_graph = Graph()
    for subject, predicate, obj in graph:
        subject = clean_and_validate_iri_safeIRI(subject)
        predicate = clean_and_validate_iri_safeIRI(predicate)
        obj = clean_and_validate_iri_safeIRI(obj)

        if subject and predicate and obj:  # Only add valid triples
            updated_graph.add((subject, predicate, obj))

    updated_graph.serialize(destination=output_path, format="turtle")
    print(f"Processed: {input_path} -> {output_path}")

def recheckTtlFile(invalid_iri_path):
    graph = Graph()
    graph.parse(invalid_iri_path, format="turtle")
    invalid_iris = []
    for s, p, o in graph:
        for term in [s, p, o]:
            if isinstance(term, URIRef) and not stricter_iri_validation(term):
                invalid_iris.append(term)

    # If invalid IRIs exist, save them for debugging
    invalid_iris_path = "invalid_iris_log.txt"
    with open(invalid_iris_path, "w", encoding="utf-8") as log_file:
        for iri in invalid_iris:
            log_file.write(f"{iri}\n")

def process_directory(input_dir, output_dir):
    """
    Processes all TTL files in a directory to fix invalid IRIs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the directory only if it doesn't exist
    else:
        print(f"Output directory '{output_dir}' already exists. Skipping creation.")

    for filename in os.listdir(input_dir):
        if filename.endswith(".ttl"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_ttl_file(input_path, output_path)
            recheckTtlFile(output_path)

# Stricter IRI validation function for GraphDB
def stricter_iri_validation(iri):
    """
    Validates an IRI against stricter rules that might be enforced by GraphDB.
    Includes checks for:
    - Invalid characters.
    - Misuse of spaces or control characters.
    """
    if isinstance(iri, URIRef):
        iri_str = str(iri)
        # Check for forbidden characters and spaces in the IRI
        if any(c in iri_str for c in ' "<>{}|\\^`') or ' ' in iri_str:
            return False
        # Ensure the IRI does not have control characters
        if any(ord(c) < 32 for c in iri_str):
            return False
    return True


def readOntologyCSVAndBuildDataTriples(owlFile,templateMeasurementCSVFile,fileNameWithoutExt):
    ##onto = get_ontology(owlFile).load()
    ##print(onto)

    #g = Graph()
    #g.parse(owlFile)

    #for s, p, o in g:
    #print("Subject: ",s)

    df_csv=pd.read_csv(templateMeasurementCSVFile,sep=",",quotechar='"')
    #print(df_csv)

    #g = Graph()

    #Create prefix for namespaces
    pmd = Namespace('https://w3id.org/pmd/co/')
    obo = Namespace('http://purl.obolibrary.org/obo/')
    bfo = Namespace('http://purl.obolibrary.org/obo/bfo.owl/')
    obi = Namespace('http://purl.obolibrary.org/obo/obi.owl/')
    iao = Namespace('http://purl.obolibrary.org/obo/iao.owl/')
    isk = Namespace('https://tib.eu/ontologies/isk/')
    time = Namespace('http://www.w3.org/2006/time/')
    stato = Namespace('http://purl.obolibrary.org/stato.owl/')
    rdf = Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#')
    rdfs = Namespace('https://www.w3.org/TR/rdf11-schema/')

    #Replace nan with empty strings
    df_csv = df_csv.fillna('')

    #df_csv['is_quality_measurement_of'] = df_csv['is_quality_measurement_of'].fillna()
    df_csv['quality_URI'] = df_csv['is_quality_measurement_of'].str.replace(r'\s+', '_', regex=True).str.replace('"','doubleHash')
    df_csv['quality_URI'] = df_csv['quality_URI'].str.strip()
    #print(df_csv['quality_URI'])

    #Check if units have valid characters, else replace
    df_csv['has_measurement_unit_label'] = df_csv['has_measurement_unit_label'].str.replace('[°C]','Degree-Celcius').str.replace('%','percentage').str.replace('Shore A','ShoreA').str.replace('log(ME)/log(s)','log(ME)_per_log(s)')
    df_csv['has_measurement_unit_label'] = df_csv['has_measurement_unit_label'].str.strip()
    #print(df_csv['has_measurement_unit_label'])

    g = Graph()
    g.bind('bfo', bfo)
    g.bind('obi',obi)
    g.bind('iao',iao)

    for index, row in df_csv.iterrows():
        if(row['is_about']) != '':
            # Instances of pmd:Object
            g.add((URIRef(isk+row['is_about']), RDF.type, pmd.Object))
            g.add((URIRef(isk+row['is_about']),RDFS.label,Literal(row['is_about'], datatype=XSD.string)))

            if(row['quality_URI']) != '':
                # Instances of pmd:Quality
                g.add((URIRef(isk+row['quality_URI']), RDF.type, pmd.Quality))
                g.add((URIRef(pmd.Quality), RDFS.label,Literal("Quality", datatype=XSD.string)))
                g.add((URIRef(isk+row['quality_URI']), RDFS.label, Literal(row['is_quality_measurement_of'], datatype=XSD.string)))

                #Create instances of Scalar Measurement Datum
                g.add((URIRef(isk+row['is_about']+'_SMD_'+row['quality_URI']),RDF.type, iao.IAO_0000032))
                g.add((URIRef(isk+row['is_about']+'_SMD_'+row['quality_URI']),RDFS.label,Literal(row['is_about']+'_SMD_'+row['quality_URI'], datatype=XSD.string)))
                g.add((URIRef(iao.IAO_0000032), RDFS.label,Literal("iao: Scalar Measurement Datum", datatype=XSD.string)))

                # Relation between Scalar Measurement Datum instance and pmd:Object - is_about instance
                g.add((URIRef(isk+row['is_about']+'_SMD_'+row['quality_URI']),iao.IAO_0000136,URIRef(isk+row['is_about'])))
                g.add((iao.IAO_0000136,RDFS.label,Literal("iao:is about", datatype=XSD.string)))
                # Relation between Scalar Measurement Datum instance and pmd:Quality instance
                g.add((URIRef(isk+row['is_about']+'_SMD_'+row['quality_URI']),iao.IAO_0000221,URIRef(isk+row['quality_URI'])))
                g.add((iao.IAO_0000221,RDFS.label,Literal("iao:is quality measurement of", datatype=XSD.string)))
                # Relation between pmd:Quality instance and Scalar Measurement Datum instance
                g.add((URIRef(isk+row['quality_URI']),iao.IAO_0000417,(URIRef(isk+row['is_about']+'_SMD_'+row['quality_URI']))))
                g.add((iao.IAO_0000417,RDFS.label,Literal("iao:is quality measured as", datatype=XSD.string)))

                # Create instance of Scalar Value Specification
                validateFloat = row['has_specified_numeric_value']
                #print("Float value being validated ",validateFloat)
                #if row['has_specified_numeric_value'] != '':
                if validateFloat != '':
                    g.add((URIRef(isk+row['is_about']+'_SVS_'+row['quality_URI']),RDF.type, obi.OBI_0001931))
                    g.add((URIRef(isk+row['is_about']+'_SVS_'+row['quality_URI']),RDFS.label,Literal(row['is_about']+'_SVS_'+row['quality_URI'], datatype=XSD.string)))
                    g.add((URIRef(obi.OBI_0001931), RDFS.label,Literal("obi: Scalar Value Specification", datatype=XSD.string)))

                    # Create relation between Scalar value specification instance and pmd:Object instance - is_about
                    g.add((URIRef(isk+row['is_about']+'_SVS_'+row['quality_URI']),iao.IAO_0000136,URIRef(isk+row['is_about'])))
                    g.add((iao.IAO_0000136,RDFS.label,Literal("iao:is about", datatype=XSD.string)))
                    #g.add((iao.IAO_0000136,RDFS.label,Literal("iao:is about", datatype=XSD.string)))
                    # Create relation between Scalar value specification instance and pmd:Quality instance - specifies value of
                    g.add((URIRef(isk+row['is_about']+'_SVS_'+row['quality_URI']),obi.OBI_0001927,URIRef(isk+row['quality_URI'])))
                    g.add((obi.OBI_0001927,RDFS.label,Literal("obi:specifies value of", datatype=XSD.string)))
                    # Create relation between Scalar value specification instance and Literal value - has specified numeric value


                    floatVal = validate_float(validateFloat)
                    #print("Float value valid ",floatVal)
                    g.add((URIRef(isk+row['is_about']+'_SVS_'+row['quality_URI']),obi.OBI_0001937,Literal(floatVal, datatype=XSD.float)))
                    g.add((obi.OBI_0001937,RDFS.label,Literal("obi:has specified numeric value", datatype=XSD.string)))

                    #Relation between Scalar Measurement Datum and Scalar Value Specification
                    g.add((URIRef(isk+row['is_about']+'_SMD_'+row['quality_URI']),obi.OBI_0001938,URIRef(isk+row['is_about']+'_SVS_'+row['quality_URI'])))
                    g.add((obi.OBI_0001938,RDFS.label,Literal("obi:has value specification", datatype=XSD.string)))

                    #Add has quality relation between pmd:Object and pmd:Quality
                    g.add((URIRef(isk+row['is_about']),obo.RO_0000086,(URIRef(isk+row['quality_URI']))))
                    g.add((obo.RO_0000086,RDFS.label,Literal("ro:has quality", datatype=XSD.string)))

                    #Create Measurement Unit Label Instances
                    if(row['has_measurement_unit_label']) != '':
                        #print(row['has_measurement_unit_label'])
                        g.add((URIRef(isk+row['has_measurement_unit_label']), RDF.type, iao.IAO_0000003))
                        g.add((URIRef(isk+row['has_measurement_unit_label']),RDFS.label,Literal(row['has_measurement_unit_label'], datatype=XSD.string)))

                        #Add relation between Scalar Value Specification and Measurement Unit Label instances
                        g.add((URIRef(isk+row['is_about']+'_SVS_'+row['quality_URI']),iao.IAO_0000039,(URIRef(isk+row['has_measurement_unit_label']))))
                        g.add((iao.IAO_0000039,RDFS.label,Literal("iao:has measurement unit label", datatype=XSD.string)))
                        #else:
                        #print("Skipping the Scalar Value Specification triple creation - found empty values in file ",templateMeasurementCSVFile, "at row : ", row)
                # empty value for specified numeric value field
                else:
                    print("________________________________")
                    print("Numeric value missing in sheet ",templateMeasurementCSVFile)
                    print(row)
                    print("Please check, triples not created for row where value is empty")
                    print("________________________________")

    #Validate URI
    for subj, pred, obj in g:
        # Check if subject, predicate, or object is an IRI and validate it
        if isinstance(subj, rdflib.URIRef) and not validate_iri(str(subj)):
            print(f"Invalid IRI in subject: {subj}")

        if isinstance(pred, rdflib.URIRef) and not validate_iri(str(pred)):
            print(f"Invalid IRI in predicate: {pred}")

        if isinstance(obj, rdflib.URIRef) and not validate_iri(str(obj)):
            print(f"Invalid IRI in object: {obj}")



    output_filename = generate_output_filename_ttl(fileNameWithoutExt)
    #print(output_filename)
    #output_filepath = os.path.join(app.config['OUTPUT_GRAPH_FOLDER'], output_filename)
    outputFilePath = current_app.config['OUTPUT_GRAPH_FOLDER']+"//"+output_filename+".ttl"
    ##g.serialize('output/scalarmeasurement_PolymerBasicProperties_RawArlanxeo.ttl',format='turtle',encoding='UTF-8')
    g.serialize(destination=outputFilePath,format="turtle", encoding="utf-8")

def upload_to_graphdb(repo_url, ttl_file_path, username=None, password=None):
    """
    Securely uploads a TTL file to a specified GraphDB repository.

    Args:
        repo_url (str): The base URL of the GraphDB repository.
        ttl_file_path (str): Path to the TTL file to upload.
        username (str, optional): Username for GraphDB authentication.
        password (str, optional): Password for GraphDB authentication.

    Raises:
        Exception: If the upload fails.
    """
    with open(ttl_file_path, 'rb') as ttl_file:
        headers = {'Content-Type': 'application/x-turtle'}
        auth = (username, password) if username and password else None
        response = requests.post(
            f"{repo_url}/statements",
            data=ttl_file,
            headers=headers,
            auth=auth
        )
    if response.status_code != 204:
        raise Exception(
            f"Failed to upload {ttl_file_path}. Status code: {response.status_code}, Response: {response.text}"
        )
    print(f"Securely uploaded {ttl_file_path} to GraphDB.")


import requests
from urllib.parse import urlencode

def run_sparql_query(repository_url, query, username=None, password=None):
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    auth = (username, password) if username and password else None
    data = {"query": query}  # Send as form-encoded data

    print(f"Sending request to: {repository_url}")
    print(f"Headers: {headers}")
    print(f"Query: {query}")

    try:
        response = requests.post(repository_url, headers=headers, data=data, auth=auth)

        print(f"Response Code: {response.status_code}")
        print(f"Response Content: {response.text}")

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"SPARQL query failed: HTTP {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"An error occurred while running the SPARQL query: {e}")
