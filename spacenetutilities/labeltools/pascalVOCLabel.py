import os
import numpy as np
from PIL import Image
from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom
import geopandas as gpd
import shapely
import json
import rasterio
from rasterio import features
from spacenetutilities.labeltools import coreLabelTools as clT
from fiona.errors import DriverError
from fiona._err import CPLE_OpenFailedError  # old versions of fiona threw this error instead

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def writePacalVocObject(objectDict, top):

    childObject = SubElement(top, 'object')
    SubElement(childObject, 'name').text = objectDict['objectType']
    SubElement(childObject, 'pose').text = objectDict['pose']
    SubElement(childObject, 'truncated').text = str(objectDict['truncated'])
    SubElement(childObject, 'difficult').text = str(objectDict['difficult'])
    # write bounding box
    childBoundBox = SubElement(childObject, 'bndbox')
    SubElement(childBoundBox, 'xmin').text = str(objectDict['bndbox']['xmin'])
    SubElement(childBoundBox, 'ymin').text = str(objectDict['bndbox']['ymin'])
    SubElement(childBoundBox, 'xmax').text = str(objectDict['bndbox']['xmax'])
    SubElement(childBoundBox, 'ymax').text = str(objectDict['bndbox']['ymax'])

    return top

def writePascalVocHeader(imageDescriptionDict, top):
    ## write header


    childFolder = SubElement(top, 'folder')
    childFolder.text = imageDescriptionDict['folder']
    childFilename = SubElement(top, 'filename')
    childFilename.text = imageDescriptionDict['filename']

    # write source block
    childSource = SubElement(top, 'source')
    SubElement(childSource, 'database').text = imageDescriptionDict['source']['database']
    SubElement(childSource, 'annotation').text = imageDescriptionDict['source']['annotation']

    # write size block
    childSize = SubElement(top, 'size')
    SubElement(childSize, 'width').text = str(imageDescriptionDict['size']['width'])
    SubElement(childSize, 'height').text = str(imageDescriptionDict['size']['height'])
    SubElement(childSize, 'depth').text = str(imageDescriptionDict['size']['depth'])

    SubElement(top, 'segmented').text = str(imageDescriptionDict['segmented'])

    return top

def writeToPascalVOCLabel(xmlFilename, imageDescriptionDict, objectDictList):



    top = Element('annotation')

    top = writePascalVocHeader(imageDescriptionDict, top)

    for objectDict in objectDictList:
        top = writePacalVocObject(objectDict, top)


    with open(xmlFilename, 'w') as f:
        f.write(prettify(top))


    return xmlFilename

def geoJsonToPASCALVOC2012Label(xmlFileName, geoJson, rasterImageName, im_id='',
                           dataset ='SpaceNet',
                           folder_name='spacenet',
                           annotationStyle = 'PASCAL VOC2012',
                           segment=True,
                           bufferSizePix=2.5,
                           convertTo8Bit=True,
                           outputPixType='Byte',
                           outputFormat='GTiff',
                           bboxResize=1.0,
                           objectType='building',
                           objectTypeField='',
                           truncatePercent=0,
                           truncatedPercentField='partialDec'):

    imageDescriptionDict, objectDictList = clT.geoJsontoDict(geoJson, rasterImageName, datasetName=dataset,
                  annotationStyle=annotationStyle,
                  bboxResize=bboxResize,
                  objectType=objectType,
                  objectTypeField=objectTypeField,
                  objectPose='Left',
                  objectTruncatedField='',
                  objectDifficultyField='',
                                                             truncatePercent=truncatePercent,
                                                             truncatedPercentField=truncatedPercentField
                  )

    xmlFileName = writeToPascalVOCLabel(xmlFileName, imageDescriptionDict, objectDictList)

    return xmlFileName


def geoJsonToPASCALVOC2012SegmentCls(geoJson, src_meta, bufferSizePix=2.5,
                                  innerShapeValue=100,
                                  borderValue=255
                                  ):

    #TODO Implement multi object class segmentation

    bufferDist = bufferSizePix*src_meta['transform'].a
    if isinstance(geoJson, list):
        geoJson = geoJson[0]
    try:
        source_layer = gpd.read_file(geoJson)
        outerShapes = list((geom, borderValue) for geom in source_layer.geometry.buffer(bufferDist) if not geom.is_empty)
        innerShapes = list((geom, innerShapeValue) for geom in source_layer.geometry.buffer(-bufferDist) if not geom.is_empty)
    except (DriverError, CPLE_OpenFailedError):
        #empty_geojson = '{"type": "FeatureCollection", "features": []}'
        #with open('__empty.geojson', 'w') as f:
        #    f.write(empty_geojson)
        #source_layer = gpd.read_file('__empty.geojson')
        outerShapes = list()
        innerShapes = list()
    if len(outerShapes) > 0:
        outerShapesImage = features.rasterize(outerShapes,
                                   out_shape=(src_meta['width'], src_meta['height']),
                                   transform=src_meta['transform'])
    else:
        outerShapesImage = np.ones((src_meta['width'], src_meta['height']))
    if len(innerShapes) > 0:
        innerShapesImage = features.rasterize(innerShapes,
                                       out_shape=(src_meta['width'], src_meta['height']),
                                       transform=src_meta['transform'])
    else:
        innerShapesImage = np.ones((src_meta['width'], src_meta['height']))

    totalImage = outerShapesImage + innerShapesImage
    # set interior value to be innerValue
    totalImage[totalImage==innerShapeValue+borderValue]=innerShapeValue

    return totalImage


def geoJsonToPASCALVOC2012SegmentObj(geoJson, src_meta, bufferSizePix=2.5,
                                  innerShapeValue=100,
                                  borderValue=255
                                  ):
    # TODO Implement multi object class segmentation

    bufferDist = bufferSizePix * src_meta['transform'].a
    if isinstance(geoJson, list):
        geoJson = geoJson[0]
    try:
        source_layer = gpd.read_file(geoJson)
        outerShapes = list((geom, borderValue) for geom in source_layer.geometry.buffer(bufferDist) if not geom.is_empty)
        innerShapes = list((geom, value) for value, geom in enumerate(source_layer.geometry.buffer(-bufferDist)) if not geom.is_empty)
    except (DriverError, CPLE_OpenFailedError):
        #empty_geojson = '{"type": "FeatureCollection", "features": []}'
        #with open('__empty.geojson', 'w') as f:
        #    f.write(empty_geojson)
        #source_layer = gpd.read_file('__empty.geojson')
        outerShapes = list()
        innerShapes = list()
    if len(outerShapes) > 0:
        outerShapesImage = features.rasterize(outerShapes,
                                   out_shape=(src_meta['width'], src_meta['height']),
                                   transform=src_meta['transform'])
    else:
        outerShapesImage = np.ones((src_meta['width'], src_meta['height']))
    #if len(innerShapes) > 0:
    #    innerShapesImage = features.rasterize(innerShapes,
    #                                   out_shape=(src_meta['width'], src_meta['height']),
    #                                   transform=src_meta['transform'])
    #else:
    #    innerShapesImage = np.zeros((src_meta['width'], src_meta['height']))

    totalImage = outerShapesImage #+ innerShapesImage
    # set interior value to be innerValue
    totalImage[totalImage > 255] = totalImage[totalImage>255]-borderValue

    return totalImage

def geoJsonToPASCALVOC2012(xmlFileName, geoJson, rasterImageName, im_id='',
                           dataset ='SpaceNet',
                           folder_name='spacenet',
                           annotationStyle = 'PASCAL VOC2012',
                           segment=True,
                           bufferSizePix=2.5,
                           convertTo8Bit=True,
                           outputPixType='Byte',
                           outputFormat='GTiff',
                           bboxResize=1.0,
                           objectType='building',
                           objectTypeField='',
                           clsPNGName='',
                           objPNGName='',
                           outputRasterName='',
                           truncatePercent=0,
                           truncatedPercentField='partialDec'
                           ):



    geoJsonToPASCALVOC2012Label(xmlFileName, geoJson, rasterImageName, im_id='',
                                dataset=dataset,
                                folder_name=folder_name,
                                annotationStyle=annotationStyle,
                                segment=segment,
                                bufferSizePix=bufferSizePix,
                                convertTo8Bit=convertTo8Bit,
                                outputPixType=outputPixType,
                                outputFormat=outputFormat,
                                bboxResize=bboxResize,
                                objectType=objectType,
                                objectTypeField=objectTypeField,
                                truncatePercent=truncatePercent,
                                truncatedPercentField=truncatedPercentField
                                )

    with rasterio.open(rasterImageName) as src:
        src_meta = src.meta.copy()
        src_profile = src.profile.copy()


    if outputFormat == 'JPEG':
        extension = 'jpg'
    else:
        extension = 'tif'
    #Write Segemeentation Images
    if segment:
        if clsPNGName=='':
            clsPNGName = xmlFileName.replace('.xml', 'segcls.{}'.format(extension))

        clsImageArray = geoJsonToPASCALVOC2012SegmentCls(geoJson, src_meta, bufferSizePix=2.5,
                                                         innerShapeValue=100,
                                                         borderValue=255
                                                         )

        clsImageArray = Image.fromarray(clsImageArray)
        if clsImageArray.mode != 'RGB' and extension == 'jpg':
            clsImageArray = clsImageArray.convert('RGB')
        clsImageArray.save(clsPNGName)

        if objPNGName=='':
            objPNGName = xmlFileName.replace('.xml', 'segobj.{}'.format(extension))

        objImageArray = geoJsonToPASCALVOC2012SegmentObj(geoJson, src_meta, bufferSizePix=2.5,
                                                         innerShapeValue=100,
                                                         borderValue=255
                                                         )
        objImageArray = Image.fromarray(objImageArray)
        if objImageArray.mode != 'RGB' and extension == 'jpg':
            objImageArray = objImageArray.convert('RGB')
        objImageArray.save(objPNGName)

    if convertTo8Bit:
        if outputRasterName=='':
            if outputFormat == 'JPEG':
                outputImageName = xmlFileName.replace('.xml', '_8bit.jpg')
            else:
                outputImageName = xmlFileName.replace('.xml', '_8bit.tif')

            clT.convertGTiffTo8Bit(rasterImageName, outputImageName, outputFormat=outputFormat)
    else:
        outputImageName = rasterImageName

    entry = {'rasterFileName': outputImageName,
                 'geoJsonFileName': geoJson,
                 'annotationName': xmlFileName,
                 'width': src_meta['width'],
                 'height': src_meta['height'],
                 'depth' : src_meta['count'],
                 'src_meta': src_meta,
                 'basename': os.path.splitext(os.path.basename(rasterImageName))[0]
                 }

    return entry