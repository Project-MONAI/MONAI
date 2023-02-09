# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np

EXPECTED_ANSWERS = [
    {  # test answers for PyTorch 1.12.1
        "integration_classification_2d": {
            "losses": [0.776835828070428, 0.1615355300011149, 0.07492854832938523, 0.04591309238865877],
            "best_metric": 0.9999184380485994,
            "infer_prop": [1029, 896, 980, 1033, 961, 1046],
        },
        "integration_segmentation_3d": {
            "losses": [
                0.5428894340991974,
                0.47331981360912323,
                0.4482289582490921,
                0.4452722787857056,
                0.4289989799261093,
                0.4359133839607239,
            ],
            "best_metric": 0.933259129524231,
            "infer_metric": 0.9332860708236694,
            "output_sums": [
                0.142167581604417,
                0.15195543400875847,
                0.1512754523215521,
                0.13962938779108452,
                0.18835719348918614,
                0.16943498693483486,
                0.1465709827477569,
                0.16806483607477135,
                0.1568844609697224,
                0.17911090857818554,
                0.16252098157181355,
                0.16806016936625395,
                0.14430124467305516,
                0.11316135548315168,
                0.16183771025615476,
                0.2009426314066978,
                0.1760258010156966,
                0.09700864497950844,
                0.1938495370314683,
                0.20319147575335647,
                0.19629641404249798,
                0.20852344793102826,
                0.16185073630020633,
                0.13184196857669161,
                0.1480959525354053,
                0.14232924377085415,
                0.23177739882790951,
                0.16094610375534632,
                0.14832771888168225,
                0.10259365443625812,
                0.11850632233099603,
                0.1294100326098242,
                0.11364228279017609,
                0.15181947897584674,
                0.16319358155815072,
                0.1940284526521386,
                0.22306137879066443,
                0.18083137638759522,
                0.1903135237574692,
                0.07402317520619131,
            ],
        },
        "integration_workflows": {
            "best_metric": 0.9219646483659745,
            "infer_metric": 0.921751058101654,
            "output_sums": [
                0.14183664321899414,
                0.1513957977294922,
                0.13804054260253906,
                0.13356828689575195,
                0.18456125259399414,
                0.16363763809204102,
                0.14090299606323242,
                0.16649389266967773,
                0.15651893615722656,
                0.17655134201049805,
                0.16116666793823242,
                0.1644763946533203,
                0.14383649826049805,
                0.11055326461791992,
                0.16080379486083984,
                0.19629907608032227,
                0.17441415786743164,
                0.053577423095703125,
                0.19043779373168945,
                0.19904804229736328,
                0.19526052474975586,
                0.20304107666015625,
                0.16030025482177734,
                0.13170623779296875,
                0.15118932723999023,
                0.13686418533325195,
                0.22668886184692383,
                0.1611471176147461,
                0.1472463607788086,
                0.10427379608154297,
                0.11962461471557617,
                0.1305704116821289,
                0.11204910278320312,
                0.15171337127685547,
                0.15962505340576172,
                0.18976259231567383,
                0.21649456024169922,
                0.17761802673339844,
                0.18516874313354492,
                0.03636503219604492,
            ],
            "best_metric_2": 0.9219559609889985,
            "infer_metric_2": 0.9217371672391892,
            "output_sums_2": [
                0.14187288284301758,
                0.15140819549560547,
                0.13802719116210938,
                0.1335887908935547,
                0.18454980850219727,
                0.1636652946472168,
                0.14091157913208008,
                0.16653108596801758,
                0.15651702880859375,
                0.17658615112304688,
                0.1611957550048828,
                0.16448307037353516,
                0.14385128021240234,
                0.1105203628540039,
                0.16085100173950195,
                0.19626951217651367,
                0.17442035675048828,
                0.053586483001708984,
                0.19042730331420898,
                0.1990523338317871,
                0.1952815055847168,
                0.20303773880004883,
                0.16034317016601562,
                0.13172531127929688,
                0.15118741989135742,
                0.1368694305419922,
                0.22667837142944336,
                0.16119050979614258,
                0.14726591110229492,
                0.10426473617553711,
                0.11961841583251953,
                0.13054800033569336,
                0.11203193664550781,
                0.15172529220581055,
                0.15963029861450195,
                0.18975019454956055,
                0.21646499633789062,
                0.17763566970825195,
                0.18517112731933594,
                0.03638744354248047,
            ],
        },
    },
    {  # test answers for PyTorch 1.8
        "integration_classification_2d": {
            "losses": [0.777176220515731, 0.16019743723664315, 0.07480076164197011, 0.045643698364780966],
            "best_metric": 0.9999418774120775,
            "infer_prop": [1030, 897, 980, 1033, 960, 1048],
        },
        "integration_segmentation_3d": {
            "losses": [
                0.5326887160539627,
                0.4685510128736496,
                0.46245276033878324,
                0.4411882758140564,
                0.4198471873998642,
                0.43021280467510226,
            ],
            "best_metric": 0.931993305683136,
            "infer_metric": 0.9326668977737427,
            "output_sums": [
                0.1418775228871769,
                0.15188869120317386,
                0.15140863737688195,
                0.1396146850007127,
                0.18784343811575696,
                0.16909487431163164,
                0.14649608249452073,
                0.1677767130878611,
                0.1568122289811143,
                0.17874181729735056,
                0.16213703658980205,
                0.16754335171970686,
                0.14444824920997243,
                0.11432402622850306,
                0.16143210936221247,
                0.20055289634107482,
                0.17543571757219317,
                0.09920729163334538,
                0.19297325815057875,
                0.2023200127892273,
                0.1956677579845722,
                0.20774045016425718,
                0.16193278944159428,
                0.13174198906539808,
                0.14830508550670007,
                0.14241105864278342,
                0.23090631643085724,
                0.16056153813499532,
                0.1480353269419819,
                0.10318719171632634,
                0.11867462580989198,
                0.12997011485830187,
                0.11401220332210203,
                0.15242746700662088,
                0.1628489107974574,
                0.19327235354175412,
                0.22184902863377548,
                0.18028049625972334,
                0.18958059106892552,
                0.07884601267057013,
            ],
        },
        "integration_workflows": {
            "best_metric": 0.9217087924480438,
            "infer_metric": 0.9214379042387009,
            "output_sums": [
                0.14209461212158203,
                0.15126705169677734,
                0.13800382614135742,
                0.1338181495666504,
                0.1850571632385254,
                0.16372442245483398,
                0.14059066772460938,
                0.16674423217773438,
                0.15653657913208008,
                0.17690563201904297,
                0.16154909133911133,
                0.16521310806274414,
                0.14388608932495117,
                0.1103353500366211,
                0.1609959602355957,
                0.1967010498046875,
                0.1746964454650879,
                0.05329275131225586,
                0.19098854064941406,
                0.19976520538330078,
                0.19576644897460938,
                0.20346736907958984,
                0.1601848602294922,
                0.1316051483154297,
                0.1511220932006836,
                0.13670969009399414,
                0.2276287078857422,
                0.1611800193786621,
                0.14751672744750977,
                0.10413789749145508,
                0.11944007873535156,
                0.1305546760559082,
                0.11204719543457031,
                0.15145111083984375,
                0.16007614135742188,
                0.1904129981994629,
                0.21741962432861328,
                0.17812013626098633,
                0.18587207794189453,
                0.03605222702026367,
            ],
            "best_metric_2": 0.9210659921169281,
            "infer_metric_2": 0.9208109736442566,
            "output_sums_2": [
                0.14227628707885742,
                0.1515035629272461,
                0.13819408416748047,
                0.13402271270751953,
                0.18525266647338867,
                0.16388607025146484,
                0.14076614379882812,
                0.16694307327270508,
                0.15677356719970703,
                0.1771831512451172,
                0.16172313690185547,
                0.1653728485107422,
                0.14413118362426758,
                0.11057281494140625,
                0.16121912002563477,
                0.19680166244506836,
                0.1748638153076172,
                0.053426265716552734,
                0.19117307662963867,
                0.19996356964111328,
                0.1959366798400879,
                0.20363712310791016,
                0.16037797927856445,
                0.13180780410766602,
                0.1513657569885254,
                0.13686084747314453,
                0.2277364730834961,
                0.16137409210205078,
                0.1476879119873047,
                0.10438394546508789,
                0.11967992782592773,
                0.13080739974975586,
                0.11226606369018555,
                0.15168476104736328,
                0.1602616310119629,
                0.190582275390625,
                0.21756458282470703,
                0.17825984954833984,
                0.18604803085327148,
                0.036206722259521484,
            ],
        },
    },
    {  # test answers for PyTorch 21.04, cuda 11.3
        "integration_classification_2d": {
            "losses": [0.7772567988770782, 0.16357883198815545, 0.0748426011840629, 0.045560025710873545],
            "best_metric": 0.9999362036681547,
            "infer_prop": [1030, 898, 981, 1033, 960, 1046],
        },
        "integration_segmentation_3d": {
            "losses": [
                0.5462346076965332,
                0.4699550330638885,
                0.4407052755355835,
                0.4473582059144974,
                0.4345871120691299,
                0.4268435090780258,
            ],
            "best_metric": 0.9325245052576066,
            "infer_metric": 0.9326683700084686,
            "output_sums": [
                0.14224469870198278,
                0.15221021012369151,
                0.15124158255724182,
                0.13988812880932433,
                0.18869885039284465,
                0.16944664085835437,
                0.14679946398855015,
                0.1681337815374021,
                0.1572538225010156,
                0.179386563044054,
                0.162734465243387,
                0.16831902111202945,
                0.1447043535420074,
                0.11343210557896033,
                0.16199135405262954,
                0.20095180481987404,
                0.17613484080473857,
                0.09717457016552708,
                0.1940439758638305,
                0.2033698355271389,
                0.19628583555443793,
                0.20852096425983455,
                0.16202004771083997,
                0.13206408917949392,
                0.14840973098125526,
                0.14237425379050472,
                0.23165483128059614,
                0.16098621485325398,
                0.14831028015056963,
                0.10317099380415945,
                0.118716576251689,
                0.13002315213569166,
                0.11436407827087304,
                0.1522274707636008,
                0.16314910792851098,
                0.1941135852761834,
                0.22309890968242424,
                0.18111804948625987,
                0.19043976068601465,
                0.07442812452084423,
            ],
        },
    },
    {  # test answers for PyTorch 1.9
        "integration_workflows": {
            "output_sums_2": [
                0.14213180541992188,
                0.15153264999389648,
                0.13801145553588867,
                0.1338348388671875,
                0.18515968322753906,
                0.16404008865356445,
                0.14110612869262695,
                0.16686391830444336,
                0.15673542022705078,
                0.1772594451904297,
                0.16174745559692383,
                0.16518878936767578,
                0.1440296173095703,
                0.11033201217651367,
                0.1611781120300293,
                0.19660568237304688,
                0.17468547821044922,
                0.053053855895996094,
                0.1909656524658203,
                0.19952869415283203,
                0.1957845687866211,
                0.2034916877746582,
                0.16042661666870117,
                0.13193607330322266,
                0.15104389190673828,
                0.13695430755615234,
                0.22720861434936523,
                0.16157913208007812,
                0.14759159088134766,
                0.10379791259765625,
                0.11937189102172852,
                0.1306462287902832,
                0.11205482482910156,
                0.15182113647460938,
                0.16006708145141602,
                0.19011592864990234,
                0.21713829040527344,
                0.17794132232666016,
                0.18584394454956055,
                0.03577899932861328,
            ]
        },
        "integration_segmentation_3d": {  # for the mixed readers
            "losses": [
                0.5645154356956482,
                0.4984356611967087,
                0.472334086894989,
                0.47419720590114595,
                0.45881829261779783,
                0.43097741305828097,
            ],
            "best_metric": 0.9325698614120483,
            "infer_metric": 0.9326590299606323,
        },
    },
    {  # test answers for PyTorch 1.13
        "integration_workflows": {
            "output_sums_2": [
                0.14264830205979873,
                0.15264129328718357,
                0.1519652511118344,
                0.14003114557361543,
                0.18870416611118465,
                0.1699260498246968,
                0.14727475398203582,
                0.16870874483246967,
                0.15757932277023196,
                0.1797779694564011,
                0.16310501082450635,
                0.16850569170136015,
                0.14472958359864832,
                0.11402527744419455,
                0.16217657428257873,
                0.20135486560244975,
                0.17627557567092866,
                0.09802074024435596,
                0.19418729084978026,
                0.20339278025379662,
                0.1966174446916041,
                0.20872528599049203,
                0.16246183433492764,
                0.1323750751202327,
                0.14830347036335728,
                0.14300732028781024,
                0.23163101813922762,
                0.1612925258625139,
                0.1489573676973957,
                0.10299491921717041,
                0.11921404797064328,
                0.1300212751422368,
                0.11437829790254125,
                0.1524755276727056,
                0.16350584736767904,
                0.19424317961257148,
                0.2229762916892286,
                0.18121074825540173,
                0.19064286213535897,
                0.0747544243069024,
            ]
        },
        "integration_segmentation_3d": {  # for the mixed readers
            "losses": [
                0.5451162219047546,
                0.4709601759910583,
                0.45201429128646853,
                0.4443251401185989,
                0.4341257899999619,
                0.4350819975137711,
            ],
            "best_metric": 0.9316844940185547,
            "infer_metric": 0.9316383600234985,
        },
    },
    {  # test answers for PyTorch 21.10
        "integration_classification_2d": {
            "losses": [0.7806222991199251, 0.16259610306495315, 0.07529311385124353, 0.04640352608529246],
            "best_metric": 0.9999369155431564,
            "infer_prop": [1030, 898, 981, 1033, 960, 1046],
        },
        "integration_segmentation_3d": {
            "losses": [
                0.5373653262853623,
                0.46776085495948794,
                0.4422474503517151,
                0.43667820692062376,
                0.42639826238155365,
                0.4158218264579773,
            ],
            "best_metric": 0.9357852935791016,
            "infer_metric": 0.9358890652656555,
            "output_sums": [
                0.14134230251963734,
                0.15126225587084188,
                0.15033401825118003,
                0.1389259850822021,
                0.18754569424488515,
                0.16839386100677756,
                0.14565994645049316,
                0.1673404545700305,
                0.1561511946878991,
                0.17825771988631423,
                0.1616607574532002,
                0.16742913895628342,
                0.14354699757474138,
                0.11215070364672397,
                0.16112241518382064,
                0.20001951769273596,
                0.17526580315958823,
                0.09564779134319003,
                0.19300729425711433,
                0.20226013883846938,
                0.1952803784613225,
                0.207563783379273,
                0.16082750039188845,
                0.13121728634981528,
                0.14741783973523187,
                0.14157844891046317,
                0.23102353186599955,
                0.15982195501286317,
                0.14750224809851548,
                0.10177519678431225,
                0.11784387764466563,
                0.12852018780730834,
                0.11300143976680752,
                0.1508621728586496,
                0.1623522601916851,
                0.19320168095077178,
                0.222086024709285,
                0.1800784736260849,
                0.18942329376838685,
                0.07354564965439693,
            ],
        },
        "integration_workflows": {
            "output_sums": [
                0.14211511611938477,
                0.1516571044921875,
                0.1381092071533203,
                0.13403034210205078,
                0.18480682373046875,
                0.16382598876953125,
                0.14140796661376953,
                0.1665945053100586,
                0.15700864791870117,
                0.17697620391845703,
                0.16163396835327148,
                0.16488313674926758,
                0.1442713737487793,
                0.11060476303100586,
                0.16111087799072266,
                0.19617986679077148,
                0.1744403839111328,
                0.052786827087402344,
                0.19046974182128906,
                0.19913578033447266,
                0.19527721405029297,
                0.2032318115234375,
                0.16050148010253906,
                0.13228464126586914,
                0.1512293815612793,
                0.1372208595275879,
                0.22692251205444336,
                0.16164922714233398,
                0.14729642868041992,
                0.10398292541503906,
                0.1195836067199707,
                0.13096046447753906,
                0.11221647262573242,
                0.1521167755126953,
                0.1599421501159668,
                0.1898345947265625,
                0.21675777435302734,
                0.1777491569519043,
                0.18526840209960938,
                0.035144805908203125,
            ],
            "output_sums_2": [
                0.14200592041015625,
                0.15146303176879883,
                0.13796186447143555,
                0.1339101791381836,
                0.18489742279052734,
                0.1637406349182129,
                0.14113903045654297,
                0.16657161712646484,
                0.15676355361938477,
                0.17683839797973633,
                0.1614980697631836,
                0.16493558883666992,
                0.14408016204833984,
                0.11035394668579102,
                0.1610560417175293,
                0.1962742805480957,
                0.17439842224121094,
                0.05285835266113281,
                0.19057941436767578,
                0.19914865493774414,
                0.19533538818359375,
                0.20333576202392578,
                0.16032838821411133,
                0.13197898864746094,
                0.1510462760925293,
                0.13703680038452148,
                0.2270984649658203,
                0.16144943237304688,
                0.1472611427307129,
                0.10393238067626953,
                0.11940813064575195,
                0.1307811737060547,
                0.11203241348266602,
                0.15186500549316406,
                0.15992307662963867,
                0.18991422653198242,
                0.21689796447753906,
                0.1777033805847168,
                0.18547868728637695,
                0.035192012786865234,
            ],
        },
    },
]


def test_integration_value(test_name, key, data, rtol=1e-2):
    for idx, expected in enumerate(EXPECTED_ANSWERS):
        if test_name not in expected:
            continue
        if key not in expected[test_name]:
            continue
        value = expected[test_name][key]
        if np.allclose(data, value, rtol=rtol):
            print(f"matched {idx} result of {test_name}, {key}, {rtol}.")
            return True
    raise ValueError(f"no matched results for {test_name}, {key}. {data}.")
