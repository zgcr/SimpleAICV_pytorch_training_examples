import os
import cv2
import collections
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

# 919 class RGB color
ImageNet_S_CLASSES_COLOR = [
    (172, 47, 117),
    (192, 67, 251),
    (195, 103, 9),
    (211, 21, 242),
    (36, 87, 70),
    (216, 88, 140),
    (58, 193, 230),
    (39, 87, 174),
    (88, 81, 165),
    (25, 77, 72),
    (9, 148, 115),
    (208, 243, 197),
    (254, 79, 175),
    (192, 82, 99),
    (216, 177, 243),
    (29, 147, 147),
    (142, 167, 32),
    (193, 9, 185),
    (127, 32, 31),
    (202, 244, 151),
    (163, 254, 203),
    (114, 183, 28),
    (34, 128, 128),
    (164, 53, 133),
    (38, 232, 244),
    (17, 79, 132),
    (105, 42, 186),
    (31, 120, 1),
    (65, 231, 169),
    (57, 35, 102),
    (119, 11, 174),
    (82, 91, 128),
    (142, 99, 53),
    (140, 121, 170),
    (84, 203, 68),
    (6, 196, 47),
    (127, 244, 131),
    (204, 100, 180),
    (232, 78, 143),
    (148, 227, 186),
    (23, 207, 141),
    (117, 85, 48),
    (49, 69, 169),
    (163, 192, 95),
    (197, 94, 0),
    (113, 178, 36),
    (162, 48, 93),
    (131, 98, 42),
    (205, 112, 231),
    (149, 201, 127),
    (0, 138, 114),
    (43, 186, 127),
    (23, 187, 130),
    (121, 98, 62),
    (163, 222, 123),
    (195, 82, 174),
    (227, 148, 209),
    (50, 155, 14),
    (41, 58, 193),
    (36, 10, 86),
    (43, 104, 11),
    (2, 51, 80),
    (32, 182, 128),
    (38, 19, 174),
    (42, 115, 184),
    (188, 232, 77),
    (30, 24, 125),
    (2, 3, 94),
    (226, 107, 13),
    (112, 40, 72),
    (19, 95, 72),
    (154, 194, 248),
    (180, 67, 236),
    (61, 14, 96),
    (4, 195, 237),
    (139, 252, 86),
    (205, 121, 109),
    (75, 184, 16),
    (152, 157, 149),
    (110, 25, 208),
    (188, 121, 118),
    (117, 189, 83),
    (161, 104, 160),
    (228, 251, 251),
    (121, 70, 213),
    (31, 13, 71),
    (184, 152, 79),
    (41, 18, 40),
    (182, 207, 11),
    (166, 111, 93),
    (249, 129, 223),
    (118, 44, 216),
    (125, 24, 67),
    (210, 239, 3),
    (234, 204, 230),
    (35, 214, 254),
    (189, 197, 215),
    (43, 32, 11),
    (104, 212, 138),
    (182, 235, 165),
    (125, 156, 111),
    (232, 2, 27),
    (211, 217, 151),
    (53, 51, 174),
    (148, 181, 29),
    (67, 35, 39),
    (137, 73, 41),
    (151, 131, 46),
    (218, 178, 108),
    (3, 31, 9),
    (138, 27, 173),
    (199, 167, 61),
    (85, 97, 44),
    (34, 162, 88),
    (33, 133, 232),
    (255, 36, 0),
    (203, 34, 197),
    (126, 181, 254),
    (80, 190, 136),
    (189, 129, 209),
    (112, 35, 120),
    (91, 168, 116),
    (36, 176, 25),
    (67, 103, 252),
    (35, 114, 30),
    (29, 241, 33),
    (146, 17, 221),
    (84, 253, 2),
    (69, 101, 140),
    (44, 117, 253),
    (66, 111, 91),
    (85, 167, 39),
    (203, 150, 158),
    (145, 198, 199),
    (18, 92, 43),
    (83, 177, 41),
    (93, 174, 149),
    (201, 89, 242),
    (224, 219, 73),
    (28, 235, 209),
    (105, 186, 128),
    (214, 63, 16),
    (106, 164, 94),
    (24, 116, 191),
    (195, 51, 136),
    (184, 91, 93),
    (123, 238, 87),
    (160, 147, 72),
    (199, 87, 13),
    (58, 81, 120),
    (116, 183, 64),
    (203, 220, 164),
    (25, 32, 170),
    (14, 214, 28),
    (20, 210, 68),
    (22, 227, 122),
    (83, 135, 200),
    (61, 141, 5),
    (0, 136, 207),
    (207, 181, 139),
    (4, 167, 92),
    (173, 26, 74),
    (52, 238, 177),
    (219, 51, 227),
    (105, 18, 117),
    (34, 51, 158),
    (181, 58, 171),
    (55, 252, 252),
    (18, 173, 87),
    (193, 70, 234),
    (53, 48, 94),
    (59, 80, 154),
    (124, 163, 58),
    (177, 106, 201),
    (44, 13, 121),
    (70, 38, 167),
    (136, 13, 248),
    (135, 208, 248),
    (22, 248, 79),
    (217, 8, 227),
    (6, 209, 199),
    (212, 217, 194),
    (60, 144, 56),
    (114, 237, 151),
    (24, 4, 100),
    (236, 49, 87),
    (30, 54, 153),
    (20, 97, 101),
    (185, 151, 155),
    (29, 161, 115),
    (53, 119, 179),
    (86, 246, 7),
    (105, 241, 137),
    (182, 128, 83),
    (120, 164, 209),
    (148, 117, 240),
    (3, 126, 42),
    (65, 20, 36),
    (68, 208, 112),
    (175, 138, 237),
    (104, 222, 91),
    (43, 63, 159),
    (148, 198, 9),
    (188, 91, 111),
    (163, 83, 76),
    (18, 113, 74),
    (226, 225, 171),
    (131, 140, 228),
    (58, 129, 113),
    (128, 39, 24),
    (186, 36, 99),
    (69, 134, 3),
    (226, 121, 168),
    (188, 161, 28),
    (68, 26, 224),
    (248, 109, 179),
    (201, 181, 197),
    (161, 135, 125),
    (94, 72, 246),
    (84, 135, 195),
    (213, 219, 108),
    (67, 102, 84),
    (71, 83, 223),
    (0, 133, 91),
    (107, 158, 201),
    (211, 7, 149),
    (229, 220, 136),
    (171, 46, 0),
    (104, 179, 38),
    (89, 74, 243),
    (226, 123, 87),
    (96, 83, 26),
    (206, 32, 115),
    (198, 97, 172),
    (59, 57, 178),
    (173, 233, 132),
    (185, 93, 91),
    (145, 163, 194),
    (148, 173, 185),
    (207, 119, 164),
    (105, 190, 4),
    (241, 242, 205),
    (158, 109, 87),
    (226, 163, 73),
    (218, 183, 26),
    (118, 22, 204),
    (207, 98, 90),
    (51, 230, 46),
    (208, 61, 188),
    (47, 250, 104),
    (128, 138, 203),
    (141, 71, 94),
    (6, 173, 245),
    (158, 15, 169),
    (166, 53, 171),
    (82, 135, 220),
    (65, 169, 66),
    (114, 92, 78),
    (229, 219, 246),
    (100, 159, 221),
    (178, 252, 174),
    (93, 114, 161),
    (12, 224, 233),
    (80, 66, 200),
    (243, 125, 138),
    (112, 218, 155),
    (184, 120, 65),
    (192, 197, 88),
    (34, 207, 3),
    (188, 238, 165),
    (171, 211, 88),
    (70, 148, 134),
    (28, 115, 134),
    (66, 92, 220),
    (102, 101, 123),
    (197, 109, 73),
    (100, 182, 77),
    (149, 251, 159),
    (81, 35, 237),
    (243, 250, 136),
    (254, 25, 21),
    (173, 229, 214),
    (144, 153, 238),
    (119, 165, 127),
    (129, 133, 198),
    (140, 90, 74),
    (251, 182, 78),
    (62, 72, 199),
    (45, 133, 47),
    (187, 170, 195),
    (138, 242, 57),
    (219, 89, 131),
    (125, 206, 82),
    (197, 186, 132),
    (17, 197, 191),
    (94, 152, 131),
    (69, 168, 164),
    (58, 177, 183),
    (152, 161, 146),
    (97, 206, 241),
    (135, 181, 235),
    (46, 240, 244),
    (127, 161, 81),
    (157, 12, 118),
    (46, 118, 32),
    (34, 115, 87),
    (124, 153, 174),
    (242, 107, 52),
    (233, 110, 64),
    (76, 118, 73),
    (146, 4, 7),
    (131, 48, 76),
    (37, 116, 237),
    (48, 109, 210),
    (179, 49, 71),
    (90, 177, 238),
    (29, 3, 74),
    (84, 234, 65),
    (119, 20, 208),
    (42, 242, 28),
    (154, 87, 180),
    (48, 194, 102),
    (9, 2, 116),
    (108, 233, 89),
    (124, 117, 100),
    (90, 68, 105),
    (10, 12, 104),
    (225, 165, 167),
    (160, 122, 33),
    (154, 99, 217),
    (50, 88, 210),
    (20, 222, 156),
    (72, 18, 153),
    (152, 39, 234),
    (123, 28, 152),
    (146, 195, 172),
    (211, 45, 54),
    (0, 138, 195),
    (134, 219, 99),
    (185, 198, 147),
    (162, 50, 242),
    (102, 20, 171),
    (118, 128, 228),
    (109, 105, 194),
    (225, 140, 15),
    (118, 33, 151),
    (140, 133, 38),
    (1, 10, 6),
    (125, 6, 102),
    (166, 75, 114),
    (85, 126, 114),
    (178, 51, 2),
    (76, 157, 9),
    (115, 200, 133),
    (116, 181, 235),
    (193, 43, 79),
    (62, 76, 47),
    (76, 149, 120),
    (18, 89, 13),
    (222, 92, 236),
    (105, 213, 6),
    (100, 48, 40),
    (154, 158, 133),
    (143, 191, 229),
    (37, 59, 113),
    (95, 96, 240),
    (208, 32, 62),
    (138, 140, 93),
    (64, 157, 90),
    (231, 150, 147),
    (133, 36, 83),
    (127, 228, 5),
    (92, 204, 154),
    (241, 83, 140),
    (230, 135, 180),
    (80, 157, 0),
    (215, 207, 74),
    (252, 131, 61),
    (67, 144, 112),
    (201, 84, 116),
    (229, 255, 56),
    (62, 184, 48),
    (145, 117, 185),
    (9, 236, 190),
    (79, 230, 206),
    (161, 132, 144),
    (228, 71, 62),
    (23, 91, 110),
    (7, 253, 22),
    (98, 158, 246),
    (80, 155, 47),
    (239, 114, 11),
    (215, 78, 231),
    (30, 90, 34),
    (54, 81, 163),
    (71, 109, 59),
    (112, 165, 68),
    (181, 13, 101),
    (19, 107, 8),
    (96, 209, 197),
    (124, 100, 129),
    (253, 157, 133),
    (208, 230, 248),
    (168, 150, 30),
    (255, 19, 4),
    (148, 106, 74),
    (123, 195, 214),
    (60, 229, 250),
    (93, 169, 40),
    (188, 209, 179),
    (40, 59, 234),
    (222, 29, 225),
    (94, 238, 165),
    (126, 216, 16),
    (99, 167, 157),
    (252, 65, 23),
    (200, 128, 87),
    (37, 111, 191),
    (154, 217, 89),
    (134, 216, 247),
    (207, 101, 41),
    (145, 208, 112),
    (43, 110, 197),
    (118, 147, 239),
    (22, 109, 139),
    (11, 161, 135),
    (119, 26, 48),
    (199, 239, 182),
    (96, 100, 82),
    (87, 149, 254),
    (2, 8, 10),
    (5, 38, 166),
    (100, 193, 117),
    (59, 164, 133),
    (5, 38, 163),
    (88, 177, 207),
    (84, 114, 9),
    (247, 132, 177),
    (24, 94, 130),
    (83, 131, 77),
    (11, 141, 228),
    (81, 154, 198),
    (175, 98, 21),
    (148, 170, 122),
    (185, 145, 101),
    (217, 244, 213),
    (183, 100, 196),
    (111, 226, 11),
    (226, 97, 238),
    (147, 112, 11),
    (225, 25, 97),
    (95, 45, 6),
    (89, 88, 237),
    (38, 51, 16),
    (151, 218, 3),
    (90, 174, 122),
    (157, 2, 133),
    (121, 199, 15),
    (78, 163, 180),
    (103, 118, 7),
    (179, 102, 249),
    (179, 157, 183),
    (113, 139, 195),
    (205, 122, 55),
    (88, 251, 252),
    (248, 68, 117),
    (115, 214, 185),
    (93, 102, 139),
    (206, 82, 217),
    (236, 3, 165),
    (135, 29, 78),
    (11, 201, 11),
    (16, 60, 123),
    (103, 191, 187),
    (129, 146, 181),
    (28, 192, 85),
    (216, 73, 136),
    (210, 139, 220),
    (117, 179, 81),
    (183, 15, 131),
    (106, 216, 213),
    (28, 58, 213),
    (78, 111, 65),
    (76, 11, 25),
    (103, 11, 90),
    (237, 162, 129),
    (254, 144, 1),
    (16, 33, 33),
    (172, 230, 40),
    (72, 205, 106),
    (83, 242, 160),
    (151, 68, 159),
    (150, 64, 229),
    (31, 79, 83),
    (15, 51, 249),
    (140, 173, 10),
    (244, 105, 80),
    (70, 21, 222),
    (195, 80, 64),
    (129, 50, 96),
    (107, 251, 210),
    (82, 185, 150),
    (15, 143, 28),
    (71, 27, 216),
    (57, 58, 216),
    (204, 13, 146),
    (78, 206, 20),
    (71, 183, 44),
    (235, 245, 91),
    (44, 15, 253),
    (87, 203, 77),
    (254, 157, 95),
    (110, 210, 132),
    (28, 193, 49),
    (177, 87, 57),
    (208, 41, 218),
    (240, 194, 175),
    (17, 20, 166),
    (64, 134, 236),
    (150, 79, 74),
    (162, 168, 166),
    (246, 149, 34),
    (117, 160, 170),
    (127, 44, 99),
    (41, 249, 103),
    (155, 251, 48),
    (127, 138, 68),
    (17, 3, 101),
    (94, 215, 29),
    (102, 123, 158),
    (203, 194, 221),
    (60, 135, 179),
    (244, 73, 215),
    (192, 240, 145),
    (168, 243, 21),
    (94, 154, 143),
    (240, 17, 10),
    (145, 131, 231),
    (73, 29, 195),
    (199, 208, 226),
    (132, 189, 90),
    (100, 134, 32),
    (81, 119, 118),
    (245, 37, 119),
    (27, 51, 78),
    (187, 86, 95),
    (8, 56, 29),
    (156, 223, 162),
    (186, 127, 126),
    (209, 220, 111),
    (144, 200, 59),
    (247, 236, 7),
    (140, 32, 212),
    (235, 75, 221),
    (40, 0, 212),
    (109, 92, 205),
    (165, 175, 61),
    (103, 178, 68),
    (238, 204, 185),
    (119, 229, 240),
    (132, 105, 36),
    (80, 210, 165),
    (222, 117, 238),
    (35, 176, 128),
    (49, 185, 9),
    (50, 225, 176),
    (241, 12, 198),
    (124, 243, 164),
    (99, 102, 36),
    (30, 114, 147),
    (166, 172, 35),
    (14, 29, 179),
    (60, 248, 81),
    (67, 29, 155),
    (131, 33, 245),
    (179, 118, 155),
    (228, 56, 21),
    (167, 234, 74),
    (29, 241, 157),
    (189, 58, 24),
    (114, 16, 192),
    (100, 226, 123),
    (13, 1, 251),
    (2, 66, 147),
    (244, 250, 204),
    (158, 39, 213),
    (77, 177, 26),
    (115, 140, 241),
    (192, 1, 164),
    (10, 106, 32),
    (254, 28, 186),
    (194, 220, 65),
    (205, 83, 47),
    (81, 164, 199),
    (53, 198, 137),
    (15, 18, 157),
    (181, 188, 42),
    (210, 130, 29),
    (145, 35, 120),
    (19, 144, 23),
    (140, 99, 109),
    (184, 194, 20),
    (131, 81, 172),
    (38, 42, 202),
    (37, 106, 40),
    (111, 27, 132),
    (179, 150, 165),
    (35, 30, 7),
    (216, 88, 239),
    (143, 138, 147),
    (77, 231, 56),
    (229, 231, 146),
    (199, 60, 24),
    (160, 108, 39),
    (116, 104, 126),
    (111, 116, 228),
    (234, 16, 180),
    (218, 232, 208),
    (46, 224, 94),
    (84, 23, 143),
    (172, 83, 119),
    (55, 98, 155),
    (45, 226, 81),
    (215, 155, 77),
    (176, 215, 94),
    (133, 121, 231),
    (225, 113, 226),
    (28, 244, 65),
    (68, 213, 200),
    (252, 114, 161),
    (118, 227, 48),
    (99, 217, 69),
    (193, 99, 71),
    (1, 129, 199),
    (153, 100, 77),
    (108, 178, 155),
    (96, 43, 19),
    (98, 45, 100),
    (80, 112, 234),
    (121, 171, 195),
    (8, 211, 174),
    (64, 116, 148),
    (96, 226, 107),
    (133, 85, 24),
    (2, 253, 253),
    (199, 108, 203),
    (243, 28, 118),
    (95, 65, 208),
    (114, 210, 142),
    (21, 111, 111),
    (197, 33, 106),
    (130, 27, 232),
    (119, 147, 143),
    (71, 236, 29),
    (163, 225, 48),
    (177, 80, 184),
    (63, 216, 186),
    (133, 51, 250),
    (61, 106, 211),
    (17, 45, 224),
    (62, 22, 225),
    (102, 137, 5),
    (76, 61, 170),
    (247, 144, 59),
    (17, 196, 60),
    (153, 142, 109),
    (61, 181, 81),
    (29, 198, 28),
    (202, 133, 228),
    (228, 84, 183),
    (242, 178, 86),
    (181, 10, 43),
    (187, 158, 59),
    (142, 243, 176),
    (120, 125, 56),
    (193, 104, 135),
    (197, 143, 119),
    (233, 52, 32),
    (87, 206, 162),
    (29, 51, 217),
    (181, 4, 192),
    (123, 154, 212),
    (133, 122, 88),
    (17, 196, 56),
    (160, 241, 81),
    (72, 73, 228),
    (159, 94, 73),
    (108, 176, 227),
    (160, 14, 61),
    (87, 160, 56),
    (156, 171, 97),
    (178, 46, 108),
    (72, 165, 120),
    (2, 17, 143),
    (195, 85, 240),
    (242, 93, 101),
    (120, 150, 210),
    (7, 55, 177),
    (26, 248, 132),
    (109, 89, 26),
    (188, 243, 111),
    (83, 78, 146),
    (121, 96, 91),
    (36, 102, 148),
    (127, 236, 131),
    (146, 205, 51),
    (49, 31, 225),
    (146, 12, 111),
    (219, 231, 98),
    (71, 94, 45),
    (208, 65, 88),
    (16, 45, 127),
    (219, 165, 178),
    (152, 176, 204),
    (4, 240, 179),
    (174, 40, 142),
    (241, 138, 155),
    (78, 102, 141),
    (4, 214, 15),
    (29, 206, 170),
    (153, 203, 223),
    (219, 22, 110),
    (43, 212, 183),
    (253, 130, 153),
    (125, 120, 128),
    (95, 184, 110),
    (198, 210, 86),
    (190, 255, 46),
    (251, 89, 205),
    (140, 62, 37),
    (211, 159, 119),
    (156, 59, 248),
    (83, 40, 104),
    (121, 89, 239),
    (29, 28, 52),
    (42, 138, 212),
    (17, 128, 167),
    (38, 226, 193),
    (48, 214, 45),
    (122, 177, 22),
    (231, 254, 92),
    (65, 85, 66),
    (32, 178, 70),
    (102, 163, 245),
    (228, 151, 243),
    (122, 82, 102),
    (89, 238, 79),
    (113, 155, 85),
    (1, 9, 115),
    (81, 139, 64),
    (247, 67, 249),
    (189, 54, 36),
    (212, 126, 19),
    (239, 197, 212),
    (125, 190, 196),
    (169, 208, 144),
    (185, 133, 96),
    (22, 131, 136),
    (62, 92, 169),
    (74, 102, 92),
    (95, 108, 139),
    (162, 91, 49),
    (71, 167, 102),
    (119, 58, 5),
    (73, 88, 140),
    (232, 225, 59),
    (206, 237, 15),
    (4, 150, 180),
    (66, 71, 109),
    (72, 5, 39),
    (79, 79, 41),
    (217, 150, 217),
    (217, 147, 62),
    (29, 9, 139),
    (125, 99, 26),
    (103, 255, 210),
    (72, 255, 129),
    (99, 114, 241),
    (174, 55, 10),
    (202, 24, 255),
    (95, 43, 219),
    (141, 206, 55),
    (5, 151, 42),
    (103, 49, 101),
    (102, 75, 117),
    (184, 171, 152),
    (81, 21, 222),
    (146, 63, 89),
    (29, 37, 177),
    (168, 241, 106),
    (223, 103, 11),
    (56, 245, 191),
    (142, 32, 193),
    (121, 117, 65),
    (36, 250, 25),
    (204, 72, 50),
    (12, 152, 147),
    (195, 138, 7),
    (180, 92, 14),
    (200, 46, 227),
    (35, 209, 151),
    (61, 117, 181),
    (12, 173, 7),
    (250, 175, 110),
    (167, 94, 147),
    (234, 109, 178),
    (65, 171, 82),
    (206, 251, 89),
    (101, 120, 184),
    (72, 178, 34),
    (218, 11, 108),
    (139, 104, 164),
    (126, 177, 204),
    (209, 223, 202),
    (48, 74, 139),
    (93, 34, 202),
    (204, 241, 39),
    (112, 10, 2),
    (121, 239, 75),
    (174, 51, 244),
    (169, 250, 91),
    (131, 16, 63),
    (64, 13, 38),
    (4, 3, 147),
    (81, 144, 126),
    (71, 152, 11),
    (150, 60, 219),
    (130, 136, 72),
    (217, 38, 52),
    (225, 88, 63),
    (149, 4, 159),
    (243, 203, 165),
    (7, 150, 124),
    (42, 187, 223),
    (54, 201, 115),
    (179, 82, 23),
    (83, 25, 57),
    (197, 89, 45),
    (111, 88, 179),
    (59, 172, 154),
    (120, 117, 171),
    (121, 160, 30),
    (142, 71, 96),
    (189, 1, 246),
    (175, 124, 109),
    (118, 171, 218),
    (15, 68, 202),
    (85, 167, 106),
    (235, 94, 214),
    (0, 81, 186),
    (12, 219, 43),
    (182, 125, 175),
    (22, 31, 116),
    (34, 226, 42),
    (68, 122, 62),
    (36, 83, 172),
    (209, 240, 111),
    (198, 6, 176),
    (137, 28, 77),
    (184, 137, 213),
    (66, 9, 110),
    (248, 164, 28),
    (244, 64, 202),
    (124, 141, 235),
    (2, 165, 199),
    (87, 222, 209),
    (3, 111, 226),
    (80, 243, 82),
    (50, 189, 234),
    (210, 28, 35),
    (53, 26, 113),
    (103, 42, 46),
    (197, 28, 127),
    (171, 83, 62),
    (74, 202, 137),
    (219, 227, 75),
    (96, 166, 31),
    (243, 150, 126),
    (91, 57, 21),
    (225, 61, 236),
    (41, 115, 107),
    (99, 169, 250),
    (151, 253, 232),
    (52, 23, 122),
    (72, 41, 237),
    (60, 47, 143),
    (36, 193, 94),
    (229, 42, 254),
    (0, 211, 134),
    (221, 67, 31),
    (29, 16, 195),
    (26, 253, 6),
    (233, 229, 11),
    (81, 55, 125),
    (43, 67, 20),
    (219, 255, 119),
    (220, 14, 14),
    (80, 227, 191),
    (117, 230, 43),
    (155, 102, 215),
    (148, 251, 23),
    (34, 32, 65),
    (22, 21, 185),
    (68, 45, 232),
    (167, 24, 86),
    (17, 180, 218),
    (136, 5, 225),
    (86, 250, 82),
    (10, 62, 29),
    (139, 158, 71),
    (58, 222, 42),
    (213, 71, 153),
    (236, 54, 118),
    (83, 25, 173),
    (121, 234, 131),
    (116, 212, 241),
    (83, 249, 37),
    (161, 188, 108),
    (217, 7, 188),
]


class ImageNetSSemanticSegmentation(Dataset):

    def __init__(self,
                 root_dir,
                 dataset_type='ImageNetS919',
                 image_sets=['train-semi', 'validation'],
                 reduce_zero_label=False,
                 transform=None):
        assert dataset_type in ['ImageNetS50', 'ImageNetS300', 'ImageNetS919']
        for per_image_set_name in image_sets:
            assert per_image_set_name in ['train-semi', 'validation']

        self.reduce_zero_label = reduce_zero_label
        self.transform = transform

        self.all_set_image_path = []
        self.all_set_mask_path = []
        for per_set_name in image_sets:
            per_set_image_path = os.path.join(root_dir, dataset_type,
                                              per_set_name)
            per_set_mask_path = os.path.join(root_dir, dataset_type,
                                             f'{per_set_name}-segmentation')
            self.all_set_image_path.append(per_set_image_path)
            self.all_set_mask_path.append(per_set_mask_path)

        annot_class_id_file_name = {
            'ImageNetS50': 'ImageNetS_categories_im50.txt',
            'ImageNetS300': 'ImageNetS_categories_im300.txt',
            'ImageNetS919': 'ImageNetS_categories_im919.txt',
        }

        self.annot_class_id_file_path = os.path.join(
            root_dir, annot_class_id_file_name[dataset_type])

        self.cats = []
        with open(self.annot_class_id_file_path, 'r') as f:
            for line in f.readlines():
                # 去掉列表中每一个元素的换行符
                line = line.strip('\n')
                self.cats.append(line)

        self.num_classes = len(self.cats)

        if self.reduce_zero_label:
            self.cat_to_label_idx = {cat: i for i, cat in enumerate(self.cats)}
            self.label_idx_to_cat = {i: cat for i, cat in enumerate(self.cats)}
        else:
            self.cat_to_label_idx = {
                cat: i + 1
                for i, cat in enumerate(self.cats)
            }
            self.label_idx_to_cat = {
                i + 1: cat
                for i, cat in enumerate(self.cats)
            }

        self.ids = []
        self.image_path_dict = collections.OrderedDict()
        self.mask_path_dict = collections.OrderedDict()
        for per_set_image_path, per_set_mask_path in zip(
                self.all_set_image_path, self.all_set_mask_path):
            for per_class_name in os.listdir(per_set_image_path):
                per_class_image_path = os.path.join(per_set_image_path,
                                                    per_class_name)
                per_class_mask_path = os.path.join(per_set_mask_path,
                                                   per_class_name)
                for per_image_name in os.listdir(per_class_image_path):
                    image_name_prefix = per_image_name.split('.')[0]
                    per_image_path = os.path.join(per_class_image_path,
                                                  per_image_name)
                    per_mask_path = os.path.join(per_class_mask_path,
                                                 f'{image_name_prefix}.png')

                    if not os.path.exists(
                            per_image_path) or not os.path.exists(
                                per_mask_path):
                        continue
                    self.ids.append(per_image_name)
                    self.image_path_dict[per_image_name] = per_image_path
                    self.mask_path_dict[per_image_name] = per_mask_path

        self.reduce_zero_label = reduce_zero_label
        self.transform = transform

        print(f'Dataset Size:{len(self.ids)}')
        if self.reduce_zero_label:
            print(f'Dataset Class Num:{self.num_classes}')
        else:
            print(f'Dataset Class Num:{self.num_classes+1}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        mask = self.load_mask(idx)

        scale = np.array(1.).astype(np.float32)
        size = np.array([image.shape[0], image.shape[1]]).astype(np.float32)

        sample = {
            'image': image,
            'mask': mask,
            'scale': scale,
            'size': size,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(
            np.fromfile(self.image_path_dict[self.ids[idx]], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_mask(self, idx):
        mask = cv2.imdecode(
            np.fromfile(self.mask_path_dict[self.ids[idx]], dtype=np.uint8),
            cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        new_zero_mask = np.zeros((mask.shape[0], mask.shape[1]),
                                 dtype=np.float32)
        new_zero_mask = mask[:, :, 0] + mask[:, :, 1] * 256

        mask = new_zero_mask
        # The ignored part is annotated as 1000, and the other category is annotated as 0.
        # set ignored area class id to 0
        mask[mask == 1000] = 0

        # If class 0 is the background class and you want to ignore it when calculating the evaluation index,
        # you need to set reduce_zero_label=True.
        if self.reduce_zero_label:
            # avoid using underflow conversion
            mask[mask == 0] = 1000
            mask = mask - 1
            # background class 0 transform to class 1000,class 1~919 transform to 0~918
            mask[mask == 999] = 1000

        return mask.astype(np.float32)


if __name__ == '__main__':
    import os
    import random
    import numpy as np
    import torch
    seed = 0
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import os
    import sys

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(BASE_DIR)

    from tools.path import ImageNet_S_path

    import torchvision.transforms as transforms
    from tqdm import tqdm

    from simpleAICV.semantic_segmentation.common import RandomCropResize, RandomHorizontalFlip, PhotoMetricDistortion, Normalize, SemanticSegmentationCollater

    # imagenetsdataset = ImageNetSSemanticSegmentation(
    #     ImageNet_S_path,
    #     dataset_type='ImageNetS919',
    #     image_sets=['train-semi', 'validation'],
    #     reduce_zero_label=True,
    #     transform=transforms.Compose([
    #         RandomCropResize(image_scale=(2048, 512),
    #                          multi_scale=True,
    #                          multi_scale_range=(0.5, 2.0),
    #                          crop_size=(512, 512),
    #                          cat_max_ratio=0.75,
    #                          ignore_index=255),
    #         RandomHorizontalFlip(prob=0.5),
    #         PhotoMetricDistortion(brightness_delta=32,
    #                               contrast_range=(0.5, 1.5),
    #                               saturation_range=(0.5, 1.5),
    #                               hue_delta=18,
    #                               prob=0.5),
    #         # Normalize(),
    #     ]))

    # count = 0
    # for per_sample in tqdm(imagenetsdataset):
    #     print('1111', per_sample['image'].shape, per_sample['mask'].shape,
    #           per_sample['scale'], per_sample['size'])
    #     print('1111', per_sample['image'].dtype, per_sample['mask'].dtype,
    #           per_sample['scale'].dtype, per_sample['size'].dtype)

    #     temp_dir = './temp1'
    #     if not os.path.exists(temp_dir):
    #         os.makedirs(temp_dir)

    #     image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     mask = per_sample['mask']
    #     mask_jpg = np.zeros((image.shape[0], image.shape[1], 3))

    #     all_classes = np.unique(mask)
    #     print("1212", all_classes)
    #     for per_class in all_classes:
    #         per_class = int(per_class)
    #         if per_class < 0 or per_class > 1000:
    #             continue
    #         if per_class != 1000:
    #             class_name, class_color = imagenetsdataset.label_idx_to_cat[
    #                 per_class], ImageNet_S_CLASSES_COLOR[per_class]
    #         else:
    #             class_name, class_color = 'background', (255, 255, 255)
    #         class_color = np.array(
    #             (class_color[0], class_color[1], class_color[2]))
    #         per_mask = (mask == per_class).astype(np.float32)
    #         per_mask = np.expand_dims(per_mask, axis=-1)
    #         per_mask = np.tile(per_mask, (1, 1, 3))
    #         mask_color = np.expand_dims(np.expand_dims(class_color, axis=0),
    #                                     axis=0)

    #         per_mask = per_mask * mask_color
    #         image = 0.5 * per_mask + image
    #         mask_jpg += per_mask

    #     cv2.imencode('.jpg', image)[1].tofile(
    #         os.path.join(temp_dir, f'idx_{count}.jpg'))
    #     cv2.imencode('.jpg', mask_jpg)[1].tofile(
    #         os.path.join(temp_dir, f'idx_{count}_mask.jpg'))

    #     if count < 10:
    #         count += 1
    #     else:
    #         break

    # from torch.utils.data import DataLoader
    # collater = SemanticSegmentationCollater(resize=512, ignore_index=1000)
    # train_loader = DataLoader(imagenetsdataset,
    #                           batch_size=4,
    #                           shuffle=True,
    #                           num_workers=2,
    #                           collate_fn=collater)

    # count = 0
    # for data in tqdm(train_loader):
    #     images, masks, scales, sizes = data['image'], data['mask'], data[
    #         'scale'], data['size']
    #     print('2222', images.shape, masks.shape, scales.shape, sizes.shape)
    #     print('2222', images.dtype, masks.dtype, scales.dtype, sizes.dtype)

    #     temp_dir = './temp2'
    #     if not os.path.exists(temp_dir):
    #         os.makedirs(temp_dir)

    #     images = images.permute(0, 2, 3, 1).cpu().numpy()
    #     masks = masks.cpu().numpy()

    #     for i, (per_image,
    #             per_image_mask_targets) in enumerate(zip(images, masks)):
    #         per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
    #         per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

    #         per_image_mask_jpg = np.zeros(
    #             (per_image.shape[0], per_image.shape[1], 3))

    #         all_classes = np.unique(per_image_mask_targets)
    #         print("2323", all_classes)
    #         for per_class in all_classes:
    #             per_class = int(per_class)
    #             if per_class < 0 or per_class > 1000:
    #                 continue
    #             if per_class != 1000:
    #                 class_name, class_color = imagenetsdataset.label_idx_to_cat[
    #                     per_class], ImageNet_S_CLASSES_COLOR[per_class]
    #             else:
    #                 class_name, class_color = 'background', (255, 255, 255)
    #             class_color = np.array(
    #                 (class_color[0], class_color[1], class_color[2]))
    #             per_image_mask = (per_image_mask_targets == per_class).astype(
    #                 np.float32)
    #             per_image_mask = np.expand_dims(per_image_mask, axis=-1)
    #             per_image_mask = np.tile(per_image_mask, (1, 1, 3))
    #             mask_color = np.expand_dims(np.expand_dims(class_color,
    #                                                        axis=0),
    #                                         axis=0)

    #             per_image_mask = per_image_mask * mask_color
    #             per_image = 0.5 * per_image_mask + per_image
    #             per_image_mask_jpg += per_image_mask

    #         cv2.imencode('.jpg', per_image)[1].tofile(
    #             os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))
    #         cv2.imencode('.jpg', per_image_mask_jpg)[1].tofile(
    #             os.path.join(temp_dir, f'idx_{count}_{i}_mask.jpg'))

    #     if count < 5:
    #         count += 1
    #     else:
    #         break

    imagenetsdataset = ImageNetSSemanticSegmentation(
        ImageNet_S_path,
        dataset_type='ImageNetS919',
        image_sets=['train-semi', 'validation'],
        reduce_zero_label=False,
        transform=transforms.Compose([
            RandomCropResize(image_scale=(2048, 512),
                             multi_scale=True,
                             multi_scale_range=(0.5, 2.0),
                             crop_size=(512, 512),
                             cat_max_ratio=0.75,
                             ignore_index=None),
            RandomHorizontalFlip(prob=0.5),
            PhotoMetricDistortion(brightness_delta=32,
                                  contrast_range=(0.5, 1.5),
                                  saturation_range=(0.5, 1.5),
                                  hue_delta=18,
                                  prob=0.5),
            # Normalize(),
        ]))

    count = 0
    for per_sample in tqdm(imagenetsdataset):
        print('1111', per_sample['image'].shape, per_sample['mask'].shape,
              per_sample['scale'], per_sample['size'])
        print('1111', per_sample['image'].dtype, per_sample['mask'].dtype,
              per_sample['scale'].dtype, per_sample['size'].dtype)

        temp_dir = './temp3'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        image = np.ascontiguousarray(per_sample['image'], dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = per_sample['mask']
        mask_jpg = np.zeros((image.shape[0], image.shape[1], 3))

        all_classes = np.unique(mask)
        print("1212", all_classes)
        for per_class in all_classes:
            if per_class == 0:
                continue
            per_class = int(per_class)
            if per_class < 0 or per_class > 919:
                continue
            if per_class != 0:
                class_name, class_color = imagenetsdataset.label_idx_to_cat[
                    per_class], ImageNet_S_CLASSES_COLOR[per_class - 1]
            else:
                class_name, class_color = 'background', (255, 255, 255)
            class_color = np.array(
                (class_color[0], class_color[1], class_color[2]))
            per_mask = (mask == per_class).astype(np.float32)
            per_mask = np.expand_dims(per_mask, axis=-1)
            per_mask = np.tile(per_mask, (1, 1, 3))
            mask_color = np.expand_dims(np.expand_dims(class_color, axis=0),
                                        axis=0)

            per_mask = per_mask * mask_color
            image = 0.5 * per_mask + image
            mask_jpg += per_mask

        cv2.imencode('.jpg', image)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}.jpg'))
        cv2.imencode('.jpg', mask_jpg)[1].tofile(
            os.path.join(temp_dir, f'idx_{count}_mask.jpg'))

        if count < 10:
            count += 1
        else:
            break

    from torch.utils.data import DataLoader
    collater = SemanticSegmentationCollater(resize=512, ignore_index=None)
    train_loader = DataLoader(imagenetsdataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collater)

    count = 0
    for data in tqdm(train_loader):
        images, masks, scales, sizes = data['image'], data['mask'], data[
            'scale'], data['size']
        print('2222', images.shape, masks.shape, scales.shape, sizes.shape)
        print('2222', images.dtype, masks.dtype, scales.dtype, sizes.dtype)

        temp_dir = './temp4'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        images = images.permute(0, 2, 3, 1).cpu().numpy()
        masks = masks.cpu().numpy()

        for i, (per_image,
                per_image_mask_targets) in enumerate(zip(images, masks)):
            per_image = np.ascontiguousarray(per_image, dtype=np.uint8)
            per_image = cv2.cvtColor(per_image, cv2.COLOR_RGB2BGR)

            per_image_mask_jpg = np.zeros(
                (per_image.shape[0], per_image.shape[1], 3))

            all_classes = np.unique(per_image_mask_targets)
            print("2323", all_classes)
            for per_class in all_classes:
                if per_class == 0:
                    continue
                per_class = int(per_class)
                if per_class < 0 or per_class > 919:
                    continue
                if per_class != 0:
                    class_name, class_color = imagenetsdataset.label_idx_to_cat[
                        per_class], ImageNet_S_CLASSES_COLOR[per_class - 1]
                else:
                    class_name, class_color = 'background', (255, 255, 255)
                class_color = np.array(
                    (class_color[0], class_color[1], class_color[2]))
                per_image_mask = (per_image_mask_targets == per_class).astype(
                    np.float32)
                per_image_mask = np.expand_dims(per_image_mask, axis=-1)
                per_image_mask = np.tile(per_image_mask, (1, 1, 3))
                mask_color = np.expand_dims(np.expand_dims(class_color,
                                                           axis=0),
                                            axis=0)

                per_image_mask = per_image_mask * mask_color
                per_image = 0.5 * per_image_mask + per_image
                per_image_mask_jpg += per_image_mask

            cv2.imencode('.jpg', per_image)[1].tofile(
                os.path.join(temp_dir, f'idx_{count}_{i}.jpg'))
            cv2.imencode('.jpg', per_image_mask_jpg)[1].tofile(
                os.path.join(temp_dir, f'idx_{count}_{i}_mask.jpg'))

        if count < 5:
            count += 1
        else:
            break