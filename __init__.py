import bpy
import bpy_extras
import bmesh
import struct
import math
from dataclasses import dataclass

@dataclass
class SchHeader:
    magic: bytes
    fileSize: int
    grid_bound_min: tuple[float, float, float]
    grid_bound_max: tuple[float, float, float]
    grid_cell_size: tuple[float, float, float]
    grid_resolution: tuple[int, int, int]
    num_polygons: int
    grid_array_offset: int
    polygon_array_offset: int

@dataclass
class SchPolygons:
    polyId: int
    attrib1: int
    flags: int
    attrib2: int
    attrib3: int
    vertex1: tuple[float, float, float]
    vertex2: tuple[float, float, float]
    vertex3: tuple[float, float, float]
    normal: tuple[float, float, float]
    distance: float

@dataclass
class SchGridCell:
    polygon_indices: list[int]
    terminator: int

@dataclass
class SchFile:
    header: SchHeader
    grid_cell_offsets: list[int] 
    grid_cells: list[SchGridCell] 
    polygons: list[SchPolygons] 

def extract_polygons_from_file(file_path: str) -> list[SchPolygons]:
    polygons = []
    with open(file_path, 'rb') as f:
        _ = f.seek(0x40)
        polygon_offset_bytes = f.read(4)
        polygon_offset = struct.unpack('>I', polygon_offset_bytes)[0]

        f.seek(polygon_offset+8)
        while True:
            data = f.read(60)
            if len(data) < 60:
                break
            unpacked = struct.unpack('>BBHHH9fffff', data)
            polygon = SchPolygons(
                polyId=unpacked[0],
                attrib1=unpacked[1],
                flags=unpacked[2],
                attrib2=unpacked[3],
                attrib3=unpacked[4],
                vertex1=(unpacked[5], unpacked[6], unpacked[7]),
                vertex2=(unpacked[8], unpacked[9], unpacked[10]),
                vertex3=(unpacked[11], unpacked[12], unpacked[13]),
                normal=(unpacked[14], unpacked[15], unpacked[16]),
                distance=unpacked[17]
            )
            polygons.append(polygon)
    return polygons


def polygons_to_mesh(polygons: list[SchPolygons], mesh_name: str = "CollisionMesh") -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(mesh_name)
    bm = bmesh.new()

    poly_id_layer = bm.faces.layers.int.new("polyId")
    attrib1_layer = bm.faces.layers.int.new("attrib1")
    flags_layer = bm.faces.layers.int.new("flags")
    attrib2_layer = bm.faces.layers.int.new("attrib2")
    attrib3_layer = bm.faces.layers.int.new("attrib3")

    vertex_map = {}  

    for poly in polygons:
        face_verts = []
        for v_coord in [poly.vertex1, poly.vertex2, poly.vertex3]:
            if v_coord not in vertex_map:
                new_vert = bm.verts.new(v_coord)
                vertex_map[v_coord] = new_vert
            face_verts.append(vertex_map[v_coord])

        try:
            face = bm.faces.new(face_verts)
            face[poly_id_layer] = poly.polyId
            face[attrib1_layer] = poly.attrib1
            face[flags_layer] = poly.flags
            face[attrib2_layer] = poly.attrib2
            face[attrib3_layer] = poly.attrib3
        except ValueError:
            print(f"Skipping invalid face from polygon data.")

    bm.to_mesh(mesh)
    bm.free() 
    mesh.update()
    mesh.validate()

    return mesh

def maya_to_blender_transform(obj: bpy.types.Object):
    obj.scale = (0.01, 0.01, 0.01)
    obj.rotation_euler = (math.radians(90), 0, 0)

def mesh_to_sch_polygons(mesh: bpy.types.Mesh) -> list[SchPolygons]:
    polygons = []
    poly_id_layer = mesh.attributes.get("polyId")
    attrib1_layer = mesh.attributes.get("attrib1")
    flags_layer = mesh.attributes.get("flags")
    attrib2_layer = mesh.attributes.get("attrib2")
    attrib3_layer = mesh.attributes.get("attrib3")

    for poly in mesh.polygons:
        v1 = mesh.vertices[poly.vertices[0]].co
        v2 = mesh.vertices[poly.vertices[1]].co
        v3 = mesh.vertices[poly.vertices[2]].co
        normal = poly.normal
        distance = -normal.dot(v1)

        polygon = SchPolygons(
            polyId=poly_id_layer.data[poly.index].value if poly_id_layer else 0,
            attrib1=attrib1_layer.data[poly.index].value if attrib1_layer else 0,
            flags=flags_layer.data[poly.index].value if flags_layer else 0,
            attrib2=attrib2_layer.data[poly.index].value if attrib2_layer else 0,
            attrib3=attrib3_layer.data[poly.index].value if attrib3_layer else 0,
            vertex1=(v1.x, v1.y, v1.z),
            vertex2=(v2.x, v2.y, v2.z),
            vertex3=(v3.x, v3.y, v3.z),
            normal=(normal.x, normal.y, normal.z),
            distance=distance
        )
        polygons.append(polygon)
    return polygons

def get_mesh_bounding_box(mesh: bpy.types.Mesh) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    for vert in mesh.vertices:
        co = vert.co
        min_x = min(min_x, co.x)
        min_y = min(min_y, co.y)
        min_z = min(min_z, co.z)
        max_x = max(max_x, co.x)
        max_y = max(max_y, co.y)
        max_z = max(max_z, co.z)

    return (min_x, min_y, min_z), (max_x, max_y, max_z)

def grid_1d_index_to_3d(index: int, grid_resolution: tuple[int, int, int]) -> tuple[int, int, int]:
    x_res, y_res, z_res = grid_resolution
    x = index // (y_res * z_res)
    remainder = index % (y_res * z_res)
    y = (remainder // z_res)
    z = remainder % z_res
    return (x, y, z)

def grid_3d_index_to_1d(coord: tuple[int, int, int], grid_resolution: tuple[int, int, int]) -> int:
    y_res, z_res = grid_resolution
    return coord[0] * (y_res * z_res) + coord[1] * z_res + coord[2]


def get_polygons_in_cell(polygons: list[SchPolygons], cell_coord: tuple[int, int, int], cell_size: tuple[float, float, float], grid_min: tuple[float, float, float]) -> list[SchPolygons]:
    cell_min = (
        grid_min[0] + cell_coord[0] * cell_size[0],
        grid_min[1] + cell_coord[1] * cell_size[1],
        grid_min[2] + cell_coord[2] * cell_size[2],
    )
    cell_max = (
        cell_min[0] + cell_size[0],
        cell_min[1] + cell_size[1],
        cell_min[2] + cell_size[2],
    )
    
    EPSILON = 0.02

    def dot_product(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def triangle_intersects_aabb(v0: tuple[float, float, float], 
                                  v1: tuple[float, float, float], 
                                  v2: tuple[float, float, float],
                                  normal: tuple[float, float, float]) -> bool:
        tri_min = (min(v0[0], v1[0], v2[0]), min(v0[1], v1[1], v2[1]), min(v0[2], v1[2], v2[2]))
        tri_max = (max(v0[0], v1[0], v2[0]), max(v0[1], v1[1], v2[1]), max(v0[2], v1[2], v2[2]))
        
        if (tri_max[0] < cell_min[0] - EPSILON or tri_min[0] > cell_max[0] + EPSILON or
            tri_max[1] < cell_min[1] - EPSILON or tri_min[1] > cell_max[1] + EPSILON or
            tri_max[2] < cell_min[2] - EPSILON or tri_min[2] > cell_max[2] + EPSILON):
            return False
        
        edge0 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        edge1 = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
        edge2 = (v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2])
        
        p = cell_min
        dp = (cell_max[0] - cell_min[0], cell_max[1] - cell_min[1], cell_max[2] - cell_min[2])
        
        c = (
            dp[0] if normal[0] > 0.0 else 0.0,
            dp[1] if normal[1] > 0.0 else 0.0,
            dp[2] if normal[2] > 0.0 else 0.0
        )
        
        d1 = dot_product(normal, (c[0] - v0[0], c[1] - v0[1], c[2] - v0[2]))
        d2 = dot_product(normal, (dp[0] - c[0] - v0[0], dp[1] - c[1] - v0[1], dp[2] - c[2] - v0[2]))
        
        if (dot_product(normal, p) + d1) * (dot_product(normal, p) + d2) > EPSILON:
            return False
        
        xym = -1.0 if normal[2] < 0.0 else 1.0
        ne0xy = (-edge0[1] * xym, edge0[0] * xym)
        ne1xy = (-edge1[1] * xym, edge1[0] * xym)
        ne2xy = (-edge2[1] * xym, edge2[0] * xym)
        
        de0xy = -(ne0xy[0] * v0[0] + ne0xy[1] * v0[1]) + max(0.0, dp[0] * ne0xy[0]) + max(0.0, dp[1] * ne0xy[1])
        de1xy = -(ne1xy[0] * v1[0] + ne1xy[1] * v1[1]) + max(0.0, dp[0] * ne1xy[0]) + max(0.0, dp[1] * ne1xy[1])
        de2xy = -(ne2xy[0] * v2[0] + ne2xy[1] * v2[1]) + max(0.0, dp[0] * ne2xy[0]) + max(0.0, dp[1] * ne2xy[1])
        
        if ((ne0xy[0] * p[0] + ne0xy[1] * p[1]) + de0xy < -EPSILON or
            (ne1xy[0] * p[0] + ne1xy[1] * p[1]) + de1xy < -EPSILON or
            (ne2xy[0] * p[0] + ne2xy[1] * p[1]) + de2xy < -EPSILON):
            return False
        
        yzm = -1.0 if normal[0] < 0.0 else 1.0
        ne0yz = (-edge0[2] * yzm, edge0[1] * yzm)
        ne1yz = (-edge1[2] * yzm, edge1[1] * yzm)
        ne2yz = (-edge2[2] * yzm, edge2[1] * yzm)
        
        de0yz = -(ne0yz[0] * v0[1] + ne0yz[1] * v0[2]) + max(0.0, dp[1] * ne0yz[0]) + max(0.0, dp[2] * ne0yz[1])
        de1yz = -(ne1yz[0] * v1[1] + ne1yz[1] * v1[2]) + max(0.0, dp[1] * ne1yz[0]) + max(0.0, dp[2] * ne1yz[1])
        de2yz = -(ne2yz[0] * v2[1] + ne2yz[1] * v2[2]) + max(0.0, dp[1] * ne2yz[0]) + max(0.0, dp[2] * ne2yz[1])
        
        if ((ne0yz[0] * p[1] + ne0yz[1] * p[2]) + de0yz < -EPSILON or
            (ne1yz[0] * p[1] + ne1yz[1] * p[2]) + de1yz < -EPSILON or
            (ne2yz[0] * p[1] + ne2yz[1] * p[2]) + de2yz < -EPSILON):
            return False
        
        zxm = -1.0 if normal[1] < 0.0 else 1.0
        ne0zx = (-edge0[0] * zxm, edge0[2] * zxm)
        ne1zx = (-edge1[0] * zxm, edge1[2] * zxm)
        ne2zx = (-edge2[0] * zxm, edge2[2] * zxm)
        
        de0zx = -(ne0zx[0] * v0[2] + ne0zx[1] * v0[0]) + max(0.0, dp[2] * ne0zx[0]) + max(0.0, dp[0] * ne0zx[1])
        de1zx = -(ne1zx[0] * v1[2] + ne1zx[1] * v1[0]) + max(0.0, dp[2] * ne1zx[0]) + max(0.0, dp[0] * ne1zx[1])
        de2zx = -(ne2zx[0] * v2[2] + ne2zx[1] * v2[0]) + max(0.0, dp[2] * ne2zx[0]) + max(0.0, dp[0] * ne2zx[1])
        
        if ((ne0zx[0] * p[2] + ne0zx[1] * p[0]) + de0zx < -EPSILON or
            (ne1zx[0] * p[2] + ne1zx[1] * p[0]) + de1zx < -EPSILON or
            (ne2zx[0] * p[2] + ne2zx[1] * p[0]) + de2zx < -EPSILON):
            return False
        
        return True
    
    def polygon_in_cell(polygon: SchPolygons) -> bool:
        return triangle_intersects_aabb(polygon.vertex1, polygon.vertex2, polygon.vertex3, polygon.normal)

    return [poly for poly in polygons if polygon_in_cell(poly)]


def mesh_to_sch_file(mesh: bpy.types.Mesh, cell_size: tuple[float, float, float]) -> SchFile:
    bounding_min, bounding_max = get_mesh_bounding_box(mesh)

    grid_resolution = (
        math.ceil((bounding_max[0] - bounding_min[0]) / cell_size[0]),
        math.ceil((bounding_max[1] - bounding_min[1]) / cell_size[1]),
        math.ceil((bounding_max[2] - bounding_min[2]) / cell_size[2]),
    )

    polygons = mesh_to_sch_polygons(mesh)

    grid_cell_offsets = []
    grid_cells = []

    total_cells = grid_resolution[0] * grid_resolution[1] * grid_resolution[2]
    for cell_index in range(total_cells):
        cell_coord = grid_1d_index_to_3d(cell_index, grid_resolution)
        cell_polygons = get_polygons_in_cell(polygons, cell_coord, cell_size, bounding_min)

        polygon_indices = [polygons.index(poly) * 60 for poly in cell_polygons]

        grid_cell = SchGridCell(
            polygon_indices=polygon_indices,
            terminator=0xFFFFFFFF
        )
        grid_cells.append(grid_cell)
    
    base_offset = 0x40 + total_cells * 4 - 4 
    current_offset = base_offset
    
    for cell in grid_cells:
        grid_cell_offsets.append(current_offset)
        cell_size_bytes = len(cell.polygon_indices) * 4 + 4
        current_offset += cell_size_bytes 

    header = SchHeader(
        magic=b'STIH', 
        fileSize=0,  
        grid_bound_min=bounding_min,
        grid_bound_max=bounding_max,
        grid_cell_size=cell_size,
        grid_resolution=grid_resolution,
        num_polygons=len(polygons),
        grid_array_offset=0x3C,  
        polygon_array_offset=0 
    )

    sch_file = SchFile(
        header=header,
        grid_cell_offsets=grid_cell_offsets,
        grid_cells=grid_cells,
        polygons=polygons
    )

    return sch_file

def polygons_to_bytes(polygons: list[SchPolygons]) -> bytes:
    data = bytearray()
    for poly in polygons:
        packed = struct.pack(
            '>BBHHH9fffff',
            poly.polyId,
            poly.attrib1,
            poly.flags,
            poly.attrib2,
            poly.attrib3,
            poly.vertex1[0], poly.vertex1[1], poly.vertex1[2],
            poly.vertex2[0], poly.vertex2[1], poly.vertex2[2],
            poly.vertex3[0], poly.vertex3[1], poly.vertex3[2],
            poly.normal[0], poly.normal[1], poly.normal[2],
            poly.distance
        )
        data.extend(packed)
    return bytes(data)

def grid_cells_to_bytes(grid_cells: list[SchGridCell]) -> bytes:
    data = bytearray()
    for cell in grid_cells:
        for index in cell.polygon_indices:
            data.extend(struct.pack('>I', index))
        data.extend(struct.pack('>I', cell.terminator))
    return bytes(data)

def write_sch_file(sch_file: SchFile, file_path: str):
    polygon_buffer = polygons_to_bytes(sch_file.polygons)
    grid_buffer = grid_cells_to_bytes(sch_file.grid_cells)
    grid_offsets_buffer = bytearray()
    for offset in sch_file.grid_cell_offsets:
        grid_offsets_buffer.extend(struct.pack('>I', offset))
    header_size = 64
    final_file_size = header_size + len(grid_offsets_buffer) + len(grid_buffer) + len(polygon_buffer)
    sch_file.header.fileSize = final_file_size
    sch_file.header.polygon_array_offset = header_size + len(grid_offsets_buffer) + len(grid_buffer) - 4
    
    with open(file_path, 'wb') as f:
        f.write(sch_file.header.magic)
        f.write(struct.pack('>I', sch_file.header.fileSize))
        f.write(struct.pack('>fff', *sch_file.header.grid_bound_min))
        f.write(struct.pack('>fff', *sch_file.header.grid_bound_max))
        f.write(struct.pack('>III', *sch_file.header.grid_cell_size))
        f.write(struct.pack('>III', *sch_file.header.grid_resolution))
        f.write(struct.pack('>I', sch_file.header.num_polygons))
        f.write(struct.pack('>I', sch_file.header.grid_array_offset))
        f.write(struct.pack('>I', sch_file.header.polygon_array_offset))
        f.write(grid_offsets_buffer)
        f.write(grid_buffer)
        f.write(polygon_buffer)

class MHTRI_OT_ColorizeByAttribute(bpy.types.Operator):
    bl_idname = "mhtri.colorize_by_attribute"
    bl_label = "Colorize by Attribute"
    bl_options = {'REGISTER', 'UNDO'}

    attribute: bpy.props.StringProperty(name="Attribute") # type: ignore

    def execute(self, context):
        obj = context.active_object
        return {'FINISHED'}


class MHTRI_OT_PaintAttribute(bpy.types.Operator):
    bl_idname = "mhtri.paint_attribute"
    bl_label = "Paint Attribute to Selection"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return {'FINISHED'}


class MHTRI_OT_CopyAttributesFromActive(bpy.types.Operator):
    bl_idname = "mhtri.copy_attributes_from_active"
    bl_label = "Copy from Active Face"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        return {'FINISHED'}


class MHTRI_PT_CollisionPanel(bpy.types.Panel):
    bl_label = "MH Tri Collision"
    bl_idname = "MHTRI_PT_collision_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MH Tri'

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        mesh = bmesh.from_edit_mesh(obj.data) if obj and obj.mode == 'EDIT' else None
        if mesh:
            selected_faces = [f for f in mesh.faces if f.select]
            if selected_faces and len(selected_faces) == 1:
                face = selected_faces[0]

                poly_id_layer = mesh.faces.layers.int.get("polyId")
                attrib1_layer = mesh.faces.layers.int.get("attrib1")
                flags_layer = mesh.faces.layers.int.get("flags")
                attrib2_layer = mesh.faces.layers.int.get("attrib2")
                attrib3_layer = mesh.faces.layers.int.get("attrib3")

                layout.label(text=f"Selected Face ID: {face.index}")
                layout.label(text=f"polyId: {face[poly_id_layer]}")
                layout.label(text=f"attrib1: {face[attrib1_layer]}")
                layout.label(text=f"flags: {face[flags_layer]}")
                layout.label(text=f"attrib2: {face[attrib2_layer]}")
                layout.label(text=f"attrib3: {face[attrib3_layer]}")

                bmesh.update_edit_mesh(obj.data)
                return

        layout.label(text="No face selected or not in edit mode.")


class ImportCollisionFile(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    bl_idname = "import_scene.mhtri_collision"
    bl_label = "Import MH Tri Collision"
    bl_options = {'PRESET', 'UNDO'}

    filter_glob: bpy.props.StringProperty(
        default="*.sch",
        options={'HIDDEN'},
    ) # type: ignore

    def execute(self, context):

        file_path = self.filepath
        polygons = extract_polygons_from_file(file_path)
        mesh = polygons_to_mesh(polygons, mesh_name="MH_Tri_Collision")
        obj = bpy.data.objects.new("MH_Tri_Collision_Object", mesh)
        context.collection.objects.link(obj)
        maya_to_blender_transform(obj)

        polygons = mesh_to_sch_polygons(mesh)
        print(f"Re-extracted {len(polygons)} polygons from the created mesh.")
        print(f"Polygon 0 data: {polygons[0]}")

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
class ExportCollisionFile(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    bl_idname = "export_scene.mhtri_collision"
    bl_label = "Export MH Tri Collision"
    bl_options = {'PRESET'}

    filename_ext = ".sch"

    filter_glob: bpy.props.StringProperty(
        default="*.sch",
        options={'HIDDEN'},
    ) # type: ignore

    cell_size: bpy.props.IntVectorProperty(
        name="Cell Size",
        description="Size of each grid cell",
        default=(1000, 1000, 1000),
        size=3,
    ) # type: ignore

    def get_mesh_objects(self, context):
        items = []
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                items.append((obj.name, obj.name, f"Export {obj.name}"))
        
        if not items:
            items.append(('NONE', "No Mesh Objects", "No mesh objects in scene"))
        
        return items
    
    mesh_object: bpy.props.EnumProperty(
        name="Mesh Object",
        description="Select the mesh object to export",
        items=get_mesh_objects,
    ) # type: ignore

    def execute(self, context):
        file_path = self.filepath
        obj = bpy.data.objects.get(self.mesh_object)
        
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Selected object is not a valid mesh.")
            return {'CANCELLED'}
        
        mesh = obj.data
        sch_file = mesh_to_sch_file(mesh, tuple(self.cell_size))
        write_sch_file(sch_file, file_path)
        
        self.report({'INFO'}, f"Exported MH Tri Collision to {file_path}")
        return {'FINISHED'}

def menu_func_import(self, context):
    self.layout.operator(ImportCollisionFile.bl_idname, text="MH Tri Collision (.sch)")

def menu_func_export(self, context):
    self.layout.operator(ExportCollisionFile.bl_idname, text="MH Tri Collision (.sch)")

classes = (
    MHTRI_OT_ColorizeByAttribute,
    MHTRI_OT_PaintAttribute,
    MHTRI_OT_CopyAttributesFromActive,
    MHTRI_PT_CollisionPanel,
    ImportCollisionFile,
    ExportCollisionFile,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
