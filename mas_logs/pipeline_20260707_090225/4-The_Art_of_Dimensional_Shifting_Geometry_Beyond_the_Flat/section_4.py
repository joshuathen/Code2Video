from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup Title and Lecture Lines
        lecture_lines = [
            'Finding the shortest path across a cube is tricky.',
            'Unfold the 3D surface into a flat 2D net.',
            'Now, a simple straight line reveals the best route.'
        ]
        self.setup_layout("The 'Unfolding' Technique: 3D to 2D Projection", lecture_lines)

        # Colors
        CUBE_COLOR = WHITE
        NET_COLOR = "#D3D3D3"
        SPIDER_COLOR = RED
        FLY_COLOR = BLUE
        PATH_COLOR = GREEN

        # Isometric projection helper
        def get_iso(pts):
            # Maps (x, y, z) to isometric 2D (x', y')
            # x' = x - z, y' = y + 0.5*(x + z)
            return [np.array([(x - z), (y + (x + z) * 0.5), 0]) for x, y, z in pts]

        # Cube geometry (1x1x1 centered at origin)
        pts_3d = [
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5], # 0, 1, 2, 3 (Back)
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]  # 4, 5, 6, 7 (Front)
        ]
        face_indices = {
            "Front": [4, 5, 6, 7], "Top": [7, 6, 2, 3], "Bottom": [0, 1, 5, 4],
            "Left": [0, 4, 7, 3], "Right": [5, 1, 2, 6], "Back": [1, 0, 3, 2]
        }
        face_names = ["Front", "Top", "Bottom", "Left", "Right", "Back"]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Build Isometric Cube Assembly
        faces = VGroup()
        for name in face_names:
            poly = Polygon(*get_iso([pts_3d[i] for i in face_indices[name]]), 
                           stroke_color=WHITE, stroke_width=2, fill_color=CUBE_COLOR, fill_opacity=0.4)
            faces.add(poly)

        # Dots at opposite corners: Spider at P7, Fly at P1
        spider_dot = Dot(get_iso([pts_3d[7]])[0], color=SPIDER_COLOR, radius=0.1)
        fly_dot = Dot(get_iso([pts_3d[1]])[0], color=FLY_COLOR, radius=0.1)
        
        cube_assembly = VGroup(faces, spider_dot, fly_dot)
        self.place_in_area(cube_assembly, "C3", "D4", scale_factor=1.2)
        
        # Labels for 3D View (Issue 37)
        spider_label = Text("Spider", font_size=20, color=SPIDER_COLOR)
        fly_label = Text("Fly", font_size=20, color=FLY_COLOR)
        self.place_at_grid(spider_label, "B2", scale_factor=0.8)
        self.place_at_grid(fly_label, "E4", scale_factor=0.8)

        self.play(FadeIn(cube_assembly), FadeIn(spider_label), FadeIn(fly_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(NET_COLOR)

        # Build Net Assembly
        # Offset centers for T-shape net
        face_net_offsets = {
            "Front": [0, 0], "Top": [0, 1], "Bottom": [0, -1], "Left": [-1, 0], "Right": [1, 0], "Back": [0, -2]
        }
        net_faces = VGroup()
        for name in face_names:
            ox, oy = face_net_offsets[name]
            sq_pts = [[ox-0.5, oy-0.5, 0], [ox+0.5, oy-0.5, 0], [ox+0.5, oy+0.5, 0], [ox-0.5, oy+0.5, 0]]
            poly = Polygon(*sq_pts, stroke_color=WHITE, stroke_width=2, fill_color=NET_COLOR, fill_opacity=0.8)
            net_faces.add(poly)

        # Dots on Net (Spider at Front Top-Left, Fly at Right Bottom-Right)
        net_spider_pos = np.array([-0.5, 0.5, 0])
        net_fly_pos = np.array([1.5, -0.5, 0])
        net_spider = Dot(net_spider_pos, color=SPIDER_COLOR, radius=0.1)
        net_fly = Dot(net_fly_pos, color=FLY_COLOR, radius=0.1)
        
        net_assembly = VGroup(net_faces, net_spider, net_fly)
        # Position net assembly in designated area (Issue 38)
        self.place_in_area(net_assembly, "A2", "F5", scale_factor=0.7)
        
        # Labels for 2D View (Issue 39)
        target_spider_label = Text("Spider", font_size=20, color=SPIDER_COLOR)
        target_fly_label = Text("Fly", font_size=20, color=FLY_COLOR)
        self.place_at_grid(target_spider_label, "A2", scale_factor=0.8)
        self.place_at_grid(target_fly_label, "E5", scale_factor=0.8)

        self.play(
            Transform(cube_assembly, net_assembly),
            Transform(spider_label, target_spider_label),
            Transform(fly_label, target_fly_label),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(PATH_COLOR)
        
        # Draw straight path on the flat net
        # Use current positions of dots within the transformed assembly
        path_line = Line(spider_dot.get_center(), fly_dot.get_center(), color=PATH_COLOR, stroke_width=6)
        
        self.play(Create(path_line))
        self.wait(3)
