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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout and lines
        title_str = "Tall Matrices: Moving from 2D to 3D"
        lines = [
            "A three-by-two matrix maps two dimensions into three.",
            "Two input basis vectors land in three-dimensional space.",
            "This creates a tilted plane within the 3D world.",
            "The input is flat, but the output has depth.",
            "We call this embedding a lower dimension into higher."
        ]
        self.setup_layout(title_str, lines)

        # Visual Configuration
        C_V1 = "#FF0000" # Red
        C_V2 = "#00FF00" # Green
        C_AXIS = "#888888" # Gray
        C_PLANE = "#0000FF" # Blue
        C_MATRIX = "#FFFFFF" # White
        
        # Simulated Projection directions for "3D" effect in a 2D scene
        # (Avoiding 3DScene functions as per mandatory constraints)
        x_dir = RIGHT * 0.9
        y_dir = UP * 0.9
        z_dir = (LEFT * 0.4 + UP * 0.4) * 0.9

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(C_MATRIX)
        matrix = MathTex(
            "A = \\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 1 & 1 \\end{bmatrix}",
            color=C_MATRIX
        )
        self.place_at_grid(matrix, 'A2', scale_factor=0.7)

        # Coordinate System setup
        center_pos = self.grid["D4"]
        ax_x = Arrow(center_pos, center_pos + x_dir * 2.2, color=C_AXIS, buff=0, stroke_width=2)
        ax_y = Arrow(center_pos, center_pos + y_dir * 2.2, color=C_AXIS, buff=0, stroke_width=2)
        ax_z = Arrow(center_pos, center_pos + z_dir * 2.2, color=C_AXIS, buff=0, stroke_width=2)
        
        v1 = Arrow(center_pos, center_pos + x_dir, color=C_V1, buff=0)
        v2 = Arrow(center_pos, center_pos + y_dir, color=C_V2, buff=0)
        
        three_d_elements = VGroup(ax_x, ax_y, v1, v2)
        # Issue 31: Place in area to avoid obstructing lecture lines
        self.place_in_area(three_d_elements, 'A2', 'F6', scale_factor=0.85)
        # Re-capture the origin after place_in_area moves everything
        scene_origin = ax_x.get_start()

        self.play(Write(matrix))
        self.play(Create(ax_x), Create(ax_y), GrowArrow(v1), GrowArrow(v2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(C_V1)
        
        # Reveal z-axis and transform vectors to their 3D images (1,0,1) and (0,1,1)
        # Vector 1 target: x_dir + z_dir
        # Vector 2 target: y_dir + z_dir
        v1_target = scene_origin + x_dir + z_dir
        v2_target = scene_origin + y_dir + z_dir

        label_v1 = MathTex("Av_1", color=C_V1)
        label_v2 = MathTex("Av_2", color=C_V2)
        # Issue 32: Positioning labels at specific grid cells
        self.place_at_grid(label_v1, 'C3', scale_factor=0.6)
        self.place_at_grid(label_v2, 'B4', scale_factor=0.6)

        self.play(Create(ax_z))
        self.play(
            v1.animate.put_start_and_end_on(scene_origin, v1_target),
            v2.animate.put_start_and_end_on(scene_origin, v2_target),
            run_time=2
        )
        self.play(Write(label_v1), Write(label_v2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(C_PLANE)
        
        # Tilted plane spanned by v1 and v2
        # Vertices: Origin, v1_target, v1_target + v2_target - origin, v2_target
        plane_vert_3 = v1_target + (v2_target - scene_origin)
        tilted_plane = Polygon(
            scene_origin, v1_target, plane_vert_3, v2_target,
            fill_color=C_PLANE,
            fill_opacity=0.4,
            stroke_color=WHITE,
            stroke_width=1
        )
        
        self.play(Create(tilted_plane))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        
        # Issue 26: Character silhouette asset integration
        char = ImageMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/character.png")
        char.scale(0.4)
        # Start character on the "flat" 2D XY plane (local z=0)
        start_char_pos = scene_origin + x_dir * 0.5 + y_dir * 0.5
        char.move_to(start_char_pos)
        
        self.play(FadeIn(char))
        self.wait(0.5)
        
        # Move onto the tilted plane (embedding)
        # Target pos: mid-point of the parallelogram
        end_char_pos = (scene_origin + plane_vert_3) / 2
        self.play(char.animate.move_to(end_char_pos), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        
        # Issue 33: Red dot target point
        red_dot = Dot(color="#FF0000")
        self.place_at_grid(red_dot, 'A6', scale_factor=0.5)
        
        self.play(FadeIn(red_dot))
        self.play(Indicate(red_dot, color=C_V1))
        self.wait(2)
