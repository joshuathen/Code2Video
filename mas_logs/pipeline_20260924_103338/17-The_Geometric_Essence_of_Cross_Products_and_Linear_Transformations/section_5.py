from manim import *

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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Cross products help define surface normal vectors.",
            "Normals are essential for calculating light reflections.",
            "This enables realistic rendering of 3D models."
        ]
        self.setup_layout("Application: Normal Vectors and Surfaces", lecture_lines)
        
        # Elements
        plane = Square(side_length=2, color="#FFFFFF", fill_opacity=0.3)
        normal_vec = Arrow(start=ORIGIN, end=UP*1.5, color="#FFFFFF", buff=0)
        plane_normal_group = VGroup(plane, normal_vec)
        
        # Assets
        flashlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/flashlight.svg", color="#00FF00")
        mirror = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg", color="#FFD700")
        
        # Labels
        n_label = Text("n", font_size=24, color=WHITE)
        n_label.next_to(normal_vec.get_end(), UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_in_area(plane_normal_group, 'C4', 'E6', scale_factor=0.6)
        self.add(n_label)
        self.play(Create(plane), GrowArrow(normal_vec))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        self.place_at_grid(flashlight, 'B4', scale_factor=0.5)
        n_dot_L = MathTex(r"n \cdot L", color="#00FF00", font_size=32)
        self.place_at_grid(n_dot_L, 'D4', scale_factor=0.8)
        self.play(FadeIn(flashlight), Write(n_dot_L))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.place_at_grid(mirror, 'B5', scale_factor=0.5)
        normal_vec.set_color("#FFD700")
        n_label.set_color("#FFD700")
        self.play(
            Rotate(plane_normal_group, angle=PI/6, about_point=plane.get_center()),
            FadeIn(mirror),
            Rotate(normal_vec, angle=PI/6)
        )
        self.wait(2)
