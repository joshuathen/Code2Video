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

class Section5Scene(TeachingScene):
    def construct(self):
        # Title and lecture lines fetched from shared state
        title = "Scalar Multiplication: Scaling the Force"
        lines = [
            "Scalars are regular numbers that scale our vectors.",
            "Multiplying a vector changes its magnitude or its direction.",
            "This scaling allows us to stretch or shrink any arrow."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors (Hex per L008)
        CYAN = "#00FFFF"
        WHITE = "#FFFFFF"
        
        # Grid positions for vector animation
        # Vector v components [1, 1]: Tail D2 -> Head C3
        # Vector 3v components [3, 3]: Tail D2 -> Head A5
        start_point = self.grid["D2"]
        end_point_v = self.grid["C3"]
        end_point_3v = self.grid["A5"]
        
        # Preparation of visual elements
        # Asset: Force icon (Issue 26)
        force_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/force.svg")
        force_icon.set_color(CYAN)
        self.place_at_grid(force_icon, "E2", scale_factor=0.6) # Placed below tail D2
        
        # Vector
        vector_v = Arrow(start_point, end_point_v, buff=0, color=CYAN)
        
        # Labels using Text (fallback for MathTex per L022)
        # Issue 39: Move label_v to B2 to avoid overlap
        label_v = Text("v = [1, 1]", color=WHITE)
        self.place_at_grid(label_v, "B2", scale_factor=0.7)
        
        # Issue 40: Move label_3v to A4 to avoid obstruction
        label_3v = Text("3v = [3, 3]", color=WHITE)
        self.place_at_grid(label_3v, "A4", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # "Scalars are regular numbers that scale our vectors."
        self.lecture[0].set_color(CYAN)
        self.play(
            Create(vector_v), 
            Write(label_v),
            FadeIn(force_icon)
        )
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # "Multiplying a vector changes its magnitude or its direction."
        self.lecture[1].set_color(CYAN)
        self.play(
            vector_v.animate.put_start_and_end_on(start_point, end_point_3v),
            run_time=2
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # "This scaling allows us to stretch or shrink any arrow."
        self.lecture[2].set_color(CYAN)
        self.play(Transform(label_v, label_3v))
        self.wait(2.0)
