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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Dot products measure how well patterns match.",
            "High values indicate strong pattern alignment.",
            "Convolution acts as a powerful feature detector."
        ]
        self.setup_layout("Prerequisite Sync: Dot Products", lecture_lines)
        
        # Define objects
        # Using SVGMobject for asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg
        v1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg", color=WHITE)
        v2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg", color=WHITE)
        label_dot = MathTex(r"a \cdot b = \sum a_i b_i", color=WHITE)
        
        group_vectors = VGroup(v1, v2)
        
        # === Animation for Lecture Line 1 ===
        # Use place_in_area as requested for group_vectors
        self.place_in_area(group_vectors, 'B2', 'D4', scale_factor=0.8)
        # Apply specific grid positions as requested
        self.place_at_grid(v1, 'C2', scale_factor=0.6)
        self.place_at_grid(v2, 'D2', scale_factor=0.6)
        
        self.play(Create(v1), Create(v2))
        self.lecture[0].set_color("#00FFFF")
        
        # === Animation for Lecture Line 2 ===
        # Shift label_dot to B5
        self.place_at_grid(label_dot, 'B5', scale_factor=0.7)
        self.play(Write(label_dot))
        self.lecture[1].set_color("#FFD700")
        
        # === Animation for Lecture Line 3 ===
        highlight = label_dot.copy().set_color("#FFD700")
        self.play(ReplacementTransform(label_dot.copy(), highlight))
        self.lecture[2].set_color("#FF69B4")
        self.wait(2)
