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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Imagine a squirrel gathering nuts daily.",
            "Velocity describes how fast it travels.",
            "The integral tracks total distance.",
            "These are inverse operations.",
            "Like unwrapping a gift, we reverse."
        ]
        self.setup_layout("Intuitive Hook: The Velocity-Position Analogy", lecture_lines)
        
        # Assets
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg", color="#FFD700")
        nuts = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/nuts.svg", color="#FF4500")
        velocity_eq = MathTex("v = \\frac{dx}{dt}", color="#FFFFFF")
        
        # === Animation for Lecture Line 1 ===
        self.add(self.lecture[0])
        self.place_at_grid(squirrel, "E3", scale_factor=0.5)
        self.play(FadeIn(squirrel))
        
        # === Animation for Lecture Line 2 ===
        self.add(self.lecture[1])
        self.place_at_grid(velocity_eq, "B5", scale_factor=1.0)
        self.play(Write(velocity_eq))
        
        # === Animation for Lecture Line 3 ===
        self.add(self.lecture[2])
        self.place_at_grid(nuts, "E4", scale_factor=0.4)
        self.play(Flash(nuts, color="#FF4500"))
        
        # === Animation for Lecture Line 4 ===
        self.add(self.lecture[3])
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.add(self.lecture[4])
        self.wait(1)
