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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "An orbit is a sequence of points.",
            "Fixed points occur where the function output repeats.",
            "Stability depends on the local derivative value."
        ]
        self.setup_layout("The Mechanism of Iteration", lecture_lines)
        
        # Elements
        func_text = MathTex("f(z) = z^2 + c", color="#FFFFFF")
        orbit_text = MathTex("z, f(z), f(f(z)), \\dots", color="#FFFF00")
        dot = Dot(color="#FF0000")
        
        # Load asset icon
        icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        # Note: Since none.svg likely doesn't exist, we skip SVG creation if path is invalid or empty
        # or use a placeholder if it were a real file. Assuming standard loading for assets.
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(func_text, "B4", scale_factor=1.0)
        self.play(Write(func_text))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_in_area(orbit_text, "C4", "C6", scale_factor=0.8)
        self.play(Write(orbit_text))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(dot, "E5", scale_factor=0.8)
        self.play(FadeIn(dot))
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        self.play(dot.animate.shift(RIGHT * 0.5).shift(UP * 0.5))
        self.wait(2)
