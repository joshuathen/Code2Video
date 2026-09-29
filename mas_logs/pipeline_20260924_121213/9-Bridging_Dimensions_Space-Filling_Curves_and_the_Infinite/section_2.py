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
        self.setup_layout("The Peano Construction: A Recursive Logic", [
            "Giuseppe Peano discovered a space-filling path.",
            "Iteratively divide the square into smaller parts.",
            "Connect the path through each iteration.",
            "The path densifies with each step.",
            "This recursion approaches the infinite limit."
        ])

        # Define assets
        square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color=WHITE)
        self.place_in_area(square, "B2", "D5", scale_factor=0.8)
        self.add(square)

        # === Animation for Lecture Line 1 ===
        line = Line(start=square.get_left(), end=square.get_right(), color="#FFFFFF")
        self.place_in_area(line, "B2", "D5", scale_factor=0.8)
        self.play(Create(line), self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        highlight = Rectangle(width=3, height=3, color="#FF00FF", fill_opacity=0.2)
        self.place_in_area(highlight, "B2", "D5", scale_factor=0.85)
        self.play(FadeIn(highlight), self.lecture[1].animate.set_color("#FF00FF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        label = Text("1D vs 2D", font_size=24, color="#00FFFF")
        self.place_at_grid(label, "A4", scale_factor=0.9)
        self.play(Write(label), self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(line.animate.scale(0.5), self.lecture[3].animate.set_color("#FF0000"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Re-using square asset as requested in animation for lecture line 5
        square_inf = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg", color=WHITE)
        self.place_in_area(square_inf, "B2", "D5", scale_factor=1.0)
        self.play(FadeIn(square_inf), self.lecture[4].animate.set_color("#FFFFFF"))
        self.wait(1)
