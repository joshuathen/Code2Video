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
        self.setup_layout("Summary and Synthesis", [
            "Real numbers focus on total magnitude.",
            "2-adic numbers focus on divisibility structure.",
            "Perspective changes what convergence means."
        ])
        
        # --- Visual Elements ---
        circle = Circle(radius=0.5, color=YELLOW).set_stroke(width=4)
        square = Square(side_length=0.8, color=BLUE).set_stroke(width=4)
        
        real_lbl = Text("Real", color=YELLOW, font_size=24)
        padic_lbl = Text("2-adic", color=BLUE, font_size=24)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(circle, 'B2', scale_factor=0.8)
        self.place_at_grid(real_lbl, 'C2', scale_factor=0.6)
        self.play(Create(circle), Write(real_lbl))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.place_at_grid(square, 'B5', scale_factor=0.8)
        self.place_at_grid(padic_lbl, 'C5', scale_factor=0.6)
        self.play(Create(square), Write(padic_lbl))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        connector = Line(self.grid['B2'], self.grid['B5'], color=WHITE)
        self.place_in_area(connector, 'B2', 'B5', scale_factor=1.0)
        self.play(Create(connector))
        self.wait(2)
