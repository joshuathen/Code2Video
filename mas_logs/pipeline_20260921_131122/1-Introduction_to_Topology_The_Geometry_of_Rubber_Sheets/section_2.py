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
        self.setup_layout("Prerequisite Concept: Homeomorphism", [
            "Homeomorphism defines topological equivalence.",
            "Transform objects by continuous deformation.",
            "We cannot tear or glue.",
            "Square stretches into circle.",
            "They are equivalent shapes."
        ])
        
        # Load assets
        square_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        circle_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        square_label = Text("Square", color=WHITE, font_size=24)
        circle_label = Text("Circle", color=WHITE, font_size=24)
        x_mark = Cross(color=RED).scale(0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(square_img, 'C3', scale_factor=0.6)
        self.place_at_grid(square_label, 'C4', scale_factor=0.6)
        square_label.next_to(square_img, DOWN)
        self.play(FadeIn(square_img), Write(square_label))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(circle_img, 'D3', scale_factor=0.6)
        self.place_at_grid(circle_label, 'D4', scale_factor=0.6)
        circle_label.next_to(circle_img, DOWN)
        self.play(FadeIn(circle_img), Write(circle_label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_at_grid(x_mark, 'C3', scale_factor=1.0)
        self.play(FadeIn(x_mark))
        self.play(FadeOut(x_mark))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.play(
            ReplacementTransform(square_img, circle_img),
            square_img.animate.set_color("#32CD32"),
            circle_img.animate.set_color("#32CD32")
        )
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        homeo_label = Text("Homeomorphic", color="#00CED1", font_size=32)
        self.place_at_grid(homeo_label, 'B4', scale_factor=0.8)
        self.play(Write(homeo_label))
        self.wait(1)
