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
            "Time signatures look like math fractions.",
            "The top number counts beats per measure.",
            "The bottom number shows the note value.",
            "Four-four time means four quarter beats.",
            "The pie chart shows the measure's capacity."
        ]
        self.setup_layout("The Time Signature: The Mathematical Fraction", lecture_lines)
        
        # Create Fraction 4/4 and Asset
        fraction = MathTex(r"{4 \over 4}", font_size=120)
        pie_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pie.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(fraction, 'B5', scale_factor=0.6)
        self.place_at_grid(pie_asset, 'B3', scale_factor=0.8)
        self.play(Write(fraction), FadeIn(pie_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(RED))
        top_digit = MathTex("4", font_size=120, color="#FF4500")
        top_digit.move_to(self.grid["B5"])
        self.play(FadeOut(fraction), Write(top_digit))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(BLUE))
        bottom_digit = MathTex("4", font_size=120, color="#1E90FF")
        bottom_digit.move_to(self.grid["B5"])
        self.play(FadeOut(top_digit), Write(bottom_digit))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(GREEN))
        fraction_full = MathTex(r"{4 \over 4}", font_size=120)
        self.place_at_grid(fraction_full, 'B5', scale_factor=0.7)
        self.play(FadeOut(bottom_digit), Write(fraction_full))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(PURPLE))
        pie = VGroup(*[Sector(radius=1.5, angle=TAU/4, start_angle=i*TAU/4, color=c) for i, c in enumerate([BLUE, RED, GREEN, YELLOW])])
        self.place_at_grid(pie, 'E5', scale_factor=0.6)
        self.play(FadeIn(pie_asset), Create(pie))
        self.wait(2)
