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
        lecture_lines = ["Roots are just inverse growth.", "Roots are fractional exponents.", "Example: Square root of 9 is 9^(1/2)."]
        self.setup_layout("Roots as Fractional Exponents", lecture_lines)
        
        # Load Assets
        seed = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/seed.svg")
        plant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plant.svg")
        
        # === Animation for Lecture Line 1 ===
        # Display radical sqrt(x) in #FF99FF alongside seed.
        radical = MathTex(r"\\sqrt{x}", color="#FF99FF")
        self.place_in_area(radical, 'B3', 'C5', scale_factor=1.2)
        self.place_at_grid(seed, 'A4', scale_factor=0.3)
        
        self.play(Write(radical), FadeIn(seed))
        self.play(self.lecture[0].animate.set_color("#FF99FF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show conversion: sqrt(x) = x^(1/2) in #FFFFFF.
        conversion = MathTex(r"\\sqrt{x} = x^{1/2}", color=WHITE)
        self.place_in_area(conversion, 'B3', 'C5', scale_factor=1.2)
        
        self.play(Transform(radical, conversion))
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight the exponent 1/2 in #FFCC00 alongside plant.
        example_highlight = MathTex(r"\\sqrt{9} = 9^{1/2}", color=WHITE)
        example_highlight.set_color_by_tex("1/2", "#FFCC00")
        
        self.place_in_area(example_highlight, 'D3', 'E5', scale_factor=1.0)
        self.place_at_grid(plant, 'F4', scale_factor=0.3)
        
        self.play(Write(example_highlight), FadeIn(plant))
        self.play(self.lecture[2].animate.set_color("#FFCC00"))
        self.wait(2)
