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

class Section7Scene(TeachingScene):
    def construct(self):
        # Configuration
        TITLE = "Summary: Efficiency through Geometry"
        LINES = [
            "Pure series exploit the inherent symmetry of physical systems.",
            "Symmetry reduces our computational work by half.",
            "Elegant geometry makes complex math much more efficient."
        ]
        
        # Colors based on storyboard/lecture consistency
        COLOR_COMPLEX = WHITE
        COLOR_SIMPLE = WHITE
        COLOR_SAVING = "#00FF00"
        COLOR_STRING = "#ADD8E6"
        COLOR_FINAL = WHITE

        self.setup_layout(TITLE, LINES)
        
        # === Animation for Lecture Line 1 ===
        # Formulas comparing complexity
        complex_tex = MathTex(
            r"f(x) = a_0 + \sum_{n=1}^\infty \left[ a_n \cos\left(\frac{n\pi x}{L}\right) + b_n \sin\left(\frac{n\pi x}{L}\right) \right]",
            color=COLOR_COMPLEX
        )
        simple_tex = MathTex(
            r"f(x) = \sum_{n=1}^\infty b_n \sin\left(\frac{n\pi x}{L}\right)",
            color=COLOR_SIMPLE
        )
        
        # Visual Anchor Positioning (Applying Fixes for Issues 44, 45, 46)
        self.place_in_area(complex_tex, 'B1', 'C2', scale_factor=0.35)
        self.place_in_area(simple_tex, 'B4', 'C5', scale_factor=0.45)
        
        vs_label = Text("vs", color=WHITE, font_size=20)
        self.place_at_grid(vs_label, 'B3', scale_factor=0.8)
        
        self.lecture[0].set_color(COLOR_COMPLEX)
        self.play(
            Write(complex_tex), 
            Write(vs_label), 
            Write(simple_tex), 
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlighting the 50% reduction in work
        less_math_label = Text("50% Less Math", color=COLOR_SAVING, font_size=24)
        self.place_at_grid(less_math_label, 'D2', scale_factor=0.9)
        
        # Arrow from complex side towards simple side to indicate reduction
        arrow = Arrow(
            start=self.grid["C2"],
            end=self.grid["C4"],
            color=COLOR_SAVING,
            buff=0.1
        )
        
        self.lecture[1].set_color(COLOR_SAVING)
        self.play(FadeIn(less_math_label), Create(arrow), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Physical symmetry with a vibrating string
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg]
        try:
            string_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg")
        except Exception:
            # Fallback for local development or missing assets
            string_svg = Line(LEFT, RIGHT, color=COLOR_STRING)
            
        string_svg.set_color(COLOR_STRING)
        self.place_in_area(string_svg, "E2", "E5", scale_factor=1.2)
        
        # Closing text: 'Symmetry is efficiency'
        closing_text = Text("Symmetry is efficiency", color=COLOR_FINAL, font_size=28)
        self.place_in_area(closing_text, "F2", "F5", scale_factor=0.8)
        
        self.lecture[2].set_color(COLOR_STRING)
        
        # Animation: Create string and text, then pulse string to simulate vibration
        self.play(
            Create(string_svg),
            Write(closing_text)
        )
        
        # Simulate vibration using stretch
        for _ in range(2):
            self.play(
                string_svg.animate.stretch(1.3, 1),
                run_time=0.4,
                rate_func=there_and_back
            )
            self.play(
                string_svg.animate.stretch(0.7, 1),
                run_time=0.4,
                rate_func=there_and_back
            )
        
        self.wait(2)
