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

class Section1Scene(Scene):
    """
    Fixed scene class named Section1Scene. 
    Replaced Tex and MathTex with Text/MarkupText and manual matrix construction 
    to resolve the FileNotFoundError: 'latex' which occurs when no LaTeX 
    distribution is found in the environment.
    """
    def construct(self):
        # 1. Title Setup - Using MarkupText to simulate math styling without LaTeX
        title = MarkupText("Unlocking the Matrix Exponential <i>e<sup>A</sup></i>", font_size=44)
        title.to_edge(UP, buff=0.5)
        
        underline = Line(LEFT, RIGHT).scale(4).next_to(title, DOWN, buff=0.2)
        underline.set_stroke(BLUE, opacity=0.5)

        # 2. Key Definitions - Replacing MathTex with Text and MarkupText
        # We use Unicode characters for Sigma (Σ) and Infinity (∞) for compatibility
        definition_text = VGroup(
            Text("The matrix exponential is defined by the power series:", font_size=32),
            MarkupText(
                "<i>e</i><sup><b>A</b></sup> = Σ<sub><i>n</i>=0</sub><sup>∞</sup> "
                "(<b>A</b><sup><i>n</i></sup> / <i>n</i>!) = <b>I</b> + <b>A</b> + "
                "<b>A</b><sup>2</sup>/2! + <b>A</b><sup>3</sup>/3! + ...", 
                font_size=30
            )
        ).arrange(DOWN, buff=0.4)
        
        definition_text.next_to(underline, DOWN, buff=0.8)

        # 3. Properties Section - Using Text and MarkupText for robust rendering
        properties = VGroup(
            Text("• Fundamental solution for linear differential systems", font_size=28),
            Text("• Generalizes the standard scalar exponential function", font_size=28),
            MarkupText("• If <b>AB</b> = <b>BA</b>, then <i>e</i><sup><b>A+B</b></sup> = <i>e</i><sup><b>A</b></sup><i>e</i><sup><b>B</b></sup>", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        properties.next_to(definition_text, DOWN, buff=1.0)

        # 4. Animation Sequence
        self.play(Write(title))
        self.play(Create(underline))
        self.wait(0.5)
        
        self.play(FadeIn(definition_text[0], shift=UP * 0.2))
        self.play(Write(definition_text[1]))
        self.wait(1)
        
        self.play(
            AnimationGroup(
                *[FadeIn(p, shift=RIGHT * 0.2) for p in properties],
                lag_ratio=0.5
            )
        )
        self.wait(3)
