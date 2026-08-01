from manim import *
import numpy as np

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
        # Data from shared state
        title_text = "The Concept of 'Inverse' Operations"
        lecture_lines = [
            "Mathematics often uses operations that reverse each other.",
            "Addition and subtraction are perfect examples of this.",
            "Can we find a reverse partner for calculus?"
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors for matching
        COLOR_ADD = "#00FF00"  # Green
        COLOR_SUB = "#FF0000"  # Red
        COLOR_CALC = "#00FFFF" # Cyan
        
        # Asset Path
        char_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png"
        
        # === Animation for Lecture Line 1 ===
        # Highlight: Mathematics often uses operations that reverse each other.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Visual for "Reverse" concept: A bidirectional loop
        rev_arrow_1 = Arrow(start=LEFT, end=RIGHT, color=WHITE).shift(UP*0.2)
        rev_arrow_2 = Arrow(start=RIGHT, end=LEFT, color=WHITE).shift(DOWN*0.2)
        rev_group = VGroup(rev_arrow_1, rev_arrow_2)
        # FIX: Issue 49: Positioning and scaling
        self.place_in_area(rev_group, 'C3', 'C4', scale_factor=1.5)
        
        self.play(Create(rev_group))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Highlight: Addition and subtraction are perfect examples of this.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_ADD),
            FadeOut(rev_group)
        )
        
        # Create Number Line
        nl = NumberLine(
            x_range=[0, 10, 1],
            length=4.5,
            include_numbers=True,
            font_size=20,
            color=WHITE
        )
        # FIX: Issue 50: Expand area coverage
        self.place_in_area(nl, 'D1', 'D6')
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/character.png]
        # FIX: Issue 29: Integrate character asset
        try:
            character = ImageMobject(char_asset_path).scale(0.3)
        except Exception:
            # Fallback if image fails to load
            character = Triangle(color=BLUE, fill_opacity=1).scale(0.2)
            
        # Position character at start of number line (n=0)
        character.move_to(nl.n2p(0) + UP * 0.5)
        
        start_label = Text("Start", font_size=18).next_to(nl.n2p(0), DOWN, buff=0.2)
        
        self.play(Create(nl), FadeIn(character), Write(start_label))
        self.wait(0.5)
        
        # Addition: Move character +5 units forward
        add_arrow = Arrow(
            start=nl.n2p(0) + UP * 1.0, 
            end=nl.n2p(5) + UP * 1.0, 
            color=COLOR_ADD, 
            buff=0
        )
        add_text = Text("+5", color=COLOR_ADD, font_size=24).next_to(add_arrow, UP, buff=0.1)
        
        self.play(
            GrowArrow(add_arrow),
            Write(add_text),
            character.animate.move_to(nl.n2p(5) + UP * 0.5),
            run_time=1.5,
            rate_func=rate_functions.smooth
        )
        self.wait(0.5)
        
        # Subtraction: Move character back -5 units to start
        self.play(self.lecture[1].animate.set_color(COLOR_SUB))
        
        sub_arrow = Arrow(
            start=nl.n2p(5) + UP * 1.8, 
            end=nl.n2p(0) + UP * 1.8, 
            color=COLOR_SUB, 
            buff=0
        )
        sub_text = Text("-5", color=COLOR_SUB, font_size=24).next_to(sub_arrow, UP, buff=0.1)
        
        self.play(
            GrowArrow(sub_arrow),
            Write(sub_text),
            character.animate.move_to(nl.n2p(0) + UP * 0.5),
            run_time=1.5,
            rate_func=rate_functions.smooth
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Highlight: Can we find a reverse partner for calculus?
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_CALC),
            FadeOut(nl), FadeOut(character), FadeOut(start_label),
            FadeOut(add_arrow), FadeOut(add_text),
            FadeOut(sub_arrow), FadeOut(sub_text)
        )
        
        # Calculus symbols: Derivative and Integral
        # FIX: Issue 51: Scale symbols
        deriv_sym = MathTex(r"\frac{d}{dx}", color=COLOR_CALC, font_size=60)
        integ_sym = MathTex(r"\int", color=COLOR_CALC, font_size=72)
        
        self.place_at_grid(deriv_sym, 'C2', scale_factor=1.5)
        self.place_at_grid(integ_sym, 'C5', scale_factor=1.5)
        
        # Question mark
        question_mark = Text("?", color=WHITE, font_size=72)
        self.place_in_area(question_mark, "C3", "C4")
        
        # Connecting Double Arrow
        connect_arrow = DoubleArrow(
            start=self.grid["C2"] + RIGHT * 0.7,
            end=self.grid["C5"] + LEFT * 0.7,
            color=WHITE,
            stroke_width=2
        )
        
        self.play(
            FadeIn(deriv_sym),
            FadeIn(integ_sym)
        )
        self.play(
            Create(connect_arrow),
            Write(question_mark)
        )
        self.wait(2)
