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
        # Fetching storyboard data
        title = "Prerequisite: The Three Pillars"
        lines = [
            "Let's define the three pillars of our numerical relationship.",
            "The Base is the value we are multiplying.",
            "The Exponent counts how many times we multiply it."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_BASE = "#00CCFF"
        COLOR_EXPONENT = "#FF9900"
        COLOR_RESULT = "#CC00FF"

        # Asset path
        CIRCLE_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg"

        # === Animation for Lecture Line 1 ===
        # Let's define the three pillars of our numerical relationship.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        base_label = Text("Base", color=COLOR_BASE, font_size=24)
        exp_label = Text("Exponent", color=COLOR_EXPONENT, font_size=24)
        res_label = Text("Result", color=COLOR_RESULT, font_size=24)
        
        # Grid positions for labels
        self.place_at_grid(base_label, "C2")
        self.place_at_grid(exp_label, "C4")
        self.place_at_grid(res_label, "C6")
        
        # Base Icon using Asset
        base_icon = SVGMobject(CIRCLE_ASSET).set_color(COLOR_BASE)
        self.place_at_grid(base_icon, "D2", scale_factor=0.6)
        
        # Exponent symbol (count symbol)
        exp_symbol = MathTex("3", color=COLOR_EXPONENT, font_size=48)
        self.place_at_grid(exp_symbol, "D4")
        
        # Result Icon (Growth Result)
        res_icon = Circle(radius=0.35, color=COLOR_RESULT, fill_opacity=0.8)
        self.place_at_grid(res_icon, "D6")
        
        self.play(
            FadeIn(base_label), FadeIn(exp_label), FadeIn(res_label),
            FadeIn(base_icon), FadeIn(exp_symbol), FadeIn(res_icon)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The Base is the value we are multiplying.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_BASE)
        )
        # Pulse animation for the Base
        self.play(base_icon.animate.scale(1.2), run_time=0.4)
        self.play(base_icon.animate.scale(1/1.2), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The Exponent counts how many times we multiply it.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_EXPONENT)
        )
        
        # Create copies of base_icon to represent multiplying
        # To avoid overlapping with exp_symbol at D4, we place copies at E4 (Issue 28)
        copy1 = base_icon.copy()
        copy2 = base_icon.copy()
        copy3 = base_icon.copy()
        
        copies = VGroup(copy1, copy2, copy3).arrange(RIGHT, buff=0.1)
        self.place_at_grid(copies, "E4", scale_factor=0.6)
        
        self.play(
            TransformFromCopy(base_icon, copies),
            exp_symbol.animate.set_color(YELLOW).scale(1.1),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Move copies to the Result position and transform into Result Icon
        self.play(
            ReplacementTransform(copies, res_icon),
            exp_symbol.animate.set_color(COLOR_EXPONENT).scale(1/1.1),
            res_icon.animate.scale(1.3),
            run_time=1.5
        )
        self.play(res_icon.animate.scale(1/1.3))
        
        self.wait(2)
