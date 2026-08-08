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
            "Integration is the reverse process of differentiation.",
            "Accumulation function tracks area under the curve.",
            "Area growth rate equals the function height.",
            "This confirms the Fundamental Theorem of Calculus.",
            "It bridges local change and total accumulation."
        ]
        self.setup_layout("The Core Mechanism: The Fundamental Theorem of Calculus", lecture_lines)
        
        # Setup visual elements
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        func = axes.plot(lambda x: 0.2 * x**2 + 0.5, color=BLUE)
        area = axes.get_area(func, x_range=[0, 4], color=GREEN, opacity=0.3)
        label = MathTex(r"f(t)", color=BLUE).next_to(func, UP)
        
        visual_group = VGroup(axes, func, area, label)
        # Applying requested layout fix from issues 28 and 43: B4 to E6
        self.place_in_area(visual_group, 'B4', 'E6', scale_factor=0.55)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(axes), Create(func))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(Create(area))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        v_line = Line(axes.c2p(4, 0), axes.c2p(4, func.underlying_function(4)), color=WHITE)
        self.play(Create(v_line))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GOLD))
        ftc = MathTex(r"\int_a^x f(t) dt = F(x) - F(a)", color=GOLD)
        # Applying requested layout fix from issues 27 and 42: D4
        self.place_at_grid(ftc, 'D4', scale_factor=0.7)
        self.play(Write(ftc))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.play(Indicate(ftc))
