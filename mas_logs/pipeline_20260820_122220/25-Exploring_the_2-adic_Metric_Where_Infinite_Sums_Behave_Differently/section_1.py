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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Usual Notion of Convergence", 
                          ["A series is the limit of partial sums.", 
                           "Standard distance shrinks towards zero.", 
                           "Geometric series converge on a number line."])
        
        # Initially hide non-first lines
        for i in range(1, 3):
            self.lecture[i].set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        s_n_label = Text("S_n", font_size=24, color=WHITE)
        # Placeholder asset: none.svg
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        icon_group = VGroup(s_n_label, asset_icon).arrange(RIGHT, buff=0.2)
        self.place_at_grid(icon_group, 'B4', scale_factor=0.7)
        
        number_line = NumberLine(x_range=[0, 2, 0.5], length=4, include_numbers=True)
        # Updated per feedback (Issue 20)
        self.place_in_area(number_line, 'A3', 'B6', scale_factor=0.9)
        
        dot = Dot(color=YELLOW)
        dot.move_to(number_line.n2p(0))
        
        self.play(FadeIn(icon_group))
        self.play(Create(number_line))
        self.play(FadeIn(dot))
        
        partial_sums = [0.5, 0.75, 0.875, 0.9375, 1.0]
        for val in partial_sums:
            self.play(dot.animate.move_to(number_line.n2p(val)), run_time=0.3)

        self.lecture[0].set_color(BLUE)
        self.lecture[1].set_opacity(1)

        # === Animation for Lecture Line 2 ===
        # Highlight distance shrinking
        dist_brace = Brace(Line(start=number_line.n2p(0.9375), end=number_line.n2p(1)), UP)
        dist_label = Text("d(s_n, s)", font_size=18).next_to(dist_brace, UP)
        
        # Arrow for distance
        arrow = Arrow(start=number_line.n2p(0.9375), end=number_line.n2p(1), color=GREEN)
        
        self.play(Create(dist_brace), Write(dist_label), Create(arrow))
        self.play(FadeOut(dist_brace), FadeOut(dist_label), FadeOut(arrow))
        
        self.lecture[1].set_color(GREEN)
        self.lecture[2].set_opacity(1)

        # === Animation for Lecture Line 3 ===
        # Showing the geometric series segment
        rect = Rectangle(width=4, height=0.2, color=YELLOW_D, fill_opacity=0.5)
        # Updated per feedback (Issues 21, 22, 35)
        self.place_in_area(rect, 'D3', 'E6', scale_factor=0.8)
        self.play(Create(rect))
        self.wait(1)
        
        self.lecture[2].set_color(RED)
        self.wait(2)
