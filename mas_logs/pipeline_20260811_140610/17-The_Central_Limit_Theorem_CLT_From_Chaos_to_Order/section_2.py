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
        self.setup_layout("The Problem: What if the Source is Non-Normal?", ["What if the source is not normal?", "Consider a skewed distribution.", "How do sample means behave then?"])
        
        # Data representation: Skewed bars
        skewed_dist = BarChart([1, 2, 3, 5, 8], bar_colors=["#FF8800"], y_axis_config={"include_tip": False})

        # Asset loading
        q_mark_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        q_mark_svg.set_color(WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF8800"))
        # Fix: chart in A3-C6, scale 0.5
        self.place_in_area(skewed_dist, 'A3', 'C6', scale_factor=0.5)
        self.play(Create(skewed_dist))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF8800"))
        self.play(skewed_dist.animate.set_opacity(0.5))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        # Fix: question mark at D4, scale 0.8
        self.place_at_grid(q_mark_svg, 'D4', scale_factor=0.8)
        self.play(FadeIn(q_mark_svg))
        self.wait(2)
